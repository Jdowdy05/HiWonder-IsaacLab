import hashlib
import io
import json
import tarfile
from pathlib import Path
import zipfile
import os
import shutil
import subprocess
import sys

import pytest


def sha(data):
    return hashlib.sha256(data).hexdigest()


def make_cache_archive(path, bad_name=None, symlink=False):
    contract = b'{"normalization":"original"}\n'
    receipt = json.dumps({'contract_sha256': sha(contract), 'files': [
        {'path': 'cache/manifest.json', 'bytes': len(contract), 'sha256': sha(contract)}]}).encode()
    with tarfile.open(path, 'w:gz') as archive:
        for name, data in [('cache/manifest.json', contract), ('cache/portable-cache-receipt.json', receipt)]:
            info = tarfile.TarInfo(name); info.size = len(data); archive.addfile(info, io.BytesIO(data))
        if bad_name:
            info = tarfile.TarInfo(bad_name)
            if symlink:
                info.type = tarfile.SYMTYPE; info.linkname = '../../outside'
            archive.addfile(info, io.BytesIO(b''))
    return sha(contract)


def test_download_checks_hash_before_committing_file(tmp_path, monkeypatch):
    import vast_assets
    monkeypatch.setattr(vast_assets.urllib.request, 'urlopen', lambda *a, **k: io.BytesIO(b'correct'))
    target = tmp_path/'asset.bin'
    vast_assets.download_checked('https://example.org/asset', target, sha(b'correct'), 7)
    assert target.read_bytes() == b'correct'
    with pytest.raises(ValueError, match='hash'):
        vast_assets.download_checked('https://example.org/asset', tmp_path/'bad.bin', sha(b'wrong!!'), 7)
    assert not (tmp_path/'bad.bin').exists()


def test_existing_unexpected_asset_is_preserved(tmp_path):
    from vast_assets import download_checked
    target = tmp_path/'asset';target.write_bytes(b'user data')
    with pytest.raises(ValueError):
        download_checked('https://example.org/asset', target, sha(b'other'), 5)
    assert target.read_bytes() == b'user data'


def test_download_retries_transient_network_failure(tmp_path, monkeypatch):
    import vast_assets
    import urllib.error
    calls=[]
    def open_response(*args, **kwargs):
        calls.append(1)
        if len(calls)==1:raise urllib.error.URLError('temporary network failure')
        return io.BytesIO(b'correct')
    monkeypatch.setattr(vast_assets.urllib.request,'urlopen',open_response)
    monkeypatch.setattr(vast_assets.time,'sleep',lambda seconds:None)
    target=tmp_path/'asset'
    vast_assets.download_checked('https://example.org/asset',target,sha(b'correct'),7)
    assert len(calls)==2 and target.read_bytes()==b'correct'


def test_cache_restore_preserves_contract_and_is_idempotent(tmp_path):
    from vast_assets import restore_cache
    archive = tmp_path/'cache.tar.gz'; contract_sha = make_cache_archive(archive)
    workspace = tmp_path/'act';workspace.mkdir()
    restore_cache(archive, workspace, contract_sha)
    before = (workspace/'cache/manifest.json').read_bytes()
    restore_cache(archive, workspace, contract_sha)
    assert sha(before) == contract_sha
    assert (workspace/'cache/manifest.json').read_bytes() == before


@pytest.mark.parametrize('name,symlink', [('cache/../../outside',False),('/outside',False),('other/file',False),('cache/link',True)])
def test_archive_cannot_escape_or_create_links(tmp_path, name, symlink):
    from vast_assets import restore_cache
    archive = tmp_path/'cache.tar.gz';contract_sha=make_cache_archive(archive,name,symlink)
    workspace=tmp_path/'act';workspace.mkdir()
    with pytest.raises(ValueError):restore_cache(archive,workspace,contract_sha)
    assert not (workspace/'cache').exists()


def test_checkpoint_restores_only_four_model_files(tmp_path):
    from vast_assets import restore_checkpoint
    archive=tmp_path/'model.zip'; model=b'test weights';contract=b'original contract'
    with zipfile.ZipFile(archive,'w') as z:
        for name,data in [('config.json',b'{}'),('model.safetensors',model),('data_contract.json',contract),('experiment_state.json',b'{}')]:
            z.writestr('trial-02/best/'+name,data)
        z.writestr('../../outside.py',b'never extracted')
        z.writestr('train_act.py',b'old archive code is not installed')
    workspace=tmp_path/'act';workspace.mkdir()
    restore_checkpoint(archive,workspace,sha(model),sha(contract))
    assert len(list((workspace/'checkpoint').iterdir()))==4
    assert not (tmp_path/'outside.py').exists()
    assert (workspace/'checkpoint/model.safetensors').read_bytes()==model


def test_act_launch_is_guarded_and_uses_separate_environment(tmp_path):
    from run_act import build_plan
    workspace=tmp_path/'act';source=tmp_path/'source'
    plan=build_plan(workspace,source,steps=100,smoke=False,from_scratch=False)
    command=plan['command']
    assert command[0]==str(workspace/'.venv/bin/python')
    assert 'guard_trial.py' in command[2]
    assert command[command.index('--seconds')+1]=='1800'
    assert '--initialize-from' in command
    assert not Path(plan['output']).exists()
    with pytest.raises(ValueError):build_plan(workspace,source,steps=0)


def test_template_startup_pins_source_and_verifies_download():
    from render_template import render
    result=render('1'*40,'a'*64)
    assert '/'+('1'*40)+'/deploy/vast/provision.sh' in result['onstart']
    assert 'sha256sum -c -' in result['onstart'] and 'a'*64 in result['onstart']
    assert result['environment']['OMNI_KIT_ACCEPT_EULA']=='NO'
    assert result['private'] is True and result['public_ports']==[]
    assert result['disk_gb']==250
    with pytest.raises(ValueError):render(';unexpected shell','a'*64)


def test_template_filters_match_web_editor_resource_units():
    import re
    from render_template import render
    values=render('1'*40,'a'*64)
    fields={name:(op,value) for name,op,value in re.findall(r'(\w+)(>=|=)(\S+)',values['extra_filters'])}
    # The web editor stores RAM in MiB and cannot round-trip the dotted driver predicate.
    assert 'driver_version' not in fields and 'driver_vers' not in fields
    assert float(fields['gpu_ram'][1])/1024 == 23
    assert float(fields['cpu_ram'][1])/1024 == 64
    assert fields['reliability2'] == ('>=','0.99')


@pytest.mark.parametrize('accept_license,completed,failed_fetch', [(False,False,False),(True,False,False),(False,True,False),(True,True,False),(False,False,True)])
def test_dual_provision_stages_workspaces_and_gates_only_rl(tmp_path, accept_license, completed, failed_fetch):
    script=Path(__file__).with_name('provision.sh').resolve()
    assert 'HIWONDER_ROOT' in script.read_text(), 'Provisioner must support a confined test workspace'
    bash=Path('C:/Program Files/Git/bin/bash.exe') if os.name=='nt' else Path(shutil.which('bash') or '')
    if not bash.is_file():pytest.skip('Bash is unavailable')
    tools=tmp_path/'bin';tools.mkdir();root=tmp_path/'workspace';calls=tmp_path/'calls.log'
    programs={
      'uname': '#!/bin/bash\nif [[ $1 == -s ]]; then echo Linux; else echo x86_64; fi\n',
      'id': '#!/bin/bash\necho 0\n', 'flock':'#!/bin/bash\nexit 0\n', 'vulkaninfo':'#!/bin/bash\nexit 0\n',
      'git': '''#!/bin/bash
if [[ $1 == init ]]; then mkdir -p "$2/.git"; exit 0; fi
path=$2; shift 2
case "$1" in
 remote) [[ $2 != get-url ]] || echo https://github.com/Jdowdy05/HiWonder-IsaacLab.git ;;
 fetch)
  if [[ $FAIL_FIRST_FETCH == 1 && $path == */hiwonder-project && ! -f "$path/.git/failed-once" ]]; then touch "$path/.git/failed-once"; exit 1; fi
  printf '%s' "${@: -1}" > "$path/.git/fetched" ;;
 checkout) cp "$path/.git/fetched" "$path/.git/pin" ;;
 rev-parse) [[ -f "$path/.git/pin" ]] || exit 128; cat "$path/.git/pin" ;;
esac
exit 0
''',
      'python3': '''#!/bin/bash
if [[ $1 == -c ]]; then exec "$REAL_PYTHON" "$@"; fi
if [[ $1 == *verify_install.py ]]; then echo verify-install >> "$CALL_LOG"; exit 0; fi
phase=unknown
while [[ $# -gt 0 ]]; do
 if [[ $1 == --workspace ]]; then [[ -d "$2" ]] || exit 41; fi
 if [[ $1 == --phase ]]; then phase=$2; fi
 shift
done
echo preflight:$phase >> "$CALL_LOG"
''',
      'bash': '#!/bin/bash\nprintf "%s\\n" "$*" >> "$CALL_LOG"\n',
    }
    for name,body in programs.items():
        p=tools/name;p.write_bytes(body.encode());p.chmod(0o755)
    env=os.environ.copy();env.update(HIWONDER_ROOT=root.as_posix(),CALL_LOG=calls.as_posix(),
                                   OMNI_KIT_ACCEPT_EULA='YES' if accept_license else 'NO',
                                   FAIL_FIRST_FETCH='1' if failed_fetch else '0',REAL_PYTHON=sys.executable)
    if completed:
        sdk=root/'hiwonder/IsaacLab-HiWonder/.venv/bin';sdk.mkdir(parents=True)
        (sdk/'python').write_bytes(b'#!/bin/bash\necho preflight:runtime >> "$CALL_LOG"\n');(sdk/'python').chmod(0o755)
        (root/'hiwonder/bootstrap-status.json').write_text('{"exit_code":0}')
    wrapper='export PATH="$(cygpath -u "$1"):$PATH"; exec /usr/bin/bash "$2" "$3"' if os.name=='nt' else 'export PATH="$1:$PATH"; exec /bin/bash "$2" "$3"'
    result=subprocess.run([str(bash),'-c',wrapper,'fixture',str(tools),str(script),'1'*40],
                          env=env,capture_output=True,text=True,timeout=20)
    if failed_fetch:
        assert result.returncode != 0
        result=subprocess.run([str(bash),'-c',wrapper,'fixture',str(tools),str(script),'1'*40],
                              env=env,capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stdout+result.stderr
    lines=calls.read_text().splitlines()
    assert lines[0]==('verify-install' if completed else 'preflight:bootstrap')
    assert any('bootstrap_act.sh' in line for line in lines)
    assert ('cloud/bootstrap.sh' in '\n'.join(lines)) == (accept_license and not completed)
    if completed:assert 'preflight:runtime' in lines and 'preflight:bootstrap' not in lines
    assert (root/'act').is_dir() and (root/'hiwonder').is_dir()
