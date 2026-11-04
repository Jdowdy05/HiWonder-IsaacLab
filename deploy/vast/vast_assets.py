"""Restore the pinned public ACT cache/checkpoint without changing its data contract."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def destination(root, name):
    relative = PurePosixPath(name)
    if not name or '\\' in name or relative.is_absolute() or '..' in relative.parts or ':' in name:
        raise ValueError('Unsafe archive/cache path')
    root = Path(root).resolve()
    target = root.joinpath(*relative.parts)
    if not target.resolve().is_relative_to(root):
        raise ValueError('Archive/cache path escapes its workspace')
    return target


def download_checked(url, target, expected_sha, expected_size):
    target = Path(target)
    if target.exists():
        if target.stat().st_size != expected_size or sha256(target) != expected_sha:
            raise ValueError(f'Existing asset size/hash differs; preserved {target}')
        return target
    if not url.startswith('https://'):
        raise ValueError('Expected an HTTPS asset URL')
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.part')
    request = urllib.request.Request(url, headers={'User-Agent': 'HiWonder-Vast-Setup/1'})
    for attempt in range(3):
        received = 0
        try:
            with urllib.request.urlopen(request, timeout=90) as response, partial.open('wb') as out:
                for block in iter(lambda: response.read(1024 * 1024), b''):
                    received += len(block)
                    if received > expected_size:
                        raise ValueError('Asset exceeds expected byte count')
                    out.write(block)
            break
        except (OSError, urllib.error.URLError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    if received != expected_size or sha256(partial) != expected_sha:
        raise ValueError('Downloaded asset byte count/hash mismatch')
    if target.exists():
        raise FileExistsError(target)
    partial.rename(target)
    return target


def validate_cache(workspace, contract_sha):
    workspace = Path(workspace).resolve()
    cache = workspace / 'cache'
    receipt = json.loads((cache / 'portable-cache-receipt.json').read_text(encoding='utf-8'))
    if receipt['contract_sha256'] != contract_sha or sha256(cache / 'manifest.json') != contract_sha:
        raise ValueError('Prepared cache contract differs from the checkpoint')
    for entry in receipt['files']:
        if not entry['path'].startswith('cache/'):
            raise ValueError('Unexpected cache receipt path')
        path = destination(workspace, entry['path'])
        if not path.is_file() or path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
            raise ValueError(f'Prepared cache file differs: {entry["path"]}')


def restore_cache(archive, workspace, contract_sha):
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    target = workspace / 'cache'
    if target.exists():
        validate_cache(workspace, contract_sha)
        return
    # All recursively cleaned/moved paths are newly created inside this explicit workspace.
    with tempfile.TemporaryDirectory(prefix='.vast-cache-', dir=workspace) as temporary:
        stage = Path(temporary).resolve()
        assert stage.is_relative_to(workspace) and stage != workspace
        with tarfile.open(archive, 'r:gz') as tar:
            members = tar.getmembers()
            for member in members:
                if not member.name.startswith('cache/') or not member.isfile():
                    raise ValueError('Unexpected archive member or link')
                destination(stage, member.name)
            tar.extractall(stage, members=members, filter='data')
        validate_cache(stage, contract_sha)
        if target.exists():
            raise FileExistsError(target)
        (stage / 'cache').rename(target)


CHECKPOINT_FILES = ('config.json', 'model.safetensors', 'data_contract.json', 'experiment_state.json')


def restore_checkpoint(archive, workspace, model_sha, contract_sha):
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    target = workspace / 'checkpoint'

    def validate(path):
        if not all((path / name).is_file() for name in CHECKPOINT_FILES):
            raise ValueError('Incomplete checkpoint; existing files preserved')
        if sha256(path / 'model.safetensors') != model_sha or sha256(path / 'data_contract.json') != contract_sha:
            raise ValueError('Checkpoint weights/data contract differ')

    if target.exists():
        validate(target)
        return
    with tempfile.TemporaryDirectory(prefix='.vast-model-', dir=workspace) as temporary:
        stage = Path(temporary).resolve()
        assert stage.is_relative_to(workspace) and stage != workspace
        with zipfile.ZipFile(archive) as zipped:
            for name in CHECKPOINT_FILES:
                # Only these four fixed filenames are restored; archive code is never installed.
                with zipped.open('trial-02/best/' + name) as src, (stage / name).open('wb') as out:
                    shutil.copyfileobj(src, out, length=1024 * 1024)
        validate(stage)
        if target.exists():
            raise FileExistsError(target)
        stage.rename(target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', required=True, type=Path)
    args = parser.parse_args()
    assets = json.loads(Path(__file__).with_name('assets.json').read_text(encoding='utf-8'))
    workspace = args.workspace.resolve()
    downloads = workspace / 'downloads'
    for name in ('cache', 'model'):
        item = assets[name]
        path = downloads / item['filename']
        download_checked(item['url'], path, item['sha256'], item['bytes'])
        if name == 'cache':
            restore_cache(path, workspace, assets['contract_sha256'])
        else:
            restore_checkpoint(path, workspace, assets['model_sha256'], assets['contract_sha256'])
    print(json.dumps({'status': 'ready', 'workspace': str(workspace), 'contract_sha256': assets['contract_sha256']}))


if __name__ == '__main__':
    main()
