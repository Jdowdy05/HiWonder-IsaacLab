#!/usr/bin/env bash
# Independent ACT environment. Never installs into the Isaac Lab or Vast main venv.
set -Eeuo pipefail
workspace=${1:-/workspace/act}
source_root=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
requirements="$source_root/deploy/vast/requirements-act.txt"
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || { printf 'ACT cloud setup requires Linux x86_64.\n' >&2; exit 2; }
python3 -c 'import sys; assert sys.version_info[:2] == (3,12), "Python 3.12 is required"'
mkdir -p "$workspace"
workspace=$(CDPATH= cd -- "$workspace" && pwd)
fingerprint=$(sha256sum "$requirements" | cut -d ' ' -f 1)
owner="$workspace/.bootstrap-owner"
if [[ -e "$owner" ]]; then
    [[ $(cat "$owner") == "$fingerprint" ]] || { printf 'ACT dependency fingerprint changed; use a new workspace.\n' >&2; exit 2; }
elif [[ -e "$workspace/.venv" || -e "$workspace/launcher" ]]; then
    printf 'Refusing to change a pre-existing ACT environment without its ownership marker.\n' >&2
    exit 2
else
    printf '%s\n' "$fingerprint" > "$owner"
fi

export UV_CACHE_DIR="$workspace/package-cache"
export HF_HOME="$workspace/huggingface"
export TORCH_HOME="$workspace/torch-cache"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export PYTHONUNBUFFERED=1
python="$workspace/.venv/bin/python"
ready="$workspace/environment-ready.json"
if [[ ! -f "$ready" ]]; then
    if [[ ! -x "$workspace/launcher/bin/python" ]]; then python3 -m venv "$workspace/launcher"; fi
    "$workspace/launcher/bin/python" -m pip install 'uv==0.13.1'
    uv="$workspace/launcher/bin/uv"
    if [[ ! -x "$python" ]]; then "$uv" venv --python "$(command -v python3)" "$workspace/.venv"; fi
    "$uv" pip install --python "$python" 'torch==2.11.0+cu130' 'torchvision==0.26.0+cu130' --index-url https://download.pytorch.org/whl/cu130
    "$uv" pip install --python "$python" -r "$requirements"
    "$python" - "$ready" "$fingerprint" <<'PY'
import importlib.metadata as m,json,sys
from pathlib import Path
expected={'torch':'2.11.0+cu130','torchvision':'0.26.0+cu130','lerobot':'0.6.1','numpy':'2.2.6','pyarrow':'26.0.0','av':'15.1.0'}
observed={k:m.version(k) for k in expected}
assert observed==expected,(observed,expected)
Path(sys.argv[1]).write_text(json.dumps({'requirements_sha256':sys.argv[2],'versions':observed},indent=2)+'\n',encoding='utf-8')
PY
fi
"$python" - "$ready" "$fingerprint" <<'PY'
import importlib.metadata as m,json,sys
from pathlib import Path
state=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
assert state['requirements_sha256']==sys.argv[2],'ACT requirements changed'
assert all(m.version(name)==version for name,version in state['versions'].items()),'ACT environment packages changed'
PY
"$python" "$source_root/deploy/vast/vast_assets.py" --workspace "$workspace"
printf '\nACT ready in %s. Start a bounded smoke with:\n' "$workspace"
printf 'python3 %s/deploy/vast/run_act.py --workspace %s --smoke\n' "$source_root" "$workspace"
