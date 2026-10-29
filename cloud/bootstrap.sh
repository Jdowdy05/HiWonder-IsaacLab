#!/usr/bin/env bash
# Pinned user-space setup for a provisioned Linux GPU instance. Creates no rental.
set -euo pipefail
if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
    printf 'Run this on the rented Linux x86_64 host, not Windows or the Jetson.\n' >&2
    exit 2
fi
workspace=${1:-/workspace/hiwonder}
system_deps=false
older_gpu=false
for flag in "${@:2}"; do
    case "$flag" in
        --install-system-deps) system_deps=true ;;
        --allow-older-gpu) older_gpu=true ;;
        *) printf 'Unknown option: %s\n' "$flag" >&2; exit 2 ;;
    esac
done
if [[ ${OMNI_KIT_ACCEPT_EULA:-} != YES ]]; then
    printf 'Review the Isaac Sim license and export OMNI_KIT_ACCEPT_EULA=YES before installation.\n' >&2
    exit 2
fi
overlay=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p "$workspace"
workspace=$(CDPATH= cd -- "$workspace" && pwd)
if "$system_deps"; then
    admin=()
    if [[ $(id -u) -ne 0 ]]; then admin=(sudo); fi
    . /etc/os-release
    [[ $ID == ubuntu && ( $VERSION_ID == 24.04 || $VERSION_ID == 22.04 ) ]] || { printf 'Expected Ubuntu22.04/24.04.\n' >&2; exit 2; }
    glib=libglib2.0-0
    if [[ $VERSION_ID == 24.04 ]]; then glib=libglib2.0-0t64; fi
    "${admin[@]}" apt-get update
    "${admin[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip git curl ca-certificates build-essential \
        libvulkan1 vulkan-tools libgl1 libegl1 libglvnd0 libglx0 "$glib" \
        libx11-6 libxext6 libxrender1 libxrandr2 libxi6 libsm6 libgomp1 \
        libfontconfig1 libnss3 libxkbcommon0 libxcb-xinerama0 ffmpeg unzip
fi
bootstrap_started=$(date +%s)
record_bootstrap() {
    result=$?
    python3 - "$workspace" "$bootstrap_started" "$result" <<'PY'
import json,sys,time
from pathlib import Path
p=Path(sys.argv[1])/'bootstrap-status.json'
p.write_text(json.dumps(dict(started_unix_time=int(sys.argv[2]),elapsed_seconds=time.time()-int(sys.argv[2]),exit_code=int(sys.argv[3]),scope='Preflight+SDK source/dependency setup; initial apt phase excluded'),indent=2)+'\n')
PY
    return "$result"
}
trap record_bootstrap EXIT
checker=(python3 "$overlay/cloud/preflight.py" --phase bootstrap --workspace "$workspace" --json "$workspace/host-bootstrap.json")
if "$older_gpu"; then checker+=(--allow-older-gpu); fi
"${checker[@]}"
export UV_CACHE_DIR="$workspace/cache/uv"
export WARP_CACHE_PATH="$workspace/cache/warp"
export XDG_CACHE_HOME="$workspace/cache/xdg"
mkdir -p "$UV_CACHE_DIR" "$WARP_CACHE_PATH" "$XDG_CACHE_HOME"
if [[ ! -x "$workspace/launcher/bin/uv" ]]; then
    python3 -m venv "$workspace/launcher"
fi
"$workspace/launcher/bin/python" -m pip install 'uv==0.13.1'
uv="$workspace/launcher/bin/uv"
target="$workspace/IsaacLab-HiWonder"
export UV_PROJECT_ENVIRONMENT="$target/.venv"
if [[ ! -e "$target" ]]; then
    python3 "$overlay/tools/setup_isaaclab.py" --target "$target" --skip-sync --uv "$uv"
else
    python3 "$overlay/cloud/verify_install.py" --installed "$target"
fi
cd "$target"
"$uv" sync --locked --extra isaacsim --extra rsl-rl --extra test --inexact
checker=("$target/.venv/bin/python" "$overlay/cloud/preflight.py" --phase runtime --workspace "$workspace" --json "$workspace/host-runtime.json")
if "$older_gpu"; then checker+=(--allow-older-gpu); fi
"${checker[@]}"
printf '\nPinned setup ready. Next: python3 %s/cloud/run.py smoke --workspace %s\n' "$overlay" "$workspace"
