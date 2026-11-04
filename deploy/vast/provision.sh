#!/usr/bin/env bash
# Vast SSH-mode dual setup; no instance purchase and no training jobs are launched.
set -Eeuo pipefail
source_revision=${1:?Pass the immutable deployment commit from the template}
[[ "$source_revision" =~ ^[0-9a-f]{40}$ ]] || { printf 'Expected a full Git commit SHA.\n' >&2; exit 2; }
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || { printf 'Require Linux x86_64.\n' >&2; exit 2; }
repository=https://github.com/Jdowdy05/HiWonder-IsaacLab.git
rl_revision=6e327702ca4a25bf5b800eaccfdf247fe90a1167
base=${HIWONDER_ROOT:-/workspace}
mkdir -p "$base"
base=$(CDPATH= cd -- "$base" && pwd)
rl_source="$base/hiwonder-project"
act_source="$base/hiwonder-act-project"
state="$base/hiwonder-template"
mkdir -p "$state" "$base/hiwonder" "$base/act"
exec > >(tee -a "$state/provision.log") 2>&1
exec 9>"$state/provision.lock"
if ! flock -n 9; then printf 'Another provisioner is running.\n'; exit 0; fi
trap 'code=$?; printf "Provisioner exit code: %s\n" "$code"' EXIT

admin=()
if [[ $(id -u) -ne 0 ]]; then admin=(sudo -n); fi
if ! command -v git >/dev/null || ! command -v vulkaninfo >/dev/null; then
    "${admin[@]}" apt-get update
    "${admin[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        git curl ca-certificates python3 python3-venv python3-pip libvulkan1 vulkan-tools \
        libgl1 libegl1 libglvnd0 libglx0 libglib2.0-0t64 libx11-6 libxext6 libxrender1 \
        libxrandr2 libxi6 libsm6 libgomp1 libfontconfig1 libnss3 libxkbcommon0 libxcb-xinerama0 ffmpeg unzip
fi

checkout() {
    local path=$1 revision=$2
    if [[ ! -e "$path" ]]; then
        git init "$path"
        git -C "$path" remote add origin "$repository"
    fi
    [[ -d "$path/.git" ]] || { printf 'Not a project checkout: %s\n' "$path" >&2; return 2; }
    [[ $(git -C "$path" remote get-url origin) == "$repository" ]] || { printf 'Unexpected project remote.\n' >&2; return 2; }
    if ! git -C "$path" rev-parse --verify HEAD >/dev/null 2>&1; then
        [[ -z $(git -C "$path" status --porcelain --untracked-files=no) ]] || { printf 'Incomplete checkout has edits; preserved.\n' >&2; return 2; }
        git -C "$path" fetch --depth 1 origin "$revision"
        git -C "$path" checkout --detach FETCH_HEAD
    fi
    [[ $(git -C "$path" rev-parse HEAD) == "$revision" ]] || { printf 'Different source revision; existing files preserved.\n' >&2; return 2; }
    [[ -z $(git -C "$path" status --porcelain --untracked-files=no) ]] || { printf 'Source has edits; existing files preserved.\n' >&2; return 2; }
}
checkout "$rl_source" "$rl_revision"
checkout "$act_source" "$source_revision"

# Completed installations use the runtime reserve and verification, not the cold-install disk requirement.
rl_ready=false
runtime_python="$base/hiwonder/IsaacLab-HiWonder/.venv/bin/python"
if [[ -x "$runtime_python" && -f "$base/hiwonder/bootstrap-status.json" ]]; then
    if python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["exit_code"] == 0 else 1)' "$base/hiwonder/bootstrap-status.json"; then
        rl_ready=true
    fi
fi
if "$rl_ready"; then
    python3 "$rl_source/cloud/verify_install.py" --installed "$base/hiwonder/IsaacLab-HiWonder"
    "$runtime_python" "$rl_source/cloud/preflight.py" --phase runtime --workspace "$base/hiwonder" --json "$state/host-runtime.json"
else
    python3 "$rl_source/cloud/preflight.py" --phase bootstrap --workspace "$base/hiwonder" --json "$state/host-bootstrap.json"
fi

bash "$act_source/deploy/vast/bootstrap_act.sh" "$base/act"

if "$rl_ready"; then
    printf '\nExisting Isaac Lab environment verified; dependency synchronization skipped.\n'
elif [[ ${OMNI_KIT_ACCEPT_EULA:-NO} == YES ]]; then
    bash "$rl_source/cloud/bootstrap.sh" "$base/hiwonder" --install-system-deps
else
    printf '\nACT is ready. Isaac Lab SDK setup awaits your license acceptance.\n'
    printf 'After reviewing the Isaac Sim license, run over SSH:\n'
    printf 'export OMNI_KIT_ACCEPT_EULA=YES\n'
    printf 'bash %s/cloud/bootstrap.sh %s --install-system-deps\n' "$rl_source" "$base/hiwonder"
fi
printf '\nSetup finished. Workloads are started manually; use the template ReadMe.\n'
