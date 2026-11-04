# Installation

[Documentation](README.md) · [Project overview](../README.md)

The installer prepares a fresh Isaac Lab checkout with the HiWonder extension and pinned dependencies.

## Pinned runtime

| Component | Version |
|---|---|
| Isaac Lab | `v3.0.0`, commit `e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4` |
| Python | 3.12 |
| Isaac Sim | 6.2.0.0 |
| PyTorch / CUDA wheel | 2.11.0 / cu130 |
| Newton / Warp | 1.6.1 / 1.17.0 |
| RSL-RL | 5.5.1 |
| NumPy | Locked 2.4.6; override `>=2,<2.5` |
| uv | Installed version >=0.13.1 |

Use a supported NVIDIA GPU/driver and OS for the pinned upstream release. Installation downloads the substantial Isaac Sim SDK; allow sufficient disk/RAM. The lock targets Windows AMD64 and Linux x86_64/aarch64, but this publication check used Windows and did not establish platform-wide runtime qualification.

## Install into a fresh checkout

Install Git and Python 3.12 first. Install or update uv with `python -m pip install --upgrade "uv>=0.13.1"` (use `python3.12` on Linux), then confirm `uv --version`. Clone this repository. The helper verifies the bundled checksums, clones [official Isaac Lab](https://github.com/isaac-sim/IsaacLab) at the exact commit above, copies the extension and pinned integration manifests, and runs locked sync. Its default target is the sibling directory `IsaacLab-HiWonder`. It refuses every existing target, preserving existing installations and user files.

Windows PowerShell, from the directory where you want both repositories:

```powershell
git clone https://github.com/Jdowdy05/HiWonder-IsaacLab.git
Set-Location HiWonder-IsaacLab
python tools/setup_isaaclab.py
Set-Location ../IsaacLab-HiWonder
```

Linux:

```bash
git clone https://github.com/Jdowdy05/HiWonder-IsaacLab.git
cd HiWonder-IsaacLab
python3.12 tools/setup_isaaclab.py
cd ../IsaacLab-HiWonder
```

Use `--target /path/to/new/IsaacLab-HiWonder` to select another **new** directory; its parent must exist. Use `--uv /path/to/uv` if the uv on `PATH` is older than 0.13.1. With `--skip-sync`, the helper prepares only the source checkout. Complete setup later from that new checkout:

```text
uv sync --locked --extra isaacsim --extra rsl-rl --extra test --inexact
```

If SDK sync fails, the prepared sources remain in the target; correct the environment/network issue and rerun that sync command there. The helper does not reuse another checkout's `.venv` or upgrade its packages. Isaac Sim's license must be accepted before use; set `OMNI_KIT_ACCEPT_EULA=YES` only after reviewing the license.

`integration/pyproject.toml` is the official tag manifest plus the HiWonder editable dependency/source and NumPy `<2.5` override. `integration/uv.lock` retains the preserved HiWonder-only dependency resolution with the four upstream editable package versions updated for GA. It is copied to the **new Isaac Lab root**, not used as a standalone environment in this overlay repository. `integration/checksums.json` covers every copied extension file and the setup manifests; it provides integrity checking, not a signature or rights grant.
