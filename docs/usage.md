# Usage

[Documentation](README.md) · [Project overview](../README.md)

## Inspect the arm and wrist camera

All following commands run from the installed `IsaacLab-HiWonder` checkout. Windows uses `.venv/Scripts/python.exe`; Linux uses `.venv/bin/python` with the same arguments.

```powershell
$env:OMNI_KIT_ACCEPT_EULA = 'YES'
& ./.venv/Scripts/python.exe source/isaaclab_hiwonder/tools/inspect_hiwonder.py --viz kit --num-envs 2 --port 8769
```

```bash
export OMNI_KIT_ACCEPT_EULA=YES
./.venv/bin/python source/isaaclab_hiwonder/tools/inspect_hiwonder.py --viz kit --num-envs 2 --port 8769
```

Open the printed loopback address, normally [http://127.0.0.1:8769](http://127.0.0.1:8769), for raw/policy RGB, room selection, pause and lighting controls. Omit `--viz kit` for browser-only inspection. `--port 0` chooses a free port. The default inspection profile is `rgb_spots.json`; profiles can be selected with `--profile`. The listener is local; it is not a phone/LAN server. While paused, raw RGB updates as a lighting preview and policy RGB retains the last simulation observation.

To inspect the Eka model with the same lighting controls:

```powershell
& ./.venv/Scripts/python.exe source/isaaclab_hiwonder/tools/inspect_hiwonder.py --profile source/isaaclab_hiwonder/configs/eka_v6_rgb_spots.json --task peg_insert --viz kit --num-envs 2 --port 8769
```

Eka retains 640x480 capture at a configured 30 Hz and the original wrist-relative camera pose/optics. Its new physical camera attachment is provisional and needs mount measurement and optical calibration.

## Train, evaluate and play

PowerShell examples (replace the Python executable with `.venv/bin/python` on Linux):

```powershell
# Two short optimizer iterations; an engineering smoke, not proof of learning.
& ./.venv/Scripts/python.exe source/isaaclab_hiwonder/tools/train_hiwonder.py train --task peg_insert --num-envs 2 --smoke

# Full RGB PPO with SGS and two-frame history.
& ./.venv/Scripts/python.exe source/isaaclab_hiwonder/tools/train_hiwonder.py train --profile source/isaaclab_hiwonder/configs/sgs_peg.json --num-envs 2 --frame-stack 2 --iterations 1000

# Visual nuisance profile; an explicit task overrides its profile's task.
& ./.venv/Scripts/python.exe source/isaaclab_hiwonder/tools/train_hiwonder.py train --task cube_stack --profile source/isaaclab_hiwonder/configs/rgb_disco.json --num-envs 2 --iterations 1000

# Eka nine-action setup smoke, followed by an example RGB training launch.
& ./.venv/Scripts/python.exe source/isaaclab_hiwonder/tools/train_hiwonder.py train --profile source/isaaclab_hiwonder/configs/eka_v6.json --task peg_insert --num-envs 2 --smoke
& ./.venv/Scripts/python.exe source/isaaclab_hiwonder/tools/train_hiwonder.py train --profile source/isaaclab_hiwonder/configs/eka_v6_rgb_spots.json --task peg_insert --num-envs 2 --frame-stack 2 --iterations 1000

# Use the actual run path printed by training.
& ./.venv/Scripts/python.exe source/isaaclab_hiwonder/tools/train_hiwonder.py train --resume '<run>/final.pt'
& ./.venv/Scripts/python.exe source/isaaclab_hiwonder/tools/train_hiwonder.py train --initialize-from '<run>/final.pt' --seed 123
& ./.venv/Scripts/python.exe source/isaaclab_hiwonder/tools/train_hiwonder.py eval --checkpoint '<run>/final.pt' --episodes 100
& ./.venv/Scripts/python.exe source/isaaclab_hiwonder/tools/train_hiwonder.py play --checkpoint '<run>/final.pt' --viz kit
```

Rendering still runs for the wrist camera when no desktop visualizer is selected. Isaac Lab 3.0 uses `--viz`, not the old `--headless` flag. The Windows launch tools load Kit before USD implementation modules. Use their entry points rather than importing simulator internals first.

The RGB actor uses 640x480 images, 18 proprioceptive values for stock or 27 for Eka, and the same 40-value public task/goal command. The privileged critic capacity is 256 for stock and 320 for Eka. `--frame-stack 2` or `4` adds image history. `--actor-mode state_debug` selects the explicit debugging actor. Checkpoints embed the selected robot/profile and sampler/history state. Exact `--resume` rejects incompatible learner/profile/implementation settings; `--initialize-from` starts a fresh optimizer and SGS bank for compatible same-model policy transfer. Evaluation uses held-out uniform reset sampling, independently of SGS training histories. Training writes checkpoints, TensorBoard and provenance to its printed run directory; generated run outputs are excluded from this repository. The Eka commands show supported launch syntax. A bounded two-iteration RGB PPO/SGS smoke passed; learned task-solving policies are not included.

## Experimental ACT public-data trial

The separate [public SO-100 ACT experiment](../experiments/act-public-pickplace-2026-10-29/README.md) includes training/evaluation code, pinned dataset provenance, configs, held-out comparisons and a virtual four-joint Eka gripper remap. Its [prerelease](https://github.com/Jdowdy05/HiWonder-IsaacLab/releases/tag/act-public-pickplace-2026-10-29) contains the trained checkpoint bundle. Results are mixed and remain offline; the model has no validated grasp success or physical/wrist-camera transfer. It is separate from the PPO/SGS learner described below.
