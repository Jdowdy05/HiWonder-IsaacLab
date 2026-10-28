# HiWonder Isaac Lab

Custom HiWonder SO-ARM101 follower environments, robot assets, task configurations and RGB reinforcement-learning tools for **Isaac Lab v3.0.0**. This private project provides the original `stock` follower and an explicit `eka_v6` hand variant. No pretrained task-solving policies are included.

The actor uses wrist RGB and proprioception; a privileged state critic supports PPO training. The package includes 30 task presets, Success Guided Sampling (SGS), lighting/image/actuator randomization, camera calibration tools, checks and historical validation records. These tools simulate the arm and do not drive physical motors.

## Choose the robot

| Profile selection | Robot | Actions / rigid bodies | Proprioception / critic capacity |
|---|---|---|---|
| Default profiles, `robot_model="stock"` | Five arm joints and the original one-joint gripper | 6 / 7 | 18 / 256 |
| `configs/eka_v6.json` or `configs/eka_v6_rgb_spots.json` | Five arm joints and four independent finger joints | 9 / 10 | 27 / 320 |

The Eka hand has two left/right finger chains, each with independently actuated root and tip joints. The stock model remains the default for legacy profile selection. Select Eka deliberately with `--profile source/isaaclab_hiwonder/configs/eka_v6.json`; the stock example commands below retain their original robot selection.

Retaining the stock default does not guarantee exact checkpoint or SGS-bank replay across implementation fingerprint changes. Old six-action stock checkpoints are incompatible with the nine-action Eka model, including through `--initialize-from`. Same-model policy initialization still requires compatible network dimensions and starts a fresh optimizer/bank.

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

Install Git and Python 3.12 first. Install or update uv with `python -m pip install --upgrade "uv>=0.13.1"` (use `python3.12` on Linux), then confirm `uv --version`. Clone this private repository using your own GitHub access. The helper verifies the bundled checksums, clones [official Isaac Lab](https://github.com/isaac-sim/IsaacLab) at the exact commit above, copies the extension and pinned integration manifests, and runs locked sync. Its default target is the sibling directory `IsaacLab-HiWonder`. It refuses every existing target, preserving existing installations and user files.

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

## Task presets and profiles

The 30 presets consist of:

| Group | Exact preset names |
|---|---|
| Rigid objects | `peg_insert`, `cube_stack`, `nut_pick_place`, `washer_ring`, `connector_insert`, `shape_sort`, `gear_assembly`, `circuit_restore`, `catch_ball`, `roll_ball`, `roll_ball:table` |
| Mechanisms | `knob`, `switch`, `latch`, `bottle_cap`, `bottle_cap:close`, `key_lock` |
| Cables/rope | `barrel_jack`, `rope_thread:thread`, `rope_thread:overhand`, `rope_thread:figure_eight`, `cable_untangle` |
| Cloth | `cloth_fold:towel`, `cloth_fold:handtowel`, `cloth_fold:shirt`, `cloth_fold:rag`, `cloth_fold:shorts`, `cloth_fold:underwear`, `cloth_fold:sock` |
| Plastic paper | `paper_airplane` |

Rigid/mechanism tasks use PhysX; cables, cloth and paper use native Newton VBD. Select with `--task NAME`. The strict JSON schema rejects unknown fields. `task_parameters` controls supported commanded goals such as cube order or target mechanism state. The [extension guide](source/isaaclab_hiwonder/README.md) describes the mechanics, ten randomization groups, rewards and calibration commands in detail.

| Profile in `source/isaaclab_hiwonder/configs/` | Purpose |
|---|---|
| `default.json` | Complete reference profile |
| `clean.json` | Nuisance-disabled ablation |
| `rgb_disco.json` | Combined visual nuisance and moving room lights |
| `rgb_spots.json` | Moving physical wrist-view spots and varying color saturation |
| `occlusion.json` | Increased visual table coverage |
| `sgs_peg.json` | SGS peg reset families with bundled simulated stable-grasp seed |
| `eka_v6.json` | Explicit Eka robot with nine actuator joints and ten rigid bodies |
| `eka_v6_rgb_spots.json` | Eka robot plus moving wrist-view RGB lighting |

SGS uses a fixed reset bank and completed Boolean outcome histories. Interrupted/manual resets do not count as failures. `--sampling uniform` provides a training ablation. The bundled stable-grasp JSON/NPZ and `sgs_peg.json` qualify only the nominal stock simulated peg cohort. They cannot supply an Eka grasp seed; Eka needs a new receipt for its exact robot asset before enabling `stable_grasp` resets. The bundled seed is not a trained policy or physical grasp calibration.

## Checks and validation limits

Source-only setup checks, run from this overlay repository:

```text
python -m unittest discover -s tests -v
```

After SDK installation, run from the installed upstream checkout:

```powershell
& ./.venv/Scripts/python.exe -m pytest source/isaaclab_hiwonder -q
& ./.venv/Scripts/python.exe source/isaaclab_hiwonder/tools/check_hiwonder_tasks.py --all --output-directory '<new-output>'
```

The second command launches simulator processes and selects stock; `--robot-model eka_v6` explicitly selects the new hand for that checker. Training and inspection instead select their robot through `--profile`. Run GPU checks serially with measured host headroom and an external process-tree memory limit. The historical [validation](source/isaaclab_hiwonder/VALIDATION.md), [optimization](source/isaaclab_hiwonder/OPTIMIZATION_VALIDATION.md), and [collision](source/isaaclab_hiwonder/COLLISION_VALIDATION.md) records retain their original dates, source digests and failed/incomplete checks. All historical all-30 GPU, contact and RGB training receipts in those records concern the stock robot. Their large external evidence bundles are not included and they do not qualify the Eka model or new integrated implementation.

The staged Eka runtime has 106 passing focused CPU tests, including construction of all 30 task/critic layouts at capacity 320 and a small CPU PPO update/save/restore contract. Integrated full-suite and saved-asset/runtime validation are recorded separately when complete. CPU construction and learner contracts do not establish GPU execution, rendering, contact stability, grasp success or task learning with Eka.

Full camera/cloth startup has previously exceeded an external 8 GiB private-allocation guard before scene setup completed. Compute-only collision probes do not establish full-scene memory savings. Strict cross-process native Newton trajectory comparisons remain rejected in the historical optimization record. Camera/renderer and physics allocations must be measured separately. The new [Eka evidence](validation/eka_v6-2026-10-28/README.md) records separate bounded simulation checks; package tests alone do not establish policy success.

The CAD camera mount and 640x480/30 Hz capture settings are represented, but lens intrinsics, distortion, optical center/roll, servo gains, inertias and contact parameters remain provisional or estimated. Calibration needs independent physical measurements and held-out captures. A configured camera rate does not establish real-time FPS. Finite task smoke checks do not establish learning, reliable manipulation or sim-to-real transfer.

## Assets and rights

Generated robot USDs, scale variants, registration/camera manifests, collision profiles and the stock stable-grasp seed are included under the extension's `isaaclab_hiwonder/assets/`. Stock assets live in `generated/`; the explicit Eka model lives in `eka_v6/`. The original large stock CAD conversion input is identified by hash in its manifest and is not bundled. The Eka directory includes the supplied replacement `source.step`, SHA-256 `141ab8a721d850ed88da9d97f17257ebce1f85cfca7a0403f4c6e20c1f60e944`, preserved unchanged.

The Eka derivative uses exact NURBS reparameterization for 13 printed parts and boundary patches for two caps. Those caps remain faceted CAD; their original parametric construction was not recovered. Geometry checks found the resulting printable meshes watertight, consistently wound and positive-volume at actual USD float precision across nominal/0.99/1.01 scale, with retained contact triangles. Earlier native V6 data supplies reference hardware/case meshes only; changed printed geometry comes from the new STEP. Matching four-hole mounting patterns on a 14 mm pitch circle place the hand after `wrist_roll`; physical fastener fit, joint travel and dynamics remain unmeasured. These geometry checks do not establish physical assembly or contact acceptance.

Software retains its existing BSD-3-Clause headers; see [LICENSE](LICENSE). Source CAD and derived asset redistribution rights have not been established. This repository is private for the owner's authorized use; making it public or redistributing assets requires a separate rights review. See [ASSET_PROVENANCE.md](ASSET_PROVENANCE.md). Optional CAD conversion uses a separate environment with `usd-core`; do not install that provider into the Isaac Sim SDK environment.

See [Eka repair and validation](source/isaaclab_hiwonder/EKA_V6_VALIDATION.md), [repaired CAD exports](cad/eka_v6/README.md), and the [bounded simulation receipts](validation/eka_v6-2026-10-28/summary.json).

Final package suite:383 passed,1 CUDA-only check skipped. After the final visualizer/API fix and import formatting,19 focused tests passed; independent review has no open findings. A fresh source-only installation matched168 copied hashes at the pinned GA commit and passed the offline lock check. SDK binaries were not reinstalled.
