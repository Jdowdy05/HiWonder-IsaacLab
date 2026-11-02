# Public SO-100 pick-and-place ACT experiment

This is an actual offline imitation-learning experiment using the official LeRobot ACT architecture, with a virtual four-joint Eka gripper remap. It does not change Isaac Lab, the HiWonder simulation repository, the Jetson, motor settings, or the running TinyDog training job.

## Data

Three independent Apache-2.0 datasets, 200 episodes, 81,290 frames at 30 Hz:

| Source | Episodes | Frames | Selected exterior camera |
|---|---:|---:|---|
| [lerobot/svla_so100_pickplace](https://huggingface.co/datasets/lerobot/svla_so100_pickplace) | 50 | 19,631 | top |
| [jmrog/so100_sweet_pick](https://huggingface.co/datasets/jmrog/so100_sweet_pick) | 50 | 32,068 | laptop |
| [nbaron99/so100_pick_and_place4](https://huggingface.co/datasets/nbaron99/so100_pick_and_place4) | 100 | 29,591 | laptop |

Only the selected camera videos, trajectory Parquet files, README/license card and metadata were downloaded. The original video resolution/encoding and trajectory values are preserved. Approximately 1.65 GB decimal of source files were downloaded. Each dataset is pinned to an immutable source commit; `datasets/*/download_manifest.json` records SHA256 hashes and remote content hashes where available.

Raw data: `../../.work/act-public-pickplace-2026-10-29/datasets/` relative to this directory. Derived cache: `../../.work/act-public-pickplace-2026-10-29/cache/`. These are ordinary local directories, not removable temporary download placeholders.

The cache uses bounded SQLite JPEG reads. Images are resized from 640x480 to 224x168 for this experiment, preserving aspect ratio. Their timestamps are matched to the original 30 Hz trajectories, including offsets within concatenated v3 videos. This is an exterior-camera policy; it is not trained on the custom wrist camera.

## Splits, remapping and normalization

- Deterministic 80/20 split by complete episode within each source: 160 train episodes, 40 validation episodes. Sources are sampled equally during training, then episodes and frames uniformly within a source.
- Validation uses four fixed anchors per held-out episode and observation-only, zero-latent ACT inference. Targets are not passed into the validation VAE encoder.
- Future targets cover 30 frames (one second) and never cross an episode boundary. Repeated end padding is masked out of loss/metrics.
- Normalization statistics and observed gripper endpoints are fitted on training episodes only. ImageNet RGB normalization is applied externally. No extra synthetic image augmentation was added; this first test uses recorded visual variation.
- The five arm coordinates remain in their recorded source reference frames. Missing source motor calibrations have **not** been reconstructed or silently guessed. These are not calibrated HiWonder simulator/hardware joint angles.
- Source gripper direction is inspected separately: larger value means closing for the first two datasets and opening for `nbaron99`. See the image audits. Training-only 0.5/99.5 percentiles define approximate observed open/closed ranges; these are not measured jaw gaps or full physical travel.
- The normalized opening is mapped to virtual finger angles in degrees, in order **left root, left tip, right root, right tip**. Open is `[-20,0,-20,0]`; closed is `[0,18,0,18]`. Public images still show stock grippers. Virtual finger state and action labels are derived from one original gripper value, not four independently observed motors.

## Model and saved-policy interface

Official LeRobot v0.6.1, commit `7e241bd630a3719a56157a497ce5d08f244784f1`; upstream source is unmodified. ACT uses ImageNet ResNet18, 512 hidden dimensions, four encoder layers, one decoder layer, four VAE encoder layers, 32 latent dimensions, 30-frame chunks and 51,567,497 parameters. One scene image and nine state values produce nine action values per future frame.

Training uses batch eight, AdamW, learning rate 1e-4 (backbone 1e-5), original ACT KL weight 10, BF16 autocast with FP32 parameters/optimizer, and gradient clipping at 10. There is no physics simulation in this experiment.

Each `best/` and `last/` directory includes ordinary LeRobot `config.json` and `model.safetensors`, plus:

- `data_contract.json`: joint order, all normalization values, source splits and gripper conventions.
- `experiment_state.json`: optimizer-step count and held-out metrics at that checkpoint.

The saved model does **not** embed input/output normalization. `train_act.PublicEpisodes` applies the preprocessing and `evaluate` reverses action normalization. Raw network outputs are normalized absolute targets; inverse normalization gives source-frame arm values and virtual finger degrees. They are not our Isaac Lab incremental actions.

`last/optimizer_state.pt` records optimizer/sampler state for future development. The runner currently supports fresh runs and `--initialize-from` warm starts, not exact optimizer-state resume. Warm starts require an identical dataset/normalization contract, strictly load ACT weights and record the initial weights' SHA256; they intentionally reset optimizer/RNG.

## Reproduce or continue the experiment

Use `Train-ACT-Trial.ps1` to launch a new guarded run. It uses the private environment and existing downloaded cache, creates a new output directory, and never starts hardware or Isaac Sim. Example:

```powershell
& .\Train-ACT-Trial.ps1 -Steps 10000
# Optional warm start from a checkpoint in this experiment:
& .\Train-ACT-Trial.ps1 -Steps 10000 -InitializeFrom .\trial-02\best
```

The private environment is at `.work/act-public-pickplace-2026-10-29/.venv` under the workspace. It reads the existing IsaacLab-30 PyTorch packages through a `.pth` path and adds separate ACT/data dependencies locally; shared packages were not replaced. `runtime-versions.json` records versions. It also points to the pinned standalone LeRobot source clone under `.work/act-public-pickplace-2026-10-29/lerobot`.

Download/preparation can be reproduced using this environment:

```powershell
$python = '..\..\.work\act-public-pickplace-2026-10-29\.venv\Scripts\python.exe'
& $python .\experiment_data.py download --root '..\..\.work\act-public-pickplace-2026-10-29\datasets'
& $python .\experiment_data.py prepare --root '..\..\.work\act-public-pickplace-2026-10-29\datasets' --cache '..\..\.work\act-public-pickplace-2026-10-29\cache'
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
& $python -m pytest .\test_experiment.py -q
```

## Resource limits and acceptance boundary

The external guard monitors only its explicitly spawned process tree, including initialization and checkpoint I/O. It caps process RSS at 6 GiB, maintains 4 GiB available-RAM reserve, and limits total wall time to 30 minutes (individual trial invocations may be shorter). CUDA allocation is limited to at most 6 GiB; the actual shared-GPU allocator limit is approximately 5.7 GiB on this 5090. The trainer has a shorter graceful stop before the outer wall limit. No data-loader subprocesses, full decoded-video RAM cache, or simulator is used.

This trial tests whether an ACT policy learns from these varied recordings. Held-out prediction error is **not** measured pick-and-place success, unseen-camera generalization, independent four-finger dexterity, or physical/simulation acceptance. The current-state hold baseline and mean-image ablation are reported alongside ACT. Mean-image ablation is an out-of-distribution diagnostic, not proof that the policy tracks objects correctly.

Using the model on the actual HiWonder/Eka robot requires motor/frame calibration, physical gripper aperture/contact retargeting, camera adaptation, and controlled closed-loop validation. ACT weights also cannot be loaded directly as the existing PPO actor: later RL requires an ACT-compatible learner or a separate ACT-to-RL-policy distillation stage. None of those deployment or RL integration steps is claimed by this experiment.
