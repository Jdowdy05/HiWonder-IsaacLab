# Vast.ai: Isaac Lab RL and ACT

[Documentation](README.md) · [Project overview](../README.md)

One Vast CUDA image supports both workloads, with separate Python environments. You do not need to build your own Docker image. First boot installs ACT and restores its prepared public dataset/checkpoint. Isaac Lab installation starts after you accept its license. Training starts explicitly after setup.

## Template fields

| Field | Value |
|---|---|
| Name | HiWonder - Isaac Lab RL + ACT |
| Image Path:Tag | `vastai/base-image:cuda-13.0.2-cudnn-devel-ubuntu24.04-py312` |
| Version tag | Keep that exact tag; avoid Automatic |
| Launch mode | Interactive shell server, SSH |
| Direct SSH | Enabled, with Vast proxy SSH available as backup |
| Extra application ports | None; Vast supplies SSH access |
| Container disk | 250 GB, allowing room for both environments and caches |
| Visibility | Private |
| Registry server / username / token | Leave blank; the image and project are public |
| On-start Script | Generate the immutable startup text below |

Environment variables:

```text
NVIDIA_DRIVER_CAPABILITIES=all
PYTHONUNBUFFERED=1
HF_HUB_DISABLE_IMPLICIT_TOKEN=1
WORKSPACE=/workspace
OMNI_KIT_ACCEPT_EULA=NO
```

`NVIDIA_DRIVER_CAPABILITIES=all` must be present at container creation for the wrist-camera renderer's driver libraries. Extra Filters use the Vast **CLI** units (GB for RAM), not the API's raw MB:

```text
verified=true rentable=true num_gpus=1 gpu_ram>=23 cpu_ram>=64 cpu_cores_effective>=8 cuda_max_good>=13.0 driver_version>=595.58.03 reliability>=0.99 cpu_arch=amd64
```

The console may normalize the displayed driver filter to `595.58` when saving. The host preflight still enforces the full `595.58.03` minimum before installation or training.

Select an **RTX 4090, RTX 5090 or supported recent Ada/Blackwell RTX Pro with 24 GB+ VRAM**. Prefer 64 GB+ RAM, or 128 GB if similarly priced. A100/H100 compute cards do not satisfy this project's RTX rendering preflight. The preflight also verifies actual cgroup capacity, at least 60 GiB usable RAM, eight effective CPUs, driver, storage and Vulkan.

Vast's Docker Options field accepts environment variables and port mappings; arbitrary `--shm-size` or `--ipc` text there is ignored. The template does not expose unauthenticated camera/TensorBoard ports.

## Generate the startup text

From a committed and published copy of this repository:

```text
python deploy/vast/render_template.py --onstart-only
```

Paste that output into **On-start Script**. The command pins the selected Git commit and verifies the downloaded startup script's SHA256 before executing it. `python deploy/vast/render_template.py` prints the other field values as JSON. An unpublished local commit will not be downloadable by the instance.

Image tag verification on 2026-11-04 found registry digest `sha256:4fd686b5ab149cbfb1e6e5ec20f8903f1da061d81027846c105d7757d710fe8f`, with Linux AMD64 image digest `sha256:7c7b2be62c5497a3aea08f87cf8a8796eb9c55c7badb890709af10dc2bbd50cd`.

Save with **Create**, which stores the template without renting. **Create & Use** also opens offer selection. Review the current offer and total hourly/storage price before renting.

## First boot

Connect using the SSH command from Vast's **Connect** button. Startup logs are at `/workspace/hiwonder-template/provision.log`; the first install downloads several GB of packages.

If your Vast account has no SSH key yet, add the public `.pub` portion of your existing SSH key through its Keys page before connecting. Keep the private key on your PC.

| Workload | Source checkout | Environment and data |
|---|---|---|
| Isaac Lab PPO + SGS | `/workspace/hiwonder-project`, pinned cloud revision `6e327702ca4a25bf5b800eaccfdf247fe90a1167` | `/workspace/hiwonder/IsaacLab-HiWonder/.venv` and `/workspace/hiwonder/runs` |
| ACT | `/workspace/hiwonder-act-project`, pinned deployment commit | `/workspace/act/.venv`, `/workspace/act/cache`, `/workspace/act/checkpoint`, `/workspace/act/runs` |

RL uses the previously prepared, guarded cloud setup with the custom Eka hand, native wrist camera and randomization. ACT uses its own PyTorch 2.11/cu130, LeRobot 0.6.1 and NumPy 2.2.6 environment; it does not share Isaac Lab's NumPy/SDK installation.

The prepared ACT cache restores the same 200 episodes/81,290 frames and exact normalization/split contract used by the archived checkpoint. Archive and individual cache hashes are checked. The loader handles Windows path separators without rewriting the saved contract. Only the four checkpoint files are restored from the original model bundle; old archive code does not overwrite the current source. Original public dataset licenses/provenance are included.

### Accept the Isaac Sim license

The template leaves `OMNI_KIT_ACCEPT_EULA=NO`. After reviewing [NVIDIA's license and container installation instructions](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html), you may set it to `YES` yourself before launch to enable automatic SDK installation. Alternatively, accept and install over SSH:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
bash /workspace/hiwonder-project/cloud/bootstrap.sh /workspace/hiwonder --install-system-deps
```

ACT setup can finish while the SDK awaits acceptance. Source checkouts and completed environments are verified on repeat starts; unexpected revisions or edits are preserved and reported instead of being overwritten. Completed RL installations use runtime checks and skip dependency synchronization.

## Start ACT

```bash
# Three-step, guarded smoke initialized from the archived best checkpoint.
python3 /workspace/hiwonder-act-project/deploy/vast/run_act.py --smoke

# Additional training with the same cache and normalization.
python3 /workspace/hiwonder-act-project/deploy/vast/run_act.py --steps 10000

# Optional fresh model on the same dataset.
python3 /workspace/hiwonder-act-project/deploy/vast/run_act.py --steps 10000 --from-scratch
```

Default initialization continues the archived **weights with a fresh optimizer/RNG**; it is not exact optimizer-state resume. Each run has a new output directory. The existing ACT process-tree guard enforces 6 GiB RSS, a 4 GiB RAM reserve, and 30 minutes maximum wall time. Longer requested runs may be stopped by that guard. Outputs and guard reports live under `/workspace/act/runs` and `/workspace/act/jobs`.

## Start Isaac Lab RL

Use the license acceptance environment above, then:

```bash
python3 /workspace/hiwonder-project/cloud/run.py smoke --workspace /workspace/hiwonder
python3 /workspace/hiwonder-project/cloud/run.py pilot --workspace /workspace/hiwonder --envs 4 --precision fp32
python3 /workspace/hiwonder-project/cloud/run.py train --workspace /workspace/hiwonder --task peg_insert --envs 4 --precision fp32 --iterations 1000
```

The smoke checks execution, not task learning. Run it before a long job. Optional `--precision bf16` affects learner updates while preserving FP32 simulation/camera tensors; compare it with a matching FP32 pilot. This template does not install the experimental Linux tensor-rate unlock or automatically transfer ACT into PPO.

## View the wrist camera

Add this forwarding option to the SSH command supplied by Vast:

```text
-L 8769:127.0.0.1:8769
```

On the instance, after SDK setup:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
cd /workspace/hiwonder/IsaacLab-HiWonder
.venv/bin/python source/isaaclab_hiwonder/tools/inspect_hiwonder.py --profile source/isaaclab_hiwonder/configs/eka_v6_rgb_spots.json --task peg_insert --num-envs 1 --port 8769
```

Open `http://127.0.0.1:8769` on your PC. TensorBoard can similarly bind to loopback and use SSH forwarding; avoid exposing the raw application ports publicly. Run workloads one at a time until measured GPU/RAM headroom supports concurrency.

Download checkpoints before deleting an instance. Stopping a training process does not stop rental/storage billing.

## Validation scope

The setup helpers have CPU/fixture tests for safe archive extraction, checksums, process launch budgets, license gating and restarts. An actual restoration check opened all 200 episodes and decoded all three source cameras with the original data contract and weights intact. No paid cloud instance or new cloud GPU/rendering test was run while preparing this template. The rented host still needs its preflight and smoke checks.

See the [ACT report](../experiments/act-public-pickplace-2026-10-29/REPORT.md) for mixed offline policy results and the [RL validation guide](validation.md) for robot/simulation evidence limits.
