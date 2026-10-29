# HiWonder/Eka hourly GPU training

Start with one **RTX 4090, 64 GB RAM,  8+ effective CPU threads and 200 GB SSD**.
The current default is a stock Vast 4090 on demand. A later October 29 refresh
found offer 52582393 in Thailand, TH at **$0.4542/hour including 200 GB**,
with about 97 GB RAM, 64 effective CPUs and a 450 W power cap. A US fallback
was offer 58188739 in Washington, US at about **$0.5111/hour**.
Inventory changes quickly; these are snapshot candidates, not reserved capacity.
The cheapest qualifying 3090 was about $0.349/hour; it is below NVIDIA's current
advertised minimum GPU class for Isaac Sim 6.2 and needs an explicit trial before
relying on it. See [research and live-source limits](RESEARCH.md).

These are time-stamped available offers, not reservations. The exact console
quote and driver/graphics smoke test decide whether a host is usable. Transfer,
tax and stopped-storage costs are additional. An inexpensive GPU with too little
RAM,  old drivers, low power limit or slow CPU/storage can cost more per finished
training job. Our preparation performs no purchase, deposit, kernel unlock or
automatic provider shutdown.

## Find and provision a candidate

From this overlay repository on your PC or any Python 3.10+ environment:

```bash
python cloud/offers.py --output current-offers.json --region US
```

This only searches the public Vast marketplace. It requests 200 GB storage in the
quote and filters one exclusive GPU, 64 GB RAM,  8 effective CPUs, driver
595.58.03+, verified/rentable status and reported reliability>=0.99. It saves the
three cheapest candidates per GPU. Use the console to check the current final
price and contract duration before renting. A chosen offer can disappear.

[Vast console](https://cloud.vast.ai/create/) settings:

- One 4090, on demand; 64 GB RAM minimum, 128 GB preferred if similarly priced.
- 200 GB disk; 16-32 effective CPU threads preferred. No reduction of camera,
  randomization or solver fidelity is made to fit a cheaper host.
- CUDA 13/Ubuntu 24.04 image from [provider-spec.json](provider-spec.json). Set
  `NVIDIA_DRIVER_CAPABILITIES=all` **at instance creation** so Vulkan/graphics
  libraries are mounted. Exporting this later cannot add missing driver mounts.
- SSH access with your dedicated public key; keep dashboards bound to loopback.
- Accept the Isaac Sim license before setting `OMNI_KIT_ACCEPT_EULA=YES`.

TensorDock is a full-VM fallback: its live API priced a 16-vCPU/64 GB/200GB4090 VM
at about $0.666/hour. Guest drivers still need verification. Full VM access alone
does not prove GPU-unlock permission or compatibility. RunPod's low-price
community availability varied; refresh before choosing. The research contains
the exact snapshot and resource-price arithmetic.

## Transfer without cloud GitHub credentials

The prepared archive contains committed project code, assets and installation
pins. It excludes `.git`, credentials, local environments, runs and model weights.
The private SSH key stays on your PC. Add only the public key to your provider.
In PowerShell:

```powershell
Get-Content "$env:USERPROFILE\.ssh\hiwonder_cloud_ed25519_20261029.pub"
.\cloud\upload.ps1 -Bundle 'C:\path\HiWonder-cloud-COMMIT.tar.gz' -SshTarget root@YOUR_HOST -SshPort YOUR_PORT
```

Verify the server's SSH fingerprint against the provider before accepting it.
The uploader checks the transferred archive SHA256 and does not start anything.
Alternatively use ordinary `scp`. On the rented machine:

```bash
cd /workspace/hiwonder-upload
sha256sum HiWonder-cloud-COMMIT.tar.gz
tar -xzf HiWonder-cloud-COMMIT.tar.gz
cd HiWonder-IsaacLab
export OMNI_KIT_ACCEPT_EULA=YES
bash cloud/bootstrap.sh /workspace/hiwonder --install-system-deps
```

The bootstrap installs user-space OS libraries and a dedicated pinned uv
environment. It does not replace the host NVIDIA driver. It clones exact
Isaac Lab v3.0.0, verifies the overlay/lock hashes and uses the existing locked
Isaac Sim 6.2.0.0/PyTorch 2.11/CUDA 13.0/Newton 1.6.1/Warp 1.17/RSL 5.5.1 stack. Repeating it
only reuses an installation when source and lock match; changed installs are
refused. The bootstrap and runtime preflight check actual cgroup memory/CPU,
GPU/driver, free disk, Vulkan RTX extensions and the selected PyTorch GPU.
Unknown physical SSD backing is reported as a warning; check the provider's
advertisement and pilot I/O performance. No Windows SDK environment is copied.

Keep `/workspace/hiwonder` on the paid persistent disk. Install caches and job
outputs are kept there where supported. Container-local disk survives stop/start
only under provider rules; deleting an instance can delete its local disk.
Download checkpoints before destroying it. Stopped disk may continue billing.

## Smoke, benchmark, then train

Commands below run from this overlay directory. Omit `--viz` for headless wrist
camera rendering; Isaac Lab 3 does not use the old `--headless` option.

```bash
python3 cloud/run.py smoke --workspace /workspace/hiwonder
python3 cloud/run.py pilot --workspace /workspace/hiwonder --envs 4 --precision fp32
python3 cloud/run.py pilot --workspace /workspace/hiwonder --envs 4 --precision bf16
```

Before a long run, test a short resume against the smoke checkpoint on the same host:

```bash
python3 cloud/run.py resume --workspace /workspace/hiwonder --checkpoint /workspace/hiwonder/runs/peg_insert/RUN/final.pt --iterations 2 --wall-seconds 600
```

The one-environment smoke runs two optimizer iterations with a shortened
0.2-second episode, 64 reset entries and native640x480/30 Hz capture. It verifies
execution, not task learning. Pilots keep the original task horizon and use 1024
SGS bank entries; compare the same environment count/seed and warm measurements.
Choose the precision/host by measured cost and numerics, not advertised TFLOPS.
BF16 is opt-in learner autocast; model parameters and simulation/camera/control
tensors remain FP32. It changes learning numerics, so task-quality validation is
still needed. Default FP32 behavior is retained. An exact resume preserves the
saved precision and rejects an explicit change.

After a successful pilot, a fresh full run uses the existing 32768-entry SGS
bank and the full task horizon:

```bash
python3 cloud/run.py train --workspace /workspace/hiwonder --task peg_insert --envs 4 --precision bf16 --iterations 1000 --wall-seconds 3600
```

Use `--precision fp32` if preferred. Every canonical preset is listed in
[task_presets.json](task_presets.json); for example `--task cloth_fold:rag` selects
the native deformable backend. Do not assume the largest environment count fits
24 GB VRAM: test 1, 2, 4 separately. A failed memory guard stops the owned job; it
does not quietly lower physics, camera resolution or randomization.

The default job guard allows24 GiB aggregate process-group RSS, reserves8 GiB
available instance memory, and limits smoke/pilot/train to 600/1800/3600 seconds.
`--max-rss-gib` and `--wall-seconds` are explicit overrides. Detached helpers outside
the owned process group are not accounted; provider cgroup quotas remain the
outer limit. Job metadata/logs live in `workspace/jobs`, and training output paths
are printed. Periodic/final checkpoints are made by the existing trainer at
optimizer boundaries; forced interruption may lose work after its last save.

**A finished or timed-out process does not stop provider charges.** Save output
and explicitly stop the rented instance. Deleting it and stopping it have
different storage/data-retention consequences. No provider credentials are
stored on the instance by these tools.

## Measure cost and resume

Using the exact run directory printed by training and your current compute+
storage hourly quote:

```bash
/workspace/hiwonder/IsaacLab-HiWonder/.venv/bin/python cloud/report.py /workspace/hiwonder/runs/peg_insert/RUN --hourly-usd 0.523
python3 cloud/run.py resume --workspace /workspace/hiwonder --checkpoint /workspace/hiwonder/runs/peg_insert/RUN/final.pt --iterations 1000
```

The report drops initial warmup samples and uses RSL's measured collection+
learning FPS. It includes cost per million transitions, learning fraction and
the theoretical upper bound if **only** learning doubled. Training-process wall
cost and time to first logged iteration are separate from steady-state cost.
Initial apt, downloads, shader warmup, transfers and idle time are additional.
No short benchmark establishes reliable task success. Resume uses the saved
profile/precision/environment count unless explicitly overridden and validated.
Use the same bundle/runtime for a run; old source-fingerprint checkpoints may
require fresh weight initialization. Cross-GPU bit-identical trajectories are
not promised.

For TensorBoard or the wrist inspection server, use SSH tunnels, for example
`ssh -L6006:127.0.0.1:6006 -L8769:127.0.0.1:8769 -p PORT root@HOST`.
Launch TensorBoard with `--host 127.0.0.1` and port 6006. Inspection is a separate
scene allocation; run it separately from training on a small GPU.

## Optional tensor-rate unlock

See [unlock.md](unlock.md). Normal cloud containers cannot change the host's
boot/kernel. Full VM or provider-host support, explicit permission, compatible
PCI passthrough and a recovery path are prerequisites. The tools do not install
or claim an unlock. An unlocked VM's higher hourly price may erase its gain.

## What preparation verifies

Unit tests cover cost/resource filters, precision/checkpoint behavior, cgroup
preflight, safe command planning, output separation, archive exclusions and
Linux process-group handling. Local GPU smokes and receipts are recorded with
their OS/device. Paid Linux4090 deployment, GPU unlock and learning success are
not established by those tests; the smoke/pilot gates are required on the
selected host before a long run.

See [VALIDATION.md](VALIDATION.md) for the exact tests and verification boundaries.
