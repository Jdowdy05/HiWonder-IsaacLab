# Optional Linux tensor-rate unlock

This is the project referred to in the request:

- [Zagitta/gpu-unlock](https://github.com/Zagitta/gpu-unlock): unified Linux
  RTX20/30/40 support; Ampere is the more established path, Ada experimental.
- [Pinned Ada/4090 source](https://github.com/JustAResearcher/gpu-unlock-ada/tree/d28c485418c3d9d034cd4bac19200bc33dd3453b).

It changes a boot-time SM throughput restriction on FP16/BF16/TF32 math with
FP32 accumulation. It does not create additional tensor cores, VRAM or ray
tracing cores. It does not double physics, rendering, Python or arbitrary FP32
CUDA work. The matched4090 evidence reports about 2x instruction issue rate but
only 12-34% FP16 cuBLAS improvement and shape-dependent BF16 improvement including
an unchanged case. No speedup for this HiWonder RL workload has been measured.
The repositories do not cover the RTX 5090.

## Rental constraints

Ordinary Vast/RunPod containers share the host kernel. Container root is not
permission or access to modify the host GPU boot process. Vast also offers full
VMs; TensorDock uses KVM VMs. Those are possible investigation platforms, not
proof that this module will work or that a provider permits it.

Before considering it, ask the provider, without sending account credentials:

> Does this exact VM provide a dedicated RTX 4090 via full PCI passthrough,
> guest-kernel module/bootloader control and a console/rollback path? Is the
> upstream gpu-unlock pre-driver GPU modification explicitly permitted on this
> rental? Does a guest reset provide the PCI/GPU reset behavior it requires?

No provider was contacted by this preparation. At the researched prices, a
$0.666/hour TensorDock 4090 VM needs about 47% more whole-job throughput to beat
the $0.4542/hour stock Vast 4090. A doubled tensor microbenchmark alone cannot
establish that break-even point. Host-supplied supported unlocking at the same
price would change the comparison; remeasure rather than assume.

## Prepare a matched experiment on an approved host

1. Establish a working stock driver/Vulkan stack and complete the normal smoke.
2. Save host/GPU UUID, driver, power limit and exact project/runtime versions.
3. Run the same profile/env count/seed/precision pilot and save its cost report.
4. Record a baseline GEMM test using the installed Python:

```bash
/workspace/hiwonder/IsaacLab-HiWonder/.venv/bin/python cloud/tensor_bench.py --output baseline-gemm.json --label baseline
```

5. An authorized operator follows the pinned upstream single-card quick start,
   build/boot instructions and recovery procedure. The module runs before the
   stock NVIDIA driver; running `make install` inside a normal rental container
   is not an installation path. The4090 path requires exact PCI ID 10de:2684;
   use the specific BDF, never target every GPU on a shared host.
6. Retain the operator's actual boot/driver-handoff evidence. Module presence
   or a faster PyTorch run alone is not proof that the restriction is removed.
7. Repeat both GEMM and the real RL pilot at the same power limit/driver. Check
   finite outputs, task behavior, error logs and renderer/solver stability, then
   compare total cost per transition. Keep an accessible stock boot path.

The benchmark is bounded to at most4096-square matrices and verifies numerical
error. Its label is user-supplied host state, not an unlock attestation. It never
loads a kernel module or changes a driver. The standard launcher's BF16 option
affects PPO learner autocast only; it leaves simulation precision untouched.
