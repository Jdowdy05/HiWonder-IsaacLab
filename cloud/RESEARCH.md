# Hourly GPU market and Linux tensor-unlock assessment

Research snapshot: 2026-10-29, approximately 05:41–05:45 UTC (01:41–01:45 America/New_York). USD, on demand, one GPU. Inventory is a snapshot, not a reservation or a successful deployment. No paid instances, account changes, provider messages, driver/kernel changes, or unlock execution occurred.

## Latest included refresh

The later included public query at2026-10-29T06:09:45.181074+00:00 found a lower global4090 candidate: offer52582393, Thailand, TH, **$0.4542/hour including200GB**. The US list starts with offer58188739, Washington, US, **$0.5111/hour**. These newer files govern the current provider-spec; the detailed earlier snapshot below remains historical evidence. No offer is reserved.

## Decision at the earlier snapshot

Use a **stock RTX 4090 on Vast first**, then select by measured dollars per fixed amount of the existing 640×480 RGB plus physics workload. The lowest qualifying quote observed was **$0.4822/hour with 200 GB** in Pakistan. For a practical US shortlist, Pennsylvania offer **55917115 at $0.5230/hour** has approximately 128 GB RAM, 32 API effective CPUs, a 450 W GPU limit and 0.9970 reported reliability. Washington offer **58188739 at $0.5111/hour** is slightly cheaper but has approximately 64 GB RAM and a 410 W limit. These are candidates for compatibility and throughput checks, not measured Isaac Lab winners. [Vast search API](https://docs.vast.ai/api-reference/search/search-offers)

The **lowest qualifying cash rate is an RTX 3090 at $0.3489/hour with 200 GB** in Massachusetts. Its older Xeon CPU and 300 W GPU limit make the lowest price alone insufficient to establish best training value. The 4090 only needs **1.382× the end-to-end throughput** of that 3090 to cost less per completed workload at the global minimum quotes; the practical Pennsylvania 4090 needs approximately **1.499×**. Measure the same environment count, resolution, rendering cadence, precision and update settings before choosing.

**TensorDock is the strongest currently priced full-VM alternative:** 3090 at **$0.4920/hour** and 4090 at **$0.6660/hour**, each configured with 16 vCPUs, 64 GB RAM and 200 GB SSD. Its full OS and guest-driver control are documented. That makes it a credible stock Isaac Sim fallback when container graphics/driver integration is troublesome, and a potential future unlock research host. It does **not** establish permission or technical readiness for a pre-driver GPU boot modification. [Current host API](https://console.tensordock.com/api/v0/client/deploy/hostnodes), [KVM and OS control](https://www.tensordock.com/), [guest-driver instructions](https://docs.tensordock.com/virtual-machines/updating-nvidia-drivers-on-ubuntu)

Do not pay the VM premium solely for a claimed 2× tensor unlock. The relevant 4090 evidence distinguishes nearly 2× instruction throughput from **12–34% FP16 cuBLAS gains and shape-dependent BF16 gains, including an unchanged case**. There is no measured speedup for this Isaac Lab workload. [Matched 4090 evidence](https://github.com/JustAResearcher/gpu-unlock-ada/blob/d28c485418c3d9d034cd4bac19200bc33dd3453b/docs/evidence/rtx4090-pr1-matched/README.md)

## Strict Vast query and rental subtotal

The fresh public POST requests used `https://console.vast.ai/api/v0/bundles/` with:

```json
{
  "limit": 1000,
  "type": "on-demand",
  "verified": {"eq": true},
  "rentable": {"eq": true},
  "rented": {"eq": false},
  "gpu_name": {"eq": "RTX 4090"},
  "num_gpus": {"eq": 1},
  "cpu_ram": {"gte": 64000},
  "cpu_cores_effective": {"gte": 8},
  "disk_space": {"gte": 200},
  "reliability2": {"gte": 0.99},
  "driver_version": {"gte": "595.58.03"},
  "allocated_storage": 200,
  "order": [["dph_total", "asc"]]
}
```

The GPU name changed for the 3090/5090 queries. VM queries additionally set `vms_enabled: {eq: true}`. Returned driver versions and resource fields were also inspected. Counts were 9 matching 3090 offers, 22 matching 4090 offers and 24 matching 5090 offers. Strict VM queries returned zero 3090 and three 4090 offers. This minimum is **64,000 MB as reported by the API**, a 64 GB class allocation rather than a promise of 64 GiB usable RAM. “Effective CPUs” is the API field, not a count of dedicated physical cores. [API field definitions and operators](https://docs.vast.ai/api-reference/search/search-offers)

All rows below include **200 GB allocated disk in the returned quote**, not the initial default 8 GB. `dph_total` equals `dph_base + storage_total_cost` for these rows. CPU and RAM are included in Vast's base rate. Traffic and any tax are additional.

| Candidate | Offer ID | GPU/base $/h | 200 GB $/h | Rental subtotal $/h | RAM MB / effective CPU | Driver | Reliability2 | API egress $/TB |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| 3090, Massachusetts | 47766459 | 0.2933 | 0.0556 | **0.3489** | 128893 / 24 | 595.71.05 | 0.9972401 | 4.00 |
| 3090, Arizona | 57435749 | 0.2533 | 0.1111 | **0.3644** | 64235 / 12 | 595.91.07 | 0.9932586 | 20.48 |
| 4090, Pakistan | 58101059 | 0.4267 | 0.0556 | **0.4822** | 64207 / 16 | 595.91.07 | 0.9949188 | 2.67 |
| 4090, Washington | 58188739 | 0.4667 | 0.0444 | **0.5111** | 64006 / 16 | 610.57.04 | 0.9958370 | 5.33 |
| 4090, Pennsylvania | 55917115 | 0.4933 | 0.0296 | **0.5230** | 128979 / 32 | 595.91.07 | 0.9970327 | 4.00 |
| 4090 VM, France | 53655469 | 0.6667 | 0.0556 | **0.7222** | 128569 / 24 | 595.84 | 0.9975706 | 4.00 |
| 4090 VM, Romania | 54902719 | 0.6667 | 0.0926 | **0.7593** | 64465 / 32 | 595.84 | 0.9976550 | 13.33 |
| 5090, Slovakia | 55265147 | 0.6533 | 0.0370 | **0.6904** | 64226 / 16 | 595.71.05 | 0.9960005 | 2.67 |

The executor retained full raw inventory outside this transfer bundle. Included [global](quotes-global-2026-10-29.json) and [US](quotes-us-2026-10-29.json) reports retain queries, timestamps, requested200GB storage and selected offer fields. Refresh withcloud/offers.py before renting; the public API is linked above.

Useful distinctions between the cheapest 3090s: Massachusetts reports Xeon E5-2690 v3, 300 W GPU limit, 11.8 GB/s PCIe bandwidth and 209.5 GB free disk; Arizona reports Ryzen 9 3900X, 420 W limit and 25.1 GB/s PCIe bandwidth. The Arizona offer only had about **2.47 days** remaining in its advertised availability at the snapshot, and higher transfer pricing. The Massachusetts disk is only slightly above the requested 200 GB, making an install/cache growth check particularly important. These are host-advertised measurements, not locally verified sustained performance.

Approximate uninterrupted 24-hour rental subtotals are $8.37 for the cheapest 3090, $11.57 for the cheapest 4090, $12.55 for Pennsylvania 4090, and $16.57 for the cheapest 5090. Installation, cache compilation, benchmarking and idle time are billable while running. Vast bills storage while the instance exists, including stopped time, and charges host-dependent ingress and egress. [Vast pricing and billing components](https://docs.vast.ai/guides/instances/pricing)

## TensorDock: current resource-inclusive prices

The public host inventory and the same session endpoint used by the deployment page were both accessible without credentials. The second query included `minGPUCount=1&minRAM=64&minvCPUs=16&minStorage=200&minVRAM=24&requiresRTX=true`. The page's `calculateVMPrice` sums each component, rounded to four decimals, using the larger of the requested resource count and each host's minimum restriction. The 16 CPU / 64 GB / 200 GB configurations below are inside their one-GPU maximum restrictions.

| GPU, location | Host ID | GPU $/h | 16 vCPU $/h | 64 GB RAM $/h | 200 GB SSD $/h | Total $/h | Reported uptime | Available GPUs |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 3090, Dallas | `49487c87-9544-4ae7-b8ff-ed512ed80257` | 0.2000 | 0.0640 | 0.1280 | 0.1000 | **0.4920** | 0.9955 | 6 |
| 4090, Orlando | `2d77d865-c5e8-412a-862d-7120bc236ced` | 0.4800 | 0.0480 | 0.1280 | 0.0100 | **0.6660** | 0.9985 | 8 |
| 5090, Orlando | `59184cd9-2c52-4aa9-ba33-9c6b0073ca2c` | 0.5800 | 0.0480 | 0.1280 | 0.0140 | **0.7700** | 0.9915 | 8 |

All three were listed, online and unreserved. CPU types are respectively EPYC 75F3, EPYC 7713 and EPYC 7J13. With eight rather than sixteen requested vCPUs the arithmetic falls to $0.4600, $0.6420 and $0.7460 per hour, subject to the same final configuration check. The public inventory does **not** report the installed NVIDIA driver, power cap, or a guarantee of physical CPU exclusivity. Therefore these are priced VM candidates, not fully driver-qualified strict Vast equivalents. [Public host inventory](https://console.tensordock.com/api/v0/client/deploy/hostnodes), [deployment page and its current calculation](https://console.tensordock.com/deploy)

The official deployment page's “from” cards showed 3090 $0.23, 4090 $0.50 and 5090 $0.60/hour; these include tiny default resource allocations and are not the requested 64 GB configuration. Older promotional pages showing $0.35/$0.37 for a 4090 were not used as current quotes. The current TensorDock home page advertises no ingress/egress fees and KVM, root access, dedicated GPUs, full OS control and user-managed drivers. Its driver guide explicitly documents updating a guest driver and rebooting the VM. Neither statement is explicit permission for `gpu_unlock`. [TensorDock platform capabilities](https://www.tensordock.com/), [Linux driver management](https://docs.tensordock.com/virtual-machines/updating-nvidia-drivers-on-ubuntu)

TensorDock raw captures were retained outside this bundle. The public host API and deployment-page links above are the reproducible sources; no instance was deployed.

## Runpod cross-check

The official product catalog displayed 3090 $0.50/hour with 125 GB RAM and 16 vCPU, versus 4090 $0.89/hour with 41 GB RAM and 6 vCPU. That displayed 4090 configuration fails the requested host RAM/CPU floor. The executor's public GraphQL snapshot (retained outside this bundle) returned null Community Cloud price/stock fields for both cards; that does not prove they are unavailable everywhere, but supplies no live bookable cheap quote. The same response advertised 5090 Community at $0.69/hour, 146 GB RAM / 37 vCPU, stock “Low.” At $0.10/GB/month for 200 GB running local storage, add approximately $0.0278/hour using 720 hours/month. Ingress/egress is advertised free. These are alternatives to recheck, not reserved capacity. [Runpod official catalog and storage pricing](https://www.runpod.io/product/cloud-gpus)

Runpod's technical documentation describes templates as Docker images and supports custom containers. Do not infer host kernel/initramfs control from a marketing FAQ's broad “VM/drivers” language. For the tensor-unlock workflow, an ordinary Pod remains an unqualified candidate unless Runpod explicitly supplies a supported full-VM/bare-metal arrangement and permission. [Pods deployment model](https://docs.runpod.io/pods/overview)

## What the new Linux unlock actually establishes

Pinned research references:

- `JustAResearcher/gpu-unlock-ada`, main commit **d28c485418c3d9d034cd4bac19200bc33dd3453b**, committed 2026-10-26. [Pinned README](https://github.com/JustAResearcher/gpu-unlock-ada/blob/d28c485418c3d9d034cd4bac19200bc33dd3453b/README.md)
- `Zagitta/gpu-unlock`, main commit **7eb51bccae0a0b7e599e084d971170e9a1e82d0c**, committed 2026-10-28, README version 2.0.0. It labels Ampere stable, Ada experimental and Blackwell unsupported. [Pinned README](https://github.com/Zagitta/gpu-unlock/blob/7eb51bccae0a0b7e599e084d971170e9a1e82d0c/README.md)

This is a Linux kernel/initramfs intervention that runs before NVIDIA binds the GPU, changes the applicable speed selectors and performs a GPU function-level reset before the stock driver starts. It requires kernel headers/build tooling, an early-load path and boot command-line control. It is not an in-container PyTorch switch. The fork's RTX 4090 route targets PCI ID `10de:2684`; the downloaded pinned firmware input version is not a general requirement to replace the installed NVIDIA driver with that same version. [Mechanism and supported targets](https://github.com/JustAResearcher/gpu-unlock-ada/blob/d28c485418c3d9d034cd4bac19200bc33dd3453b/README.md)

The matched owner-controlled 4090 run used the same card, 450 W cap, driver and benchmark binaries. It found FP16/FP32-accumulate 160.4→320.4 TF/s and BF16/FP32 159.7→320.2 TF/s at the instruction level. FP16 cuBLAS ratios ranged 1.120–1.344; BF16 ranged 0.995–1.196. Exercised numerical checks passed, but this was not general numerical proof or a long stability test. One warm-repeat BF16 shape was slower. These results justify controlled evaluation; they do not justify assuming 2× simulation, rendering, PPO or wall-clock training. [Exact matched evidence and limits](https://github.com/JustAResearcher/gpu-unlock-ada/blob/d28c485418c3d9d034cd4bac19200bc33dd3453b/docs/evidence/rtx4090-pr1-matched/README.md)

Vast **does now offer full VMs**, selected with `vms_enabled=true`; the earlier Docker-only assumption is obsolete. Its docs establish a Linux VM, systemd, nested Docker and SSH access, but do not explicitly guarantee all GPU PCI configuration space/reset/ROM operations required by this module. The France and Romania offers above had `vms_enabled=true` and `is_vm_deverified=false`; this establishes advertised VM eligibility, not a successful guest unlock. [Current Vast VM documentation](https://docs.vast.ai/guides/instances/virtual-machines)

A published **4070 Ti** test succeeded in a VFIO/KVM guest only after the host bound VFIO at boot and kept host NVIDIA/nouveau away from the card. It is useful evidence that a VM route can work, but is not proof for a cloud 4090's passthrough/reset policy. [VFIO host/guest conditions](https://github.com/JustAResearcher/gpu-unlock-ada/blob/d28c485418c3d9d034cd4bac19200bc33dd3453b/docs/ada-2782-report.md)

Before any future rented-host attempt, obtain explicit provider authorization covering the pre-driver kernel module, GPU firmware execution/selector modification, reset operations, and required recovery. Also establish the exact guest GPU PCI function, boot ownership, safe rollback and whether a fresh GPU power state can be provided. A guest reboot may not power-cycle the physical GPU. The fork's first-test quick start calls for a fresh full power-off and an exact-card success log plus a healthy driver and workload test. None of these provider-specific gates was established in this research. [Quick-start prerequisites](https://github.com/JustAResearcher/gpu-unlock-ada/blob/d28c485418c3d9d034cd4bac19200bc33dd3453b/QUICKSTART.md)

That provider-permission gate is an operational requirement for touching rented hardware, not a claim that all kernel changes are forbidden. TensorDock documents ordinary driver changes, while its AUP prohibits unauthorized system access/disruption and bypassing system limitations; no published `gpu_unlock` exception was found. Use the provider's supported channel if the user later authorizes that discussion. [TensorDock AUP](https://docs.tensordock.com/legal-information/acceptable-use-policy-aup.md)

## Cost and speed break-even

For equal completed work and negligible fixed setup cost, the higher-rate machine wins when `speed_fast / speed_slow > hourly_fast / hourly_slow`.

| Comparison | Required whole-workload throughput ratio |
|---|---:|
| Cheapest stock Vast 4090 vs cheapest stock Vast 3090 | **1.382×** |
| Washington stock 4090 vs cheapest 3090 | **1.465×** |
| Pennsylvania stock 4090 vs cheapest 3090 | **1.499×** |
| Cheapest 5090 vs cheapest 4090 | **1.432×** |
| TensorDock 4090 VM vs cheapest Vast stock 4090 | **1.381×** |
| Vast France 4090 VM vs cheapest Vast stock 4090 | **1.498×** |
| TensorDock 3090 VM vs cheapest Vast stock 3090 | **1.410×** |

This makes the VM premium important: the stock-to-unlocked 4090 must improve the **entire job by more than 38.1%** merely to offset TensorDock's 16-vCPU VM rate against the cheapest stock container, or 49.8% for the cheapest qualifying Vast VM. Comparing against a US stock offer narrows the premium, but does not eliminate it. An unlock with the same host/rate has a different break-even and may still be worth researching after permission and correctness checks.

If only fraction `f` of elapsed time benefits from a tensor speedup `k`, ideal end-to-end speedup is `1 / ((1-f) + f/k)`. Even assuming `k=2`, tensor work consuming 20%, 50% or 80% of baseline time yields only 1.111×, 1.333× or 1.667× overall. Rendering, physics, reset logic, CPU work, transfers and optimizer operations do not automatically gain that tensor rate. The affected fraction for the actual AMP/TF32 execution path is unmeasured.

For short runs include setup explicitly: `total = rate * (setup_hours + training_hours) + stopped_storage + traffic + taxes`. Keep simulated camera tensors and rollouts on the rented machine; export checkpoints, scalar logs and selected compressed videos. Repeated source downloads, shader-cache builds and environment repair can erase small hourly savings. Compare dollars per fixed training steps and deterministic evaluation gates, including any precision-induced learning difference, before scaling up.

## Verification boundary

The research verifies current public quotes and documented environment capabilities. It does not verify account-specific checkout, allocated runtime resources, Vulkan/RTX rendering, native 640×480 sensor correctness, physics parity, sustained temperature/clock behavior, kernel-module admission, successful unlock, numerical equivalence or RL convergence on any rental. No measured cloud Isaac Lab throughput is claimed.
