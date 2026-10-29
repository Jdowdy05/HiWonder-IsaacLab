# Independent cloud training review

**Closure status: all three P2 findings resolved; no outstanding P0/P1/P2 finding in the reviewed scope. See the closure section below.**

Reviewed 2026-10-29 against base `769e098b78d2dcb56baa831d1693194bd613a80f`, working branch `codex/cloud-training`. Review was read only on the checkout. No SSH, hardware, GPU/Kit, paid cloud, driver or kernel action was performed.

## Strengths

- `cloud/run.py` keeps job sidecars outside the trainer's output path, which is created exclusively by `train_hiwonder.py`. Runtime preflight uses the installed `.venv/bin/python`.
- BF16 is explicitly opt in; the saved learner precision is preserved for exact resume and an explicit mismatch is rejected. Inspection of pinned RSL-RL confirms autocast is inside the PPO update; the added code does not change physics, camera resolution, render cadence, randomization or solver settings.
- Offer ranking includes the returned disk charge, preserves quote time and resource fields, rejects stale offline snapshots, and never calls rental/billing endpoints.
- The unlock writeup distinguishes instruction throughput, measured cuBLAS gains and unmeasured RL speed. The pinned primary repositories support the stated experimental Ada/unsupported Blackwell boundary. It does not install an unlock or infer provider permission.
- The archive is built from committed explicit files and excludes the listed credential/run/environment paths. Upload verifies bytes and performs no launch.

## Initial findings (resolved at closure)

### [P2] The memory reserve misses nested process and ancestor cgroups

Location: `cloud/guard.py:45-58`.

`available_memory()` reads host MemAvailable and only two fixed cgroup mount roots. Unlike the preflight collector, it never reads `/proc/self/cgroup` or walks the process's current cgroup and its ancestors. On a VM or container without a cgroup namespace rooted exactly at its allocation, a valid >=60 GiB ancestor limit can become nearly full while the host still has ample free RAM. The advertised 8 GiB reserve then does not stop the job before an allocation OOM; the separate 24 GiB process-group RSS budget does not account for unrelated processes within the allocation.

CPU fixture: host MemAvailable=80 GiB; `/proc/self/cgroup`=`0::/tenant/job`; root memory.max=max; `/tenant/memory.max`=64 GiB and memory.current=60 GiB. The current function returned **80 GiB**, although only **4 GiB** remained inside the applicable cgroup. This is a concrete missed reserve threshold.

Fix: walk the visible current memory hierarchy and each ancestor, for v1 and v2, and take the minimum of host MemAvailable and every finite `limit-current` pair. Add a nested-ancestor regression. Preserve the owned-process-group termination boundary.

### [P2] Installation reuse accepts changed tracked upstream source

Location: `cloud/verify_install.py:13-24`; claimed behavior in `cloud/README.md:75-80`.

The manifest verifies the overlay payload and lock/pyproject replacements, then `git rev-parse HEAD` checks the upstream commit. An unchanged HEAD does not mean the upstream working tree matches that commit. Edits to tracked upstream code such as `source/isaaclab/app/launcher.py` are outside the overlay manifest and are accepted. A later bootstrap thus reports matching pinned source and reuses a modified runtime despite the documented refusal contract.

CPU reproduction used an actual isolated scratch Git repository: commit a minimal upstream launcher; make pins reference that exact HEAD; make all manifest hashes match; modify the tracked launcher without committing. `git status --porcelain` showed `M source/isaaclab/app/launcher.py`, but `verify_install()` returned success. No project source was changed by this reproduction.

Fix: also reject tracked upstream changes relative to the pinned commit, including index changes, except the exact intentionally overlaid files whose bytes are separately verified. Retain refusal rather than overwriting an edited installation.

### [P2] The storage quote is GB but the public interface labels it GiB

Location: `cloud/offers.py:65-85,151-157,174-182,202`; `cloud/provider-spec.json:10`; `cloud/README.md:3,6,27,38`.

The launcher sends `allocated_storage=200` and filters `disk_space>=200`, then reports `allocated_storage_gib=200`, `storage_gib=200` and describes prices as including 200 GiB. Vast's official API defines both input fields in **GB**, and `cloud/RESEARCH.md` correctly calls these 200 GB quotes. The present price labels therefore claim about 7.4% more storage than was actually requested. This also makes the documented rental target inconsistent with the quote evidence.

Primary source checked live: [Vast search offers API](https://docs.vast.ai/api-reference/search/search-offers), `allocated_storage` and `disk_space` field definitions.

Fix: consistently call the existing target and quotes 200 GB (including CLI/schema names), or convert 200 GiB to at least 214.75 GB, round up as required by the provider, and refresh quotes. Do not relabel existing quote evidence as a larger allocation.

## Documentation follow-up

`cloud/RESEARCH.md:53,73,77` names `market-*.json` and `runpod-public.json` as local evidence; none of those files is in the `cloud/` checkout. They exist in the separate workspace research folder. The bundle's two `quotes-*.json` reports preserve the selected stock Vast fields, but do not supply the cited full VM/TensorDock/Runpod raw snapshots. Include sanitized evidence in the bundle, or rewrite the references to the evidence actually delivered and state the boundary. This is a reproducibility/document-location issue, not evidence that the prices were fabricated.

## Verification performed

With canonical Python, `CUDA_VISIBLE_DEVICES=-1`, bytecode disabled and pytest cache disabled:

```text
python -m pytest -q -p no:cacheprovider tests/test_cloud_run.py tests/test_cloud_install.py tests/test_cloud_report.py tests/test_cloud_bundle.py tests/test_cloud_guard.py
24 passed, 2 skipped in 0.24s
```

The skips were the Windows-inapplicable Linux process-group integration checks. Two additional CPU fixtures reproduced the nested-cgroup reserve and dirty-upstream acceptance findings above. The full suite, actual Linux process behavior, local GPU smoke and final checksum refresh belong to the integrator's separate receipts.

## Declined to judge

- Actual provider GPU passthrough, guest boot/reset behavior and kernel-module permission: no paid host or provider permission was established; no unlock action was authorized in this review.
- Linux RTX 4090 renderer/physics compatibility, throughput and convergence: no such deployment was performed, and CPU fixtures or a Windows 5090 cannot establish it.
- Sustained numerical or task-quality equivalence of BF16: the small finite-update tests establish plumbing, not learning-quality equivalence.
- Unrelated legacy upstream implementation and generalized security hardening: outside the scoped cloud/AMP changes.

## Initial assessment (superseded by closure)

**Ready to publish: with fixes.** No P0/P1 issue was found in the reviewed launch/precision flow. Resolve the two resource/source verification defects and the storage-unit mismatch before presenting the branch and bundle as ready for a rental pilot. Then rerun affected checks, refresh installer hashes, and retain the explicit requirement for actual selected-host smoke/pilot qualification.


## Closure review, 2026-10-29

The implementer corrected all three P2 findings and the documentation follow-up. This closure pass independently reread the current source and tests, ran the complete focused cloud tests with CUDA disabled, checked quote consistency, read the actual Linux fixture receipt, and read saved local GPU result/profile/benchmark receipts. The checkout was not edited by the reviewer.

- **Nested cgroup reserve: resolved.** `guard.available_memory()` now uses the same current-process/ancestor path parser as preflight and computes the minimum finite remaining capacity. Fresh v1 and v2 regressions both assert the original host-80-GiB/nested-4-GiB case returns 4 GiB. Monitor errors now produce an unsuccessful status and owned-group cleanup.
- **Upstream reuse: resolved.** Reuse now compares tracked files with HEAD, refusing changes except the exact `pyproject.toml` and `uv.lock` replacements whose bytes are separately verified by the overlay manifest. The real scratch Git regression passes and confirms the edited file is preserved rather than replaced.
- **Storage units: resolved.** The CLI is now `--storage-gb`; reports and provider spec use `storage_gb`/`allocated_storage_gb`; the unchanged 200 GB API allocation is stated accurately in the docs and refreshed quote evidence. No 200 GB snapshot was relabeled as 200 GiB.
- **Evidence locations: resolved.** Research now explicitly states that full raw captures are outside the transfer bundle and links the included stock Vast quote reports and public APIs.

Fresh independent command:

```text
CUDA_VISIBLE_DEVICES=-1
PYTHONDONTWRITEBYTECODE=1
python -m pytest -q -p no:cacheprovider tests/test_cloud_guard.py tests/test_cloud_install.py tests/test_cloud_offers.py tests/test_cloud_preflight.py tests/test_cloud_run.py tests/test_cloud_report.py tests/test_cloud_bundle.py
101 passed, 2 skipped, 19 subtests passed in 0.70s
```

`git diff --check` also returned success. The recorded Linux CPU fixture run (`linux-fixture-final.log`) reports **102 passed, 1 deselected** in 1.43 seconds, including the Linux process checks; the catalog-only test was deselected on that fixture. This review inspected that receipt and did not initiate SSH or rerun Linux commands remotely.

The refreshed included global report and provider spec agree on **offer 52582393, Thailand, RTX 4090, $0.4542222222/hour including 200 GB**, advertised CPU RAM 96539 MB, 64 effective CPUs and 450 W power. The included US report starts with **58188739 at $0.5111111111/hour**. All included selected rows were checked for `base + disk == total`, matching 200 GB request/schema fields, and the selected spec row's location/rate. These remain timestamped quotes, not reserved capacity. The detailed older price arithmetic is explicitly introduced as historical snapshot evidence.

The saved local smoke result reports `bf16_autocast`, two completed iterations, four optimizer steps and finite parameters. Its saved profile is Eka V6 with native 640x480/30 Hz camera settings. The separate GEMM receipt identifies Windows RTX 5090 and finite, passing numeric checks for FP32/FP16/BF16. These receipts establish local plumbing/numerical smoke evidence only. Linux cloud 4090 compatibility, unlock success, cloud throughput, exact cross-GPU replay and learned task quality remain outside the demonstrated evidence.

**Final assessment: ready to publish the reviewed cloud-preparation branch and credential-free bundle. No outstanding P0/P1/P2 issue was identified in the reviewed scope.** Final commit/archive creation, archive byte verification and installer checksum refresh remain the integrator's release checks; the selected paid host still requires its documented stock smoke and pilot before training. No paid instance or unlock was created or applied during this review.

Closure source fingerprints (working tree at review time):

- `cloud/guard.py`: `9c39dfc7b7d7d70d48887a3e2592b52efdf2512d156f29c8d2f6270ea0011c3a`
- `cloud/verify_install.py`: `ba24951ade18b1dac6ce2af8197cdfedeed163464f624a39c55c407a895e8bca`
- `cloud/offers.py`: `7ba4e7a29b36a22607dcebd7eb7acfdc79757c71dd136aa03bd2aae9f3fa48ae`
- `cloud/README.md`: `47ffdb2b3527a3ab0679b6e2363839879eea9ef02c675bb057d9288ac33a2d6c`
- `cloud/RESEARCH.md`: `1259055012d7c1cf6850739c71a8ea011e7897f1ab38c5aeedad1c2a703420e0`
- `cloud/provider-spec.json`: `22130ee0a0aedb9dd362508fde7dc8af77360fa605f8b188772a337f91373f54`
- `cloud/quotes-global-2026-10-29.json`: `40ef56934dab69b4ff07d435a1ea230afa50dc5c51423e718b5c5c898a19e426`
- `cloud/quotes-us-2026-10-29.json`: `745d3856b7bae10a55182dac86470aec15d010c0bc2093a3e9b886a2c80ff5ec`
