# Runtime optimization evidence, 2026-10-19

> Historical validation record. Dates, source digests and results below refer to the original recorded experiments. `external-evidence/` denotes owner-retained evidence outside this repository; those large bundles are not included. This publication did not rerun simulator or hardware qualification.

All six optimizations are implemented. All 30 presets passed the finite-step execution/reset/restore fixtures in both source versions (32 main control steps; these historical workers were not the new memory-capped processes). Full preservation qualification remains incomplete: strict native trajectories, renderer acceptance and the supplemental live checks have the limits below.

Tested package: `3c1a2a34179b8362c3bc7a616be5978f581ae986f6a48ecb7721df6485c4a403` (48 Python files). Frozen reference: `a0f1a99c330b20ab8ba373ff7ed1e2a2ff1f2a3d80748d9af7dd74ca2d9f1951` (47 files). Eight production files changed. Evidence directory: `external-evidence/optimization-2026-10-19`.

## Changes and preserved settings

1. Select live GPU body/joint rows and batch host transfers; preserve per-component dtypes and the original arithmetic rounding.
2. Collect only mechanism inputs needed at the original physics cadence; retain full-state fallback for custom/deformable mechanics and fresh dependency derivation.
3. Cache invariant root/contact mappings and cable rest lengths with model, definition and content invalidation. Refresh live centers of mass, states, forces and contacts.
4. Reuse bounded image grids/masks/color constants. Preserve operation order, black borders, autocast/dtype/device behavior and returned-buffer ownership. Guard unversioned tensor alias mutations.
5. Reuse exact SGS probabilities while complete history/counts/parameters match. Preserve sampling RNG consumption, FIFO/outcome order and checkpoint behavior.
6. Reuse immutable geometry and exact texture bytes through per-run content-addressed assets. Keep local materials, lights and physical properties independent; retain existing references, metadata and time samples.

The Warp-to-Torch path uses the existing DLPack producer/consumer stream handshake. Camera remains 640x480 at configured 30 Hz, control 30 Hz, dt=1/240 s and eight physics ticks per control interval. Native VBD remains 20 iterations with eight solver substeps. Solver/backend configuration, collision cadence/coverage, task mechanics, authored physical CAD and randomization ranges were not reduced. Geometry cohorts retain their existing launch-time cadence.

Implementation changes deliberately change the SGS bank identity. Use `--initialize-from` to transfer an older policy instead of importing incompatible old sampling histories.

## Tests and review

- 304 CPU package tests and the retained CUDA stream regression passed (305 total). Ruff source/test checks passed.
- Independent and crossed review reproduced and repaired stale alias caches, autocast/dtype/rounding changes, live COM and mechanism dependency errors, cable/contact invalidation, USD relocation/references/time samples and the Warp/Torch stream race. Exact negative reproductions were retained and replayed against the final source.
- Each of 30 presets ran in two RGB environments: 32 main control steps, eight warmups and 24 timed samples, running checkpoint/interrupted reset, eight restored steps, indexed reset and eight more steps. All 60 processes passed existing finite/health/reset/restore contracts with no unexpected failure hidden by autoreset.
- All 30 composed two-room scenes match in typed geometry/physics/material bindings, camera/light properties and ownership. All 13 native CPU imports match in 159 ordered captured fields, including rigid bodies/joints, particles, triangles, edges, tets, groups and custom attributes. CPU import parity does not establish opaque finalized GPU state identity.
- GPU image/SGS differential tests passed at full resolution, batches 1/2/8, none/one/all transforms, smudge drift, mutation, checkpoint and autocast cases. Counters: 83 image calls, 11 parameter records, nine probability vectors, 36 sample sequences and nine sampler states. Every global RNG remained unchanged by the component benchmark.
- Actual RGB/SGS PPO passed on PhysX peg and native Newton shirt: two iterations, then fresh-process checkpoint continuation to iteration three. The fixture enabled all ten randomization groups, moving RGB spots/vibrancy, drifting smudge, physical distractors and confetti. Consumed images/history, sampler/group/camera/global RNGs, motor/lighting state, policy weights and optimizer restored exactly before continuing; learner values stayed finite.

## Strict comparison and incomplete checks

Seventeen PhysX presets have exact recorded physical/logical trajectories. Thirteen native Newton presets failed the strict cross-process trajectory comparison. Repeating unchanged baseline and unchanged optimized barrel runs also diverged. These failures are retained, without a new tolerance, a nondeterminism exemption, or a claim that every observed difference lies inside one baseline envelope. Initial pose/configuration checks, CPU imports and bounded contracts pass, but do not turn trajectory failures into passes.

Ordinary all-30 traces record randomization/SGS RNG coverage, not every environment generator. Full environment RNG coverage is separately checked by the PPO contracts and applicable additional diagnostics. Rendered pixels are not bit-identical across processes; exact composition and exact augmentation are distinct from renderer acceptance. No numerical pixel threshold has been used to hide a physical-state difference.

The user terminated an additional same-live native cloth diagnostic after reporting RAM growth above 15 GB. Its partial archive is incomplete. The prior baseline barrel same-live diagnostic passed all 12 anchors. Remaining same-live native cases, finalized GPU initial-input comparison and longer quiet repeats must be read from separate completed receipts; the interrupted run provides none of those passes. `memory-incident.md` records the event and the diagnostic-only memory amplification found afterward. A replacement must be memory bounded before another simulator launch.

Raw native quiet-window/resting flags were not uniformly true, including unchanged baseline repeats. The existing optional raw-rest metric was not made required or relaxed to manufacture acceptance. This report establishes bounded execution, not learned manipulation success, resting equivalence, physical camera calibration, servo readiness or sim-to-real transfer.

## Observed end-to-end timing

Median per-preset p50 step-time reduction: 5.06%. All 30 recorded p50 values were lower in this sample (range 0.42% to 36.02%). Several p95 values increased, and construction times did not uniformly improve. Some windows overlapped CPU review work; treat these as descriptive samples, not stable universal speedups. Native windows also follow differing trajectories. No claim of unchanged native trajectory cost follows from them.

Times are synchronized `env.step` measurements including physics, RGB capture and policy augmentation, excluding Kit startup, trace copies, exports and PPO. Hardware/runtime: RTX 5090, Ryzen 7950X3D, Windows 11, Isaac Sim 6.2.0, Torch 2.11.0+cu130, Newton 1.6.1, Warp 1.17.0. Two environments were used; configured 30 Hz does not imply real-time execution.

| Preset | Reference p50 / p95 ms | Optimized p50 / p95 ms | p50 reduction | Strict physical/logical trace |
|---|---:|---:|---:|---|
| peg_insert | 179.75 / 194.79 | 174.99 / 183.60 | 2.65% | Exact |
| cube_stack | 189.24 / 204.36 | 175.97 / 185.20 | 7.01% | Exact |
| nut_pick_place | 215.72 / 229.30 | 212.43 / 269.24 | 1.53% | Exact |
| washer_ring | 188.62 / 195.82 | 181.23 / 186.06 | 3.92% | Exact |
| connector_insert | 173.27 / 180.67 | 172.55 / 223.70 | 0.42% | Exact |
| shape_sort | 229.83 / 316.97 | 215.86 / 231.10 | 6.08% | Exact |
| knob | 246.54 / 259.33 | 177.22 / 186.30 | 28.11% | Exact |
| switch | 244.50 / 344.35 | 195.23 / 214.62 | 20.15% | Exact |
| latch | 281.36 / 288.97 | 198.60 / 246.28 | 29.41% | Exact |
| bottle_cap | 319.13 / 329.97 | 294.48 / 310.64 | 7.72% | Exact |
| key_lock | 344.31 / 358.99 | 270.31 / 292.52 | 21.49% | Exact |
| barrel_jack | 313.54 / 334.21 | 304.72 / 315.09 | 2.81% | Rejected; native |
| cable_untangle | 285.12 / 306.81 | 270.80 / 287.33 | 5.02% | Rejected; native |
| paper_airplane | 1565.87 / 1635.21 | 1553.69 / 1666.91 | 0.78% | Rejected; native |
| gear_assembly | 323.57 / 339.17 | 224.54 / 245.50 | 30.60% | Exact |
| circuit_restore | 386.11 / 399.29 | 247.04 / 269.49 | 36.02% | Exact |
| catch_ball | 172.58 / 221.84 | 157.89 / 165.42 | 8.51% | Exact |
| roll_ball | 179.77 / 188.54 | 172.51 / 180.55 | 4.04% | Exact |
| cloth_fold:towel | 1166.54 / 1267.17 | 1127.91 / 1174.08 | 3.31% | Rejected; native |
| cloth_fold:handtowel | 1068.28 / 1106.80 | 1036.93 / 1075.32 | 2.94% | Rejected; native |
| cloth_fold:shirt | 1870.14 / 2018.78 | 1847.04 / 1993.98 | 1.24% | Rejected; native |
| cloth_fold:rag | 1100.66 / 1145.44 | 1081.39 / 1128.32 | 1.75% | Rejected; native |
| cloth_fold:shorts | 1770.06 / 1922.27 | 1735.22 / 1868.62 | 1.97% | Rejected; native |
| cloth_fold:underwear | 1628.18 / 1790.05 | 1612.62 / 1785.63 | 0.96% | Rejected; native |
| cloth_fold:sock | 1700.21 / 1764.17 | 1676.80 / 1701.41 | 1.38% | Rejected; native |
| rope_thread:thread | 305.82 / 325.18 | 290.25 / 305.47 | 5.09% | Rejected; native |
| rope_thread:overhand | 402.94 / 474.94 | 379.57 / 394.21 | 5.80% | Rejected; native |
| rope_thread:figure_eight | 388.10 / 410.58 | 351.09 / 386.73 | 9.54% | Rejected; native |
| bottle_cap:close | 335.52 / 426.20 | 291.66 / 302.91 | 13.07% | Exact |
| roll_ball:table | 177.02 / 186.09 | 166.35 / 229.36 | 6.03% | Exact |

Component image caching was 1.31–1.64x faster by wall median across nine full-resolution cases (five warmups, 51 repeats). This is a component result. A CPU transport fixture reduced 12 host copies/851,968 bytes to one/8,192 bytes while its elapsed median did not improve; it does not establish a CPU/GPU training speedup.

## Memory controls for supplemental diagnostics

The old diagnostic retained duplicate host buffers, repeated Mesh metadata and large JSON serialization temporaries. A separate replacement streams typed data into bounded chunks and SQLite metadata, memoizes repeated Mesh objects and returns compact receipts. These diagnostic changes do not alter the tested package source. The original run had no memory telemetry; its entire user-reported peak cannot be attributed to a particular allocation from surviving files.

The independent Windows launcher passed 15 small-process tests: assignment to an owned Job Object before the suspended child resumes, hard total-job allocation denial, descendant containment, unrelated-process survival, host headroom/admission refusal, failure/interrupt cleanup, abrupt launcher death, exact argument handling, uv descendants and bounded logging. Simulator policy is 10 GiB hard private allocation limit, an 8 GiB sampled working-set stop, 12 GiB admission headroom and a 6 GiB reserve for both physical RAM and commit headroom. The resident threshold is sampled (0.25 seconds), so it is not an instantaneous hard RSS cap. Windows' private peak-accounting counter can include a denied allocation attempt; sampled accepted usage and denial notifications are separate evidence.

The launcher has refused live work during low headroom from unrelated IsaacLab-231 profiling jobs. Those processes were left untouched. Small capture tests use a separate CPU-only 2 GiB hard / 1.5 GiB sampled budget, 4 GiB admission and 2 GiB reserve. An admission refusal creates no child and remains incomplete. No simulator rerun is credited on the basis of a passing containment test. See `bounded-process-report.md`, `bounded-process-tests.xml` and the memory/admission receipts in the evidence directory.

The new capture/adapter suite was refused admission twice; its latest source has not passed that CPU suite. Its live entry point is disabled until CPU and metadata-scaling qualification passes. The 2 MiB chunk size is an I/O bound, not a proof of total process RAM: the prototype still retains metadata proportional to unique objects/views. A metadata-scaling test and observed guarded runtime memory are pending. The simulator has not been restarted after the user's interruption. The existing environment's 305 tests and all-30 finite-step results precede this diagnostic work and do not validate the replacement harness.

## Evidence

The evidence directory contains source manifests, raw JSON/NPZ comparisons and failure arrays, package/CUDA test logs, `review.md`, `review-final-cpu-rechecks.json`, `comparator-review-final.json`, `catalog-comparison.json`, `scene-asset-comparison.json`, `native-model-input-comparison.json`, `native-observed-trajectory-spreads.json`, `images-sgs-gpu.json` and the four PPO contract receipts. Original negative and interrupted evidence is retained.
