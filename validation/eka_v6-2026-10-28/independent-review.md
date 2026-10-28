# Eka V6 independent review addendum

Reviewed updated source under `C:\Users\VR2\Documents\LeRobot-HiWonder-Jetson_ws\.work\hiwonder-rl\eka-v6-integration-2026-10-28\staged-extension` and the seven updated portable documentation files under `C:\Users\VR2\Documents\HiWonder-IsaacLab`. The baseline remains published private main commit `3bc34009e7ae8fff53a235f480ae65ae369f2c38`.

This pass was read-only apart from this report. No GPU, Kit, FreeCAD, CAD changes, hardware, installation or publication was used. CPU Python subprocesses masked CUDA and disabled bytecode; pytest disabled its cache provider.

## Previously reported findings

1. **Resolved: model-aware qualification tools.** `validate_hiwonder_asset.py` now requires explicit selected-model ownership, verifies the model-specific body/joint sets, joint hierarchy, axes, limits, inertials, camera and source-backed visual bounds. The collision checker uses the selected assets, reference directory, full joint order and nine-dimensional pose sweeps; protected hand groups must be complete and correctly classified.
2. **Resolved: model selection and qualification documentation.** Root README, asset provenance, extension README, asset guide and three historical evidence records now distinguish stock from Eka, describe the four independent root/tip controls, 9/10 dimensions, 27/320 observation capacities, profiles, checkpoint incompatibility, derived caps and provisional physical camera/dynamics. Historical all-30 GPU/contact/training evidence is explicitly stock-only. No all-30 Eka solver or physical-success claim was found.
3. **Resolved: source STEP package data.** The built wheel contains the unchanged supplied STEP. Independently compared all 27 expected packaged asset files against staging, byte for byte; every file matches. STEP SHA-256 remains `141ab8a721d850ed88da9d97f17257ebce1f85cfca7a0403f4c6e20c1f60e944`.

## Fresh independent CPU checks

- Selected asset validator and collision-certificate suite: **12 passed in 13.46 seconds**. All three Eka scales validate; ownership, disconnected-pivot, negative-mass and displaced-visual mutations reject. Actual nominal stock and Eka collision certificates retain the correct protected groups.
- Raw-environment utility action paths: **6 passed in 1.72 seconds**. The real inspection, indexed-reset/task and runtime-tool paths send six or nine actions from `cfg.action_space` without assuming the raw SDK environment has `num_actions`. Source reread confirms the additional task rollout/restore allocations use the same corrected field.
- Direct stock asset validation still returns `PASS_USD_REGISTRATION`.
- Existing-output refusal checked through actual CLI calls: USD builder rejects the current manifest, cap stitcher rejects the existing repair directory, and collision cooker rejects existing nominal references. None overwrites those artifacts.
- Reviewed `stitch_eka_caps.py`: uses only boundary filling rather than global MeshFix cleanup; preserves every original source triangle before float conversion; then requires closure, winding, positive volume, one nontrivial component, and exact original contact triangles at float32 precision. The discarded components are bounded by `1e-14 m^3` and their volumes are reported. The completed reproduction receipt records six closed cap variants with contact triangles preserved; largest recorded discarded sliver is about `1.48e-16 m^3`.
- Reproduction cap vertex sets are exactly identical to shipped float32 sets in all six cases. Face sets match in five cases; nominal right-cap triangulation differs in the newly filled noncontact patch. This is geometric reproduction, not byte-identical archive or triangle-order reproduction.

## Resolved follow-up: P2 cooker input ownership/completeness preflight

The concern below was found in the preceding snapshot and is now resolved. The final follow-up independently ran `test/test_eka_reference_bake_preflight.py`: **6 passed in 10.97 seconds**, with CUDA masked and no model cooking. Positive preflight validates all three candidates and inherited cache identities; negative cases reject a self-rehashed wrong STEP, rehashed wrong USD root ownership, disabled hand collider, corrupt stock-cache envelope, and empty/partial cooked groups. Existing bake receipts also cannot be overwritten.

The revised cooker pins the STEP and both mesh archives to the registered inputs, checks every selected USD's model/source root metadata and full selected-model validation, verifies stock reference envelope/source/runtime identity through `_read(..., _identity(...))`, compares the complete enabled decomposition inventory against its registered manifest, and validates cooked group paths and body ownership before the first reference write. Existing reference or receipt files reject before cooking. These checks address the original concern without changing the checked geometry or dynamics.

The following paragraphs retain the original finding for traceability:

Files:

- `C:\Users\VR2\Documents\LeRobot-HiWonder-Jetson_ws\.work\hiwonder-rl\eka-v6-integration-2026-10-28\staged-extension\tools\bake_eka_collision_references.py:29`
- `C:\Users\VR2\Documents\LeRobot-HiWonder-Jetson_ws\.work\hiwonder-rl\eka-v6-integration-2026-10-28\staged-extension\tools\bake_eka_collision_references.py:91`
- `C:\Users\VR2\Documents\LeRobot-HiWonder-Jetson_ws\.work\hiwonder-rl\eka-v6-integration-2026-10-28\staged-extension\tools\bake_eka_collision_references.py:163`

The portable cooker checks existing outputs but trusts arbitrary files under `--asset-directory`: it does not preflight the Eka root ownership, 10-body/9-joint topology, matching source STEP/manifest identity, or expected decomposition-path completeness before importing/cooking. `read_stock()` also consumes retained stock reference payloads without checking their envelope/source identity. `new_groups` may be empty or incomplete, yet the code writes references and a `BAKED_NATIVE_REFERENCES` receipt.

This does not invalidate the currently reviewed, matching staged assets. It affects a fresh rebuild using the portable CLI: wrong-model, stale or partial inputs can create misleading candidate references after expensive cooking. The later selected-model validators can reject many such cases, but the cooker should reject them before emitting its own success artifacts.

Add a CPU preflight of the supplied source/manifest, each selected USD's model/topology/source identity, stock reference payload and source hashes, plus the actual expected decomposition paths from the USD/metadata. Reject an empty/missing/duplicate cooked component set before writing the first reference. Keep hardware/spacer convex-hull proxies distinct from decomposition paths. A documented recipe that mandates the selected-model validators remains useful, but early ownership checks should also protect direct cooker use.

The final saved recipe at `C:\Users\VR2\Documents\LeRobot-HiWonder-Jetson_ws\.work\hiwonder-rl\eka-v6-integration-2026-10-28\staged-extension\EKA_V6_VALIDATION.md` was subsequently reviewed. It builds USD from the shipped frozen archives, copies `source.step`, `source_geometry.npz` and `precision_caps.npz` to the fresh candidate directory, then runs the guarded cooker and model-aware profile builder. Optional boundary stitching is explicitly a separate geometry-reproduction check that does not promote changed archives. It distinguishes geometric reproduction from archive/crate byte identity, faceted caps from retained imported solids, provisional camera/dynamics from digital checks, and representative GPU receipts from all-30 CPU layout coverage. No all-30 Eka solver claim was found.

The collision checker's temporary stage now explicitly uses Z up, matching the source asset convention. Its configurable contact buffer is confined to the isolated diagnostic query, records its capacity, and fails explicitly on saturation with the pose index and joint values. This does not alter environment solver/contact settings. The completed expanded query receipt is reviewed below; it was not rerun by this reviewer.

## Reviewed parent-owned runtime receipts

These are direct reads of saved receipts, not independent GPU reruns:

- `scene/scene-check.json`: two `640x480` RGB images, configured 30 Hz, material/direct-light room isolation passing, maximum marker projection error `0.667094 px`. Physical optical calibration and shadow-only isolation remain unqualified.
- `grasp-final/receipt.json`: nominal Eka peg fixture passes measured two-tip dwell `0.5 s`, lift `23.934543 mm`, relative slip `13.750021 micrometers`, and final two-jaw fraction `1.0`. The source-assembly zero hand and 50 ms closure are explicit fixture conditions; the wider training reset was not changed. This is a simulated exact-asset fixture, not learned grasp acquisition, randomized robustness or physical acceptance.
- `ppo-rgb/result.json`: finite RGB/SGS learner parameters after two completed iterations/four optimizer steps, with `final.pt` recorded. This is a short learner smoke, not task learning.
- `cloth-final/cloth_fold_rag.json`: `PASS_PHYSICS_RESET_EXECUTION`, 16 primary plus eight post-reset/restore steps, zero unexpected failures. The isolated compute probe explicitly omitted the camera. This does not extend to all-30 Eka native/rigid execution.

## Final packaged CLI closure

Reviewed the final two-file fix under `C:\Users\VR2\Documents\HiWonder-IsaacLab\source\isaaclab_hiwonder`. `tools/inspect_hiwonder.py:171` now honors the explicit `kit` visualizer without accessing the absent Isaac Lab 3 `args.headless` attribute. `test/test_raw_env_tool_actions.py` supplies `visualizer=["kit"]` without a headless field and exercises the actual inspection path for both stock and Eka, while replacing only native panel construction. A repository tools search found no remaining `args.headless`/`.headless` access.

Independent post-fix run using the packaged source on `PYTHONPATH`, CUDA masked and pytest cache disabled: **6 passed in 1.54 seconds**. This re-verifies the existing six tool-path cases against the final branch; it is not six new distinct contracts. The change is scoped to CLI compatibility and its regression fixture; no further finding.

The parent-owned `packaged-suite.log` was read directly: **383 passed, 1 skipped, 42 warnings in 156.46 seconds**, before this final CLI fix. The post-fix focused result above supplies fresh evidence for the changed path; a fresh full-suite pass after that fix is not claimed here.

Also read the completed `collision-final.json`: `PASS_SAMPLED_COLLISION_PROFILE`, selected model `eka_v6`, all three scale variants, **222 poses per variant**, diagnostic contact capacity `8,388,608`, and **zero additional penetrating body pairs** in each manipulation-profile comparison. Largest recorded before/after contact counts are 2,735,827 / 2,048,844, below capacity. These are geometric subset and sampled collision-query comparisons against the retained reference; they do not establish collision-free travel, solver/SDF qualification, all-30 task execution, or physical safety.

## Declined to judge

- Full GPU/CAD work, all-30 Eka solver coverage, throughput and physical behavior: parent-owned or not supplied; bounded receipts above retain their separate limits.
- Bit-identical regeneration of NPZ ordering or convex decompositions: not promised; geometric/source identity and contact preservation are the relevant demonstrated contracts.
- Immutable input artifact re-cooking in this lane: not rerun because the parent owns the bounded expensive jobs; review covered the code, source/cache preflight regressions, existing-output refusals, recipe and saved evidence. Optional helper-environment dependency installation was not rerun.

## Assessment

**No open review findings. All reported P2s are resolved; source, asset and documented rebuild workflow are ready for the parent's final integration/publication checks.** Independent follow-up verification covers 24 CPU cases: 12 selected-validator/certificate tests, six raw-tool action tests and six cooker preflight tests; the six tool-path cases additionally pass against the final packaged CLI fix. Core source/asset checks, package contents, corrected utility action widths, ownership guards and documentation boundaries pass this review. Parent-owned full packaged-suite and expanded collision results were read with their exact temporal and qualification limits above. This report does not convert digital geometry or bounded simulation receipts into physical acceptance.
