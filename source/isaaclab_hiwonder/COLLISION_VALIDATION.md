# HiWonder collision profile validation

> Historical validation record. Dates, source digests and results below refer to the original recorded experiments. `external-evidence/` denotes owner-retained evidence outside this repository; those large bundles are not included. This publication did not rerun simulator or hardware qualification.

Date: 2026-10-19. SDK: Isaac Sim 6.2.0.0, Newton 1.6.1, Warp 1.17.0; GPU: RTX 5090. This change optimizes robot collision representation, with an intentional bounded body-contact approximation. It does not claim exact whole-task trajectory equivalence.

## Geometry and backend choice

`robot_collision_profile="manipulation"` is the new default. It applies the packaged recipe only to native Newton tasks. PhysX retains its original CAD collision authoring: the installed PhysX 110.4.4 cooker returned 176 structural hulls for each of the three variants, versus the much larger Newton decomposition. Reusing the Newton recipe on PhysX was rejected during review because it would increase hull count and involve a different convex cooker.

| Geometry variant | Newton structural hulls before | After | Protected gripper hulls, unchanged |
|---|---:|---:|---:|
| nominal | 1697 | 1260 | 295 |
| 0.99 scale | 1697 | 1322 | 312 |
| 1.01 scale | 1635 | 1271 | 273 |

All 20 servo housing colliders remain unchanged in addition to these structural counts. The nominal body-only count falls from 1402 to 965 (31.2%). Both complete gripper links retain their own variant's reference vertices, triangles, transforms and scales. Original USD files, visuals, mass, COM, inertia, joint frames/limits, camera, actuator/randomization settings, SDF voxel rule, contact settings and solver settings are unchanged. Replacement child meshes inherit the per-link physics materials.

Each new body hull is an inscribed convex subset of one original hull. No hull is expanded across a joint cavity. Redundant hulls may be omitted only with a whole-solid coverage certificate: every original convex solid lies within 2 mm of one retained solid on the same link. Selected high-vertex body hulls are reduced with a 0.5 mm bound where possible; pieces that cannot qualify, including thin/planar pieces, remain unchanged. A 5 micrometre inset protects newly authored subsets from float32 rounding.

Because distance to a convex solid is convex, vertex coverage bounds extend to complete convex solids. Rigid transforms preserve subset membership, so this geometric no-expansion property does not depend on sampled joint poses. It is relative to the original decomposed collision model, not a physical metrology qualification. Engine contact generation, margins and numerical tolerances still require runtime checks.

## Self collisions and native import

`robot_self_collisions=null` preserves the backend default; `true` enables nonadjacent self collisions; `false` disables articulation self collisions. The original directly connected joint-interface exclusions remain in place. No blanket pair filter hides nonadjacent contacts. The original CAD path remains selectable with `robot_collision_profile="cad"`; switching profiles requires a fresh scene.

Real Newton imports passed for nominal, 0.99 and 1.01 scale: protected geometry is compared against the immutable reference captures, and mass/COM/inertia against the current USD MassAPI. Restoration runs before SDF generation and validates unique body ownership, contact flags, mesh type, frames and unchanged inertials. It restores authored topology after automatic convex re-hulling. Tests explicitly verify enabled and disabled nonadjacent self-collision masks and retained joint-interface exclusions.

Every recipe group was independently rechecked against its reference: subset membership, complete-solid coverage witnesses and exact protected-link data. Assets and recipe are checksum-bound; the selected effective profile/checksum is part of the SGS bank identity. Use `--initialize-from` to transfer older policies rather than restore histories under a different collision model.

## Sampled contact queries and isolated performance

255 poses per variant (zero pose, 21 samples for each of six joints, and 128 seeded combined poses) were queried with self collisions enabled on both the exact original references and the optimized recipe. There were zero new penetrating body-pair contacts across all 765 configurations, at a 2 micrometre signed-penetration threshold. The original model already has penetrating nonadjacent contacts in 57-59 extreme/combined sampled configurations per variant; this change does not certify every allowed pose as collision-free. The comparison checks body pairs, not identical contact points, depths, counts or trajectories.

| Variant | Median original query, ms | Optimized, ms | Mesh source bytes reduction |
|---|---:|---:|---:|
| nominal | 4.459 | 2.832 | 29.1% |
| scale099 | 4.641 | 2.962 | 26.3% |
| scale101 | 4.396 | 2.716 | 28.9% |

These are synchronized, single-arm collision queries with unchanged query settings. They exclude rooms, cameras, SDFs and solver integration. The nominal median reduction is 36.5%; this is not an end-to-end RL throughput claim. Mesh-source bytes include unaffected visual sources and do not measure total process or SDF memory. Contact buffers were sized for the isolated worst-case folded poses; no product contact budget was changed.

## Runtime and memory

The normal native randomized `cloth_fold:rag` compute scene passed 16 zero-action control steps, an interrupted reset, running-episode checkpoint restore and eight additional steps, with finite observations/rewards and no unexpected failure. It used the nominal manipulation recipe and restored all 1260 structural meshes before the unchanged SDF provisioning callback. Imported rest error was 2.10e-8 m and maximum measured edge strain 4.16e-6. This proves execution/reset integration for this fixture, not task-solving ability or all-30 runtime qualification.

The successful compute probe omitted the camera/renderer solely to isolate collision/SDF/solver integration. Its process-tree peak was 2.28 GiB resident and 5.66 GiB private allocation. Product camera settings remain unchanged. A camera-enabled cloth startup was stopped by the external 8 GiB private/6 GiB resident memory guard; its sampled resident peak was 4.86 GiB and its OS private peak 8.47 GiB. It spawned up to 27 processes during startup and stopped before environment setup completed, so no completed camera or full-scene RAM-savings claim is made. This does not attribute that startup spike to collision meshes.

The same compute-only original-CAD fixture initialized its solver, but hit the identical 6 GiB private-allocation cap during subsequent execution (2.31 GiB sampled peak resident; 6.01 GiB OS private peak). The optimized fixture completed below that cap. This is a bounded comparison, not a completed original-CAD peak measurement; resident usage was similar in these probes, so a large whole-process RAM reduction is not established. No higher-limit retry was made. See `cloth-cad-compute-02-memory.json`.

Earlier incomplete probes remain in the evidence directory: an undersized diagnostic contact buffer, a 4 GiB compute startup limit, a copied camera diagnostic calling the deliberately omitted camera, and a transient memory-capped first regression run. The corrected contact sweep, compute probe and full regression rerun completed successfully. Simulator qualification probes used external process-tree memory limits and host-headroom admission checks; no physical robot was controlled.

## Regression, packaging and review

- Full staged package: 322 tests passed in 118.45 seconds. The complete 322-test suite was repeated successfully against the installed canonical IsaacLab-30 files; its process guard completed successfully (1.53 GiB peak resident, 2.64 GiB peak private).
- New tests cover analytic joint cavities/adversarial filling, Euclidean distance, float32 subsets, thin/planar fallbacks, protected surfaces, mechanics, explicit self-collision masks and source corruption/missing assets.
- Wheel build and isolated extracted-wheel import passed, including all three required gzip references and recipe/manifest.
- Independent initial review caught PhysX cooking/cost and missing gzip package-data issues; the follow-up source review found no remaining blocking issue after their fixes.
- Lint and formatting use the repository configuration and pinned Ruff 0.14.10.

## Reproduction and evidence

Use the existing pinned uv environment. A standalone geometry/contact recheck is available as:

```powershell
uv run --no-sync --offline python source/isaaclab_hiwonder/tools/check_hiwonder_collision_profile.py --device cuda:0 --output <fresh-output.json>
```

Run GPU checks serially in an external memory-limited process. The checker deliberately does not start a renderer or change solver settings. Offline recipe generation is available in `tools/build_hiwonder_collision_profile.py`; it requires the assets extra (SciPy/trimesh) and refuses to overwrite an existing artifact.

Evidence directory:

`external-evidence/collision-2026-10-19`

Key receipts: `clearance-sweep-02.json` and its memory receipt; `installed-tests-01.log` and memory receipt; `full-tests-02.log` and memory receipt; `cloth-compute-03/cloth_fold_rag.json` and memory receipt; `physx-reference-extraction.json`; `independent-collision-review-followup.md`. The PhysX extraction itself returned valid cooked geometry and exit code zero, but its guard marked cleanup incomplete because a startup helper survived the parent; that helper was contained and terminated. It was not counted as a full runtime pass.
