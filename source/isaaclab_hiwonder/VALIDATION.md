# Validation record

> Historical validation record. Dates, source digests and results below refer to the original recorded experiments. `external-evidence/` denotes owner-retained evidence outside this repository; those large bundles are not included. This publication did not rerun simulator or hardware qualification.

## Runtime optimization review, 2026-10-19

The current package Python digest is
`3c1a2a34179b8362c3bc7a616be5978f581ae986f6a48ecb7721df6485c4a403`
(48 files). See [OPTIMIZATION_VALIDATION.md](OPTIMIZATION_VALIDATION.md) for
the six implemented optimizations, 305 passing package tests, both all-30
bounded execution/reset/restore matrices and per-preset benchmark measurements.
Seventeen recorded PhysX traces match exactly. Thirteen native Newton strict
trajectory comparisons remain rejected; unchanged native repeats also diverge.
The additional native cloth diagnostic was terminated by the user after RAM
growth above 15 GB, and its partial evidence is incomplete. Remaining live
qualification and longer quiet repeats are not claimed as passed.

## RGB wrist-view lighting extension, 2026-10-19

The preceding lighting package Python digest (47 files) is
`a0f1a99c330b20ab8ba373ff7ed1e2a2ff1f2a3d80748d9af7dd74ca2d9f1951`.
Benchmarks and PPO smokes below identify their preceding digest separately.
Evidence is retained at
`external-evidence/lighting-2026-10-19`.
The new modes preserve 640x480 capture at a configured 30 Hz, 30 Hz control,
eight physics ticks per control interval and dt=1/240 s. Solver/contact settings
and authored physical geometry were not reduced for these changes.

`runtime-final-reset/result.json` passes actual two-room PhysX wrist RGB acceptance:
saturation zero versus one changes raw pixels by 3.679/0.532 mean levels. Disabling
room0 lights changes its image by 9.525 levels; room1 changes 0.0253 against
0.0389 baseline render noise. Across 16 nonzero wrist-motion steps, all 112 active
valid-plane targets project inside the actual camera image; maximum difference
from targets recomputed using the SDK camera pose is 3.50e-7 m. These are bounded
support-plane checks, not guarantees against occlusion or arbitrary camera angles.

Preview edits integrate zero physics steps and leave joint/root/task-body state
exact. Running-episode checkpoint restore reproduces every saved visual parameter
and preview override. Manual reset after inference-mode stepping also passes
without integrating physics. The latter check reproduced and repaired persistent
command/camera-age/action/timeout/reward-log buffers that had become inference tensors.
The final direct reset runs before any checkpoint load, so restored normal tensors
cannot mask the inference-storage regression. The retained catalog checker now
collects rollouts under inference mode and resets/restores outside it. Its actual
two-room 16-step RGB peg run also passes full and indexed reset/restore, with zero
integration steps during indexed reset and zero retained-room state change
(`inference-reset-regression`).

Actual renderer ablations exposed legacy cone softness extinguishing narrow
beams. Explicit per-source `omni:rtx:usdluxVersion=2505` selects the documented
current mapping for wrist-view sources; the same narrow beams render at softness
0, 0.2, 0.5 and 1. Kelvin tint is disabled on these RGB sources. Existing room-sweep
sources preserve their legacy lighting specification and temperature behavior.
No empirical cone-softness remapping, extra physics integration or solver change
was used to repair rendering.

The fresh coherent extension suite reports **247 passed, 42 SDK/Torch warnings,
zero failures/errors/skips in 80.51 s** (`final-current-tests.log` and XML). Pinned
Ruff 0.14.10 check and format checks pass for all 85 Python files. Independent
source review confirms the same 47-file package digest and finds no new confirmed
P1/P2 issue. This regression scope does not repeat the historical 30-preset matrix.

`rl-smoke-audit.json` qualifies actual two-room RGB SGS/PPO on PhysX peg and
Newton shirt with the new wrist-view profile. Each performs two learning
iterations/four optimizer updates and records four true terminal SGS outcomes,
then resumes in a fresh process through iteration 3. Saved visual parameters,
consumed full-resolution `[2,6,480,640]` images, frame history and SGS history
restore exactly. Actor and critic tensors remain finite in both original and
resumed checkpoints. Training environments reject held-out preview overrides.
These short engineering smokes establish execution/resume, not learned task success.
They use the benchmark-source digest below; the subsequent reward-log allocation
repair is separately covered by direct UI/reset, the strengthened simulator
checker and the final CPU rerun. It changes persistent storage ownership only.

The inspection viewer retains the native full-resolution GPU buffers, bounds
asynchronous PNG work and exports sequence-pinned raw/policy image pairs. While
paused, raw renders can change under lighting edits while the policy observation
remains the last real simulation observation. Export cadence is independent of
the camera's configured simulation cadence. These lifecycle and HTTP ownership
contracts have real USD/loopback/CPU regressions; live viewer and throughput
measurements are recorded separately below.

Final live UI acceptance (`viewer-proof/receipt.json`) directly resets a running
inference rollout without an intervening checkpoint load. Saturation 0 versus 1
while paused changes the raw image by 6.828 mean levels, preserves the policy image
exactly (max difference 0) and holds the displayed step at 183. Native GPU images
were visually inspected, including both full-resolution raw/policy views.
`live-capture/receipt.json` records 20 actual sequential 640x480 raw PNG frames and
an animated preview of moving spots/vibrancy. The live scene uses a neutral
matte-material inspection cohort to reveal the illumination; the training
`rgb_spots.json` preset retains the default material randomization domains.

### Matched throughput and physical-state comparison

Package Python digest is
`b44acc4ea0ece279c43e458757813a5d295e8b89fa203e5aac1411e28c34f0a5`.
`benchmark-comparison.json` compares the historical source above against this
source on the same two-room profile and runtime, with 20 warm-up and 60 timed
control steps per process. Every timed step includes a final CUDA synchronization.
Startup, shader compilation, PPO optimization and browser PNG export are excluded.

| Mode | Before median/p95 ms | After median/p95 ms | Median step-time change |
|---|---|---|---|
| Headless with actual wrist RGB and physics | 100.521 / 106.022 | 96.175 / 99.492 | 4.32% lower |
| Desktop with CPU image upload | 138.921 / 147.379 | 139.720 / 145.014 | 0.58% higher, no demonstrated gain |
| Desktop with retained GPU image upload | 138.921 / 147.379 | 102.787 / 107.631 | 26.01% lower |

Saved step80 joint positions/velocities, root pose/velocity and peg/socket poses
have exactly zero maximum difference in each matched run. Camera/control/dt and
physics configuration metadata are identical. Profiling separately over ten
steps reduces nuisance snapshot calls from six to four per control interval;
the cached light update also avoids repeated stage lookups. Nested instrumented
phase means are diagnostic and cannot be added as independent throughput costs.

The new wrist-view profile itself measures 95.844 ms median / 98.023 ms p95
headlessly in the same bounded setup. This is a separate illumination profile,
so its pixel work differs and it is not used as the matched speedup estimate.
The benchmark establishes a local engineering improvement, not 30 real-time FPS,
a PPO-training throughput result or a guarantee across scene sizes/hardware.

## Historical full catalog qualification, 2026-10-16

Measured on 2026-10-16, Windows 11, RTX 5090. This file distinguishes implementation execution from task-solving or hardware acceptance. The software uses estimated servo/inertial parameters and provisional physical lens calibration.

Quality review and requalification are complete on source `d050ea78b92459f5bb7932cdd6e0b8ee042f8502a570984e6ab80a8acf49a272`. The complete reviewed package suite passes 188 tests. All 30 catalog presets, three physical-nuisance profiles, full RGB PPO collection/fresh-process resume, native shirt RGB PPO and rendered scene checks pass on this source. Earlier component experiments below retain their recorded scope; they are not new measurements of the reviewed source.

Rows explicitly marked reviewed-source or F9 are fresh review runs. Other numerical component measurements are retained implementation evidence. The catalog section identifies its own common source separately.

| Check | Result | Evidence |
|---|---|---|
| Pinned Isaac Lab 3.0 runtime | Installed and executable | Upstream `903ea8ebe362ac3ad82b853ff1995bbf53dc0cab`; Python3.12/IsaacSim6.2/Torch2.11cu130/Newton1.6.1 |
| Original installation preservation | Original2.3.2 directory retained | `external-evidence/original-IsaacLab-2.3.2-snapshot` |
| Runtime NumPy/Numba compatibility | PASS actual import and JIT execution | NumPy2.5.1 reproduced an import failure; locked override now selects2.4.6 with Numba0.64.0 |
| Package regression suite | PASS188 tests on reviewed source | Actual CPU/SDK tests,85.52s; seven emitted SDK warnings described below |
| Actual supplied CAD articulation | PASS, 240 PhysX steps | Maximum registered FK error2.06e-7m; fixed-base drift3.58e-8m; six correct joint names/order |
| Portable mechanism world anchors | PASS actual two-room PhysX knob regression | Pre-fix both rooms failed atstep1; composed Task-frame anchors give32steps plus8restore and8indexed-reset steps, zero failures and retained-room delta0 |
| Camera frame and projection | PASS reviewed-source configured simulation model | 640×480 uint8,30Hz; three rendered marker errors .503/.732/.638px; CAD mount and xyzw/OpenCV/OpenGL conventions reviewed |
| First RGB frame after full/indexed reset | PASS actual two-room rendered markers, zero integration steps | Maximum projection error .911px; camera/FK .233µm; physics counters[0,0,0]; retained room joint/root/camera pose exact |
| Physical optical equivalence | Pending measured calibration | Default60°HFOV and lens parameters are illustrative; f-stop unknown/DoF disabled; calibration tools included |
| Two-room material ownership | PASS reviewed-source rendered pixels | Changing only room0 table:19.60 mean absolute pixel levels; retained room1 .076; baseline noise .182/.058 |
| Direct-light/disco isolation | PASS reviewed-source rendered pixels | Own lights off:12.63/.038; white spot rotation:17.81/.036; RGB color:14.61/.045 mean absolute pixel levels across rooms0/1 |
| Measured rigid support contact | PASS | Peg-table force .29431N, consistent with a .03kg object's weight |
| Simulated stable grasp seed | PASS actual two-jaw hold and lift | Both contacts for0.5s;20.34mm lift;0.72µm relative slip; measured jaw.05158rad and target.12rad stored separately |
| Physical clutter and bounded confetti | PASS finite supported scene | Two physical distractors and two thin cards per room; fixed-capacity wrappers initialize; remaining confetti is rendered instancing |
| Dense visual confetti | PASS reviewed-source rendered effect and isolation | 688 visual pieces plus two physical cards per room; hiding room0 visual cards changes 17.96% of its image by more than eight levels; retained-room mean .048 below baseline .058 |
| Direct visual PPO collection/update | PASS reviewed-source PhysX peg and Newton shirt engineering smokes | Two iterations/four optimizer steps each, finite actor/critic weights; full-resolution actual augmented RGB [2,6,480,640]; native CLI selects shirt from explicit profile while initializing the peg checkpoint with fresh optimizer/SGS |
| SGS distribution and outcome bookkeeping | PASS numeric/cohort/history regressions and actual PPO episode updates | Valid-history counts and appendix probability floor, duplicate completions across wraparound/resume, invalid-batch atomic rejection and cohort normalization; current F9 actual terminal histories 4→8 |
| Fresh-process PPO resume | PASS reviewed-source F9 engineering smoke | Two-frame RGB [2,6,480,640]; iteration 2→3, optimizer 4→6; exact consumed images/camera/history buffers; real terminal SGS histories 4→8; evidence in `review-2026-10-16/f9-ppo` |
| Physical actuator restore after episode resets | PASS reviewed-source F9 readbacks | Exact zero max error for stiffness/damping/effort/speed/armature/friction after real terminal resets and fresh-process restore; these are simulator property readbacks, not physical servo accuracy |
| Uniform held-out evaluation | Executed one full episode with final camera/reset path | Independent seed1000003,900control steps;0/1success. This is workflow evidence and is not a useful policy-performance estimate |
| Native paper constitutive forces | PASS CPU solver checks | Persistent paper crease state and selective reset; per-substep native constitutive hook |
| Native cable/plug frame bond | PASS two-room GPU execution with authored inertias | Three noncollinear native attachment sites; position error .436/1.138µm, angle .000874/.000651rad, cable strain1.039%/1.043%;32steps plus8post-reset/restore steps |
| Native cable inertia preservation | PASS80 actual GPU body tensors and mass readbacks | Authored positive geometric tensors restored before solver construction; bit-equal after reset/restore physics; per-body eigenvalue audit saved |
| Native rope local contact filtering | PASS CPU geometry/dynamics and strict GPU threading fixture | Excludes unavoidable capsule end-cap overlap by material arclength; distant self-contact retained; physical mass/I unchanged; threading32+8steps with zero unexpected failures |
| Native rope stationary-success component | PASS continuous actual GPU resting-fixture windows | Straight rope raw speed .00347–.00841m/s; overhand/figure-eight control-frame speed maxima .01630/.01126m/s; both final0.5s windows100% qualified and longest2.033s; raw spikes retained separately |
| Native indexed cable reset | PASS actual two-room GPU state isolation | Reset room0 only; retained room1 q/qd/root/body/node/velocity/contact state exact; physics counter320→320;8subsequent control steps stable |
| Native indexed garment reset | PASS actual two-room shirt GPU state isolation | Retained room1 physical state exact; zero integration during indexed reset;8subsequent steps finite; sheet edge strain1.385%/.790% |
| Native motor property dynamics | PASS8 actual VBD dynamics tests and two-room integration | Armature-dependent acceleration, frictional decay/stopping, bounded motor torque/speed envelope, limit crossing, tiny-jaw damping stability; first explicit-damping limit cycle reproduced and repaired |
| Native fixed-root randomized reset | PASS focused solver regression and two-room GPU scene | Updated VBD rest reference preserves reset yaw; shirt camera/FK error .310/.319mm after32 control steps |
| Native heterogeneous garment cohorts | PASS two-room shirt execution with new motor model | Imported rest-mesh errors .120/.073µm, distinct per-room geometry; maximum sheet edge strain2.077%/.631%; reset/checkpoint checks |
| Native paper scene | PASS32 control-step execution | Persistent crease state present; camera/FK error .306mm, maximum sheet edge strain .00990%; reset/checkpoint checks passed |
| Exact collision-hull cache | PASS CPU reimport and native GPU use | 1697 original hulls, zero vertex error; body frames/flags/inertials retained; independent cached import9.834s |
| Task catalog GPU execution | PASS all 30 presets on reviewed source | 17 PhysX and 13 native cases; 32 control steps plus eight after restoring a running episode; zero unexpected failures or terminals; full matrix below |
| Physical HiWonder robot deployment | Outside this simulator verification | No real motor reads, calibration, torque enable or motion performed |

Detailed logs and JSON evidence are in `external-evidence`. Run artifacts record source, asset and configuration hashes. Source-only fixtures and finite smoke runs do not establish stable manipulation, sim-to-real transfer or task learning. Precision insertions, knot manipulation, garment folds and complete paper-airplane sequences require learned-policy/contact acceptance experiments.

The locked environment retains upstream Isaac Lab overrides for Isaac Sim's older NumPy, fsspec, urllib3 and USD pins. `uv pip check` reports those metadata differences plus Windows wheel pywin32/ONNX Runtime requirements; it is not a clean compatibility receipt. The executed simulation, renderer, NumPy/Numba JIT and PPO checks establish the exercised runtime paths only.

The pinned cable importer emits a legacy material-schema deprecation and a generic parallel-joint warning for the three-site VBD bond. That bond is qualified specifically through the recorded VBD load/twist/reset tests. Newton's default regularizer inflated small rod eigenvalues by roughly15770–45266×; the active path now restores validated positive authored tensors and their inverses through public model arrays before solver construction. CPU load/contact/reset experiments and actual two-room GPU readbacks verify this correction. The original startup warning can still occur before restoration; run metadata records authored, regularized and restored eigenvalues. Material and servo parameters remain engineering priors requiring physical identification.

## Review

Three fresh independent reviews covered the complete task catalog, all ten randomization groups, CAD/camera conversion, RL/SGS, launch/checkpoint workflows and evaluation. Root integration review and cross-reviews inspected the repairs. The review found and fixed material correctness issues; this is bounded evidence, not a claim that no possible defect remains.

| Reviewed area | Confirmed repairs and verification |
|---|---|
| Task mechanics and goals | Correct SI angular-drive conversion; persistent roll-only shortcut rejection; configurable cube count/colors/order and knob/switch/latch targets; positive/negative success fixtures and ten production-predicate mutations detected |
| Public environment configuration | Profile-selected task and parameters forwarded; full 40-value task/goal command with overflow rejection; episode horizon applied to both task and environment; reverse near-goal starts remain nontrivial |
| PPO/SGS semantics | Genuine terminal outcomes take precedence over a simultaneous deadline; completed uniform-training episodes populate coverage histories; interrupted resets do not; actual 500 ms delay reaches tick 15 at 30 Hz; post-reset critic observations are fresh |
| Resume and policy initialization | Embedded profile survives copied checkpoints; incompatible exact-resume overrides reject; explicit profile task overrides inherited task during policy initialization; source/profile/bank identity protects old histories |
| Held-out evaluation | Independent seed required; fixed per-cohort episode quotas and bank-slice weighting; incomplete coverage produces null aggregate scores instead of a misleading estimate |
| Visual and actuator nuisance | Independent checkpointed camera-drop RNG per environment; configured motor-delay capacity; coherent geometry/scale and supported camera-frame validation; meaningful material-family priors; explicit fixed structural cadence |
| Supported initial layouts | Physical nuisance excludes task/reset geometry and CAD base; table support includes loose jitter and stack goals; bounded base-only reset rejection preserves other draws and rolls back atomically on exhaustion |
| Runtime efficiency | Ordinary post-step task-state reads reduced from two to one with step/reset lifetime guards; this is a read-count measurement, not a training-throughput benchmark |

Review evidence is under `external-evidence/review-2026-10-16`. Owner findings, causal reproductions, cross-reviews, source freeze, test XML and simulator receipts are retained there. Structural geometry and appearance remain fixed per-room launch cohorts; a new seed/cohort redraws them. Unsupported episode cadence now rejects explicitly.

The original ten requested controls map to these implemented paths:

| Request | Implementation and bounded evidence |
|---|---|
| 1. Table variation | Shape/dimensions, texture/PBR/color and contact properties; sampler/USD tests, task-support checks and actual material-change render |
| 2. Separate rooms and disco lighting | Own 3.6576 m four-wall rooms with table gap, wall backgrounds/materials and local white/RGB spot controls; USD ownership tests and actual two-room light/material renders |
| 3. Robot appearance and size | Per-link materials and coherent nominal/0.99/1.01 CAD geometry with inertia scaling; manifest/asset registration tests; live structural scaling is not promised |
| 4. Camera smudge | Parameterized spots/streaks/haze/blur/tint/local contrast and drift; deterministic image regressions and actual transformed RGB PPO collection |
| 5. Task selection | Twenty task families and 30 presets listed below, dedicated articulated/cable/garment/paper mechanics, configurable commanded goals and independent positive/negative terminal fixtures |
| 6. Motors | Six physical/adapted motor properties plus delay, encoder noise and optional play/dead-band; actual readback/restore and native dynamics component tests; physical identification pending |
| 7. Base pose | Supported XYZ/yaw sampling with CAD position IK, initial-clearance rejection and atomic bounded retries; actual wide-base/indexed-reset integration |
| 8. Image transforms | Shear/crop/rotation with exact black padding, color jitter and none/one/all selection; image/frame/channel regressions and actual stored RGB/history resume |
| 9. Unrelated table objects | Bounded visual or physical distractors, appearance and layout controls; task/base/reset clearance regressions and six-physical-box execution profiles |
| 10. Lightweight confetti | Rendered instances with configurable density/appearance/geometry and capped physical subset; 688-piece rendered ablation and native two-card physical reset/restore |

All ten controls are available, with structural variation sampled at launch and supported sensor/actuator/base variation sampled at episode resets. This coverage does not establish every parameter combination as solvable, an identified physical model, or a learned policy for any task.

The final package suite reports **188 passed, zero failures/errors/skips**, in 85.52 s. Seven SDK warnings concern the legacy cable material schema, parallel attachment joints and Newton's initial rod-inertia regularization before the validated authored-tensor correction. Ruff check and format pass for all 78 Python files.

Additional reviewed-source physical-nuisance execution covers two-room PhysX peg with a wide randomized base and six physical boxes, six-cube stacking with six boxes, and Newton rope with six boxes plus two physical cards. The wide-base profile performs 50 manual resets and recovers two rejected base candidates; interrupted reset history remains empty. Each profile runs 16 control steps, saves/restores state, resets one room without integrating physics, then runs eight more steps. Retained-room state change is zero; nuisance checkpoint state errors are zero for PhysX and about 1.14e-13 numerically for Newton. These receipts qualify reset/state handling, not task success.

The current rendered scene proof includes table material, local white/RGB spot changes, configured projection, physical support and nuisance state. The confetti ablation renders the same scene with only room0 visual pieces hidden. Warm render plus sensor-read timings were 13.69 ms visible and 19.20 ms hidden over 12 cycles; they exclude physics/startup/shader compilation and do not establish training throughput or a confetti speed benefit. Pixel effect is not geometric table-coverage ground truth. Direct-light isolation is measured with shadows enabled; an isolated shadow-only perturbation remains unqualified.

The component experiments in the following paragraphs were performed during implementation before this fresh review. Current regression tests retain their contracts; their specific numerical measurements are historical component evidence.

A bounded review also checked goal geometry against actual convex collision pieces. It reproduced a 1mm circuit-plug collar/rim intersection and a concave key notch filled by single-hull convexification (1.173mm inscribed overlap with the socket wall). Raising the collar and splitting the key into two convex prisms repaired these fixtures without relaxing success tolerances. The insertion, shape, gear, washer, stacking, nut, cap, mechanism and ball checks found no further material goal-clearance contradictions. This review does not establish collision-free manipulation paths.

The knot variants' endpoint goals were also lowered from25mm to the rope's1.5mm supported center height. Previously, released tails resting on the table failed the strict20mm 3D gate even at exact goal XY. The real-state regression now reaches topology classification, still rejects an unknot, and still rejects endpoint errors beyond20mm. Knot determinant, release, strain, speed and dwell requirements were preserved.

The stricter pre-reset checker exposed two additional integration defects. Mechanism world joints had task-local world-side anchors; binding them to the non-rigid Task reference frame composes the room and table transforms correctly. All six mechanism families now pass guarded execution. Fine rope capsules also overlapped local material neighbors and produced unstable contact. The bounded arclength filter removes those artificial contacts while retaining distant self-contact, and the production free-rope discretization now uses 40 segments with preserved total length and mass. Its measured centerline approximation is at most 1.21 mm and its discrete axial response differs about 1.28% from the fine reference.

Rope stationarity is explicitly measured from actual node displacement over the control interval, with the same 20 mm/s threshold and 0.5-second dwell. High-frequency instantaneous motion remains in the raw-velocity diagnostics; a passing control-frame window does not establish substep immobility. First samples cannot credit quiet dwell, motion history resets per episode, checkpoints preserve it, and nonfinite raw velocities remain failures. Actual overhand and figure-eight GPU windows and independent motion/history regressions qualify this component. They do not demonstrate a learned knot policy.

The first guarded catalog sweep then exposed an invalid randomized base placement: the initial rope intersected the CAD base, producing enormous finite coordinates on the first control step. Initial task/base-envelope rejection now resamples only the base, with capsule radius, contact margin and loose-object jitter reserved. An actual matched first-substep comparison reduced rope/robot contacts from15 to0, maximum rod linear speed from1.342 to.005158m/s and angular speed from207.33 to.000029rad/s; masses, inertias, forces, gravity and motor parameters were identical. Ordinary unsynchronized one-room and two-room graph runs subsequently passed, including indexed reset and running-episode restore. The two-room64-step fixture qualified the last0.5s in both raw and control-frame motion metrics, with zero unexpected failures and zero retained-room state change. Evidence is in `native-initial-base-overlap-red-green.json` and `native-base-clearance-graph-n1`/`native-base-clearance-graph-n2`.

## Catalog matrix

All 30 presets passed the fresh review matrix after all confirmed repairs. Every process exited zero using source SHA256 `d050ea78b92459f5bb7932cdd6e0b8ee042f8502a570984e6ab80a8acf49a272`. Each ran 32 control steps, saved a checkpoint from a running episode, performed an interrupted manual reset, restored that checkpoint and ran eight more steps. All receipts enable the pre-autoreset failure guard, report zero unexpected failures and zero terminals, and pass camera/FK and native geometry checks. No failed child was retried into acceptance. This is bounded zero-action execution/reset/restore acceptance, not a learned-policy success result.

There are 17 PhysX cases (nine rigid, eight articulated) and 13 native cases (five cable, seven cloth, one plastic shell). The largest measured camera/FK error was **0.328864 mm** and the largest final native edge strain was **0.380685%**. Native import/rest and authored-inertia checks also passed. Earlier implementation matrices remain historical and do not substitute for this reviewed-source sweep.

The aggregate is `task-matrix-30/catalog-results.json`; the worker audit is `gpu-matrix-independent-summary.json`. Root independently reread every individual receipt, checked the exact expected 30-preset set, compared the aggregate and recomputed the current source digest: `final-catalog-independent-audit.json`, exit zero. These paths are relative to the review evidence directory above. The ordinary final sweep leaves optional synchronized native diagnostics disabled.

| Preset | Physics | Execution/reset/restore |
|---|---|---|
| `peg_insert` | rigid | PASS |
| `cube_stack` | rigid | PASS |
| `nut_pick_place` | rigid | PASS |
| `washer_ring` | rigid | PASS |
| `connector_insert` | rigid | PASS |
| `shape_sort` | rigid | PASS |
| `knob` | articulated | PASS |
| `switch` | articulated | PASS |
| `latch` | articulated | PASS |
| `bottle_cap` | articulated | PASS |
| `bottle_cap:close` | articulated | PASS |
| `key_lock` | articulated | PASS |
| `barrel_jack` | cable | PASS |
| `rope_thread:thread` | cable | PASS |
| `rope_thread:overhand` | cable | PASS |
| `rope_thread:figure_eight` | cable | PASS |
| `cable_untangle` | cable | PASS |
| `paper_airplane` | plastic_shell | PASS |
| `gear_assembly` | articulated | PASS |
| `circuit_restore` | articulated | PASS |
| `catch_ball` | rigid | PASS |
| `roll_ball` | rigid | PASS |
| `roll_ball:table` | rigid | PASS |
| `cloth_fold:towel` | cloth | PASS |
| `cloth_fold:handtowel` | cloth | PASS |
| `cloth_fold:shirt` | cloth | PASS |
| `cloth_fold:rag` | cloth | PASS |
| `cloth_fold:shorts` | cloth | PASS |
| `cloth_fold:underwear` | cloth | PASS |
| `cloth_fold:sock` | cloth | PASS |
