# HiWonder SO-ARM101 visual reinforcement learning

This extension targets the supplied HiWonder follower CAD and HX-30HM servo arm. It trains a direct wrist-RGB/proprioceptive PPO actor with a privileged state critic. The leader arm is not required. No tool in this package connects to or drives real motors.

Install this extension into a fresh pinned Isaac Lab v3.0.0 checkout using the [repository setup guide](../../README.md). The runtime uses Python 3.12, Isaac Sim 6.2, PyTorch 2.11/CUDA 13.0, Newton 1.6.1 and RSL-RL 5.5.1. Run commands from the newly installed checkout using its `.venv`.

## Run

From a PowerShell terminal in the checkout:

```powershell
# Run from the installed IsaacLab-HiWonder checkout.
$env:OMNI_KIT_ACCEPT_EULA = 'YES'
& .\.venv\Scripts\python.exe source/isaaclab_hiwonder/tools/train_hiwonder.py train --task peg_insert --num-envs 4 --smoke
& .\.venv\Scripts\python.exe source/isaaclab_hiwonder/tools/train_hiwonder.py train --task cube_stack --profile source/isaaclab_hiwonder/configs/rgb_disco.json --iterations 1000
& .\.venv\Scripts\python.exe source/isaaclab_hiwonder/tools/train_hiwonder.py train --profile source/isaaclab_hiwonder/configs/sgs_peg.json --frame-stack 2 --iterations 1000
& .\.venv\Scripts\python.exe source/isaaclab_hiwonder/tools/train_hiwonder.py train --resume '<run>/final.pt'
& .\.venv\Scripts\python.exe source/isaaclab_hiwonder/tools/train_hiwonder.py train --initialize-from '<run>/final.pt' --seed 123
& .\.venv\Scripts\python.exe source/isaaclab_hiwonder/tools/train_hiwonder.py eval --checkpoint '<run>/final.pt' --episodes 100
& .\.venv\Scripts\python.exe source/isaaclab_hiwonder/tools/train_hiwonder.py play --checkpoint '<run>/final.pt' --viz kit
```

To recreate the pinned environment, use installed `uv>=0.13.1` from the installed Isaac Lab root:

```text
uv sync --locked --extra isaacsim --extra rsl-rl --extra test --inexact
```

On Linux replace `.venv/Scripts/python.exe` in the examples with `.venv/bin/python`.

Omitting `--viz` runs without a desktop visualizer. Isaac Lab 3.0 does not use the old `--headless` flag. Rendering still runs for the wrist camera. Windows must load Kit before importing USD scene/asset implementation modules; the launch tools do this automatically.

`--smoke` performs two short optimizer iterations and creates a checkpoint; it does not establish task learning. `--actor-mode state_debug` explicitly selects a proprioceptive debugging actor. The default actor consumes full 640×480 images, 18 proprioceptive values and a 40-value public command: 20 task indicators and up to 20 commanded-goal values. Goals that exceed the declared payload fail explicitly. `--frame-stack 2` or `4` enables temporal visual history for dynamic objects, occlusion and delayed capture. Its critic alone receives privileged object/mechanism/deformable state. PPO stores the actual transformed observations; augmentation is not redrawn during optimizer epochs.

Training outputs include `final.pt`, `profile.json`, source/version/asset provenance, the structural randomization snapshot, scene USDs, TensorBoard logs and `result.json`. Checkpoints embed the complete profile, so copying a checkpoint preserves its camera, timing and history configuration without an adjacent `profile.json`. Explicit `--profile` overrides still use the strict profile schema. `--task` wins over a profile's task; otherwise an explicit profile's task wins over the checkpoint. Task-specific parameters stay paired with their profile task and clear when another task is selected. Resume restores optimizer, SGS, nuisance RNGs, active reset IDs, image records and physical state. Different explicit rollout, epoch, minibatch or smoke settings reject exact resume; use `--initialize-from` to apply changed learner settings with a fresh optimizer. Backend warm-start caches are not serialized, so physical replay is subject to solver tolerances. Policy-only evaluation uses a separate seed and uniform reset sampling without loading training sampler histories; an explicit evaluation seed equal to the checkpoint's training seed is rejected.

Use `--initialize-from` with a new seed to continue policy development across newly drawn structural cohorts. This explicitly starts a fresh optimizer, SGS bank and random streams; strict `--resume` preserves the original experiment instead.

## Inspect wrist RGB and adjust physical lighting

Run a held-out, uniform-sampled inspection with zero actions and no optimizer:

```powershell
& .\.venv\Scripts\python.exe source/isaaclab_hiwonder/tools/inspect_hiwonder.py --viz kit --num-envs 2 --port 8769
```

Open the printed [loopback controls](http://127.0.0.1:8769). The default profile is `configs/rgb_spots.json`; use `--profile '<profile.json>'` to inspect another profile. Omit `--viz kit` for browser-only inspection. `--port 0` selects an available port. The tool prints its output directory; `--output '<new-directory>'` selects a new destination explicitly.

The native panel and browser show full configured-resolution raw wrist RGB beside the actual policy RGB after the camera channel, smudge and image randomization. While running, the policy image comes from the current simulation observation. **While paused, raw RGB refreshes as a render-only lighting preview, and policy RGB remains the last simulation observation.** Pause does not advance physics, nuisance clocks or policy image history. The controls also select a room, reset episodes, clear overrides and stop inspection. Lighting changes apply to all inspection rooms; selecting a room chooses the displayed camera.

Live lighting controls are restricted to held-out environments with an enabled, unfrozen room group. They do not modify a training run's profile or randomization distribution. RGB saturation ranges from 0 (white source RGB) to 1 (fully vivid hue). Wrist-view lights disable Kelvin tint, so saturation zero gives neutral source parameters; materials and the camera channel can still color the resulting pixels. Colored-light power is an independent 0–3 multiplier, motion/hue speed is an independent 0–3 multiplier, and white fill accepts 0–2000 intensity units. The browser spot-radius slider covers 0.01–0.20m; the shared validator accepts 0.005–0.25m. These controls change physical scene lights rather than applying image saturation or recoloring rendered pixels. Equal HSV value across hues does not guarantee equal perceived brightness.

Use **Export partial lighting profile** to download `lighting-preview.json`, containing `randomization.lighting` with the applied settings. Power and speed multipliers are converted to intensity and angular-speed ranges; scalar saturation, radius and fill overrides become fixed ranges. Supply that file deliberately with `--profile` on a later inspection or training launch. It is a partial lighting profile, so merge it with any other custom task/camera/randomization settings you need to retain. **Use randomized profile / clear overrides** restores the inspection's original sampled lighting behavior.

The native panel retains GPU image buffers and uses GPU uploads when supported, with a reported CPU fallback. Browser PNG encoding/export runs asynchronously with bounded pending work; `--export-hz 2` sets its default two-frame-pairs-per-wall-second request independently of camera capture and native display. Browser snapshots still require CPU readback at export cadence. These changes do not lower camera resolution, configured capture rate, physics timestep, solver substeps or solver quality. Configured camera Hz, measured display/control wall Hz and measured export Hz are separate values in the status panel; a configured 30Hz camera does not establish 30 real-time FPS, and no unmeasured throughput gain is claimed.

## Tasks

Select a task with `--task NAME`. Topology changes rebuild the scene; a single policy across every task is optional.

| Name | Task |
|---|---|
| `peg_insert` | Peg into a real compound annular socket |
| `cube_stack` | Stack cubes in commanded color order |
| `nut_pick_place` | Collect scattered nuts in a parts container |
| `washer_ring` | Place washer around a ring at its seat |
| `connector_insert` | Keyed connector and matching socket |
| `shape_sort` | Distinct polygonal pieces and apertures |
| `knob` | Rotate to a target angle with detent mechanics |
| `switch` | Operate a bistable switch |
| `latch` | Slide a latch |
| `bottle_cap`, `bottle_cap:close` | Free-cap opening/removal and closing, reciprocal screw mechanics and regrasp checks |
| `key_lock` | Insert, turn and retract the cam-driven lock bolt |
| `barrel_jack` | Plug with a coupled deformable rod cable |
| `rope_thread:thread` | Thread a deformable rope through rings |
| `rope_thread:overhand`, `rope_thread:figure_eight` | Knot variants with topology checks |
| `cable_untangle` | Separate/straighten an initially entangled cable |
| `cloth_fold:VARIANT` | One-arm garment folding with surface self-contact |
| `paper_airplane` | Ordered paper folds with persistent plastic creases |
| `gear_assembly` | Assemble meshing gears and demonstrate coupled rotation |
| `circuit_restore` | Restore a circuit using actual terminal/contact state |
| `catch_ball` | Retain a moving rolling ball |
| `roll_ball` | Push an oversized ball into a ramped pocket on a low floor platform |

Garment variants are `towel`, `handtowel`, `shirt`, `rag`, `shorts`, `underwear`, and `sock`. The default is `towel`. `roll_ball:table` uses a tabletop platform instead of the near-floor support preset. Dimensions describe miniature simulation task hardware, not measured commercial fixtures.

Use `task_parameters` in a profile to change the commanded task. Stacking accepts `count` from 1 to 6, one distinct RGB triple per cube in `colors`, and an index permutation in `order` describing bottom-to-top order. For example:

```json
{"task":"cube_stack","task_parameters":{"count":3,"colors":[[1,0,0],[0,1,0],[0,0,1]],"order":[2,0,1]}}
```

The ordered colors enter the actor command; hidden object poses do not. `knob` accepts `target_angle_rad` in [-2.6,2.6], `switch` accepts `target_state` of `on` or `off`, and `latch` accepts `engaged` or `released`. Unknown or incompatible task parameters are rejected. Initial states and near-goal families keep reverse operations nontrivial. `episode_seconds` overrides both the environment deadline and task horizon. True task success/failure takes precedence over a simultaneous deadline. Rolling tasks reject grasp/carry/lift shortcuts throughout the episode.

Rigid and articulated tasks use PhysX. Cable/cloth/paper tasks use monolithic Newton VBD for robot, rigid and deformable interactions. Paper adds per-substep constitutive forces and persistent crease state. Three noncollinear native attachment sites constrain the barrel-jack plug/cable bond in all six rigid degrees of freedom; an external spring approximation is not used by that task. Native families use eight solver substeps per physics tick; the default eight physics ticks per control decision therefore produce 64 solver substeps. These are simulation models whose contact and constitutive parameters require task-specific qualification. Task registration, a finite scene or a pure success-fixture test does not establish successful manipulation.

Free rope tasks use 40 segments sampled by rest arclength. Total length, radius, density and continuum moduli are preserved, with true segment inertias recomputed. Against the earlier fine polyline, centerline approximation error is at most 1.21 mm and discrete axial response changes about 1.28%. Local capsule overlap is filtered by material arclength; distant self-contact remains active. Rope stationarity uses actual node displacement over each control interval with a 20 mm/s threshold and continuous dwell. Raw instantaneous solver velocities remain diagnostic metrics and must be finite; episode motion history is cleared on reset and preserved by checkpoints.

## Ten randomization groups

Copy `configs/default.json` or supply a partial JSON using `--profile`. Unknown fields are rejected. Every group has `enabled`, `seed`, `cadence`, `reset_interval` and `frozen` controls under `randomization.controls`. `clean.json` disables nuisance groups for an ablation; `rgb_disco.json` demonstrates combined visual nuisance; `occlusion.json` increases table coverage without simulating every piece as a rigid body.

| Group | Parameters and application |
|---|---|
| Table | Shape, dimensions, rounded edges, PBR material/color/roughness/metallicity/specularity, texture/UV scale, friction/restitution |
| Room | Separate 3.6576×3.6576×3.6576m four-wall rooms, no ceiling; own wall materials/background image paths, light count/type/intensity/temperature/cone/shadows and stationary/white/RGB disco profile |
| Robot | Seven independently shaded links, coherent nominal/0.99/1.01 geometry variants, mass/inertia scaling |
| Smudge | Spots/streaks, size, anisotropy, tint, opacity, blur, haze, local contrast and deterministic drift |
| Task | Explicit catalog selection, garment/knot variants and task-specific topology/mechanics |
| Actuator | Effort/velocity limits, stiffness, damping, friction, armature, delay, encoder bias/noise and optional backlash |
| Base | Supported Cartesian position and yaw close to the task |
| Image | Shear, crop, rotation with strict black outside padding, brightness/contrast/saturation/hue; `none`, `one` or `all` mode |
| Distractor | Bounded count, shape, dimensions, placement, appearance, optional physical bodies |
| Confetti | Lightweight rendered pieces with count, geometry, size/aspect, color, opacity, reflectivity, placement and an optional physical subset capped at eight pieces |

Structural table/room/robot/clutter/confetti draws are baked into coherent per-environment USD cohorts at launch. A new scene seed redraws these; they are not silently scaled in live physics. Actuator/base/image/smudge draws support indexed episode resets. Disco light phase and smudge drift advance deterministically. Rooms have separate material ownership and light/shadow links. The table leaves a wall gap and each camera belongs to its own render partition.

### Physical RGB vibrancy and wrist-view spots

Set these fields under `randomization.lighting`. Each two-value numeric range must be finite and ordered; a repeated value fixes that parameter. The sampler records realized parameters and independent temporal phases in checkpoints.

| Field | Meaning, accepted domain and default |
|---|---|
| `profile` | `stationary` (default), `white_disco` or `rgb_disco`; stationary/white profiles use white source RGB |
| `motion_mode` | `room_sweep` (default directional orbit) or `wrist_view` (moving physical spotlight targets in the wrist camera's view) |
| `count`, `max_lights`, `types` | Nonnegative integer count range within capacity, defaults `[1,4]` and `4`; nonempty type pool of `sphere`/`spot`, both by default. A `[0,0]` count with zero capacity is supported. Wrist-view sources use spots. |
| `color_saturation` | HSV saturation range `[0,1]`, default `[0.85,0.85]`; 0 is white, 1 is the pure hue |
| `saturation_period_s` | Finite nonnegative seconds, default `0`: keep the sampled saturation fixed. A positive period smoothly traverses `color_saturation` using a saved deterministic phase. |
| `angular_speed` | Nonnegative motion/hue speed range in rad/s, default `[0.3,2.0]`; it does not set the saturation period |
| `intensity` | Nonnegative colored/source light intensity range in normalized USD intensity units, default `[300,1800]`; independent of saturation |
| `white_fill_intensity` | Nonnegative neutral fill range in normalized USD intensity units, default `[0,0]`; lifts overall illumination independently of colored-light power |
| `spot_radius_m` | Positive transverse beam-radius range in meters, default `[0.025,0.07]`; used by wrist-view spots |
| `emitter_radius_m` | Positive source radius in meters, default `0.004`, for wrist-view projectors; separate from beam radius |
| `edge_softness` | Cone-edge softness range `[0,1]`, default `[0.15,0.4]`; 0 gives a hard edge |
| `view_extent` | Image trajectory extent range `[0,0.9]` as a fraction of each half-frame dimension, default `[0.25,0.65]` |
| `temperature_kelvin` | Positive range in Kelvin, default `[3000,8000]`; retained in room-sweep mode, disabled for wrist-view lights so HSV saturation controls their source color |
| `beam_angle_degrees` | Cone angle range strictly between 0 and 180 degrees, default `[15,70]`; wrist-view cones are derived from radius and source-to-target distance |
| `shadows_probability` | Probability `[0,1]`, default `0.8` |

`configs/rgb_spots.json` demonstrates varying vibrancy (`color_saturation=[0.2,1.0]`, eight-second period), moving wrist-view spots and independent neutral fill. `configs/rgb_disco.json` retains the room-sweep example. Profile fields control physical illumination; `randomization.image.saturation` remains a separate image augmentation.

Wrist-view motion composes the actual camera mount with the current link pose and traces a smooth image-space path through the configured intrinsics. The corresponding world rays intersect the horizontal task support plane, without moving the camera or changing its calibration. A horizon-facing, backward or excessively distant intersection uses a small bounded orbit around the support center and reports an invalid visible-ray mask. The support plane is an approximation: raised objects and occlusion can alter where a spot is seen, and arbitrary camera angles cannot guarantee a visible patch. `spot_radius_m` describes the radius transverse to the beam; an oblique tabletop intersection naturally produces an elliptical footprint. It is not a guarantee of a circular image-space spot.

Physical clutter and cards reserve space around the scaled CAD base and the task's initial/reset geometry. Table qualification also checks loose-object reset margins and commanded stack destinations. Construction can redraw eligible nuisance groups; an incompatible frozen layout fails clearly. Episode resets make up to 32 base-only placement attempts while preserving other randomization draws and retained environments. Exhaustion rolls the reset transaction back. These initial-clearance checks do not certify a collision-free manipulation path; visual-only confetti remains free to obscure task objects.

The six structural controls (`table`, `room`, `robot`, `task`, `distractor`, `confetti`) default to `cadence="fixed"`. Enabled, unfrozen structural controls requesting episode cadence are rejected instead of silently changed. Appearance belongs to the structural draw in this implementation; choose a new launch seed/cohort to redraw it. Material families (`matte`, `wood`, `plastic`, `metal`) select normalized roughness/metallic/specular priors inside the numeric ranges you supply. Wood uses the metallic interval minimum and metal its upper quarter; constant numeric bounds override family variation. Texture and friction remain explicit independent controls. These priors are not measured material models.

Backlash is a command-side rate-independent play/dead-band approximation, not identified HiWonder gear dynamics. Servo ranges, inertias and friction are initial engineering estimates. Camera exposure/gain/white balance/gamma and frame latency/drop are explicit image-channel approximations; a physical rolling shutter is not modeled.

Motor command history capacity follows the configured maximum delay, including delays above100ms. Camera frame-drop RNG streams are independent per environment and saved individually. The reviewed command schema, nuisance version and camera checkpoint schema reject incompatible exact resumes; policy initialization also requires compatible network input dimensions.

PhysX applies its native joint properties. Newton VBD uses the local motor adapter because its native armature/friction/effort/speed fields are unsupported: stabilized scalar-rotor PD, bounded motor torque, a no-load speed torque envelope, stopping-bounded smooth Coulomb drag, and a generalized inertial drive. Friction is a native PhysX joint coefficient or a Newton-adapter Coulomb torque in Nm, respectively. External loads can backdrive the speed envelope. The scalar controller approximates coupled/contact-loaded servo response; its parameters are not measured physical gains. Actual solver tests cover acceleration versus armature, frictional decay, torque/speed response, joint-limit crossing and the tiny-jaw damping stability boundary.

For thin native cables, the default Newton inertia regularizer would inflate rotational inertia substantially. The extension validates and restores the authored geometric tensors and their inverses before solver construction, preserving mass. Run metadata and task-check sidecars record the original, regularized and restored eigenvalues; these geometric values still require physical identification for a particular cable.

Rewards are configurable under `reward` in the JSON profile. Dense mode uses bounded reaching/task-progress and measured rigid grasp/lift potentials, with `gamma * potential_next - potential_previous` shaping. This avoids rewarding a policy indefinitely for hovering near the goal. Success/failure are task-defined terminal outcomes; time/action/velocity/effort costs are recorded separately. Set `reward.dense=false` for the sparse ablation. The shaping discount is tied to PPO's discount. Ground-truth reward features do not enter the actor.

## SGS and evaluation

The sampler follows [Success Guided Sampling, sections 3.2/B.2](https://arxiv.org/html/2610.12465v1): a fixed bank, rolling Boolean outcome histories, target 0.5, concentration 1, epsilon 1e-4 and temperature 2. Default bank size is 32768, history length 100, independent of environment count. Empty histories use the target prior. Simultaneous duplicate completions preserve all outcomes. Interrupted/manual resets do not add failures. Uniform training records completed outcomes for truthful coverage diagnostics while retaining a uniform sampling distribution; held-out evaluation leaves training histories untouched.

Each structural cohort has a compatible bank slice (`ID % num_envs`); probability sampling is conditional on that slice. The bank has an explicit deterministic definition/hash and immutable pose perturbations. Reaching and meaningful rigid/mechanism near-goal families are provided. `sgs_peg.json` also enables the bundled nominal simulated stable-grasp seed: measured two-jaw retention for 0.5s and a 20.34mm lift. Its measured jaw position and squeeze target are stored separately. Other geometry/task cohorts cannot silently claim this grasp qualification. Base/reset candidates require bounded five-arm-DOF CAD position IK and initial task clearance from the CAD base envelope, including cable radius and loose-object reset jitter. Only the base is resampled; an invalid fixed base fails explicitly. These checks reject invalid initial placements; they do not certify a collision-free grasp or manipulation path.

Histories average over the configured stationary actuator/base/sensor nuisances. The bank identity also pins the extension's Python source and physics runtime versions, so a repaired actuator/contact/reset implementation cannot inherit old success histories. Changing the bank/profile/implementation rejects resume rather than reusing incompatible histories; use `--initialize-from` for deliberate policy transfer. `--sampling uniform` supports the matching training ablation and ordinary resume preserves that mode. Evaluation assigns episode quotas to the fixed structural cohorts before collection, so early failures cannot dominate the score by completing more episodes. Its aggregate success and return weight cohort means by their bank-slice sizes; raw `success_count` and `sample_success_rate` remain available separately. `by_cohort` records weights, quotas and completed counts alongside the reset-family breakdown. An exhausted step budget or a budget too small to sample every cohort leaves aggregate success and return `null`, with coverage reported explicitly. Report held-out evaluation separately from SGS-weighted training success.

## Camera matching and calibration

The wrist camera is attached to the CAD `gripper_body` lens mount. Capture is 640×480 at 30Hz, with explicit OpenCV-to-OpenGL frame conversion. The default 60° horizontal FoV, focal length, filmback, distortion and optical center/roll are provisional. Physical f-number is unknown and depth of field is disabled. A CAD lens plane and USB capture descriptor cannot establish the actual lens intrinsics.

HiWonder's [camera product page](https://www.hiwonder.com/products/hd-wide-angle-camera) lists several lens/FoV variants. Its [SO-ARM101 manual](https://docs.hiwonder.com/projects/LeRobot/en/latest/docs/SO_ARM101_Open_Source_6_Axis_Robotic_Arm_User_Manual.html) specifies the capture workflow without an optical calibration for this particular camera. Those sources do not identify the exact installed lens well enough to replace measurement.

`tools/calibrate_hiwonder_camera.py` fits intrinsics from at least ten varied checkerboard captures and registers the mount from measured non-coplanar link-frame 3D/pixel correspondences. Keep independent validation captures; fit RMS alone is not acceptance. No calibration captures or measured optical specifications were present in the supplied bundle, so this installation cannot yet claim an exact physical-camera match.

```powershell
& .\.venv\Scripts\python.exe source/isaaclab_hiwonder/tools/calibrate_hiwonder_camera.py intrinsics '<captures>' --inner-corners 9 6 --square-m 0.02 --output '<output>/intrinsics.json'
& .\.venv\Scripts\python.exe source/isaaclab_hiwonder/tools/calibrate_hiwonder_camera.py mount '<correspondences.json>' --profile '<output>/intrinsics.json' --output '<output>/mounted.json'
& .\.venv\Scripts\python.exe source/isaaclab_hiwonder/tools/calibrate_hiwonder_camera.py profile '<output>/mounted.json' --output '<output>/camera-profile.json'
```

The last command emits a loadable `--profile` override while retaining pending physical-validation status. Filmback/focal parameters are reconciled with the fitted intrinsics, and configured projection is tested with actual rendered markers. Frame latency/drop handling uses actual capture IDs and elapsed time independently of the control rate.

## Runtime optimization

Task measurements gather the selected live body/joint data before copying it to the host. Physics-cadence mechanism updates request their actual inputs, with a complete-state fallback for custom mechanics. Model/contact topology and cable rest geometry are cached with invalidation; live poses, velocities, local centers of mass, contacts and forces are always refreshed. Warp arrays use a DLPack stream handoff before Torch reads them.

Observation assembly batches the existing joint-cost and privileged-state transfers. Image transforms reuse bounded derived grids, masks and color constants while preserving operation order, black borders and returned-buffer ownership. SGS reuses probabilities only while the complete history and sampler parameters match; sampling still consumes the original RNG sequence. Public tensor/definition mutations, dtype changes and autocast transitions invalidate affected derived data.

Generated scenes reuse content-addressed immutable mesh sources and exact texture bytes within their run directory. Materials, lights, transforms, collision and physical properties remain independently authored. Existing USD prims retain their references, metadata and animation. Relative references support moving the complete generated directory together; keep its geometry and texture sidecars with the scene.

These changes do not reduce camera resolution/rate, physics frequency, solver iterations, collision frequency, contact coverage, randomization ranges or task mechanics. Source changes deliberately change the SGS bank identity: transfer older policies with `--initialize-from`, rather than importing histories from a different implementation. See [OPTIMIZATION_VALIDATION.md](OPTIMIZATION_VALIDATION.md) for the all-30 benchmark, review repairs, strict native trajectory failures and incomplete supplemental checks. [VALIDATION.md](VALIDATION.md) retains the earlier suite and lighting evidence.

## Robot collision profiles

`robot_collision_profile` defaults to `"manipulation"`. On native Newton tasks it keeps both gripper links and all servo housing colliders intact, while removing redundant body hulls and reducing body vertices. New body hulls stay inside their original convex pieces; the maximum inward body-surface loss is 2 mm. This is an intentional body-contact approximation, not an exact dynamics-equivalence optimization. Visuals, inertias, joint frames, travel, camera and physics/solver settings are unchanged.

PhysX keeps its existing CAD collision geometry: its own cooker already produces far fewer hulls than Newton. `"cad"` selects the original path on either backend. Profiles are chosen before scene creation; switching requires rebuilding the scene. The profile checksum/settings are included in the SGS bank identity, so use `--initialize-from` when transferring older policies.

`robot_self_collisions` accepts `true`, `false` or `null` in a JSON profile. `null` retains the backend default. `true` enables nonadjacent links while keeping the original directly connected joint-interface exclusions; it does not hide all self contacts. Extremely folded poses can still have real nonadjacent self contacts. See [COLLISION_VALIDATION.md](COLLISION_VALIDATION.md) for certificates, measured reductions, sampled clearance checks and limitations.

## Checks

```powershell
& .\.venv\Scripts\python.exe -m pytest source/isaaclab_hiwonder -q
& .\.venv\Scripts\python.exe source/isaaclab_hiwonder/tools/check_hiwonder_articulation.py --output '<output>/articulation.json'
& .\.venv\Scripts\python.exe source/isaaclab_hiwonder/tools/check_hiwonder_scene.py --output-directory '<new-output>'
& .\.venv\Scripts\python.exe source/isaaclab_hiwonder/tools/check_hiwonder_tasks.py --all --output-directory '<new-output>'
& .\.venv\Scripts\python.exe source/isaaclab_hiwonder/tools/check_hiwonder_runtime.py --output-directory '<new-output>'
& .\.venv\Scripts\python.exe source/isaaclab_hiwonder/tools/check_hiwonder_tasks.py --task rope_thread:overhand --steps 64 --quiet-window 0.5 --quiet-metric control_frame --output-directory '<new-output>'
```

The scene tool renders wrist views and checks room isolation and measured support contact. The catalog tool uses isolated processes to exercise a zero-action physics fixture, interrupted reset semantics and checkpoint restore. It rejects failures before automatic reset can hide the physical state, saving a diagnostic JSON/NPZ. Use `--num-envs 2` for additional indexed-reset isolation checks. Review `VALIDATION.md` for the measured acceptance boundary; no pretrained task-solving policy is included.
