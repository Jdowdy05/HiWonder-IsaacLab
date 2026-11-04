# Tasks and randomization

[Documentation](README.md) · [Project overview](../README.md)

## Task presets and profiles

The 30 presets consist of:

| Group | Exact preset names |
|---|---|
| Rigid objects | `peg_insert`, `cube_stack`, `nut_pick_place`, `washer_ring`, `connector_insert`, `shape_sort`, `gear_assembly`, `circuit_restore`, `catch_ball`, `roll_ball`, `roll_ball:table` |
| Mechanisms | `knob`, `switch`, `latch`, `bottle_cap`, `bottle_cap:close`, `key_lock` |
| Cables/rope | `barrel_jack`, `rope_thread:thread`, `rope_thread:overhand`, `rope_thread:figure_eight`, `cable_untangle` |
| Cloth | `cloth_fold:towel`, `cloth_fold:handtowel`, `cloth_fold:shirt`, `cloth_fold:rag`, `cloth_fold:shorts`, `cloth_fold:underwear`, `cloth_fold:sock` |
| Plastic paper | `paper_airplane` |

Rigid/mechanism tasks use PhysX; cables, cloth and paper use native Newton VBD. Select with `--task NAME`. The strict JSON schema rejects unknown fields. `task_parameters` controls supported commanded goals such as cube order or target mechanism state. The [extension guide](../source/isaaclab_hiwonder/README.md) describes the mechanics, ten randomization groups, rewards and calibration commands in detail.

| Profile in `source/isaaclab_hiwonder/configs/` | Purpose |
|---|---|
| `default.json` | Complete reference profile |
| `clean.json` | Nuisance-disabled ablation |
| `rgb_disco.json` | Combined visual nuisance and moving room lights |
| `rgb_spots.json` | Moving physical wrist-view spots and varying color saturation |
| `occlusion.json` | Increased visual table coverage |
| `sgs_peg.json` | SGS peg reset families with bundled simulated stable-grasp seed |
| `eka_v6.json` | Explicit Eka robot with nine actuator joints and ten rigid bodies |
| `eka_v6_rgb_spots.json` | Eka robot plus moving wrist-view RGB lighting |

SGS uses a fixed reset bank and completed Boolean outcome histories. Interrupted/manual resets do not count as failures. `--sampling uniform` provides a training ablation. The bundled stable-grasp JSON/NPZ and `sgs_peg.json` qualify only the nominal stock simulated peg cohort. They cannot supply an Eka grasp seed; Eka needs a new receipt for its exact robot asset before enabling `stable_grasp` resets. The bundled seed is not a trained policy or physical grasp calibration.

## Ten randomization groups

Copy `configs/default.json` or supply a partial JSON using `--profile`. Unknown fields are rejected. Every group has `enabled`, `seed`, `cadence`, `reset_interval` and `frozen` controls under `randomization.controls`. `clean.json` disables nuisance groups for an ablation; `rgb_disco.json` demonstrates combined visual nuisance; `occlusion.json` increases table coverage without simulating every piece as a rigid body.

| Group | Parameters and application |
|---|---|
| Table | Shape, dimensions, rounded edges, PBR material/color/roughness/metallicity/specularity, texture/UV scale, friction/restitution |
| Room | Separate 3.6576×3.6576×3.6576m four-wall rooms, no ceiling; own wall materials/background image paths, light count/type/intensity/temperature/cone/shadows and stationary/white/RGB disco profile |
| Robot | Seven independently shaded links for stock or ten for Eka, coherent nominal/0.99/1.01 geometry variants, mass/inertia scaling |
| Smudge | Spots/streaks, size, anisotropy, tint, opacity, blur, haze, local contrast and deterministic drift |
| Task | Explicit catalog selection, garment/knot variants and task-specific topology/mechanics |
| Actuator | Effort/velocity limits, stiffness, damping, friction, armature, delay, encoder bias/noise and optional backlash |
| Base | Supported Cartesian position and yaw close to the task |
| Image | Shear, crop, rotation with strict black outside padding, brightness/contrast/saturation/hue; `none`, `one` or `all` mode |
| Distractor | Bounded count, shape, dimensions, placement, appearance, optional physical bodies |
| Confetti | Lightweight rendered pieces with count, geometry, size/aspect, color, opacity, reflectivity, placement and an optional physical subset capped at eight pieces |

Structural table/room/robot/clutter/confetti draws are baked into coherent per-environment USD cohorts at launch. A new scene seed redraws these; they are not silently scaled in live physics. Actuator/base/image/smudge draws support indexed episode resets. Disco light phase and smudge drift advance deterministically. Rooms have separate material ownership and light/shadow links. The table leaves a wall gap and each camera belongs to its own render partition.

For RGB saturation, moving spotlights and lighting parameters, see the [physical lighting reference](../source/isaaclab_hiwonder/README.md#physical-rgb-vibrancy-and-wrist-view-spots).
