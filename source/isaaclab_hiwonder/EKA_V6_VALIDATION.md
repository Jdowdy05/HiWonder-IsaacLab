# Eka V6 HiWonder integration

The supplied `Eka_XL330_SO100_V6_Distal_Assembly_Printable_Parts.step` mounts after
`wrist_roll`. Its SHA-256 is
`141ab8a721d850ed88da9d97f17257ebce1f85cfca7a0403f4c6e20c1f60e944`.
`assets/eka_v6/source.step` preserves it byte for byte. This variant retains the
five original arm axes and adds independently controlled left/right root and tip
joints: nine actions, ten rigid bodies, 27 proprioceptive values and a 320-value
privileged critic buffer. The stock six-action model remains available.

## Geometry and repair

The adapter and wrist horn share four holes on a measured 14 mm pitch circle.
`assets/robot_models.py` records the mating-frame measurements and transforms.
CAD hole alignment is not a physical fastener, clearance or strength test.
The supplied export contains fifteen printable solids and only a right liner.
The earlier native V6 file supplied hardware reference geometry, not replacement
printed shapes. The palm, root cases/bearings, proximal cases/horns and distal
horns/caps move with their registered rigid bodies.

Thirteen printed meshes use exact-surface NURBS reparameterization to recover
complete tessellation. Both cap meshes needed boundary filling. PyMeshFix
0.18.1 `fill_small_boundaries` was used without global cleaning or component
removal at source precision. The original STEP and unpatched cap tessellation
are included. At body-local float32 precision, coincident seam vertices and
zero-area triangles were resolved, small remaining boundaries stitched, and one
negligible four-face zero-volume sliver per cap removed. The final three scale
variants are closed, consistently wound, single-component positive-volume
meshes; tests verify that the original gripping-surface triangles remain.

The repaired CAD exports have two faceted cap solids and thirteen retained
imported STEP solids. They do not recover Fusion sketches or parametric history.
Both repaired caps passed OCCT validity, closedness and positive-volume checks;
the native copy was reopened and the STEP reimported with fifteen valid solids.

The original arm, output horn and wrist camera are retained. The camera still
uses 640x480 at 30 Hz and the existing intrinsics/optical pose. Its attachment to
this new palm is provisional pending measurements; focal properties/distortion
and dynamics/joint travel remain estimates. The tool center follows the actual
midpoint of the moving tip contact references. Tip contacts provide the two
pinch aliases; all five hand bodies provide generic hand contact semantics.

## Select and check

Run from the installed Isaac Lab root, after accepting the SDK license:

```powershell
& ./.venv/Scripts/python.exe source/isaaclab_hiwonder/tools/validate_hiwonder_asset.py source/isaaclab_hiwonder/isaaclab_hiwonder/assets/eka_v6/hiwonder_eka_v6.usdc --robot-model eka_v6
& ./.venv/Scripts/python.exe source/isaaclab_hiwonder/tools/inspect_hiwonder.py --profile source/isaaclab_hiwonder/configs/eka_v6_rgb_spots.json --task peg_insert --num-envs 1 --viz kit --kit_args=--/app/vulkan=false
& ./.venv/Scripts/python.exe source/isaaclab_hiwonder/tools/check_hiwonder_grasp.py --robot-model eka_v6 --device cuda:0 --output-directory logs/eka-grasp --kit_args=--/app/vulkan=false
& ./.venv/Scripts/python.exe source/isaaclab_hiwonder/tools/check_hiwonder_collision_profile.py --robot-model eka_v6 --device cuda:0 --contact-capacity 8388608 --output logs/eka-collision.json
```

Linux uses `.venv/bin/python`. The Vulkan override is specific to the tested
Windows launch. Use a process memory guard for SDK checks; cameras and native
SDF provisioning can use substantial memory. Solver fidelity was not reduced.
Old six-action checkpoints cannot initialize or resume nine-action policies.

## Reproduce derived assets

The checked mesh archives are shipped, so USD rebuilding does not require
FreeCAD, Rhino or a hardware connection. Use fresh output directories throughout.
Build from the frozen, checked mesh archives:

```powershell
& ./.venv/Scripts/python.exe source/isaaclab_hiwonder/tools/build_eka_hiwonder.py --output rebuilt-eka
Copy-Item source/isaaclab_hiwonder/isaaclab_hiwonder/assets/eka_v6/source.step rebuilt-eka/source.step
Copy-Item source/isaaclab_hiwonder/isaaclab_hiwonder/assets/eka_v6/source_geometry.npz rebuilt-eka/
Copy-Item source/isaaclab_hiwonder/isaaclab_hiwonder/assets/eka_v6/precision_caps.npz rebuilt-eka/
& ./.venv/Scripts/python.exe source/isaaclab_hiwonder/tools/bake_eka_collision_references.py --asset-directory rebuilt-eka
& ./.venv/Scripts/python.exe source/isaaclab_hiwonder/tools/build_hiwonder_collision_profile.py --robot-model eka_v6 --assets rebuilt-eka --output rebuilt-eka/manipulation_collisions.json
```

The optional repair tool dependency is separate from the locked simulator env:

```powershell
uv run --no-project --python ./.venv/Scripts/python.exe --with pymeshfix==0.18.1 -- python source/isaaclab_hiwonder/tools/stitch_eka_caps.py --output repaired-eka
```

This reproduces and checks a new geometric boundary repair for inspection.
It does not replace the registered frozen inputs. The reference cooker pins
the source STEP and source archives and rejects changed inputs; promoting a
new repair requires a separate geometry/contact qualification and registration.
The native reference cooker reuses the exact stock arm/camera hulls, cooks the
new hand on CPU and derives hand scale variants by uniform affine scaling. The
manipulation profile protects all five hand bodies without further simplification
and retains the established <=2 mm body surface-loss bound on the arm links.
Different generated USD bytes require matching regenerated reference hashes.
Boundary-fill triangulation in the new noncontact patches can differ between
runs; the recipe establishes geometry/contact-surface reproducibility, not
byte-identical NPZ archives. The shipped archives remain the exact frozen inputs.
Do not overwrite the installed model with unqualified candidates.

For the repaired printable STEP/FCStd export, use FreeCAD's own Python or the
project FreeCAD MCP headless tool. Load
`tools/export_eka_repaired_step_freecad.py` with `runpy.run_path`, then call its
`export_repaired_step(new_directory)` function. It verifies the original source
hash, exports into a fresh directory, reopens native CAD and reimports STEP.
Do not install FreeCAD or pip `usd-core` into the locked Isaac Sim environment.

## Qualification scope

The final validation report and machine-readable receipts accompany this change.
The whole CPU suite covers the retained stock model and new Eka contracts; all
thirty task presets have CPU construction/critic-layout coverage. Representative
GPU checks cover articulation, wrist rendering, lighting/material isolation,
measured two-tip peg grasp/lift, native cloth execution and RGB PPO with SGS.
Passing these checks does not mean a policy solves all thirty tasks or that the
physical assembly/camera/servo dynamics are calibrated. Historical all-thirty
solver receipts in the other validation documents apply to the stock arm.
