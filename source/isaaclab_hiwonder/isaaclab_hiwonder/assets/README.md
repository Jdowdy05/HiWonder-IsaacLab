# HiWonder follower asset

`generated/hiwonder_soarm101.usdc` is a local conversion of the supplied assembled
HiWonder follower CAD. It contains seven rigid bodies, six revolute joints and a
fixed base. It does not reference the generic SO101 robot. The default prim is
`/HiWonder`; all source objects have an explicit rigid-link assignment in
`registration.py` and the generated manifest.

The six joints, in control order, are `shoulder_pan`, `shoulder_lift`,
`elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`. Zero angles reproduce the CAD
assembly. The nominal reset is `(0, 0, 0, 0, 0, -0.34906585)` radians. Negative
gripper angles open the jaw; the provisional close target is `0.20943951` radians.
These are simulation coordinates, not physical servo encoder coordinates.

The robot's support frame has CAD-aligned X/Y/Z axes and its origin at the CAD
base floor. USD units are meters. The camera exists at
`gripper_body/camera_mount/Camera`. An approximate finger-tip reference exists at
`gripper_body/tool_center`. `kinematics.forward_kinematics` provides support-frame
link, camera and tool transforms for the same registered geometry.

The camera optical +Z direction follows the lens plane toward the outside of the
camera housing. The local OpenCV rotation is X=160 degrees; OpenGL/USD is X=-20
degrees. The lens mechanical center is used as an **estimated** optical center,
with zero assumed optical offset. Image roll, optical center and distortion need
independent calibration. The VGA 640x480/30fps capture capability is known; the
preview filmback, focal length and zero distortion are illustrative estimates.
The physical f-stop is unknown, and USD depth of field is disabled. Filmback and
f-stop are separate parameters. USD lens lengths are authored in tenths of the
meter stage unit, so a 4.8mm filmback is stored as 0.048.

Structural collisions use per-component convex decomposition; servo housings
use convex hulls. These approximations require simulator contact validation,
especially at the finger pads and assembly fixtures. Simplified visuals preserve
the source components; screws and small internal electrical components have
reduced triangle budgets. Six construction/empty-extrusion objects without saved
render geometry are recorded and omitted from rendering.

Masses, box-based inertias, physical gains, continuous torque, friction and joint
travel remain estimates. The HX-30HM manual's 30kg.cm stall torque and 0.19s/60deg
no-load speed at 11.1V are recorded separately, not used as measured continuous
ratings. No port, motor, calibration register or physical actuator is accessed.

Prebuilt `scale099` and `scale101` variants coherently scale visual/collision
vertices, pivots and camera/tool attachments by ±1 percent. Mass scales cubically
and inertia scales to the fifth power. These are synthetic modeling variations,
not measured manufacturing tolerances. Choose the variant before spawning.

Conversion and validation commands from the repository root:

```powershell
python source/isaaclab_hiwonder/tools/convert_hiwonder_cad.py 'path/to/follower arm black.3dm' --variants
python source/isaaclab_hiwonder/tools/validate_hiwonder_asset.py source/isaaclab_hiwonder/isaaclab_hiwonder/assets/generated/hiwonder_soarm101.usdc
python source/isaaclab_hiwonder/tools/calibrate_hiwonder_camera.py intrinsics captures --inner-corners 9 6 --square-m 0.015 --output camera-fit.json
python source/isaaclab_hiwonder/tools/calibrate_hiwonder_camera.py mount measured-points.json --profile camera-fit.json --output camera-mounted.json
python source/isaaclab_hiwonder/tools/calibrate_hiwonder_camera.py profile camera-mounted.json --output environment-camera.json
```

The converter requires isolated `rhino3dm`, `numpy`, `trimesh`,
`fast-simplification`, and `usd-core`. Do not install pip `usd-core` into the Isaac
Sim runtime. Camera calibration requires OpenCV and already captured images;
mount points must be measured in the gripper-body frame and non-coplanar. Fitted
profiles remain pending independent validation even when fit residuals are low.
The `profile` subcommand filters fit-only evidence fields into a valid environment
`camera` override, references the complete fitted JSON for provenance, and derives
horizontal/vertical FoV from the fitted intrinsic matrix. It does not promote a
fit to `calibrated`. Its virtual filmback/focal-length pair is projection-consistent;
the physical filmback scale and f-stop still need independent optical information.

`manifest.json` records source/artifact SHA-256, link assignments, camera profile,
triangle counts, collision approximations and provenance. Source CAD licensing
has not been established; generated assets are included in this private
owner-authorized repository, with public redistribution pending rights review. Static USD registration checks establish transforms and
topology only; they do not establish contact stability or physical accuracy.
