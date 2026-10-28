# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Export the repaired printable STEP from FreeCAD's own Python interpreter.

Use ``runpy.run_path(script)["export_repaired_step"](new_directory)`` in a
FreeCAD MCP headless process or the FreeCAD Python console. NumPy is required.
The two repaired caps become faceted solids; the other thirteen imported source
solids are retained. No parametric history is recovered. Existing output and
unrelated documents are never overwritten or closed.
"""

import hashlib
import json
import sys
import uuid
from pathlib import Path


def export_repaired_step(output_directory):
    import FreeCAD as App
    import Import
    import numpy as np
    import Part

    source_root = str(Path(__file__).resolve().parents[1])
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from isaaclab_hiwonder.assets.robot_models import get_robot_model

    model = get_robot_model("eka_v6")
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError(output)
    original = model.asset_directory / "source.step"
    manifest = json.loads(model.manifest_path.read_text())
    if hashlib.sha256(original.read_bytes()).hexdigest() != manifest["source"]["sha256"]:
        raise ValueError("Supplied STEP checksum differs from the registered source")
    doc = App.newDocument("Eka_V6_Patched_" + uuid.uuid4().hex[:8])
    Import.insert(str(original), doc.Name)
    parts = [obj for obj in doc.Objects if hasattr(obj, "Shape") and len(obj.Shape.Solids) == 1]
    if len(parts) != 15:
        raise ValueError("Expected exactly fifteen supplied printable solids")
    rotation = np.asarray(model.mount["rotation"])
    translation = np.asarray(model.mount["translation_cad_mm"])
    records = []
    with np.load(model.asset_directory / "precision_caps.npz", allow_pickle=False) as caps:
        for index, body in ((6, "eka_right_tip"), (13, "eka_left_tip")):
            points = (
                caps[f"nominal_v{index}"] * 1000 + np.asarray(model.links[body]["origin_mm"]) - translation
            ) @ rotation
            faces = caps[f"nominal_f{index}"]
            shape = Part.Shape()
            shape.makeShapeFromMesh(
                ([App.Vector(*point) for point in points], [tuple(map(int, face)) for face in faces]), 1e-7
            )
            if len(shape.Shells) != 1:
                raise ValueError("Patched cap did not make exactly one shell")
            solid = Part.makeSolid(shape.Shells[0])
            if solid.Volume < 0:
                solid.reverse()
            if not (solid.isValid() and solid.isClosed() and len(solid.Solids) == 1):
                raise ValueError("Patched cap is not a valid closed solid")
            parts[index].Shape = solid
            parts[index].addProperty("App::PropertyString", "RepairNote")
            parts[
                index
            ].RepairNote = "Derived faceted boundary stitch; retained imported STEP, no native feature history"
            records.append(dict(label=parts[index].Label, volume_mm3=solid.Volume, valid=True, closed=True))
    doc.recompute()
    if not all(obj.Shape.isValid() and len(obj.Shape.Solids) == 1 for obj in parts):
        raise ValueError("A retained source solid failed CAD validity")
    output.mkdir(parents=True)
    native = output / "Eka_V6_Patched_Printable_Parts.FCStd"
    step = output / "Eka_V6_Patched_Printable_Parts.step"
    doc.saveAs(str(native))
    Import.export(parts, str(step))
    names = [obj.Name for obj in parts]
    App.closeDocument(doc.Name)
    reopened = App.openDocument(str(native))
    if not all(reopened.getObject(name).Shape.isValid() for name in names):
        raise ValueError("Saved native copy failed reopened CAD validity")
    verification = App.newDocument("Eka_V6_Verify_" + uuid.uuid4().hex[:8])
    Import.insert(str(step), verification.Name)
    solids = [obj for obj in verification.Objects if hasattr(obj, "Shape") and len(obj.Shape.Solids) == 1]
    if len(solids) != 15 or not all(obj.Shape.isValid() for obj in solids):
        raise ValueError("Exported STEP failed reimported fifteen-solid validity")
    App.closeDocument(verification.Name)
    result = dict(
        source_sha256=manifest["source"]["sha256"],
        caps=records,
        reopened_native_valid=True,
        reimported_step_solids=15,
        reimported_step_all_valid=True,
        qualification="faceted cap repair; imported STEP geometry; no physical fit or strength validation",
    )
    (output / "cad-repair-verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    return result
