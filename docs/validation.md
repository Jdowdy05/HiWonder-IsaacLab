# Validation

[Documentation](README.md) · [Project overview](../README.md)

Records retain their original dates, robot variants and evidence limits. Use them to distinguish package checks, simulator execution, contact validation and learned task performance.

## Checks and validation limits

Source-only setup checks, run from this overlay repository:

```text
python -m unittest discover -s tests -v
```

After SDK installation, run from the installed upstream checkout:

```powershell
& ./.venv/Scripts/python.exe -m pytest source/isaaclab_hiwonder -q
& ./.venv/Scripts/python.exe source/isaaclab_hiwonder/tools/check_hiwonder_tasks.py --all --output-directory '<new-output>'
```

The second command launches simulator processes and selects stock; `--robot-model eka_v6` explicitly selects the new hand for that checker. Training and inspection instead select their robot through `--profile`. Run GPU checks serially with measured host headroom and an external process-tree memory limit. The historical [validation](../source/isaaclab_hiwonder/VALIDATION.md), [optimization](../source/isaaclab_hiwonder/OPTIMIZATION_VALIDATION.md), and [collision](../source/isaaclab_hiwonder/COLLISION_VALIDATION.md) records retain their original dates, source digests and failed/incomplete checks. All historical all-30 GPU, contact and RGB training receipts in those records concern the stock robot. Their large external evidence bundles are not included and they do not qualify the Eka model or new integrated implementation.

The staged Eka runtime has 106 passing focused CPU tests, including construction of all 30 task/critic layouts at capacity 320 and a small CPU PPO update/save/restore contract. Integrated full-suite and saved-asset/runtime validation are recorded separately when complete. CPU construction and learner contracts do not establish GPU execution, rendering, contact stability, grasp success or task learning with Eka.

Full camera/cloth startup has previously exceeded an external 8 GiB private-allocation guard before scene setup completed. Compute-only collision probes do not establish full-scene memory savings. Strict cross-process native Newton trajectory comparisons remain rejected in the historical optimization record. Camera/renderer and physics allocations must be measured separately. The new [Eka evidence](../validation/eka_v6-2026-10-28/README.md) records separate bounded simulation checks; package tests alone do not establish policy success.

The CAD camera mount and 640x480/30 Hz capture settings are represented, but lens intrinsics, distortion, optical center/roll, servo gains, inertias and contact parameters remain provisional or estimated. Calibration needs independent physical measurements and held-out captures. A configured camera rate does not establish real-time FPS. Finite task smoke checks do not establish learning, reliable manipulation or sim-to-real transfer.

## Recorded package verification

Final package suite:383 passed,1 CUDA-only check skipped. After the final visualizer/API fix and import formatting,19 focused tests passed; independent review has no open findings. A fresh source-only installation matched168 copied hashes at the pinned GA commit and passed the offline lock check. SDK binaries were not reinstalled.

The separate [ACT report](../experiments/act-public-pickplace-2026-10-29/REPORT.md) describes offline prediction metrics and their limits; it does not establish closed-loop task success.
