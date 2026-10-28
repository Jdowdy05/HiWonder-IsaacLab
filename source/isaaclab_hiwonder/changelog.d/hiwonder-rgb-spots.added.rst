* Add sampled RGB saturation and independent temporal vibrancy, power and neutral fill controls.
* Add soft moving RGB spotlights targeting the wrist camera's visible support plane, with
  explicit current RTX UsdLux semantics and per-room light ownership.
* Add a held-out inspection tool with live controls, paired full-resolution browser frames,
  paused raw-render preview, profile export and retained GPU image uploads.
* Cache light handles, skip unchanged USD writes and project nuisance snapshots in hot paths;
  bound asynchronous browser encoding independently of simulation and camera cadence.
* Preserve command, camera-age, action, timeout and reward-log buffer storage across inference steps so
  manual reset and checkpoint restore can run outside inference mode.
