# Asset provenance and rights

This is a private owner-authorized copy of the custom HiWonder Isaac Lab project. Its software and CAD-derived assets have distinct provenance.

The extension's existing Python source headers declare BSD-3-Clause and are preserved unchanged. `source/isaaclab_hiwonder/pyproject.toml` declares the same software license. Root `LICENSE` preserves the upstream notice/disclaimer for the Isaac Lab-derived software and integration manifest. The small repository setup helper and its tests are provided under BSD-3-Clause as well. Dependencies such as Isaac Sim, PyTorch, Newton, RSL-RL and optional tools retain their own licenses and acceptance requirements.

The generated HiWonder follower USDs and scale variants derive from owner-supplied assembled follower CAD. The exact source hash, individual source object/link assignments, camera assumptions, conversion approximations and output hashes are recorded in `source/isaaclab_hiwonder/isaaclab_hiwonder/assets/generated/manifest.json`. Original CAD input bytes are not included in this repository. The existing generated asset and collision manifests remain byte-identical to their packaging source; absolute paths embedded as historical provenance are not runtime installation requirements.

The external regeneration input is `follower arm black.3dm`, 176,591,534 bytes, SHA-256 `f675f6049f94206c7b74ab73c40485913916babff1a6f4fc145dc9040672a2f8`. It exceeds GitHub's normal file limit and is intentionally omitted; no Git LFS dependency is required to use the included runtime assets. Keep the original input externally if CAD regeneration is needed.

CAD licensing and permission for public redistribution of the original model or its derivatives have **not** been established. A software BSD declaration does not grant rights to these CAD assets. Their inclusion here is for the owner's requested private repository; public release or redistribution requires separate rights confirmation.

`stable_peg_grasp.json` and `stable_peg_grasp.npz` are retained because SGS loads the qualified simulated reset state. They are small runtime assets, not trained policy weights. Any local evidence path inside their metadata records the historical experiment; the packaged loader uses the bundled files.

The represented robot is the stock HiWonder follower geometry present when this package was copied. No Eka geometry is incorporated. Masses, inertias, motor properties, collision approximations and optical calibration limits are documented in the extension's asset README and validation records. The assets establish simulation geometry/configuration, not physical acceptance.
