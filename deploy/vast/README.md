# Vast deployment helpers

See [the field-by-field setup and launch guide](../../docs/vast.md).

- `template.json` and `render_template.py`: UI values and immutable, checksum-verified startup text.
- `provision.sh`: stage the two pinned source checkouts and coordinate their separate environments.
- `bootstrap_act.sh`: install only the ACT environment; existing RL bootstrap remains pinned separately.
- `assets.json` and `vast_assets.py`: restore the prepared public ACT cache and archived checkpoint with hashes intact.
- `run_act.py`: start a bounded ACT smoke/training job with a fresh optimizer and optional archived-weight initialization.

Run CPU checks with `python -m pytest deploy/vast/test_vast.py -q`. Shell scripts require Linux x86_64; tests use confined mock shell commands and do not install packages, launch Kit or rent GPUs.
