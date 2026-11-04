# HiWonder Isaac Lab

SO-ARM101 manipulation environments, robot assets and RGB reinforcement learning for **Isaac Lab 3.0.0**. Supports the stock follower and the **Eka V6 hand** with four independently actuated finger joints.

- Wrist RGB and proprioception with PPO and Success Guided Sampling (SGS).
- 30 task presets covering rigid assembly, mechanisms, cables, cloth and paper.
- Ten domain randomization groups, including moving RGB lighting, camera effects and actuator variation.

## Get started

Requires Git, Python 3.12, uv >=0.13.1 and a supported NVIDIA GPU. See [Installation](docs/installation.md) for requirements and Linux commands.

```powershell
git clone https://github.com/Jdowdy05/HiWonder-IsaacLab.git
Set-Location HiWonder-IsaacLab
python tools/setup_isaaclab.py
```

The installer creates the sibling `IsaacLab-HiWonder` checkout. Continue with [camera inspection and training](docs/usage.md).

## Documentation

| Guide | What you need |
|---|---|
| [Installation](docs/installation.md) | Pinned versions, Windows/Linux setup and custom paths |
| [Robot and camera](docs/robot-and-camera.md) | Stock/Eka selection, joint dimensions and camera calibration |
| [Usage](docs/usage.md) | Inspect, train, resume, evaluate and play |
| [Tasks and randomization](docs/tasks-and-randomization.md) | Task presets, profiles, lighting and SGS resets |
| [Validation](docs/validation.md) | Check commands, recorded results and evidence limits |
| [Assets and licensing](docs/assets.md) | Robot assets, source CAD, repairs and provenance |

See the [documentation index](docs/README.md) for experiments and the detailed extension reference.

## ACT experiment

The [public-data ACT trial](experiments/act-public-pickplace-2026-10-29/README.md) includes training code, dataset provenance, a checkpoint and results. It remains an offline experiment with mixed performance; physical and wrist-camera transfer need validation.

## License

Software: [BSD-3-Clause](LICENSE). Source CAD and derived assets have [separate provenance and redistribution limits](ASSET_PROVENANCE.md).
