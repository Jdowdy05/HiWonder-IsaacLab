# Robot and camera

[Documentation](README.md) · [Project overview](../README.md)

## Choose the robot

| Profile selection | Robot | Actions / rigid bodies | Proprioception / critic capacity |
|---|---|---|---|
| Default profiles, `robot_model="stock"` | Five arm joints and the original one-joint gripper | 6 / 7 | 18 / 256 |
| `configs/eka_v6.json` or `configs/eka_v6_rgb_spots.json` | Five arm joints and four independent finger joints | 9 / 10 | 27 / 320 |

The Eka hand has two left/right finger chains, each with independently actuated root and tip joints. The stock model remains the default for legacy profile selection. Select Eka deliberately with `--profile source/isaaclab_hiwonder/configs/eka_v6.json`; the stock example commands below retain their original robot selection.

Retaining the stock default does not guarantee exact checkpoint or SGS-bank replay across implementation fingerprint changes. Old six-action stock checkpoints are incompatible with the nine-action Eka model, including through `--initialize-from`. Same-model policy initialization still requires compatible network dimensions and starts a fresh optimizer/bank.

## Wrist camera

Both variants use 640x480 wrist capture configured at 30 Hz. The Eka model retains the original wrist-relative camera pose and optics; its new physical camera attachment remains provisional and needs mount measurement and optical calibration.

For operating the viewer, see [Usage](usage.md#inspect-the-arm-and-wrist-camera). For the calibration workflow and optical parameters, see the [camera reference](../source/isaaclab_hiwonder/README.md#camera-matching-and-calibration).
