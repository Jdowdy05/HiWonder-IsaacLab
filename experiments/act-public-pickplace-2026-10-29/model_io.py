"""Strict, offline ACT checkpoint loading for evaluation artifacts."""
from pathlib import Path


def load_policy_checked(checkpoint, *, device='cpu'):
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy

    checkpoint=Path(checkpoint).resolve()
    config=ACTConfig.from_pretrained(checkpoint)
    config.device=str(device)
    # The saved checkpoint must contain every trained tensor; fetching ImageNet is unnecessary.
    config.pretrained_backbone_weights=None
    return ACTPolicy.from_pretrained(checkpoint, config=config, strict=True).to(device).eval()
