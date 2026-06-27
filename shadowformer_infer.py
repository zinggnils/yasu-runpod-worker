"""Path B: ShadowFormer GPU inference (placeholder until GPU worker image ships).

ShadowFormer (AAAI 2023) needs:
  - CUDA + PyTorch
  - Pretrained weights (ISTD+ recommended)
  - Shadow mask per frame (auto-estimated from luminance if not provided)

Set REFINE_ENHANCEMENT=shadowformer only after GPU Dockerfile + weights are deployed.
"""

from __future__ import annotations

import os

from PIL import Image

_WEIGHTS = os.environ.get("SHADOWFORMER_WEIGHTS", "/models/shadowformer_istd_plus.pth")


def remove_shadows_shadowformer(img_rgb: Image.Image) -> Image.Image:
    if not os.path.isfile(_WEIGHTS):
        raise RuntimeError(
            f"ShadowFormer weights not found at {_WEIGHTS}. "
            "Deploy GPU worker image with weights before enabling REFINE_ENHANCEMENT=shadowformer."
        )
    raise RuntimeError(
        "ShadowFormer inference module not yet wired in this worker build. "
        "Use REFINE_ENHANCEMENT=off until GPU pipeline is deployed."
    )
