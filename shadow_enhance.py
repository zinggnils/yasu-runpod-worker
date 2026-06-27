"""Local shadow / illumination enhancement for editor refine (no generative AI).

Inspired by structure-guided appearance enhancement: estimate illumination on the
foreground, flatten harsh facial shadows, then re-inject edge detail so pores and
hair stay crisp. Pure OpenCV/numpy — runs on CPU in ~100–300 ms per portrait.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image


def _foreground_mask(rgb: np.ndarray) -> np.ndarray:
    """Soft mask from black-background clinical portraits."""
    alpha = rgb.max(axis=2).astype(np.float32) / 255.0
    mask = np.clip((alpha - 0.04) / 0.96, 0.0, 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=2.5)
    return mask


def _structure_map(l_channel: np.ndarray) -> np.ndarray:
    """High-frequency detail map (structure branch)."""
    blur = cv2.GaussianBlur(l_channel, (0, 0), sigmaX=3.0)
    detail = l_channel - blur
    return detail


def remove_face_shadows(img_rgb: Image.Image, strength: float = 0.72) -> Image.Image:
    """Reduce harsh facial shadows while preserving identity and black background.

    Appearance branch: divide luminance by a large-scale illumination estimate.
    Structure branch: blend back fine detail so the result does not look plastic.
    """
    strength = float(np.clip(strength, 0.0, 1.0))
    rgb = np.array(img_rgb.convert("RGB")).astype(np.float32)
    mask = _foreground_mask(rgb)
    if float(mask.max()) < 0.05:
        return img_rgb

    lab = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[:, :, 0]

    # Multi-scale illumination (appearance enhancement).
    illum = cv2.GaussianBlur(L, (0, 0), sigmaX=55)
    illum = np.maximum(illum, 8.0)
    target = float(np.percentile(L[mask > 0.15], 62)) if np.any(mask > 0.15) else float(np.median(L))
    L_flat = L * (target / illum)
    L_flat = np.clip(L_flat, 0, 255)

    # Structure-guided synthesis: keep local detail from original L.
    detail = _structure_map(L)
    L_out = L_flat + detail * 0.85
    L_mixed = L * (1.0 - strength * mask) + L_out * (strength * mask)
    L_mixed = np.clip(L_mixed, 0, 255)

    lab[:, :, 0] = L_mixed
    out_rgb = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)

    # Lock pure background to black.
    bg = mask < 0.06
    out_rgb[bg] = 0.0

    return Image.fromarray(np.clip(out_rgb, 0, 255).astype(np.uint8), mode="RGB")
