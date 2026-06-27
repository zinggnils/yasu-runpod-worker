"""Face shadow removal for editor refine (no generative AI).

Task definition: identify dark obscured facial areas, level illumination in those
regions while preserving plausible skin texture (inpainting-style correction, not
global recolor). Uses mask-guided Retinex + self-exemplar luminance from lit skin.
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

# Mild global evening — half of the earlier 0.72 default; applied only near shadows.
DEFAULT_STRENGTH = 0.36
# How strongly to lift pixels inside the detected shadow mask.
DEFAULT_SHADOW_STRENGTH = 0.85


def _foreground_mask(rgb: np.ndarray) -> np.ndarray:
    """Soft mask from black-background clinical portraits."""
    alpha = rgb.max(axis=2).astype(np.float32) / 255.0
    mask = np.clip((alpha - 0.04) / 0.96, 0.0, 1.0)
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=2.5)
    return mask


def _structure_map(l_channel: np.ndarray) -> np.ndarray:
    """High-frequency detail map (structure branch)."""
    blur = cv2.GaussianBlur(l_channel, (0, 0), sigmaX=3.0)
    return l_channel - blur


def _multi_scale_retinex_l(l_channel: np.ndarray, sigmas: tuple[float, ...] = (20.0, 80.0, 200.0)) -> np.ndarray:
    """Illumination-normalized luminance for deep shadow cores."""
    l = np.maximum(l_channel.astype(np.float32), 1.0)
    acc = np.zeros_like(l)
    for sigma in sigmas:
        blur = cv2.GaussianBlur(l, (0, 0), sigmaX=sigma)
        acc += np.log(l) - np.log(np.maximum(blur, 1.0))
    retinex = acc / len(sigmas)
    out = l * np.exp(np.clip(retinex * 0.35, -1.2, 1.2))
    return np.clip(out, 0, 255)


def _face_shadow_mask(
    l_channel: np.ndarray,
    s_channel: np.ndarray,
    fg_mask: np.ndarray,
    local_illum: np.ndarray,
) -> np.ndarray:
    """Detect facial shadow regions: locally dark + desaturated vs lit skin."""
    fg = fg_mask > 0.18
    if not np.any(fg):
        return np.zeros_like(l_channel, dtype=np.float32)

    lit_vals = l_channel[fg]
    p55 = float(np.percentile(lit_vals, 55))
    s_med = float(np.median(s_channel[fg]))

    rel_dark = l_channel < np.maximum(local_illum * 0.90, p55 * 0.86)
    absolute_dark = l_channel < p55
    desaturated = s_channel < s_med * 0.90

    raw = (fg & rel_dark & (absolute_dark | desaturated)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel)
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, kernel)
    raw = cv2.dilate(raw, kernel, iterations=1)

    smooth = cv2.GaussianBlur(raw.astype(np.float32) / 255.0, (0, 0), sigmaX=7.0)
    return np.clip(np.power(smooth, 1.35), 0.0, 1.0)


def _self_exemplar_l_target(l_channel: np.ndarray, fg_mask: np.ndarray, shadow_mask: np.ndarray) -> float:
    """Well-lit facial skin as illumination reference (non-shadow regions)."""
    lit = (fg_mask > 0.25) & (shadow_mask < 0.12)
    if np.sum(lit) < 64:
        lit = fg_mask > 0.25
    if not np.any(lit):
        return float(np.median(l_channel))
    return float(np.percentile(l_channel[lit], 58))


def _lift_shadows_in_mask(
    l_channel: np.ndarray,
    shadow_mask: np.ndarray,
    l_retinex: np.ndarray,
    target_l: float,
    local_illum: np.ndarray,
) -> np.ndarray:
    """Level light inside shadow mask toward self-exemplar lit skin."""
    illum = np.maximum(local_illum, 8.0)
    l_even = np.clip(l_channel * (target_l / illum), 0, 255)
    core = np.clip(1.0 - l_channel / np.maximum(target_l, 1.0), 0.0, 1.0)
    return np.clip(l_even * (1.0 - core * 0.4) + l_retinex * (core * 0.4), 0, 255)


def remove_face_shadows(
    img_rgb: Image.Image,
    strength: float = DEFAULT_STRENGTH,
    shadow_strength: float = DEFAULT_SHADOW_STRENGTH,
) -> Image.Image:
    """Remove facial shadows: mask-guided light leveling + texture preservation."""
    strength = float(np.clip(strength, 0.0, 1.0))
    shadow_strength = float(np.clip(shadow_strength, 0.0, 1.0))
    rgb = np.array(img_rgb.convert("RGB")).astype(np.float32)
    fg_mask = _foreground_mask(rgb)
    if float(fg_mask.max()) < 0.05:
        return img_rgb

    lab = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    hsv = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    l_ch = lab[:, :, 0]
    s_ch = hsv[:, :, 1]

    local_illum = cv2.GaussianBlur(l_ch, (0, 0), sigmaX=55)
    shadow_mask = _face_shadow_mask(l_ch, s_ch, fg_mask, local_illum)
    l_retinex = _multi_scale_retinex_l(l_ch)
    target_l = _self_exemplar_l_target(l_ch, fg_mask, shadow_mask)
    l_corrected = _lift_shadows_in_mask(l_ch, shadow_mask, l_retinex, target_l, local_illum)

    # Texture synthesis inside shadow regions (inpaint plausible skin detail).
    detail = _structure_map(l_ch)
    l_lifted = l_corrected + detail * 0.88

    w = shadow_mask * shadow_strength
    l_out = l_ch * (1.0 - w) + l_lifted * w

    # Mild residual evening at half prior global strength — shadow-adjacent only.
    feather = cv2.GaussianBlur(shadow_mask, (0, 0), sigmaX=18)
    g_w = np.clip(feather * strength * 0.5, 0.0, strength * 0.5)
    l_flat = np.clip(l_ch * (target_l / np.maximum(local_illum, 8.0)) + detail * 0.75, 0, 255)
    l_mixed = l_out * (1.0 - g_w) + l_flat * g_w
    l_mixed = np.clip(l_mixed, 0, 255)

    lab[:, :, 0] = l_mixed
    out_rgb = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)

    bg = fg_mask < 0.06
    out_rgb[bg] = 0.0

    return Image.fromarray(np.clip(out_rgb, 0, 255).astype(np.uint8), mode="RGB")
