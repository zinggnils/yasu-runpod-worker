"""CLAHE + dark-pixel lift for capture shadow normalization (tunable via env)."""
from __future__ import annotations

import os
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ShadowNormParams:
    """Defaults tuned on ~/IMG_3897.png (bad side-shadow reference); override via SHADOW_* env."""

    clahe_clip_limit: float = 2.4
    clahe_tile_grid: int = 8
    dark_threshold: int = 102
    dark_mult: float = 1.16
    dark_lift: float = 10.0

    def label(self) -> str:
        return (
            f"clip={self.clahe_clip_limit:.2f}"
            f" thr={self.dark_threshold}"
            f" mult={self.dark_mult:.2f}"
            f" lift={self.dark_lift:.1f}"
        )


def load_shadow_params_from_env() -> ShadowNormParams:
    defaults = ShadowNormParams()
    return ShadowNormParams(
        clahe_clip_limit=float(os.environ.get("SHADOW_CLAHE_CLIP", str(defaults.clahe_clip_limit))),
        clahe_tile_grid=int(os.environ.get("SHADOW_CLAHE_TILE", str(defaults.clahe_tile_grid))),
        dark_threshold=int(os.environ.get("SHADOW_DARK_THRESHOLD", str(defaults.dark_threshold))),
        dark_mult=float(os.environ.get("SHADOW_DARK_MULT", str(defaults.dark_mult))),
        dark_lift=float(os.environ.get("SHADOW_DARK_LIFT", str(defaults.dark_lift))),
    )


# Small matrix for local A/B on real bad-shadow captures (see scripts/shadow_ab_matrix.py).
SHADOW_AB_PRESETS: dict[str, ShadowNormParams] = {
    "A_baseline": ShadowNormParams(1.9, 8, 95, 1.12, 7.0),
    "B_mild": ShadowNormParams(1.5, 8, 90, 1.08, 5.0),
    "C_balanced": ShadowNormParams(2.1, 8, 98, 1.13, 8.0),
    "D_strong": ShadowNormParams(2.4, 8, 102, 1.16, 10.0),
    "E_aggressive": ShadowNormParams(2.8, 8, 108, 1.20, 12.0),
}


def normalize_shadows(img: Image.Image, params: ShadowNormParams | None = None) -> Image.Image:
    """Lightweight shadow-lift while preserving overall contrast."""
    p = params or load_shadow_params_from_env()
    rgb = np.array(img.convert("RGB"))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    tile = max(2, int(p.clahe_tile_grid))
    clahe = cv2.createCLAHE(clipLimit=float(p.clahe_clip_limit), tileGridSize=(tile, tile))
    l2 = clahe.apply(l)
    dark = l2 < int(p.dark_threshold)
    l2f = l2.astype(np.float32)
    l2f[dark] = np.clip(l2f[dark] * float(p.dark_mult) + float(p.dark_lift), 0, 255)
    merged = cv2.merge([l2f.astype(np.uint8), a, b])
    out = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return Image.fromarray(out, mode="RGB")
