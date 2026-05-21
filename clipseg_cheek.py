"""CLIPSeg text-prompt cheek mask at 352px (CPU)."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

CLIPSEG_MODEL_ID = os.environ.get("CLIPSEG_MODEL_ID", "CIDAS/clipseg-rd64-refined")
CLIPSEG_SIZE = int(os.environ.get("CLIPSEG_SIZE", "352"))
CLIPSEG_THRESHOLD = float(os.environ.get("CLIPSEG_THRESHOLD", "0.42"))

_processor = None
_model = None
_ready = False


def clipseg_ready() -> bool:
    return _ready


def _load() -> bool:
    global _processor, _model, _ready
    if _ready:
        return True
    try:
        import torch
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

        torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
        _processor = CLIPSegProcessor.from_pretrained(CLIPSEG_MODEL_ID)
        _model = CLIPSegForImageSegmentation.from_pretrained(CLIPSEG_MODEL_ID)
        _model.eval()
        _ready = True
        print(f"[clipseg] loaded {CLIPSEG_MODEL_ID}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[clipseg] load failed: {exc}")
        return False


def predict_mask(rgb: np.ndarray, prompt: str, *, threshold: float | None = None) -> np.ndarray | None:
    """Boolean mask (H, W) from text prompt; inference at CLIPSEG_SIZE."""
    if not _load():
        return None
    import torch

    thr = CLIPSEG_THRESHOLD if threshold is None else threshold
    h, w = rgb.shape[:2]
    pil = Image.fromarray(rgb).resize((CLIPSEG_SIZE, CLIPSEG_SIZE), Image.BILINEAR)
    inputs = _processor(text=[prompt], images=[pil], return_tensors="pt", padding=True)
    with torch.inference_mode():
        logits = _model(**inputs).logits
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    small = prob > thr
    return (
        cv2.resize(small.astype(np.uint8), (w, h), interpolation=cv2.INTER_LINEAR) > 0
    )
