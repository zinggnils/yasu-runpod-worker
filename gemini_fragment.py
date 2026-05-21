"""Gemini image model — cheek/jaw fragment from VISIA BONE map (Nano Banana family)."""

from __future__ import annotations

import os
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

# Exact prompt from product spec (Gemini 3.1 / Nano Banana workflow).
GEMINI_FRAGMENT_PROMPT = """Deconstruct the portrait into fragmented, isolated skin texture pieces.
Extract only the cheek and jaw area as a single irregular polygon shape —
no eyes, no nose, no full face. Float the fragment on pure black background.
Keep the blue-tinted monochrome duotone color grading.
The fragment should appear as a geometric cutout with sharp, angular edges,
like a torn or shattered piece of the face. No other facial features visible.
lastly remove all shapes but the biggest one completely from the picture"""

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
# Nano Banana 2 = gemini-3.1-flash-image-preview (override on RunPod if needed).
GEMINI_FRAGMENT_MODEL = os.environ.get(
    "GEMINI_FRAGMENT_MODEL", "gemini-3.1-flash-image-preview"
)
GEMINI_FRAGMENT_MAX_EDGE = int(os.environ.get("GEMINI_FRAGMENT_MAX_EDGE", "1536"))


def _keep_largest_fragment(rgb: Image.Image) -> Image.Image:
    """Drop smaller blobs so only the biggest cheek cutout remains on black."""
    arr = np.array(rgb)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    _, fg = cv2.threshold(gray, 12, 255, cv2.THRESH_BINARY)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    if n_labels <= 2:
        return rgb
    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.size == 0:
        return rgb
    best = 1 + int(np.argmax(areas))
    mask = labels == best
    out = arr.copy()
    out[~mask] = 0
    return Image.fromarray(out, mode="RGB")


def _prepare_input(img: Image.Image) -> Image.Image:
    rgb = img.convert("RGB")
    w, h = rgb.size
    edge = max(w, h)
    if edge <= GEMINI_FRAGMENT_MAX_EDGE:
        return rgb
    scale = GEMINI_FRAGMENT_MAX_EDGE / float(edge)
    return rgb.resize(
        (max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS
    )


def run_gemini_fragment(visia: Image.Image) -> tuple[Image.Image | None, str | None]:
    """
    Send VISIA (BONE duotone) + prompt to Gemini; return edited image or (None, error).
    """
    if not GEMINI_API_KEY:
        return None, "GEMINI_API_KEY not set"

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        return None, f"google-genai not installed: {exc}"

    client = genai.Client(api_key=GEMINI_API_KEY)
    prepared = _prepare_input(visia)

    config = types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        temperature=0.4,
    )

    try:
        response = client.models.generate_content(
            model=GEMINI_FRAGMENT_MODEL,
            contents=[prepared, GEMINI_FRAGMENT_PROMPT],
            config=config,
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"Gemini API error: {exc}"

    if not response.candidates:
        return None, "Gemini returned no candidates"

    for part in response.candidates[0].content.parts:
        inline = getattr(part, "inline_data", None)
        if inline is not None and inline.data:
            out = Image.open(BytesIO(inline.data)).convert("RGB")
            return _keep_largest_fragment(out), None
        # Some SDK versions expose as part.as_image()
        try:
            out = part.as_image()
            if out is not None:
                return _keep_largest_fragment(out.convert("RGB")), None
        except Exception:  # noqa: BLE001
            pass

    return None, "Gemini response had no image part"
