"""Gemini cheek fragment — used by calibrate script; production runs in Edge Function."""

from __future__ import annotations

import os
from io import BytesIO

import cv2
import numpy as np
from PIL import Image

GEMINI_FRAGMENT_PROMPT = """Transform this portrait into a single floating skin texture fragment on a pure
black background.

WHAT TO SHOW:
- One single polygon shape only — the cheek zone specifically,
  meaning the flat area between the ear and nose, below the eye socket,
  above the jawline
- Pure skin texture fill inside the shape: pores, subtle discoloration,
  fine hair, realistic skin surface
- Blue-grey cold monochrome duotone color grading matching the original photo
- Sharp straight geometric edges on the polygon (5-7 sides)
- Pure black void everywhere outside the single shape

WHAT MUST NOT APPEAR — STRICT EXCLUSIONS:
- NO eye, eyelid, eyelash, eyebrow — not even partially cropped
- NO ear, ear canal, earlobe — not even the edge of one
- NO nose, nostril, nose bridge — not even a sliver
- NO mouth, lips, chin — not even partially
- NO neck, hair, clothing
- NO multiple shapes, fragments, or secondary pieces — only ONE polygon
- NO face outline or silhouette recognizable as a face

COMPOSITION:
- The single polygon floats centered on pure black
- It should look like a skin sample or geological cross-section,
  not like a face crop
- If any facial feature appears at the edge of the polygon,
  crop it out entirely — shrink the shape inward until only
  featureless skin texture remains"""

GEMINI_REFINE_PROMPT = """Don't change the face, identity, expression, facial features, skin texture,
hair shape, or clinical appearance.

Only clean the image presentation:
- remove the background completely
- make the background pure clean black, including around hair edges
- keep the person natural and realistic
- keep the original exposure and skin brightness; do not darken the subject
- recenter the head to the center of the image
- do not beautify, retouch, smooth skin, remove wrinkles, remove redness, or alter any facial detail

Return one clean portrait image only."""

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
GEMINI_FRAGMENT_MODEL = os.environ.get(
    "GEMINI_FRAGMENT_MODEL", "gemini-2.5-flash-image"
)
GEMINI_FRAGMENT_IMAGE_SIZE = os.environ.get("GEMINI_FRAGMENT_IMAGE_SIZE", "1K")
GEMINI_FRAGMENT_TIMEOUT_S = int(os.environ.get("GEMINI_FRAGMENT_TIMEOUT_S", "150"))


def _keep_largest_fragment(rgb: Image.Image) -> Image.Image:
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


def run_gemini_fragment_from_url(visia_url: str) -> tuple[Image.Image | None, str | None]:
    """VISIA by public URL (file_uri) + 1K output cap — mirrors edge function."""
    if not GEMINI_API_KEY:
        return None, "GEMINI_API_KEY not set"

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        return None, f"google-genai not installed: {exc}"

    http_options = types.HttpOptions(timeout=GEMINI_FRAGMENT_TIMEOUT_S * 1000)
    client = genai.Client(api_key=GEMINI_API_KEY, http_options=http_options)

    config = types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        temperature=0.4,
        image_config=types.ImageConfig(image_size=GEMINI_FRAGMENT_IMAGE_SIZE),
    )

    try:
        response = client.models.generate_content(
            model=GEMINI_FRAGMENT_MODEL,
            contents=[
                types.Part.from_uri(file_uri=visia_url, mime_type="image/jpeg"),
                GEMINI_FRAGMENT_PROMPT,
            ],
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
        try:
            out = part.as_image()
            if out is not None:
                return _keep_largest_fragment(out.convert("RGB")), None
        except Exception:  # noqa: BLE001
            pass

    return None, "Gemini response had no image part"


def run_gemini_fragment(visia: Image.Image) -> tuple[Image.Image | None, str | None]:
    """Local file path: upload bytes inline (calibrate only)."""
    if not GEMINI_API_KEY:
        return None, "GEMINI_API_KEY not set"

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        return None, f"google-genai not installed: {exc}"

    buf = BytesIO()
    visia.convert("RGB").save(buf, format="JPEG", quality=92, optimize=True)
    client = genai.Client(api_key=GEMINI_API_KEY)
    config = types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        temperature=0.4,
        image_config=types.ImageConfig(image_size=GEMINI_FRAGMENT_IMAGE_SIZE),
    )
    try:
        response = client.models.generate_content(
            model=GEMINI_FRAGMENT_MODEL,
            contents=[
                types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"),
                GEMINI_FRAGMENT_PROMPT,
            ],
            config=config,
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"Gemini API error: {exc}"

    if not response.candidates:
        return None, "no candidates"
    for part in response.candidates[0].content.parts:
        inline = getattr(part, "inline_data", None)
        if inline is not None and inline.data:
            return _keep_largest_fragment(
                Image.open(BytesIO(inline.data)).convert("RGB")
            ), None
    return None, "no image part"


def run_gemini_refine(img: Image.Image) -> tuple[Image.Image | None, str | None]:
    """Gemini image edit for clean black background + centered head."""
    if not GEMINI_API_KEY:
        return None, "GEMINI_API_KEY not set"

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        return None, f"google-genai not installed: {exc}"

    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=95, optimize=True)

    http_options = types.HttpOptions(timeout=GEMINI_FRAGMENT_TIMEOUT_S * 1000)
    client = genai.Client(api_key=GEMINI_API_KEY, http_options=http_options)
    config = types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        temperature=0.15,
        image_config=types.ImageConfig(image_size=GEMINI_FRAGMENT_IMAGE_SIZE),
    )

    try:
        response = client.models.generate_content(
            model=GEMINI_FRAGMENT_MODEL,
            contents=[
                types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"),
                GEMINI_REFINE_PROMPT,
            ],
            config=config,
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"Gemini API error: {exc}"

    if not response.candidates:
        return None, "Gemini returned no candidates"

    for part in response.candidates[0].content.parts:
        inline = getattr(part, "inline_data", None)
        if inline is not None and inline.data:
            return Image.open(BytesIO(inline.data)).convert("RGB"), None
        try:
            out = part.as_image()
            if out is not None:
                return out.convert("RGB"), None
        except Exception:  # noqa: BLE001
            pass

    return None, "Gemini response had no image part"
