"""Unified image enhancement for editor Refine.

Paths (env REFINE_ENHANCEMENT):
  off          — no enhancement; halo cleanup only (safe default)
  shadowformer — Path B: GPU ShadowFormer inference (when deployed)
  claid        — Path A: Claid.ai restoration API (when configured)

The previous OpenCV shadow_enhance pass is removed — it caused blown highlights,
halos, and harsh artifacts on clinical portraits.
"""

from __future__ import annotations

import os
from typing import Literal

import requests
from PIL import Image

EnhancementMode = Literal["off", "shadowformer", "claid"]

MODE: EnhancementMode = os.environ.get("REFINE_ENHANCEMENT", "off").strip().lower()  # type: ignore[assignment]
if MODE not in ("off", "shadowformer", "claid"):
    MODE = "off"


def run_enhancement_pipeline(img_rgb: Image.Image) -> tuple[Image.Image, str]:
    """Shadow normalization + optional detail restore. Returns (image, method tag)."""
    if MODE == "off":
        return img_rgb, "halo_only"

    if MODE == "shadowformer":
        from shadowformer_infer import remove_shadows_shadowformer  # noqa: PLC0415

        return remove_shadows_shadowformer(img_rgb), "shadowformer"

    if MODE == "claid":
        return _claid_restore(img_rgb), "claid_api"

    return img_rgb, "halo_only"


def _claid_restore(img_rgb: Image.Image) -> Image.Image:
    """Path A: Claid restoration (faces + polish). Requires CLAID_API_KEY."""
    api_key = os.environ.get("CLAID_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("CLAID_API_KEY is not configured")

    import io

    buf = io.BytesIO()
    img_rgb.convert("RGB").save(buf, format="JPEG", quality=95)
    buf.seek(0)

    # Claid accepts multipart or URL; we use storage upload pattern via temp public URL
    # when CLAID_INPUT_URL is set, otherwise base64 via their storage workflow.
    input_url = os.environ.get("CLAID_INPUT_URL", "").strip()
    if not input_url:
        raise RuntimeError(
            "Claid path requires CLAID_INPUT_URL (public image URL). "
            "Prefer backend proxy in refine-scan for HIPAA routing."
        )

    resp = requests.post(
        "https://api.claid.ai/v1/image/edit",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "input": input_url,
            "operations": {
                "restorations": {
                    "upscale": "faces",
                    "polish": True,
                    "decompress": "auto",
                },
            },
        },
        timeout=120,
    )
    if not resp.ok:
        raise RuntimeError(f"Claid API {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    out_url = (
        data.get("data", {}).get("output", {}).get("tmp_url")
        or data.get("output", {}).get("tmp_url")
    )
    if not out_url:
        raise RuntimeError("Claid API returned no output URL")

    out_resp = requests.get(out_url, timeout=60)
    out_resp.raise_for_status()
    return Image.open(io.BytesIO(out_resp.content)).convert("RGB")
