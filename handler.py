import base64
import os
import uuid
from io import BytesIO

import cv2
import numpy as np
import requests
from PIL import Image, ImageOps

try:
    import runpod
except ImportError:  # Allows local helper tests without the RunPod package installed.
    runpod = None


SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "scans")

ANGLE_KEYS = ["frontal", "left_45", "left_90", "right_45", "right_90"]
ANALYSIS_ANGLE = "right_90"
PORTRAIT_WIDTH = 2160
PORTRAIT_HEIGHT = 2700
ANALYSIS_CROP_SIZE = 1000


def decode_image(image_b64: str) -> Image.Image:
    """Decode a base64 image or data URI and apply EXIF orientation."""
    if "," in image_b64 and image_b64.lstrip().startswith("data:"):
        image_b64 = image_b64.split(",", 1)[1]
    data = base64.b64decode(image_b64)
    return ImageOps.exif_transpose(Image.open(BytesIO(data))).convert("RGB")


def crop_to_aspect(img: Image.Image, target_width: int, target_height: int) -> Image.Image:
    target_aspect = target_width / target_height
    width, height = img.size
    source_aspect = width / height

    if source_aspect > target_aspect:
        crop_width = round(height * target_aspect)
        left = (width - crop_width) // 2
        box = (left, 0, left + crop_width, height)
    else:
        crop_height = round(width / target_aspect)
        top = (height - crop_height) // 2
        box = (0, top, width, top + crop_height)

    return img.crop(box)


def normalize_portrait(img: Image.Image) -> Image.Image:
    """Normalize client capture to the app contract: 2160 x 2700 portrait."""
    portrait = crop_to_aspect(img, PORTRAIT_WIDTH, PORTRAIT_HEIGHT)
    if portrait.size != (PORTRAIT_WIDTH, PORTRAIT_HEIGHT):
        portrait = portrait.resize((PORTRAIT_WIDTH, PORTRAIT_HEIGHT), Image.Resampling.LANCZOS)
    return portrait


def fixed_analysis_crop(img: Image.Image) -> Image.Image:
    """Deterministic center crop matching the scanner's right-profile target."""
    width, height = img.size
    side = min(ANALYSIS_CROP_SIZE, width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    crop = img.crop((left, top, left + side, top + side))
    if crop.size != (ANALYSIS_CROP_SIZE, ANALYSIS_CROP_SIZE):
        crop = crop.resize((ANALYSIS_CROP_SIZE, ANALYSIS_CROP_SIZE), Image.Resampling.LANCZOS)
    return crop


def save_webp_bytes(img: Image.Image, quality: int = 94) -> bytes:
    buf = BytesIO()
    img.save(buf, format="WEBP", quality=quality, method=4)
    return buf.getvalue()


def upload_to_supabase(img: Image.Image, filename: str) -> str:
    path = f"processed/{filename}"
    url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{path}"
    resp = requests.put(
        url,
        data=save_webp_bytes(img),
        headers={
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "Content-Type": "image/webp",
            "x-upsert": "true",
        },
        timeout=60,
    )
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Storage upload failed: {resp.status_code} {resp.text[:200]}")
    return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{path}"


def skin_mask(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lightness = lab[..., 0] * (100.0 / 255.0)
    return (lightness > 18.0) & (lightness < 98.0)


def compute_redness_score(crop: Image.Image) -> int:
    rgb = np.array(crop.convert("RGB"))
    mask = skin_mask(rgb)
    if not np.any(mask):
        return 0

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    a_star = lab[..., 1] - 128.0
    redness = np.clip((a_star - 8.0) / 26.0, 0.0, 1.0)
    return int(round(float(redness[mask].mean()) * 100))


def compute_white_score(crop: Image.Image) -> int:
    rgb = np.array(crop.convert("RGB"))
    mask = skin_mask(rgb)
    if not np.any(mask):
        return 0

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lightness = lab[..., 0] * (100.0 / 255.0)
    a_star = lab[..., 1] - 128.0
    b_star = lab[..., 2] - 128.0
    chroma = np.sqrt(a_star * a_star + b_star * b_star)
    white = (lightness > 72.0) & (chroma < 16.0) & mask
    return int(round(float(white.sum()) / float(mask.sum()) * 100))


def compute_quality_score(portrait: Image.Image, crop: Image.Image) -> int:
    gray = cv2.cvtColor(np.array(crop.convert("RGB")), cv2.COLOR_RGB2GRAY)
    blur = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast = float(gray.std())
    mean = float(gray.mean())

    resolution_score = 100 if portrait.size == (PORTRAIT_WIDTH, PORTRAIT_HEIGHT) else 70
    blur_score = min(100, int(blur / 3.0))
    contrast_score = min(100, int(contrast * 2.5))
    exposure_score = max(0, 100 - int(abs(mean - 132.0) * 1.4))
    return int(round(0.35 * resolution_score + 0.35 * blur_score + 0.15 * contrast_score + 0.15 * exposure_score))


def make_analysis_preview(crop: Image.Image) -> Image.Image:
    gray = cv2.cvtColor(np.array(crop.convert("RGB")), cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    bone = cv2.applyColorMap(contrast, cv2.COLORMAP_BONE)
    return Image.fromarray(cv2.cvtColor(bone, cv2.COLOR_BGR2RGB))


def process_right90(image_b64: str) -> dict:
    original = decode_image(image_b64)
    portrait = normalize_portrait(original)
    crop = fixed_analysis_crop(portrait)
    preview = make_analysis_preview(crop)

    return {
        "portrait": portrait,
        "crop": crop,
        "preview": preview,
        "redness_score": compute_redness_score(crop),
        "white_score": compute_white_score(crop),
        "quality_score": compute_quality_score(portrait, crop),
        "crop_box": {
            "x": (PORTRAIT_WIDTH - ANALYSIS_CROP_SIZE) // 2,
            "y": (PORTRAIT_HEIGHT - ANALYSIS_CROP_SIZE) // 2,
            "width": ANALYSIS_CROP_SIZE,
            "height": ANALYSIS_CROP_SIZE,
        },
    }


def update_supabase_scan(scan_id: str, processed_angles: dict, right90: dict) -> None:
    url = f"{SUPABASE_URL}/rest/v1/scans?id=eq.{scan_id}"
    body = {
        "status": "done",
        "processed_angles": processed_angles,
        "clean_image_url": right90.get("crop_image_url"),
        "redness_image_url": right90.get("visia_image_url"),
        "image_url": right90.get("visia_image_url"),
        "redness_severity": right90.get("redness_score", 0),
    }
    resp = requests.patch(
        url,
        json=body,
        headers={
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "apikey": SUPABASE_SERVICE_KEY,
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        },
        timeout=60,
    )
    if resp.status_code not in (200, 201, 204):
        raise RuntimeError(f"DB update failed: {resp.status_code} {resp.text[:200]}")


def handler(job):
    job_input = job.get("input", {})
    scan_id = job_input.get("scan_id")
    images = job_input.get("images") or {}

    if not isinstance(images, dict):
        return {"error": "input.images must be an object keyed by angle"}
    if ANALYSIS_ANGLE not in images:
        return {"error": f"Missing required {ANALYSIS_ANGLE} image"}

    print(f"[handler] scan_id={scan_id} analyzing={ANALYSIS_ANGLE} available={list(images.keys())}")

    result = process_right90(images[ANALYSIS_ANGLE])
    uid = uuid.uuid4().hex[:10]

    right90 = {
        "crop_box": result["crop_box"],
        "redness_score": result["redness_score"],
        "white_score": result["white_score"],
        "quality_score": result["quality_score"],
    }

    if SUPABASE_URL and SUPABASE_SERVICE_KEY:
        right90["crop_image_url"] = upload_to_supabase(result["crop"], f"right90_crop_{uid}.webp")
        right90["visia_image_url"] = upload_to_supabase(result["preview"], f"right90_visia_{uid}.webp")
    else:
        print("[handler] Supabase env missing; returning metrics without uploads")

    processed_angles = {ANALYSIS_ANGLE: right90}

    if scan_id and SUPABASE_URL and SUPABASE_SERVICE_KEY:
        update_supabase_scan(scan_id, processed_angles, right90)
        print(f"[handler] DB updated OK for scan_id={scan_id}")

    return {
        "status": "done",
        "scan_id": scan_id,
        "analysis_angle": ANALYSIS_ANGLE,
        "processed_angles": processed_angles,
    }


if __name__ == "__main__":
    if runpod is None:
        raise RuntimeError("runpod package is required to start the worker")
    runpod.serverless.start({"handler": handler})
