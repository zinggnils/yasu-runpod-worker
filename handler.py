import base64
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import cv2
import numpy as np
import requests
from PIL import Image, ImageOps

# Register the HEIF/HEIC opener with Pillow so Image.open() auto-detects HEIC
# bytes (the format the iOS app uploads). Pillow's auto-detect reads the file
# header, so no other code in this module needs to know the input format.
from pillow_heif import register_heif_opener

register_heif_opener()

try:
    import runpod
except ImportError:  # Allows local helper tests without RunPod installed.
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
    if "," in image_b64 and image_b64.lstrip().startswith("data:"):
        image_b64 = image_b64.split(",", 1)[1]
    data = base64.b64decode(image_b64)
    return ImageOps.exif_transpose(Image.open(BytesIO(data))).convert("RGB")


def download_storage_image(path: str) -> Image.Image:
    url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{path}"
    resp = requests.get(url, headers={"Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}, timeout=90)
    if resp.status_code != 200:
        raise RuntimeError(f"Storage download failed: {resp.status_code} {resp.text[:200]}")
    return ImageOps.exif_transpose(Image.open(BytesIO(resp.content))).convert("RGB")


def public_storage_url(path: str) -> str:
    return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{path}"


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
    portrait = crop_to_aspect(img, PORTRAIT_WIDTH, PORTRAIT_HEIGHT)
    if portrait.size != (PORTRAIT_WIDTH, PORTRAIT_HEIGHT):
        portrait = portrait.resize((PORTRAIT_WIDTH, PORTRAIT_HEIGHT), Image.Resampling.LANCZOS)
    return portrait


def fixed_analysis_crop(img: Image.Image) -> Image.Image:
    left = (img.width - ANALYSIS_CROP_SIZE) // 2
    top = (img.height - ANALYSIS_CROP_SIZE) // 2
    return img.crop((left, top, left + ANALYSIS_CROP_SIZE, top + ANALYSIS_CROP_SIZE))


def jpeg_bytes(img: Image.Image, quality: int = 95) -> bytes:
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def webp_lossless_bytes(img: Image.Image) -> bytes:
    """Encode image as bit-exact lossless WebP (analysis-grade).

    Reserved for the right_90 ROI artifacts where pixel-exact fidelity matters
    for re-analysis. method=6 is the strongest (slowest) entropy coder.
    """
    buf = BytesIO()
    img.convert("RGB").save(
        buf,
        format="WEBP",
        lossless=True,
        quality=100,
        method=6,
    )
    return buf.getvalue()


def webp_visual_bytes(img: Image.Image, quality: int = 95) -> bytes:
    """Encode image as high-quality lossy WebP for display use.

    For the four non-analysis angles we only render them; we never re-run
    skin math on them. q=95 method=4 is visually indistinguishable from
    lossless on skin imagery and encodes ~5-10x faster than lossless.
    """
    buf = BytesIO()
    img.convert("RGB").save(
        buf,
        format="WEBP",
        quality=quality,
        method=4,
    )
    return buf.getvalue()


def upload_bytes(data: bytes, filename: str, content_type: str) -> str:
    path = f"processed/{filename}"
    url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{path}"
    resp = requests.put(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "Content-Type": content_type,
            "x-upsert": "true",
        },
        timeout=120,
    )
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Storage upload failed: {resp.status_code} {resp.text[:200]}")
    return public_storage_url(path)


def upload_jpeg(img: Image.Image, filename: str, quality: int = 95) -> str:
    return upload_bytes(jpeg_bytes(img, quality=quality), filename, "image/jpeg")


def upload_webp_lossless(img: Image.Image, filename: str) -> str:
    return upload_bytes(webp_lossless_bytes(img), filename, "image/webp")


def upload_webp_visual(img: Image.Image, filename: str, quality: int = 95) -> str:
    return upload_bytes(webp_visual_bytes(img, quality=quality), filename, "image/webp")


def make_analysis_map(img: Image.Image, mode: str = "redness") -> Image.Image:
    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)

    if mode == "texture":
        inv = 255 - contrast
        return Image.fromarray(cv2.cvtColor(inv, cv2.COLOR_GRAY2RGB))

    bone = cv2.applyColorMap(contrast, cv2.COLORMAP_BONE)
    return Image.fromarray(cv2.cvtColor(bone, cv2.COLOR_BGR2RGB))


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


def compute_quality(crop: Image.Image) -> dict:
    gray = cv2.cvtColor(np.array(crop.convert("RGB")), cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    mean = float(gray.mean())
    exposure = max(0, 100 - abs(mean - 132.0) * 1.4)
    contrast_score = min(100, contrast * 2.5)
    quality = int(round(0.5 * exposure + 0.5 * contrast_score))
    warnings = []
    if blur < 80:
        warnings.append("Right_90 crop may be blurry")
    if mean < 80:
        warnings.append("Right_90 crop is underexposed")
    if mean > 190:
        warnings.append("Right_90 crop is overexposed")
    if contrast < 25:
        warnings.append("Right_90 crop has low contrast")
    return {"quality_score": quality, "quality_warnings": warnings}


def load_angle_image(images: dict, image_paths: dict, label: str) -> tuple[Image.Image | None, str | None]:
    if image_paths.get(label):
        path = image_paths[label]
        return download_storage_image(path), public_storage_url(path)
    if images.get(label):
        return decode_image(images[label]), None
    return None, None


def update_supabase_scan(scan_id: str, processed_angles: dict, right90: dict) -> None:
    url = f"{SUPABASE_URL}/rest/v1/scans?id=eq.{scan_id}"
    body = {
        "status": "done",
        "processed_angles": processed_angles,
        "clean_image_url": right90.get("crop_image_url") or right90.get("clean_image_url"),
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


def _process_angle(label: str, images: dict, image_paths: dict, mode: str, uid: str) -> tuple[str, dict] | None:
    """Process one angle. Designed to be safe to run in a worker thread:
    Pillow image encoding releases the GIL during native libwebp calls, so
    concurrent encodes of multiple angles overlap on multi-core CPUs.
    """
    original, original_url = load_angle_image(images, image_paths, label)
    if original is None:
        return None

    clean = normalize_portrait(original)
    visia = make_analysis_map(clean, mode)

    if label == ANALYSIS_ANGLE:
        # Analysis-grade: lossless WebP for both the full clean frame and the
        # ROI crop, so re-analysis from storage is bit-exact.
        clean_url = upload_webp_lossless(clean, f"clean_{label}_{uid}.webp")
    else:
        # Display-only angles: q=95 WebP is visually indistinguishable and
        # ~5-10x faster to encode than lossless method=6.
        clean_url = upload_webp_visual(clean, f"clean_{label}_{uid}.webp", quality=95)

    visia_url = upload_jpeg(visia, f"visia_{label}_{uid}.jpg", quality=92)

    angle_data: dict = {
        "original_image_url": original_url,
        "clean_image_url": clean_url,
        "visia_image_url": visia_url,
    }

    if label == ANALYSIS_ANGLE:
        crop = fixed_analysis_crop(clean)
        crop_url = upload_webp_lossless(crop, f"right90_crop_{uid}.webp")
        angle_data.update(
            {
                "crop_image_url": crop_url,
                "crop_box": {"x": 580, "y": 850, "width": ANALYSIS_CROP_SIZE, "height": ANALYSIS_CROP_SIZE},
                "redness_score": compute_redness_score(crop),
                "white_score": compute_white_score(crop),
                **compute_quality(crop),
            }
        )

    return label, angle_data


def process_images(images: dict, image_paths: dict, mode: str = "redness") -> dict:
    uid = uuid.uuid4().hex[:10]
    processed: dict = {}

    # Process all 5 angles in parallel. Each angle does: download -> normalize
    # -> encode WebP/JPEG -> upload. Pillow + requests release the GIL during
    # native calls, so we get real parallelism on storage I/O and libwebp.
    with ThreadPoolExecutor(max_workers=len(ANGLE_KEYS)) as pool:
        futures = [pool.submit(_process_angle, label, images, image_paths, mode, uid) for label in ANGLE_KEYS]
        for fut in futures:
            result = fut.result()
            if result is not None:
                label, data = result
                processed[label] = data

    return processed


def handler(job):
    job_input = job.get("input", {})
    scan_id = job_input.get("scan_id")
    images = job_input.get("images") or {}
    image_paths = job_input.get("image_paths") or {}
    mode = job_input.get("mode", "redness")

    print(f"[handler] scan_id={scan_id} image_paths={list(image_paths.keys())} images={list(images.keys())}")

    if not isinstance(images, dict) or not isinstance(image_paths, dict):
        return {"error": "input.images and input.image_paths must be objects keyed by angle"}
    if ANALYSIS_ANGLE not in images and ANALYSIS_ANGLE not in image_paths:
        return {"error": f"Missing required {ANALYSIS_ANGLE} image"}
    if (scan_id or image_paths) and (not SUPABASE_URL or not SUPABASE_SERVICE_KEY):
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY are required")

    processed_angles = process_images(images, image_paths, mode)
    right90 = processed_angles.get(ANALYSIS_ANGLE, {})

    if scan_id:
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
