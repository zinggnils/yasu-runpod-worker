import runpod
import base64
import os
import uuid
import requests
from io import BytesIO
from PIL import Image, ImageFilter
import numpy as np
import cv2
import onnxruntime as ort
from rembg import new_session, remove

print(f"ORT version: {ort.__version__}")
print(f"Available providers: {ort.get_available_providers()}")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
SUPABASE_BUCKET = "scans"

try:
    session = new_session("u2net", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print("Rembg session initialized with CUDA/CPU.")
except Exception as e:
    print(f"Failed to initialize GPU session, falling back: {e}")
    session = new_session("u2net")


def upload_to_supabase(img: Image.Image, filename: str) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    path = f"processed/{filename}"
    url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{path}"
    resp = requests.put(url, data=buf.read(), headers={
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "image/png",
        "x-upsert": "true",
    })
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Storage upload failed: {resp.status_code} {resp.text[:200]}")
    return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{path}"


def update_supabase_scan(scan_id: str, processed_angles: dict, frontal: dict):
    url = f"{SUPABASE_URL}/rest/v1/scans?id=eq.{scan_id}"
    body = {
        "status": "done",
        "processed_angles": processed_angles,
        "clean_image_url": frontal.get("clean_image_url"),
        "redness_image_url": frontal.get("redness_image_url"),
        "image_url": frontal.get("visia_image_url"),
    }
    resp = requests.patch(url, json=body, headers={
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "apikey": SUPABASE_SERVICE_KEY,
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    })
    if resp.status_code not in (200, 201, 204):
        raise RuntimeError(f"DB update failed: {resp.status_code} {resp.text[:200]}")


def on_black(clean_rgba: Image.Image) -> Image.Image:
    clean_rgba = clean_rgba.convert("RGBA")
    alpha = clean_rgba.split()[3]
    rgb = clean_rgba.convert("RGB")
    # Strong sharpening for crisp clinical output
    rgb = rgb.filter(ImageFilter.UnsharpMask(radius=2.0, percent=160, threshold=2))
    result = Image.new("RGB", clean_rgba.size, (0, 0, 0))
    result.paste(rgb, mask=alpha)
    return result


def make_visia_duotone(clean_rgba: Image.Image) -> Image.Image:
    clean_rgba = clean_rgba.convert("RGBA")
    alpha_arr = np.array(clean_rgba.split()[3])
    rgb_arr = np.array(clean_rgba.convert("RGB"))
    bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    bone = cv2.applyColorMap(contrast, cv2.COLORMAP_BONE)
    bone_rgb = cv2.cvtColor(bone, cv2.COLOR_BGR2RGB)
    duotone = Image.fromarray(bone_rgb, mode="RGB")
    result = Image.new("RGB", clean_rgba.size, (0, 0, 0))
    result.paste(duotone, mask=Image.fromarray(alpha_arr, mode="L"))
    return result


def compute_redness_overlay(clean_rgba: Image.Image) -> tuple:
    """Smooth Gaussian-blob redness overlay. Excludes hair (brightness) and ears (mask erosion).
    Returns (overlay RGBA image, score 0-100). Neon cyan (40, 220, 255)."""
    clean_rgba = clean_rgba.convert("RGBA")
    rgb_arr = np.array(clean_rgba)[..., :3].astype(np.float32)
    alpha_arr = np.array(clean_rgba)[..., 3].astype(np.float32) / 255.0
    h, w = alpha_arr.shape

    r, g, b = rgb_arr[..., 0], rgb_arr[..., 1], rgb_arr[..., 2]
    redness = r - (g + b) / 2.0
    # Perceptual brightness in [0, 1]
    brightness = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0

    # Erode alpha mask to exclude ears and peripheral skin protrusions
    alpha_uint8 = (alpha_arr * 255).astype(np.uint8)
    erode_px = max(15, int(min(h, w) * 0.04))
    kernel = np.ones((erode_px, erode_px), np.uint8)
    eroded = cv2.erode(alpha_uint8, kernel, iterations=1)
    inner_face = eroded > 100

    # Combine: must be inner face AND not dark hair
    skin_mask = inner_face & (brightness > 0.24)

    if not np.any(skin_mask):
        return Image.new("RGBA", (w, h), (0, 0, 0, 0)), 0

    # Normalize redness on inner skin only — tight range highlights actual redness
    lo, hi = np.percentile(redness[skin_mask], [60, 98])
    redness_n = np.clip((redness - lo) / (hi - lo + 1e-6), 0.0, 1.0)

    # Build smooth mask: redness × face alpha × brightness weight
    mask = redness_n * alpha_arr
    mask *= np.where(skin_mask, 1.0, 0.0)
    mask *= np.clip((brightness - 0.15) / 0.85, 0.0, 1.0)

    # Threshold: only genuinely elevated redness
    mask = np.clip((mask - 0.25) / 0.75, 0.0, 1.0)

    # Gaussian blur → smooth neon blobs (no hard edges)
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=6))

    # Score: fraction of face pixels with visible overlay
    mask_arr = np.array(mask_img) / 255.0
    face_pixels = alpha_arr > 0.2
    score = int((mask_arr[face_pixels] > 0.1).mean() * 100)

    # Neon cyan — exact same colour as the reference best result
    neon = np.zeros((h, w, 4), dtype=np.uint8)
    neon[..., 0] = 40
    neon[..., 1] = 220
    neon[..., 2] = 255
    neon[..., 3] = np.array(mask_img)

    return Image.fromarray(neon, mode="RGBA"), score


def apply_overlay(base_rgb: Image.Image, overlay_rgba: Image.Image) -> Image.Image:
    base = base_rgb.convert("RGBA")
    return Image.alpha_composite(base, overlay_rgba).convert("RGB")


ANGLE_KEYS = ["frontal", "left_45", "left_90", "right_45", "right_90"]


def process_single(image_b64: str, label: str) -> dict:
    img_data = base64.b64decode(image_b64)
    original = Image.open(BytesIO(img_data)).convert("RGB")
    clean_rgba = remove(original, session=session).convert("RGBA")
    uid = uuid.uuid4().hex[:8]

    clean_img = on_black(clean_rgba)
    visia_img = make_visia_duotone(clean_rgba)

    overlay, redness_score = compute_redness_overlay(clean_rgba)
    redness_on_clean = apply_overlay(clean_img, overlay)
    redness_on_visia = apply_overlay(visia_img, overlay)

    return {
        "clean_image_url":        upload_to_supabase(clean_img,        f"clean_{label}_{uid}.png"),
        "visia_image_url":        upload_to_supabase(visia_img,        f"visia_{label}_{uid}.png"),
        "redness_image_url":      upload_to_supabase(redness_on_clean, f"redness_{label}_{uid}.png"),
        "redness_visia_image_url":upload_to_supabase(redness_on_visia, f"redness_visia_{label}_{uid}.png"),
        "redness_score": redness_score,
    }


def handler(job):
    import traceback
    job_input = job.get("input", {})
    scan_id = job_input.get("scan_id")
    images = job_input.get("images")

    print(f"[handler] scan_id={scan_id}")
    print(f"[handler] images type={type(images)}, keys={list(images.keys()) if isinstance(images, dict) else 'N/A'}")
    print(f"[handler] SUPABASE_URL set={bool(SUPABASE_URL)}, SERVICE_KEY set={bool(SUPABASE_SERVICE_KEY)}")

    if images and isinstance(images, dict):
        processed_angles = {}
        for key in ANGLE_KEYS:
            b64 = images.get(key)
            if b64:
                print(f"[handler] Processing {key}, b64 length={len(b64)}")
                try:
                    result = process_single(b64, key)
                    processed_angles[key] = result
                    print(f"[handler] {key} OK — score={result.get('redness_score')}")
                except Exception as e:
                    print(f"[handler] {key} FAILED: {e}")
                    traceback.print_exc()
                    processed_angles[key] = {"error": str(e)}
            else:
                print(f"[handler] {key} — no b64 data, skipping")

        print(f"[handler] Processed angles: {list(processed_angles.keys())}")

        if scan_id and SUPABASE_URL and SUPABASE_SERVICE_KEY:
            try:
                frontal = processed_angles.get("frontal", {})
                update_supabase_scan(scan_id, processed_angles, frontal)
                print(f"[handler] DB updated OK for scan_id={scan_id}")
            except Exception as e:
                print(f"[handler] DB UPDATE FAILED: {e}")
                traceback.print_exc()
                raise  # Re-raise so RunPod marks the job FAILED (not COMPLETED)
            return {"status": "done", "scan_id": scan_id}
        else:
            print(f"[handler] Skipping DB update — scan_id={bool(scan_id)}, url={bool(SUPABASE_URL)}, key={bool(SUPABASE_SERVICE_KEY)}")

        return {"angles": processed_angles}

    print(f"[handler] No images provided — images={images}")
    return {"error": "No images provided."}


runpod.serverless.start({"handler": handler})
