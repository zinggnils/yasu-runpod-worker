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
    """Grid-rectangle redness overlay on skin pixels only (hair excluded via brightness filter).
    Returns (overlay RGBA image, score 0-100). Strictly blue/cyan palette."""
    clean_rgba = clean_rgba.convert("RGBA")
    rgb_arr = np.array(clean_rgba)[..., :3].astype(np.float32)
    alpha_arr = np.array(clean_rgba)[..., 3].astype(np.float32) / 255.0
    h, w = alpha_arr.shape

    r, g, b = rgb_arr[..., 0], rgb_arr[..., 1], rgb_arr[..., 2]
    redness = r - (g + b) / 2.0
    brightness = (r + g + b) / 3.0

    face_mask = alpha_arr > 0.3
    # Exclude dark hair — hair pixels are typically brightness < 60
    skin_mask = face_mask & (brightness > 60)

    if not np.any(skin_mask):
        return Image.new("RGBA", (w, h), (0, 0, 0, 0)), 0

    lo, hi = np.percentile(redness[skin_mask], [15, 95])
    redness_n = np.clip((redness - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    redness_n[~skin_mask] = 0.0

    grid_cols, grid_rows = 32, 32
    cell_w = w / grid_cols
    cell_h = h / grid_rows

    overlay = np.zeros((h, w, 4), dtype=np.uint8)
    face_cells = 0
    affected_cells = 0
    threshold = 0.35

    for row in range(grid_rows):
        for col in range(grid_cols):
            y0 = int(row * cell_h); y1 = int((row + 1) * cell_h)
            x0 = int(col * cell_w); x1 = int((col + 1) * cell_w)
            if skin_mask[y0:y1, x0:x1].mean() < 0.15:
                continue
            face_cells += 1
            cell_redness = redness_n[y0:y1, x0:x1].mean()
            if cell_redness <= threshold:
                continue
            affected_cells += 1
            t = min(1.0, (cell_redness - threshold) / (1.0 - threshold))
            # Strictly blue/cyan: (0,180,255) at low t → (30,60,255) at high t
            cr = int(0 + 30 * t)
            cg = int(180 - 120 * t)
            cb = 255
            ca = int(100 + 130 * t)
            pad = max(1, int(min(cell_w, cell_h) * 0.06))
            overlay[y0+pad:y1-pad, x0+pad:x1-pad] = [cr, cg, cb, ca]

    score = int((affected_cells / max(face_cells, 1)) * 100)
    return Image.fromarray(overlay, mode="RGBA"), score


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
    job_input = job.get("input", {})
    scan_id = job_input.get("scan_id")
    images = job_input.get("images")

    if images and isinstance(images, dict):
        processed_angles = {}
        for key in ANGLE_KEYS:
            b64 = images.get(key)
            if b64:
                try:
                    processed_angles[key] = process_single(b64, key)
                except Exception as e:
                    processed_angles[key] = {"error": str(e)}

        if scan_id and SUPABASE_URL and SUPABASE_SERVICE_KEY:
            frontal = processed_angles.get("frontal", {})
            update_supabase_scan(scan_id, processed_angles, frontal)
            return {"status": "done", "scan_id": scan_id}

        return {"angles": processed_angles}

    return {"error": "No images provided."}


runpod.serverless.start({"handler": handler})
