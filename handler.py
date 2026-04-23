import runpod
import base64
import os
import uuid
import random
import requests
from io import BytesIO
from PIL import Image, ImageFilter, ImageDraw
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
    """Upload PIL image to Supabase Storage, return public URL."""
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
    """Write processed_angles back to the scans table."""
    import json
    url = f"{SUPABASE_URL}/rest/v1/scans?id=eq.{scan_id}"
    body = {
        "status": "done",
        "processed_angles": processed_angles,
        "clean_image_url": frontal.get("clean_image_url"),
        "redness_image_url": frontal.get("redness_image_url"),
        "image_url": frontal.get("visia_image_url"),
        "full_analysis_url": frontal.get("full_analysis_url"),
    }
    resp = requests.patch(url, json=body, headers={
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "apikey": SUPABASE_SERVICE_KEY,
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    })
    if resp.status_code not in (200, 201, 204):
        raise RuntimeError(f"DB update failed: {resp.status_code} {resp.text[:200]}")


def to_b64(img_obj: Image.Image) -> str:
    buffered = BytesIO()
    img_obj.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def on_black(clean_rgba: Image.Image, sharpen: bool = True) -> Image.Image:
    clean_rgba = clean_rgba.convert("RGBA")
    alpha = clean_rgba.split()[3]
    rgb = clean_rgba.convert("RGB")
    if sharpen:
        rgb = rgb.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3))
    result = Image.new("RGB", clean_rgba.size, (0, 0, 0))
    result.paste(rgb, mask=alpha)
    return result


def compute_redness_overlay(clean_rgba: Image.Image) -> tuple:
    """Returns (overlay PIL RGBA image, score 0-100).
    Dots placed only where VISIA shows dark patches — these are the redness zones."""
    clean_rgba = clean_rgba.convert("RGBA")
    alpha_arr = np.array(clean_rgba.split()[3]).astype(np.float32) / 255.0
    h, w = alpha_arr.shape

    # VISIA reveals redness as dark spots — use its luminance to guide dot placement
    visia_img = make_visia_duotone(clean_rgba)
    visia_lum = np.array(visia_img.convert("L")).astype(np.float32) / 255.0

    face_mask = alpha_arr > 0.3
    if not np.any(face_mask):
        return Image.new("RGBA", (w, h), (0, 0, 0, 0)), 0

    face_mean = visia_lum[face_mask].mean()
    face_std  = visia_lum[face_mask].std()
    # How many std-devs below mean — capped at 2 std, normalized to [0,1]
    darkness = np.clip((face_mean - visia_lum) / (face_std * 2.0 + 1e-6), 0.0, 1.0)
    darkness *= face_mask

    grid_cols, grid_rows = 32, 32
    cell_w = w / grid_cols
    cell_h = h / grid_rows
    dot_r = max(2, int(min(cell_w, cell_h) * 0.22))

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    rng = random.Random(42)

    face_cells = 0
    affected_cells = 0
    # Only cells ≥ 0.5 std below mean get dots — keeps overlay sparse and targeted
    threshold = 0.25

    for row in range(grid_rows):
        for col in range(grid_cols):
            y0 = int(row * cell_h); y1 = int((row + 1) * cell_h)
            x0 = int(col * cell_w); x1 = int((col + 1) * cell_w)
            if alpha_arr[y0:y1, x0:x1].mean() < 0.2:
                continue
            face_cells += 1
            cell_dark = darkness[y0:y1, x0:x1].mean()
            if cell_dark <= threshold:
                continue
            affected_cells += 1
            t = min(1.0, (cell_dark - threshold) / (1.0 - threshold))
            # Strictly blue/cyan — no pink or red
            cr = int(10 + 35 * t)
            cg = int(155 * (1.0 - t * 0.55))
            cb = 240
            ca = int(140 + 90 * t)
            n_dots = max(1, int(1 + t * 4))
            margin = dot_r + 1
            for _ in range(n_dots):
                cx = rng.randint(x0 + margin, max(x0 + margin + 1, x1 - margin))
                cy = rng.randint(y0 + margin, max(y0 + margin + 1, y1 - margin))
                draw.ellipse([cx - dot_r, cy - dot_r, cx + dot_r, cy + dot_r], fill=(cr, cg, cb, ca))

    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=dot_r * 0.5))
    score = int((affected_cells / max(face_cells, 1)) * 100)
    return overlay, score


def apply_overlay(base_rgb: Image.Image, overlay_rgba: Image.Image) -> Image.Image:
    base = base_rgb.convert("RGBA")
    return Image.alpha_composite(base, overlay_rgba).convert("RGB")


def make_redness_grid(clean_rgba: Image.Image) -> tuple:
    overlay, score = compute_redness_overlay(clean_rgba)
    base = on_black(clean_rgba, sharpen=True)
    return apply_overlay(base, overlay), score


def make_texture_grid(clean_rgba: Image.Image) -> tuple:
    clean_rgba = clean_rgba.convert("RGBA")
    rgb_arr = np.array(clean_rgba)[..., :3].astype(np.float32)
    alpha_arr = np.array(clean_rgba)[..., 3].astype(np.float32) / 255.0

    h, w = rgb_arr.shape[:2]
    grid_cols, grid_rows = 32, 32
    cell_w = w / grid_cols
    cell_h = h / grid_rows

    gray = cv2.cvtColor(rgb_arr.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    lap_abs = np.abs(lap)

    face_mask = alpha_arr > 0.3
    if np.any(face_mask):
        # Use top 45% of variance range — makes dots appear only on truly rough areas
        lo, hi = np.percentile(lap_abs[face_mask], [55, 99])
        texture_n = np.clip((lap_abs - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    else:
        texture_n = np.zeros_like(lap_abs)

    base = on_black(clean_rgba, sharpen=True)
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    rng = random.Random(42)

    face_cells = 0
    affected_cells = 0
    threshold = 0.35
    dot_r = max(2, int(min(cell_w, cell_h) * 0.22))

    for row in range(grid_rows):
        for col in range(grid_cols):
            y0 = int(row * cell_h); y1 = int((row + 1) * cell_h)
            x0 = int(col * cell_w); x1 = int((col + 1) * cell_w)
            if alpha_arr[y0:y1, x0:x1].mean() < 0.2:
                continue
            face_cells += 1
            cell_tex = texture_n[y0:y1, x0:x1].mean()
            if cell_tex <= threshold:
                continue
            affected_cells += 1
            t = min(1.0, (cell_tex - threshold) / (1.0 - threshold))
            # orange/gold palette
            cr = int(220 + 35 * t)
            cg = int(140 + 60 * (1.0 - t * 0.5))
            cb = int(20 * (1.0 - t))
            ca = int(110 + 120 * t)
            n_dots = max(1, int(1 + t * 5))
            margin = dot_r + 1
            for _ in range(n_dots):
                cx = rng.randint(x0 + margin, max(x0 + margin + 1, x1 - margin))
                cy = rng.randint(y0 + margin, max(y0 + margin + 1, y1 - margin))
                r_var = max(1, dot_r + rng.randint(-1, 1))
                draw.ellipse([cx - r_var, cy - r_var, cx + r_var, cy + r_var], fill=(cr, cg, cb, ca))

    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=dot_r * 0.5))
    score = int((affected_cells / max(face_cells, 1)) * 100)
    result = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
    return result, score


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


ANGLE_KEYS = ["frontal", "left_45", "left_90", "right_45", "right_90"]


def process_single(image_b64: str, label: str, mode: str = "redness") -> dict:
    img_data = base64.b64decode(image_b64)
    original = Image.open(BytesIO(img_data)).convert("RGB")
    clean_rgba = remove(original, session=session).convert("RGBA")
    uid = uuid.uuid4().hex[:8]

    clean_img = on_black(clean_rgba, sharpen=True)
    visia_img = make_visia_duotone(clean_rgba)

    result = {
        "clean_image_url": upload_to_supabase(clean_img, f"clean_{label}_{uid}.png"),
        "visia_image_url": upload_to_supabase(visia_img, f"visia_{label}_{uid}.png"),
    }

    if mode == "texture":
        texture_img, texture_score = make_texture_grid(clean_rgba)
        result["texture_image_url"] = upload_to_supabase(texture_img, f"texture_{label}_{uid}.png")
        result["texture_score"] = texture_score
    else:
        # Compute overlay once, apply to both clean and VISIA
        overlay, redness_score = compute_redness_overlay(clean_rgba)
        redness_on_clean = apply_overlay(clean_img, overlay)
        redness_on_visia = apply_overlay(visia_img, overlay)
        result["redness_image_url"]       = upload_to_supabase(redness_on_clean, f"redness_{label}_{uid}.png")
        result["redness_visia_image_url"] = upload_to_supabase(redness_on_visia, f"redness_visia_{label}_{uid}.png")
        result["redness_score"] = redness_score

    return result


def handler(job):
    job_input = job.get("input", {})
    scan_id = job_input.get("scan_id")

    mode = job_input.get("mode", "redness")
    images = job_input.get("images")
    if images and isinstance(images, dict):
        processed_angles = {}
        for key in ANGLE_KEYS:
            b64 = images.get(key)
            if b64:
                try:
                    processed_angles[key] = process_single(b64, key, mode)
                except Exception as e:
                    processed_angles[key] = {"error": str(e)}

        if scan_id and SUPABASE_URL and SUPABASE_SERVICE_KEY:
            frontal = processed_angles.get("frontal", {})
            update_supabase_scan(scan_id, processed_angles, frontal)
            return {"status": "done", "scan_id": scan_id}

        return {"angles": processed_angles}

    image_b64 = job_input.get("image")
    if not image_b64:
        return {"error": "No image provided."}
    try:
        buf = BytesIO()
        img_data = base64.b64decode(image_b64)
        original = Image.open(BytesIO(img_data)).convert("RGB")
        clean_rgba = remove(original, session=session).convert("RGBA")
        result_img = on_black(clean_rgba)
        result_img.save(buf, format="PNG")
        return {"image": base64.b64encode(buf.getvalue()).decode()}
    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": handler})
