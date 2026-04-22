import runpod
import base64
from io import BytesIO
from PIL import Image, ImageFilter
import numpy as np
import cv2
import onnxruntime as ort
from rembg import new_session, remove

print(f"ORT version: {ort.__version__}")
print(f"Available providers: {ort.get_available_providers()}")

try:
    session = new_session("u2net", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print("Rembg session initialized with CUDA/CPU.")
except Exception as e:
    print(f"Failed to initialize GPU session, falling back: {e}")
    session = new_session("u2net")

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

def make_redness_grid(clean_rgba: Image.Image) -> tuple:
    """
    32x32 grid overlay (1024 cells) coloured by redness intensity.
    Returns (image, score_0_to_100).
    """
    clean_rgba = clean_rgba.convert("RGBA")
    rgb_arr = np.array(clean_rgba)[..., :3].astype(np.float32)
    alpha_arr = np.array(clean_rgba)[..., 3].astype(np.float32) / 255.0

    h, w = rgb_arr.shape[:2]
    grid_cols, grid_rows = 32, 32
    cell_w = w / grid_cols
    cell_h = h / grid_rows

    r, g, b = rgb_arr[..., 0], rgb_arr[..., 1], rgb_arr[..., 2]
    redness = r - (g + b) / 2.0

    face_mask = alpha_arr > 0.3
    if np.any(face_mask):
        lo, hi = np.percentile(redness[face_mask], [15, 95])
        redness_n = np.clip((redness - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    else:
        redness_n = np.zeros_like(redness)

    base = on_black(clean_rgba, sharpen=True).convert("RGBA")
    overlay = np.zeros((h, w, 4), dtype=np.uint8)

    face_cells = 0
    affected_cells = 0
    threshold = 0.35

    for row in range(grid_rows):
        for col in range(grid_cols):
            y0 = int(row * cell_h)
            y1 = int((row + 1) * cell_h)
            x0 = int(col * cell_w)
            x1 = int((col + 1) * cell_w)

            cell_alpha = alpha_arr[y0:y1, x0:x1]
            if cell_alpha.mean() < 0.2:
                continue

            face_cells += 1
            cell_redness = redness_n[y0:y1, x0:x1].mean()

            if cell_redness > threshold:
                affected_cells += 1
                t = min(1.0, (cell_redness - threshold) / (1.0 - threshold))
                # Cyan-blue for low → orange-red for high
                cr = int(30 + 225 * t)
                cg = int(160 * (1 - t))
                cb = int(255 * (1 - t * 0.85))
                ca = int(100 + 130 * t)
                pad = max(1, int(min(cell_w, cell_h) * 0.06))
                overlay[y0+pad:y1-pad, x0+pad:x1-pad, 0] = cr
                overlay[y0+pad:y1-pad, x0+pad:x1-pad, 1] = cg
                overlay[y0+pad:y1-pad, x0+pad:x1-pad, 2] = cb
                overlay[y0+pad:y1-pad, x0+pad:x1-pad, 3] = ca

    score = int((affected_cells / max(face_cells, 1)) * 100)
    overlay_img = Image.fromarray(overlay, mode="RGBA")
    result = Image.alpha_composite(base, overlay_img).convert("RGB")
    return result, score

def make_texture_grid(clean_rgba: Image.Image) -> tuple:
    """
    32x32 grid overlay coloured by texture (Laplacian variance) per cell.
    Returns (image, score_0_to_100).
    """
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
        lo, hi = np.percentile(lap_abs[face_mask], [20, 95])
        texture_n = np.clip((lap_abs - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    else:
        texture_n = np.zeros_like(lap_abs)

    base = on_black(clean_rgba, sharpen=True).convert("RGBA")
    overlay = np.zeros((h, w, 4), dtype=np.uint8)

    face_cells = 0
    affected_cells = 0
    threshold = 0.35

    for row in range(grid_rows):
        for col in range(grid_cols):
            y0 = int(row * cell_h)
            y1 = int((row + 1) * cell_h)
            x0 = int(col * cell_w)
            x1 = int((col + 1) * cell_w)

            cell_alpha = alpha_arr[y0:y1, x0:x1]
            if cell_alpha.mean() < 0.2:
                continue

            face_cells += 1
            cell_tex = texture_n[y0:y1, x0:x1].mean()

            if cell_tex > threshold:
                affected_cells += 1
                t = min(1.0, (cell_tex - threshold) / (1.0 - threshold))
                # Green-teal for low → yellow-white for high (clinical texture look)
                cr = int(100 * t)
                cg = int(200 + 55 * t)
                cb = int(180 * (1 - t * 0.7))
                ca = int(90 + 140 * t)
                pad = max(1, int(min(cell_w, cell_h) * 0.06))
                overlay[y0+pad:y1-pad, x0+pad:x1-pad, 0] = cr
                overlay[y0+pad:y1-pad, x0+pad:x1-pad, 1] = cg
                overlay[y0+pad:y1-pad, x0+pad:x1-pad, 2] = cb
                overlay[y0+pad:y1-pad, x0+pad:x1-pad, 3] = ca

    score = int((affected_cells / max(face_cells, 1)) * 100)
    overlay_img = Image.fromarray(overlay, mode="RGBA")
    result = Image.alpha_composite(base, overlay_img).convert("RGB")
    return result, score

def make_visia_duotone(clean_rgba: Image.Image) -> Image.Image:
    """VISIA-style clinical duotone using CLAHE + COLORMAP_BONE."""
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
    alpha_mask = Image.fromarray(alpha_arr, mode="L")
    result.paste(duotone, mask=alpha_mask)
    return result

ANGLE_KEYS = ["frontal", "left_45", "left_90", "right_45", "right_90"]

def process_single(image_b64: str) -> dict:
    """Process one image: clean, redness grid + score, texture grid + score, visia, full_analysis."""
    img_data = base64.b64decode(image_b64)
    original = Image.open(BytesIO(img_data)).convert("RGB")

    clean_rgba = remove(original, session=session).convert("RGBA")

    clean_img = on_black(clean_rgba, sharpen=True)
    redness_img, redness_score = make_redness_grid(clean_rgba)
    texture_img, texture_score = make_texture_grid(clean_rgba)
    visia_img = make_visia_duotone(clean_rgba)

    # Skin mask OFF: redness grid on full original (no bg removal)
    original_rgba = original.convert("RGBA")
    orig_r, orig_g, orig_b, _ = original_rgba.split()
    opaque_alpha = Image.new("L", original_rgba.size, 255)
    original_full = Image.merge("RGBA", (orig_r, orig_g, orig_b, opaque_alpha))
    full_analysis_img, _ = make_redness_grid(original_full)

    return {
        "clean_image": to_b64(clean_img),
        "redness_image": to_b64(redness_img),
        "texture_image": to_b64(texture_img),
        "visia_image": to_b64(visia_img),
        "full_analysis": to_b64(full_analysis_img),
        "redness_score": redness_score,
        "texture_score": texture_score,
    }

def handler(job):
    job_input = job.get("input", {})

    images = job_input.get("images")
    if images and isinstance(images, dict):
        results = {}
        for key in ANGLE_KEYS:
            b64 = images.get(key)
            if b64:
                try:
                    results[key] = process_single(b64)
                except Exception as e:
                    results[key] = {"error": str(e)}
        return {"angles": results}

    image_b64 = job_input.get("image")
    if not image_b64:
        return {"error": "No image provided. Send base64 under input.image or input.images"}

    try:
        return process_single(image_b64)
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
