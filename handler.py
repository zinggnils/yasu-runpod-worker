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
    """Place background-removed face on pure black background, optionally sharpened."""
    clean_rgba = clean_rgba.convert("RGBA")
    alpha = clean_rgba.split()[3]

    rgb = clean_rgba.convert("RGB")
    if sharpen:
        rgb = rgb.filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3))

    result = Image.new("RGB", clean_rgba.size, (0, 0, 0))
    result.paste(rgb, mask=alpha)
    return result

def make_redness_overlay(clean_rgba: Image.Image) -> Image.Image:
    """
    Sharpened face on black background + neon blue overlay where redness is high.
    Left panel output (skin mask ON).
    """
    clean_rgba = clean_rgba.convert("RGBA")
    rgb_arr = np.array(clean_rgba)[..., :3].astype(np.float32)
    alpha_arr = np.array(clean_rgba)[..., 3].astype(np.float32) / 255.0

    r, g, b = rgb_arr[..., 0], rgb_arr[..., 1], rgb_arr[..., 2]
    redness = r - (g + b) / 2.0

    lo, hi = np.percentile(redness[alpha_arr > 0.2], [60, 98]) if np.any(alpha_arr > 0.2) else (0, 255)
    redness_n = np.clip((redness - lo) / (hi - lo + 1e-6), 0.0, 1.0)

    brightness = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
    mask = redness_n * alpha_arr
    mask *= np.clip((brightness - 0.15) / 0.85, 0.0, 1.0)
    mask = np.clip((mask - 0.25) / 0.75, 0.0, 1.0)

    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=6))

    neon = np.zeros((*mask.shape, 4), dtype=np.uint8)
    neon[..., 0] = 40
    neon[..., 1] = 220
    neon[..., 2] = 255
    neon[..., 3] = np.array(mask_img)

    # Composite neon overlay on sharpened face on black background
    base = on_black(clean_rgba, sharpen=True).convert("RGBA")
    overlay = Image.fromarray(neon, mode="RGBA")
    return Image.alpha_composite(base, overlay).convert("RGB")

def make_visia_duotone(clean_rgba: Image.Image) -> Image.Image:
    """
    VISIA-style clinical duotone using CLAHE + COLORMAP_BONE.
    Matches the April 18 reference output (scan_0d9e4d41 / scan_7aa3ea77).
    """
    clean_rgba = clean_rgba.convert("RGBA")
    alpha_arr = np.array(clean_rgba.split()[3])

    # Convert to BGR for OpenCV
    rgb_arr = np.array(clean_rgba.convert("RGB"))
    bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

    # Grayscale
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # CLAHE: adaptive contrast for clinical pore/texture visibility
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)

    # BONE colormap: black → blue-grey → near-white (clinical look)
    bone = cv2.applyColorMap(contrast, cv2.COLORMAP_BONE)

    # Convert back to PIL RGB
    bone_rgb = cv2.cvtColor(bone, cv2.COLOR_BGR2RGB)
    duotone = Image.fromarray(bone_rgb, mode="RGB")

    # Place on black background using face alpha
    result = Image.new("RGB", clean_rgba.size, (0, 0, 0))
    alpha_mask = Image.fromarray(alpha_arr, mode="L")
    result.paste(duotone, mask=alpha_mask)
    return result

ANGLE_KEYS = ["frontal", "left_45", "left_90", "right_45", "right_90"]

def process_single(image_b64: str) -> dict:
    """Process one image: returns clean, redness (mask ON), visia, full_analysis (mask OFF)."""
    img_data = base64.b64decode(image_b64)
    original = Image.open(BytesIO(img_data)).convert("RGB")

    clean_rgba = remove(original, session=session).convert("RGBA")

    clean_img = on_black(clean_rgba, sharpen=True)
    redness_img = make_redness_overlay(clean_rgba)
    visia_img = make_visia_duotone(clean_rgba)

    # Skin mask OFF: redness on full original (no bg removal)
    r, g, b = original.split() if False else (None, None, None)
    original_rgba = original.convert("RGBA")
    orig_r, orig_g, orig_b, _ = original_rgba.split()
    opaque_alpha = Image.new("L", original_rgba.size, 255)
    original_full = Image.merge("RGBA", (orig_r, orig_g, orig_b, opaque_alpha))
    full_analysis_img = make_redness_overlay(original_full)

    return {
        "clean_image": to_b64(clean_img),
        "redness_image": to_b64(redness_img),
        "visia_image": to_b64(visia_img),
        "full_analysis": to_b64(full_analysis_img),
    }

def handler(job):
    job_input = job.get("input", {})

    # Multi-angle mode: images dict keyed by angle
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

    # Single image fallback (backward compat)
    image_b64 = job_input.get("image")
    if not image_b64:
        return {"error": "No image provided. Send base64 under input.image or input.images"}

    try:
        return process_single(image_b64)
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
