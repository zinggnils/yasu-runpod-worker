import runpod
import base64
from io import BytesIO
from PIL import Image, ImageFilter
import numpy as np
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

def make_redness_overlay(clean_rgba: Image.Image) -> Image.Image:
    """
    VISIA-like neon blue overlay where redness is high.
    Uses background-removed alpha as ROI.
    Redness metric: R - mean(G,B)
    """
    clean_rgba = clean_rgba.convert("RGBA")
    rgb = np.array(clean_rgba)[..., :3].astype(np.float32)
    alpha = np.array(clean_rgba)[..., 3].astype(np.float32) / 255.0

    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    redness = r - (g + b) / 2.0

    lo, hi = np.percentile(redness[alpha > 0.2], [60, 98]) if np.any(alpha > 0.2) else (0, 255)
    redness_n = np.clip((redness - lo) / (hi - lo + 1e-6), 0.0, 1.0)

    brightness = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
    mask = redness_n * alpha
    mask *= np.clip((brightness - 0.15) / 0.85, 0.0, 1.0)

    mask = np.clip((mask - 0.25) / 0.75, 0.0, 1.0)
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=6))

    neon = np.zeros((*mask.shape, 4), dtype=np.uint8)
    neon[..., 0] = 40
    neon[..., 1] = 220
    neon[..., 2] = 255
    neon[..., 3] = np.array(mask_img)

    overlay = Image.fromarray(neon, mode="RGBA")
    out = Image.alpha_composite(clean_rgba, overlay).convert("RGB")
    return out

def process_image(image_b64: str, skin_mask: bool = True):
    img_data = base64.b64decode(image_b64)
    original = Image.open(BytesIO(img_data)).convert("RGB")

    # Output 1: clean background-removed image (RGBA)
    clean_rgba = remove(original, session=session).convert("RGBA")

    # Output 2: redness analysis with neon blue overlay (always uses masked face)
    redness_img = make_redness_overlay(clean_rgba)

    result = {
        "clean_image": to_b64(clean_rgba),
        "redness_image": to_b64(redness_img),
    }

    # Output 3: skin mask off — redness on full original without background removal
    if not skin_mask:
        original_rgba = original.convert("RGBA")
        full_redness_img = make_redness_overlay(original_rgba)
        result["full_analysis"] = to_b64(full_redness_img)

    return result

def handler(job):
    job_input = job.get("input", {})
    image_b64 = job_input.get("image")
    if not image_b64:
        return {"error": "No image provided. Send base64 under input.image"}

    skin_mask = job_input.get("skin_mask", True)

    try:
        return process_image(image_b64, skin_mask=skin_mask)
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
