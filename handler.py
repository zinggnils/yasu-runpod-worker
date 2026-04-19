import runpod
import base64
from io import BytesIO
from PIL import Image, ImageFilter
import numpy as np
import onnxruntime as ort
from rembg import new_session, remove

print(f"ORT version: {ort.__version__}")
print(f"Available providers: {ort.get_available_providers()}")

# Try GPU first, then CPU
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
    Produces a VISIA-like neon blue overlay where redness is high.
    - Uses background-removed alpha as ROI
    - Simple redness metric: R - mean(G,B)
    - Threshold + blur to make it aesthetic
    """
    # Ensure RGBA
    clean_rgba = clean_rgba.convert("RGBA")
    rgb = np.array(clean_rgba)[..., :3].astype(np.float32)
    alpha = np.array(clean_rgba)[..., 3].astype(np.float32) / 255.0  # 0..1

    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    # Redness score (simple but effective)
    redness = r - (g + b) / 2.0  # higher = more red relative to others

    # Normalize to 0..1 using robust percentiles to avoid over-colorizing
    lo, hi = np.percentile(redness[alpha > 0.2], [60, 98]) if np.any(alpha > 0.2) else (0, 255)
    redness_n = np.clip((redness - lo) / (hi - lo + 1e-6), 0.0, 1.0)

    # Only show in foreground (alpha) and suppress very dark pixels (hair/eyes)
    brightness = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
    mask = redness_n * alpha
    mask *= np.clip((brightness - 0.15) / 0.85, 0.0, 1.0)

    # Make it “spotty” less and more aesthetic: threshold then blur
    mask = np.clip((mask - 0.25) / 0.75, 0.0, 1.0)  # only stronger redness
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=6))

    # Neon blue color overlay (you can tweak)
    neon = np.zeros((*mask.shape, 4), dtype=np.uint8)
    neon[..., 0] = 40    # R
    neon[..., 1] = 220   # G
    neon[..., 2] = 255   # B
    neon[..., 3] = np.array(mask_img)  # alpha from mask

    overlay = Image.fromarray(neon, mode="RGBA")

    # Composite overlay on top of clean image
    out = Image.alpha_composite(clean_rgba, overlay).convert("RGB")
    return out

def process_image(image_b64: str):
    img_data = base64.b64decode(image_b64)
    img = Image.open(BytesIO(img_data)).convert("RGB")

    # 1) clean background removed (RGBA)
    clean_img = remove(img, session=session).convert("RGBA")

    # 2) redness visualization based on clean_img
    redness_img = make_redness_overlay(clean_img)

    return {
        "clean_image": to_b64(clean_img),
        "redness_image": to_b64(redness_img)
    }

def handler(job):
    job_input = job.get("input", {})
    image_b64 = job_input.get("image")
    if not image_b64:
        return {"error": "No image provided. Send base64 under input.image"}

    try:
        return process_image(image_b64)
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
