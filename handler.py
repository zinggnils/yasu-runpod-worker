import runpod
import base64
from io import BytesIO
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
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
    VISIA-style clinical duotone. Right panel output.
    - Redness-weighted grayscale: R channel weighted higher → red areas appear brighter
    - High contrast: near-black shadows, bright light-blue highlights
    - Sharpened to reveal skin texture
    - Black background
    """
    clean_rgba = clean_rgba.convert("RGBA")
    alpha = clean_rgba.split()[3]
    rgb_arr = np.array(clean_rgba.convert("RGB")).astype(np.float32)

    r, g, b = rgb_arr[..., 0], rgb_arr[..., 1], rgb_arr[..., 2]

    # Redness-weighted grayscale: weight R more so redness = brighter in duotone
    redness_gray = np.clip(0.55 * r + 0.25 * g + 0.20 * b, 0, 255).astype(np.uint8)
    gray_img = Image.fromarray(redness_gray, mode="L")

    # Sharpen to reveal skin texture
    gray_img = gray_img.filter(ImageFilter.UnsharpMask(radius=2, percent=160, threshold=2))

    # Boost contrast so shadows are very dark, highlights very bright
    gray_img = ImageEnhance.Contrast(gray_img).enhance(1.4)

    # Duotone: very dark navy → bright almost-white light blue
    duotone = ImageOps.colorize(gray_img, black=(3, 5, 18), white=(210, 225, 248))

    # Place on black background using face alpha
    result = Image.new("RGB", clean_rgba.size, (0, 0, 0))
    result.paste(duotone, mask=alpha)
    return result

def process_image(image_b64: str, skin_mask: bool = True):
    img_data = base64.b64decode(image_b64)
    original = Image.open(BytesIO(img_data)).convert("RGB")

    # Remove background once
    clean_rgba = remove(original, session=session).convert("RGBA")

    # Output 1: clean base — sharpened face on black background
    clean_img = on_black(clean_rgba, sharpen=True)

    # Output 2 (left panel): redness overlay on black bg (skin mask ON state)
    redness_img = make_redness_overlay(clean_rgba)

    # Output 3 (right panel): VISIA duotone — always produced
    visia_img = make_visia_duotone(clean_rgba)

    result = {
        "clean_image": to_b64(clean_img),
        "redness_image": to_b64(redness_img),
        "visia_image": to_b64(visia_img),
    }

    # Skin mask OFF: also return redness overlay on full original (no bg removal)
    if not skin_mask:
        original_rgba = original.convert("RGBA")
        # Make all pixels fully opaque for full-image analysis
        r, g, b, a = original_rgba.split()
        a = Image.new("L", original_rgba.size, 255)
        original_full = Image.merge("RGBA", (r, g, b, a))
        result["full_analysis"] = to_b64(make_redness_overlay(original_full))

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
