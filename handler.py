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


def guided_filter_alpha(guide_gray: np.ndarray, alpha: np.ndarray, radius: int = 6, eps: float = 1e-3) -> np.ndarray:
    """Edge-aware alpha mask refinement using guided filter.
    Snaps rembg's upsampled mask edges to actual image boundaries — fixes fringing and jagged hair edges."""
    def box(I, r):
        return cv2.boxFilter(I.astype(np.float32), -1, (2 * r + 1, 2 * r + 1))

    I = guide_gray.astype(np.float32) / 255.0
    p = alpha.astype(np.float32) / 255.0

    mean_I  = box(I, radius)
    mean_p  = box(p, radius)
    mean_Ip = box(I * p, radius)
    cov_Ip  = mean_Ip - mean_I * mean_p

    mean_II = box(I * I, radius)
    var_I   = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = box(a, radius)
    mean_b = box(b, radius)

    q = mean_a * I + mean_b
    return np.clip(q * 255, 0, 255).astype(np.uint8)


def refine_alpha(original_rgb: Image.Image, clean_rgba: Image.Image) -> Image.Image:
    """Apply guided filter to rembg's alpha mask using original image as guide.
    Eliminates fringing and sharpens hair/skin boundary without touching the face content."""
    gray = cv2.cvtColor(np.array(original_rgb.convert("RGB")), cv2.COLOR_RGB2GRAY)
    alpha = np.array(clean_rgba.split()[3])
    alpha_refined = guided_filter_alpha(gray, alpha, radius=6, eps=1e-3)
    result = clean_rgba.copy()
    result.putalpha(Image.fromarray(alpha_refined))
    return result


def on_black(clean_rgba: Image.Image) -> Image.Image:
    clean_rgba = clean_rgba.convert("RGBA")
    alpha = clean_rgba.split()[3]
    rgb = clean_rgba.convert("RGB")

    # ── Step 1: Clarity pass (large-radius high-pass) ──
    # Replicates Adobe Express / Lightroom "Clarity" — lifts midtone contrast,
    # makes skin texture, pores and fine detail visually pop without over-sharpening edges.
    arr = np.array(rgb).astype(np.float32)
    blur_large = cv2.GaussianBlur(arr, (0, 0), sigmaX=30)
    # strength=0.45 ≈ Lightroom Clarity +40 / Adobe Express texture boost
    arr_clarity = np.clip(arr + (arr - blur_large) * 0.45, 0, 255).astype(np.uint8)
    rgb = Image.fromarray(arr_clarity)

    # ── Step 2: Fine sharpening pass ──
    # Crispens edges and fine lines — radius=1.5, percent=220 ≈ Adobe Express Sharpen +25
    rgb = rgb.filter(ImageFilter.UnsharpMask(radius=1.5, percent=220, threshold=1))

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

    # Clarity pass on the grayscale before colormap — lifts skin texture in VISIA view
    contrast_f = contrast.astype(np.float32)
    blur_large = cv2.GaussianBlur(contrast_f, (0, 0), sigmaX=30)
    contrast_clarity = np.clip(contrast_f + (contrast_f - blur_large) * 0.40, 0, 255).astype(np.uint8)

    bone = cv2.applyColorMap(contrast_clarity, cv2.COLORMAP_BONE)
    bone_rgb = cv2.cvtColor(bone, cv2.COLOR_BGR2RGB)
    duotone = Image.fromarray(bone_rgb, mode="RGB")
    # Fine sharpening pass
    duotone = duotone.filter(ImageFilter.UnsharpMask(radius=1.5, percent=200, threshold=1))
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

    # Lowered brightness threshold (0.18) to better handle darker/olive skin tones
    skin_mask = inner_face & (brightness > 0.18)

    if not np.any(skin_mask):
        return Image.new("RGBA", (w, h), (0, 0, 0, 0)), 0

    # Normalize redness on inner skin only — tight range highlights actual redness
    lo, hi = np.percentile(redness[skin_mask], [55, 97])
    redness_n = np.clip((redness - lo) / (hi - lo + 1e-6), 0.0, 1.0)

    # Score computed from RAW redness signal BEFORE visualization blur — accurate, blur-independent
    # Uses mean of top-25% most-red skin pixels to reflect peak redness severity
    face_skin = skin_mask & (alpha_arr > 0.2)
    if np.any(face_skin):
        rn_face = redness_n[face_skin]
        p75 = np.percentile(rn_face, 75)
        top25_mean = float(rn_face[rn_face >= max(p75, 0.01)].mean())
        # Scale: 0.2 raw → ~20 score (mild), 0.5 raw → ~50 (moderate), 0.8+ → ~80+ (severe)
        score = int(min(100, top25_mean * 100))
    else:
        score = 0

    # Build smooth mask for visualization: redness × face alpha × brightness weight
    mask = redness_n * alpha_arr
    mask *= np.where(skin_mask, 1.0, 0.0)
    mask *= np.clip((brightness - 0.12) / 0.88, 0.0, 1.0)

    # Softer threshold (0.10 → more of the redness area covered, clinically realistic)
    mask = np.clip((mask - 0.10) / 0.90, 0.0, 1.0)

    # Gaussian blur → smooth neon blobs (no hard edges)
    mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
    mask_img = mask_img.filter(ImageFilter.GaussianBlur(radius=8))

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


def compute_texture_score(clean_rgba: Image.Image) -> int:
    """Texture severity score 0-100 using same methodology as redness score.
    Bilateral-filter diff + Laplacian → normalized texture map → top-25% mean × 100."""
    clean_rgba = clean_rgba.convert("RGBA")
    rgb_arr = np.array(clean_rgba.convert("RGB"))
    alpha_arr = np.array(clean_rgba)[..., 3].astype(np.float32) / 255.0
    h, w = alpha_arr.shape

    alpha_uint8 = (alpha_arr * 255).astype(np.uint8)
    erode_px = max(15, int(min(h, w) * 0.04))
    kernel = np.ones((erode_px, erode_px), np.uint8)
    eroded = cv2.erode(alpha_uint8, kernel, iterations=1)
    inner_face = eroded > 100

    brightness = (0.2126 * rgb_arr[..., 0].astype(np.float32)
                + 0.7152 * rgb_arr[..., 1].astype(np.float32)
                + 0.0722 * rgb_arr[..., 2].astype(np.float32)) / 255.0
    skin_mask = inner_face & (brightness > 0.18)

    if not np.any(skin_mask):
        return 0

    gray = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY).astype(np.float32)
    smooth_ref = cv2.bilateralFilter(gray.astype(np.uint8), 25, 80, 80).astype(np.float32)
    diff = np.abs(gray - smooth_ref)
    laplacian = np.abs(cv2.Laplacian(gray.astype(np.uint8), cv2.CV_32F))
    texture_map = diff * 0.55 + laplacian * 0.45
    texture_map[~skin_mask] = 0.0

    if texture_map.max() < 1e-6:
        return 0

    texture_map /= texture_map.max()
    tm_face = texture_map[skin_mask]
    p75 = np.percentile(tm_face, 75)
    return int(min(100, float(tm_face[tm_face >= max(p75, 0.01)].mean()) * 100))


ANGLE_KEYS = ["frontal", "left_45", "left_90", "right_45", "right_90"]


def process_single(image_b64: str, label: str, mode: str = "redness") -> dict:
    img_data  = base64.b64decode(image_b64)
    original  = Image.open(BytesIO(img_data)).convert("RGB")
    print(f"[process_single] Input size: {original.size}")
    # alpha_matting=True: trimap-based matting preserves fine hair strands u2net would clip.
    # Falls back to standard removal if pymatting is unavailable.
    try:
        clean_rgba = remove(
            original,
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=10,
        ).convert("RGBA")
        print("[process_single] Alpha matting OK")
    except Exception as e:
        print(f"[process_single] Alpha matting failed ({e}), falling back to standard removal")
        clean_rgba = remove(original, session=session).convert("RGBA")
    # Guided filter: snaps soft matting edges to actual image boundaries (fixes fringing)
    clean_rgba = refine_alpha(original, clean_rgba)

    # 2x upsample before all processing — sharpening + clarity run at full 2x resolution
    w, h = clean_rgba.size
    clean_rgba = clean_rgba.resize((w * 2, h * 2), Image.LANCZOS)
    print(f"[process_single] Upscaled to {clean_rgba.size}")

    uid = uuid.uuid4().hex[:8]

    clean_img = on_black(clean_rgba)

    if mode == "texture":
        visia_img = make_visia_duotone(clean_rgba)
        texture_score = compute_texture_score(clean_rgba)
        print(f"[process_single] texture_score={texture_score}")
        return {
            "clean_image_url":  upload_to_supabase(clean_img,  f"clean_{label}_{uid}.png"),
            "visia_image_url":  upload_to_supabase(visia_img,  f"visia_{label}_{uid}.png"),
            "texture_score":    texture_score,
        }

    # ── Default: redness flow ──
    visia_img = make_visia_duotone(clean_rgba)
    overlay, redness_score = compute_redness_overlay(clean_rgba)
    redness_on_clean = apply_overlay(clean_img, overlay)
    redness_on_visia = apply_overlay(visia_img, overlay)

    return {
        "clean_image_url":         upload_to_supabase(clean_img,        f"clean_{label}_{uid}.png"),
        "visia_image_url":         upload_to_supabase(visia_img,        f"visia_{label}_{uid}.png"),
        "redness_image_url":       upload_to_supabase(redness_on_clean, f"redness_{label}_{uid}.png"),
        "redness_visia_image_url": upload_to_supabase(redness_on_visia, f"redness_visia_{label}_{uid}.png"),
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

    # Fail immediately if env vars missing — worker started without config, no point processing
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        raise RuntimeError(
            f"MISSING ENV VARS: SUPABASE_URL={'SET' if SUPABASE_URL else 'EMPTY'}, "
            f"SUPABASE_SERVICE_KEY={'SET' if SUPABASE_SERVICE_KEY else 'EMPTY'}. "
            "Worker started without required environment variables."
        )

    mode = job_input.get("mode", "redness")
    print(f"[handler] mode={mode}")

    if images and isinstance(images, dict):
        processed_angles = {}
        for key in ANGLE_KEYS:
            b64 = images.get(key)
            if b64:
                print(f"[handler] Processing {key}, b64 length={len(b64)}")
                try:
                    result = process_single(b64, key, mode=mode)
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
