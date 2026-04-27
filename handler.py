import runpod
import base64
import os
import uuid
import requests
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageFilter
import numpy as np
import cv2
import onnxruntime as ort
from rembg import new_session, remove

print(f"ORT version: {ort.__version__}")  # build trigger
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
    img.save(buf, format="WEBP", quality=88, method=2)
    buf.seek(0)
    path = f"processed/{filename}"
    url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{path}"
    resp = requests.put(url, data=buf.read(), headers={
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "image/webp",
        "x-upsert": "true",
    })
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Storage upload failed: {resp.status_code} {resp.text[:200]}")
    return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{path}"


def update_supabase_scan(scan_id: str, processed_angles: dict, frontal: dict,
                         overall_score: int = 0, texture_score: int = 0):
    url = f"{SUPABASE_URL}/rest/v1/scans?id=eq.{scan_id}"
    body = {
        "status": "done",
        "processed_angles": processed_angles,
        "clean_image_url": frontal.get("clean_image_url"),
        "redness_image_url": frontal.get("redness_image_url"),
        "image_url": frontal.get("visia_image_url"),
        "redness_severity": overall_score,
        "texture_severity": texture_score,
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


def make_visia_duotone(clean_rgba: Image.Image, invert: bool = False) -> Image.Image:
    clean_rgba = clean_rgba.convert("RGBA")
    alpha_arr = np.array(clean_rgba.split()[3])
    rgb_arr = np.array(clean_rgba.convert("RGB"))
    bgr = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)

    contrast_f = contrast.astype(np.float32)
    blur_large = cv2.GaussianBlur(contrast_f, (0, 0), sigmaX=30)
    contrast_clarity = np.clip(contrast_f + (contrast_f - blur_large) * 0.40, 0, 255).astype(np.uint8)

    if invert:
        # Texture mode: inverted white — pores/texture appear bright on dark skin
        inv = 255 - contrast_clarity
        duotone = Image.fromarray(cv2.cvtColor(inv, cv2.COLOR_GRAY2RGB), mode="RGB")
    else:
        # Redness mode: BONE colormap — blue-gray tones, April 24 look
        bone = cv2.applyColorMap(contrast_clarity, cv2.COLORMAP_BONE)
        duotone = Image.fromarray(cv2.cvtColor(bone, cv2.COLOR_BGR2RGB), mode="RGB")

    duotone = duotone.filter(ImageFilter.UnsharpMask(radius=1.5, percent=200, threshold=1))
    result = Image.new("RGB", clean_rgba.size, (0, 0, 0))
    result.paste(duotone, mask=Image.fromarray(alpha_arr, mode="L"))
    return result


def compute_redness_score(clean_rgba: Image.Image) -> int:
    """Absolute erythema score 0-100 using Lab a* channel.
    Fixed universal baseline (a*=10 = neutral skin) instead of personal median —
    prevents self-cancellation for uniform redness (rosacea). REDNESS_MAX=35
    covers the full clinical range from mild flush to severe rosacea."""
    # Fixed calibration constants (skin-tone neutral in Lab a*)
    NEUTRAL_THRESHOLD = 10.0   # a* for non-reddened skin (universal)
    REDNESS_MAX       = 35.0   # a* ~35 = severe rosacea / clinical max

    clean_rgba = clean_rgba.convert("RGBA")
    rgb_arr   = np.array(clean_rgba)[..., :3]
    alpha_arr = np.array(clean_rgba)[..., 3].astype(np.float32) / 255.0
    h, w = alpha_arr.shape

    brightness = (0.2126 * rgb_arr[..., 0].astype(np.float32)
                + 0.7152 * rgb_arr[..., 1].astype(np.float32)
                + 0.0722 * rgb_arr[..., 2].astype(np.float32)) / 255.0

    alpha_uint8 = (alpha_arr * 255).astype(np.uint8)
    erode_px = max(15, int(min(h, w) * 0.04))
    eroded = cv2.erode(alpha_uint8, np.ones((erode_px, erode_px), np.uint8), iterations=1)
    skin_mask = (eroded > 100) & (brightness > 0.18)

    if not np.any(skin_mask):
        return 0

    lab    = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2LAB).astype(np.float32)
    a_star = lab[..., 1] - 128.0

    # Absolute redness above neutral threshold — uniform redness scores correctly
    ei   = np.clip(a_star - NEUTRAL_THRESHOLD, 0, None)
    ei_n = np.clip(ei / REDNESS_MAX, 0.0, 1.0)

    return int(ei_n[skin_mask].mean() * 100)


def compute_texture_score(clean_rgba: Image.Image) -> int:
    """Texture severity score 0-100.
    Speed: score on half-res (4× fewer pixels) + bilateral d=9 (7.7× faster than d=25).
    Accuracy: 99th-percentile normalization (no outlier collapse) + LoG (pre-blur kills JPEG noise)."""
    clean_rgba = clean_rgba.convert("RGBA")

    # ── Downsample to 50% for scoring — 4× fewer pixels, negligible accuracy loss ──
    w0, h0 = clean_rgba.size
    small = clean_rgba.resize((w0 // 2, h0 // 2), Image.BILINEAR)

    rgb_arr  = np.array(small.convert("RGB"))
    alpha_arr = np.array(small)[..., 3].astype(np.float32) / 255.0
    h, w = alpha_arr.shape

    alpha_uint8 = (alpha_arr * 255).astype(np.uint8)
    erode_px = max(8, int(min(h, w) * 0.04))
    eroded = cv2.erode(alpha_uint8, np.ones((erode_px, erode_px), np.uint8), iterations=1)
    inner_face = eroded > 100

    brightness = (0.2126 * rgb_arr[..., 0].astype(np.float32)
                + 0.7152 * rgb_arr[..., 1].astype(np.float32)
                + 0.0722 * rgb_arr[..., 2].astype(np.float32)) / 255.0
    skin_mask = inner_face & (brightness > 0.18)

    if not np.any(skin_mask):
        return 0

    gray = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # d=9: ~7.7× faster than d=25, better pore-scale sensitivity
    smooth_ref = cv2.bilateralFilter(gray.astype(np.uint8), 9, 80, 80).astype(np.float32)
    diff = np.abs(gray - smooth_ref)

    # LoG: pre-blur before Laplacian kills JPEG compression noise
    gray_smooth = cv2.GaussianBlur(gray.astype(np.uint8), (3, 3), 0.8)
    laplacian = np.abs(cv2.Laplacian(gray_smooth, cv2.CV_32F))

    texture_map = diff * 0.55 + laplacian * 0.45
    texture_map[~skin_mask] = 0.0

    skin_vals = texture_map[skin_mask]
    if skin_vals.max() < 1e-6:
        return 0

    # 99th-percentile ceiling: prevents a single outlier pixel from collapsing the scale
    ceil = np.percentile(skin_vals, 99)
    texture_map = np.clip(texture_map / (ceil + 1e-6), 0.0, 1.0)

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
    # Alpha matting disabled: pymatting Cholesky decomposition fails repeatedly on most inputs,
    # adding ~15–20s of wasted retries per scan. The guided filter below handles edge cleanup.
    clean_rgba = remove(original, session=session).convert("RGBA")
    print("[process_single] Background removed")
    # Guided filter: snaps soft matting edges to actual image boundaries (fixes fringing)
    clean_rgba = refine_alpha(original, clean_rgba)
    uid = uuid.uuid4().hex[:8]
    clean_img = on_black(clean_rgba)

    if mode == "texture":
        visia_img = make_visia_duotone(clean_rgba, invert=True)
        texture_score = compute_texture_score(clean_rgba)
        print(f"[process_single] texture_score={texture_score}")
        with ThreadPoolExecutor(max_workers=2) as pool:
            f_clean = pool.submit(upload_to_supabase, clean_img, f"clean_{label}_{uid}.webp")
            f_visia = pool.submit(upload_to_supabase, visia_img, f"visia_{label}_{uid}.webp")
            return {
                "clean_image_url": f_clean.result(),
                "visia_image_url": f_visia.result(),
                "texture_score":   texture_score,
            }

    # ── Default: redness flow ──
    # Score computed from raw skin signal; no overlay images generated.
    # Left panel = clean photo on black, right panel = BONE duotone.
    visia_img = make_visia_duotone(clean_rgba)
    redness_score = compute_redness_score(clean_rgba)
    print(f"[process_single] redness_score={redness_score}")

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_clean = pool.submit(upload_to_supabase, clean_img,  f"clean_{label}_{uid}.webp")
        f_visia = pool.submit(upload_to_supabase, visia_img,  f"visia_{label}_{uid}.webp")
        return {
            "clean_image_url": f_clean.result(),
            "visia_image_url": f_visia.result(),
            "redness_score":   redness_score,
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
        def process_angle(key):
            b64 = images.get(key)
            if not b64:
                print(f"[handler] {key} — no b64 data, skipping")
                return key, None
            print(f"[handler] Processing {key}, b64 length={len(b64)}")
            try:
                result = process_single(b64, key, mode=mode)
                score_log = result.get('redness_score') if result.get('redness_score') is not None else result.get('texture_score')
                print(f"[handler] {key} OK — score={score_log}")
                return key, result
            except Exception as e:
                print(f"[handler] {key} FAILED: {e}")
                traceback.print_exc()
                return key, {"error": str(e)}

        processed_angles = {}
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {pool.submit(process_angle, key): key for key in ANGLE_KEYS}
            for future in futures:
                key, result = future.result()
                if result is not None:
                    processed_angles[key] = result

        print(f"[handler] Processed angles: {list(processed_angles.keys())}")

        if scan_id and SUPABASE_URL and SUPABASE_SERVICE_KEY:
            try:
                frontal = processed_angles.get("frontal", {})

                # Overall redness score: frontal + 45° angles show the cheek butterfly
                # pattern most directly. 90° profiles see redness obliquely and underreport.
                redness_vals = [
                    s for s in [
                        processed_angles.get("frontal",   {}).get("redness_score"),
                        processed_angles.get("left_45",   {}).get("redness_score"),
                        processed_angles.get("right_45",  {}).get("redness_score"),
                    ] if isinstance(s, (int, float))
                ]
                overall_score = int(sum(redness_vals) / len(redness_vals)) if redness_vals else 0
                print(f"[handler] overall_redness_score={overall_score} (frontal+45°)")

                # Overall texture score: same angles — pores/scars face forward,
                # foreshortened at 90° profile so frontal+45° give the clearest read.
                texture_vals = [
                    s for s in [
                        processed_angles.get("frontal",   {}).get("texture_score"),
                        processed_angles.get("left_45",   {}).get("texture_score"),
                        processed_angles.get("right_45",  {}).get("texture_score"),
                    ] if isinstance(s, (int, float))
                ]
                overall_texture = int(sum(texture_vals) / len(texture_vals)) if texture_vals else 0
                print(f"[handler] overall_texture_score={overall_texture} (frontal+45°)")

                update_supabase_scan(scan_id, processed_angles, frontal, overall_score, overall_texture)
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
