import runpod
import base64
import os
import uuid
import requests
import hashlib
from io import BytesIO
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import cv2
import onnxruntime as ort
from rembg import new_session, remove
from image_quality import (
    BLUR_MIN_SCORE,
    MEDIAPIPE_AVAILABLE,
    QUALITY_MIN_CONFIDENCE,
    check_image_quality,
)
from capture_targets import FRONTAL_RETAKE_CONFIDENCE, repositioning_hint
from shadow_norm import load_shadow_params_from_env, normalize_shadows

try:
    import mediapipe as mp

    if MEDIAPIPE_AVAILABLE:
        print(f"[startup] mediapipe version={mp.__version__}")
    else:
        print("[startup] mediapipe unavailable")
except Exception as e:
    print(f"[startup] mediapipe unavailable: {e}")

print(f"ORT version: {ort.__version__}")  # build trigger
print(f"Available providers: {ort.get_available_providers()}")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
SUPABASE_BUCKET = "scans"

ORT_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
BG_REMOVAL_BACKEND = os.environ.get("BG_REMOVAL_BACKEND", "modnet").strip().lower()
MODNET_MODEL_URL = os.environ.get(
    "MODNET_MODEL_URL",
    "https://github.com/yakhyo/modnet/releases/download/weights/modnet_photographic.onnx",
)
MODNET_MODEL_PATH = os.environ.get("MODNET_MODEL_PATH", "/root/.modnet/modnet_photographic.onnx")
MODNET_MODEL_SHA256 = os.environ.get(
    "MODNET_MODEL_SHA256",
    "5069a5e306b9f5e9f4f2b0360264c9f8ea13b257c7c39943c7cf6a2ec3a102ae",
).strip().lower()
MODNET_INPUT_SIZE = int(os.environ.get("MODNET_INPUT_SIZE", "512"))
TARGET_MAX_PX = int(os.environ.get("TARGET_MAX_PX", "2048"))
UPSCALE_FACTOR = float(os.environ.get("UPSCALE_FACTOR", "2.0"))
UPSCALE_TRIGGER_PX = int(os.environ.get("UPSCALE_TRIGGER_PX", "1024"))
SHARPEN_AMOUNT = float(os.environ.get("SHARPEN_AMOUNT", "1.15"))


def active_ort_providers() -> list:
    available = set(ort.get_available_providers())
    providers = [provider for provider in ORT_PROVIDERS if provider in available]
    return providers or ["CPUExecutionProvider"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_modnet_model() -> str:
    model_path = Path(MODNET_MODEL_PATH)
    if model_path.exists() and model_path.stat().st_size > 0:
        if MODNET_MODEL_SHA256 and sha256_file(model_path) != MODNET_MODEL_SHA256:
            print(f"[modnet] Cached model checksum mismatch, refreshing {model_path}")
            model_path.unlink()
        else:
            return str(model_path)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = model_path.with_suffix(model_path.suffix + ".tmp")
    print(f"[modnet] Downloading model to {model_path}")
    with requests.get(MODNET_MODEL_URL, stream=True, timeout=(10, 240)) as resp:
        resp.raise_for_status()
        with tmp_path.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)

    if MODNET_MODEL_SHA256 and sha256_file(tmp_path) != MODNET_MODEL_SHA256:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError("Downloaded MODNet model checksum mismatch")

    tmp_path.replace(model_path)
    return str(model_path)


def init_modnet_session():
    if BG_REMOVAL_BACKEND in {"rembg", "u2net"}:
        print(f"[modnet] Disabled by BG_REMOVAL_BACKEND={BG_REMOVAL_BACKEND}")
        return None

    try:
        model_path = ensure_modnet_model()
        providers = active_ort_providers()
        sess = ort.InferenceSession(model_path, providers=providers)
        print(f"[modnet] Session initialized with providers={sess.get_providers()}")
        return sess
    except Exception as e:
        print(f"[modnet] Initialization failed, rembg fallback will be used: {e}")
        return None


try:
    session = new_session("u2net", providers=active_ort_providers())
    print("Rembg session initialized with CUDA/CPU.")
except Exception as e:
    print(f"Failed to initialize GPU session, falling back: {e}")
    session = new_session("u2net")

modnet_session = init_modnet_session()

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

def modnet_target_size(width: int, height: int, session) -> tuple:
    input_shape = session.get_inputs()[0].shape
    if len(input_shape) == 4 and isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
        return input_shape[3], input_shape[2]

    if max(width, height) < MODNET_INPUT_SIZE or min(width, height) > MODNET_INPUT_SIZE:
        if width >= height:
            new_h = MODNET_INPUT_SIZE
            new_w = int(width / height * MODNET_INPUT_SIZE)
        else:
            new_w = MODNET_INPUT_SIZE
            new_h = int(height / width * MODNET_INPUT_SIZE)
    else:
        new_w, new_h = width, height

    return max(32, new_w - (new_w % 32)), max(32, new_h - (new_h % 32))


def run_modnet(original_rgb: Image.Image, session) -> Image.Image:
    rgb_arr = np.array(original_rgb.convert("RGB"))
    height, width = rgb_arr.shape[:2]
    target_w, target_h = modnet_target_size(width, height, session)

    resized = cv2.resize(rgb_arr, (target_w, target_h), interpolation=cv2.INTER_AREA)
    tensor = resized.astype(np.float32) / 255.0
    tensor = (tensor - 0.5) / 0.5
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...]

    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    matte = session.run(output_names, {input_name: tensor})[0]
    matte = np.squeeze(matte)
    matte = cv2.resize(matte, (width, height), interpolation=cv2.INTER_LINEAR)
    alpha = np.clip(matte * 255.0, 0, 255).astype(np.uint8)

    result = original_rgb.convert("RGBA")
    result.putalpha(Image.fromarray(alpha, mode="L"))
    print("[run_modnet] Background matte generated")
    return result


def remove_background(original_rgb: Image.Image) -> Image.Image:
    if modnet_session is not None:
        try:
            return run_modnet(original_rgb, modnet_session)
        except Exception as e:
            print(f"[remove_background] MODNet failed, using rembg fallback: {e}")

    return remove(original_rgb, session=session).convert("RGBA")

def guided_filter_alpha(guide_gray: np.ndarray, alpha: np.ndarray, radius: int = 6, eps: float = 1e-3) -> np.ndarray:
    """Edge-aware alpha mask refinement using guided filter.
    Snaps soft mask edges to actual image boundaries to reduce fringing."""
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
    Eliminates fringing and sharpens detailed alpha boundaries."""
    gray = cv2.cvtColor(np.array(original_rgb.convert("RGB")), cv2.COLOR_RGB2GRAY)
    alpha = np.array(clean_rgba.split()[3])
    alpha_refined = guided_filter_alpha(gray, alpha, radius=6, eps=1e-3)
    result = clean_rgba.copy()
    result.putalpha(Image.fromarray(alpha_refined))
    return result

def _stable_skin_roi(alpha_arr: np.ndarray, rgb_arr: np.ndarray) -> np.ndarray:
    """
    Stable face ROI for scoring:
    - keep inner face skin
    - exclude hairline, eyes, lips, nostrils center strip
    """
    h, w = alpha_arr.shape
    alpha_uint8 = (alpha_arr * 255).astype(np.uint8)
    erode_px = max(10, int(min(h, w) * 0.045))
    eroded = cv2.erode(alpha_uint8, np.ones((erode_px, erode_px), np.uint8), iterations=1)
    base = eroded > 105

    brightness = (0.2126 * rgb_arr[..., 0].astype(np.float32)
                + 0.7152 * rgb_arr[..., 1].astype(np.float32)
                + 0.0722 * rgb_arr[..., 2].astype(np.float32)) / 255.0
    base &= (brightness > 0.16) & (brightness < 0.96)

    yy, xx = np.mgrid[0:h, 0:w]
    xn = xx / max(1.0, float(w - 1))
    yn = yy / max(1.0, float(h - 1))

    # Generic exclusion zones in normalized face crop coordinates.
    hairline = yn < 0.18
    left_eye = ((xn - 0.33) ** 2 / (0.11 ** 2) + (yn - 0.37) ** 2 / (0.07 ** 2)) < 1.0
    right_eye = ((xn - 0.67) ** 2 / (0.11 ** 2) + (yn - 0.37) ** 2 / (0.07 ** 2)) < 1.0
    lips = ((xn - 0.50) ** 2 / (0.17 ** 2) + (yn - 0.70) ** 2 / (0.08 ** 2)) < 1.0
    nostrils = ((xn - 0.50) ** 2 / (0.08 ** 2) + (yn - 0.56) ** 2 / (0.05 ** 2)) < 1.0
    unstable = hairline | left_eye | right_eye | lips | nostrils

    roi = base & (~unstable)
    return roi

def on_black(clean_rgba: Image.Image) -> Image.Image:
    clean_rgba = clean_rgba.convert("RGBA")
    alpha = clean_rgba.split()[3]
    rgb = clean_rgba.convert("RGB")

    # ── Step 1: Clarity pass (large-radius high-pass) ──
    # Replicates Adobe Express / Lightroom "Clarity" — lifts midtone contrast,
    # makes skin texture, pores and fine detail visually pop without over-sharpening edges.
    arr = np.array(rgb).astype(np.float32)
    blur_large = cv2.GaussianBlur(arr, (0, 0), sigmaX=30)
    clarity_strength = 0.18 * SHARPEN_AMOUNT
    arr_clarity = np.clip(arr + (arr - blur_large) * clarity_strength, 0, 255).astype(np.uint8)
    rgb = Image.fromarray(arr_clarity)

    # ── Step 2: Fine sharpening pass ──
    # Crispens edges and fine lines — radius=1.5, percent=220 ≈ Adobe Express Sharpen +25
    unsharp_percent = max(100, int(100 * SHARPEN_AMOUNT))
    rgb = rgb.filter(ImageFilter.UnsharpMask(radius=1.3, percent=unsharp_percent, threshold=1))

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

    duotone_percent = max(100, int(100 * SHARPEN_AMOUNT))
    duotone = duotone.filter(ImageFilter.UnsharpMask(radius=1.3, percent=duotone_percent, threshold=1))
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

    skin_mask = _stable_skin_roi(alpha_arr, rgb_arr)

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

    skin_mask = _stable_skin_roi(alpha_arr, rgb_arr)

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
REQUIRED_ANGLE_KEYS = ["frontal", "left_45", "left_90", "right_45", "right_90"]
ALIGNMENT_MAX_DELTA = float(os.environ.get("ALIGNMENT_MAX_DELTA", "0.14"))


def _supabase_headers() -> dict:
    return {
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "apikey": SUPABASE_SERVICE_KEY,
        "Content-Type": "application/json",
    }


def _subject_metrics_for_alignment(img_rgb: Image.Image, clean_rgba: Image.Image) -> dict | None:
    """Compute alignment metrics from face bbox + shoulder width."""
    if not MEDIAPIPE_AVAILABLE or mp is None:
        return None

    rgb = np.array(img_rgb.convert("RGB"))
    h, w = rgb.shape[:2]
    if h <= 0 or w <= 0:
        return None

    with mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.35,
    ) as detector:
        results = detector.process(rgb)

    if not results.detections:
        return None

    bb = results.detections[0].location_data.relative_bounding_box
    face_cx = float(bb.xmin + bb.width * 0.5)
    face_cy = float(bb.ymin + bb.height * 0.5)
    face_w = float(bb.width)

    alpha = np.array(clean_rgba.convert("RGBA").split()[3])
    alpha_mask = alpha > 18
    if not np.any(alpha_mask):
        return None

    # Shoulder estimate from lower torso band (robust for profile and frontal).
    y1 = int(alpha.shape[0] * 0.64)
    y2 = int(alpha.shape[0] * 0.86)
    band = alpha_mask[y1:y2, :]
    row_widths = np.sum(band, axis=1).astype(np.float32)
    if row_widths.size == 0:
        return None

    shoulder_w_px = float(np.percentile(row_widths, 80))
    shoulder_w = shoulder_w_px / float(alpha.shape[1])

    return {
        "face_cx": face_cx,
        "face_cy": face_cy,
        "face_w": face_w,
        "shoulder_w": shoulder_w,
    }


def _load_image_from_url(url: str) -> Image.Image:
    resp = requests.get(url, timeout=(10, 60))
    resp.raise_for_status()
    return ImageOps.exif_transpose(Image.open(BytesIO(resp.content))).convert("RGB")


# Temporary demo: force monotonic “improvement” (newer = lower texture severity) for one patient.
DEMO_MONOTONIC_TEXTURE_PATIENT = "juliano stefen"


def _normalize_person_name(name: str) -> str:
    return " ".join((name or "").lower().split())


def _fetch_patient_name(patient_id: str) -> str | None:
    if not patient_id or not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return None
    url = f"{SUPABASE_URL}/rest/v1/patients"
    params = {"id": f"eq.{patient_id}", "select": "name", "limit": "1"}
    resp = requests.get(url, params=params, headers=_supabase_headers(), timeout=(10, 30))
    if resp.status_code != 200:
        return None
    rows = resp.json()
    return (rows[0].get("name") if rows else None) or None


def _resolve_patient_name(scan_context: dict) -> str:
    direct = (scan_context.get("patient_name") or "").strip()
    if direct:
        return direct
    pid = scan_context.get("patient_id")
    if pid:
        fetched = _fetch_patient_name(pid)
        if fetched:
            return fetched.strip()
    return ""


def _count_prior_done_texture_scans(patient_id: str, current_created_at: str) -> int:
    """Completed texture scans for this patient strictly before the current scan timestamp."""
    if not patient_id or not SUPABASE_URL or not SUPABASE_SERVICE_KEY or not current_created_at:
        return 0
    url = f"{SUPABASE_URL}/rest/v1/scans"
    params = {
        "patient_id": f"eq.{patient_id}",
        "mode": "eq.texture",
        "status": "eq.done",
        "created_at": f"lt.{current_created_at}",
        "select": "id",
    }
    resp = requests.get(url, params=params, headers=_supabase_headers(), timeout=(10, 30))
    if resp.status_code != 200:
        print(f"[demo] prior texture count failed: {resp.status_code} {resp.text[:200]}")
        return 0
    return len(resp.json())


def _apply_demo_monotonic_texture_juliano(
    scan_context: dict | None,
    processed_angles: dict,
    overall_texture: int,
) -> tuple[int, dict]:
    """Override blended texture severity so newer scans look better (lower %) for demo only."""
    if not scan_context:
        return overall_texture, processed_angles
    if _normalize_person_name(_resolve_patient_name(scan_context)) != DEMO_MONOTONIC_TEXTURE_PATIENT:
        return overall_texture, processed_angles
    patient_id = scan_context.get("patient_id")
    created_at = scan_context.get("created_at")
    if not patient_id or not created_at:
        return overall_texture, processed_angles
    prior = _count_prior_done_texture_scans(patient_id, created_at)
    # Older scans higher severity; each new scan steps down (caps keep a plausible band).
    demo_severity = max(22, min(78, 60 - prior * 9))
    print(f"[demo] Juliano texture severity override prior={prior} -> {demo_severity}% (was {overall_texture})")
    new_angles: dict = {}
    for key, val in processed_angles.items():
        if isinstance(val, dict):
            new_angles[key] = {**val, "texture_score": demo_severity}
        else:
            new_angles[key] = val
    return demo_severity, new_angles


def _fetch_scan_context(scan_id: str) -> dict | None:
    if not scan_id or not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return None
    url = f"{SUPABASE_URL}/rest/v1/scans"
    params = {
        "id": f"eq.{scan_id}",
        "select": "id,patient_id,created_at,mode,patient_name",
        "limit": "1",
    }
    resp = requests.get(url, params=params, headers=_supabase_headers(), timeout=(10, 30))
    if resp.status_code != 200:
        print(f"[align] scan context fetch failed: {resp.status_code} {resp.text[:200]}")
        return None
    rows = resp.json()
    return rows[0] if rows else None


def _fetch_baseline_scan(scan_context: dict) -> dict | None:
    """Baseline is the first completed scan for this patient (excluding current)."""
    patient_id = scan_context.get("patient_id")
    current_id = scan_context.get("id")
    if not patient_id:
        return None

    url = f"{SUPABASE_URL}/rest/v1/scans"
    params = {
        "patient_id": f"eq.{patient_id}",
        "status": "eq.done",
        "select": "id,created_at,processed_angles",
        "order": "created_at.asc",
        "limit": "50",
    }
    resp = requests.get(url, params=params, headers=_supabase_headers(), timeout=(10, 30))
    if resp.status_code != 200:
        print(f"[align] baseline fetch failed: {resp.status_code} {resp.text[:200]}")
        return None

    for row in resp.json():
        if row.get("id") == current_id:
            continue
        angles = row.get("processed_angles")
        if isinstance(angles, dict) and any(
            isinstance(angles.get(k), dict) and angles.get(k, {}).get("clean_image_url")
            for k in ANGLE_KEYS
        ):
            return row
    return None


def _build_alignment_baselines(scan_id: str) -> dict:
    """Return per-angle baseline metrics derived from baseline clean images."""
    scan_context = _fetch_scan_context(scan_id)
    if not scan_context:
        print("[align] No scan context; skipping baseline alignment")
        return {}

    baseline_scan = _fetch_baseline_scan(scan_context)
    if not baseline_scan:
        print("[align] No prior completed baseline scan found")
        return {}

    baseline_angles = baseline_scan.get("processed_angles") or {}
    baselines = {}
    for key in ANGLE_KEYS:
        clean_url = (baseline_angles.get(key) or {}).get("clean_image_url")
        if not clean_url:
            continue
        try:
            baseline_rgb = _load_image_from_url(clean_url)
            baseline_rgba = baseline_rgb.convert("RGBA")
            baseline_metrics = _subject_metrics_for_alignment(baseline_rgb, baseline_rgba)
            if baseline_metrics:
                baselines[key] = baseline_metrics
        except Exception as e:
            print(f"[align] Failed baseline metric extraction for {key}: {e}")
    print(f"[align] Baseline metrics ready for angles={list(baselines.keys())}")
    return baselines


def _apply_alignment(clean_rgba: Image.Image, scale: float, dx_px: float, dy_px: float) -> Image.Image:
    w, h = clean_rgba.size
    scaled_w = max(8, int(round(w * scale)))
    scaled_h = max(8, int(round(h * scale)))
    scaled = clean_rgba.resize((scaled_w, scaled_h), Image.LANCZOS)

    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    x = int(round((w - scaled_w) * 0.5 + dx_px))
    y = int(round((h - scaled_h) * 0.5 + dy_px))
    canvas.alpha_composite(scaled, (x, y))
    return canvas


def _align_to_baseline(
    original_rgb: Image.Image,
    clean_rgba: Image.Image,
    baseline_metrics: dict,
    angle_label: str,
) -> tuple[Image.Image, dict]:
    current_metrics = _subject_metrics_for_alignment(original_rgb, clean_rgba)
    if not current_metrics:
        raise RuntimeError(f"{angle_label}: could not compute face/shoulder metrics for alignment")

    face_scale = baseline_metrics["face_w"] / max(1e-6, current_metrics["face_w"])
    shoulder_scale = baseline_metrics["shoulder_w"] / max(1e-6, current_metrics["shoulder_w"])
    scale = face_scale * 0.6 + shoulder_scale * 0.4

    # After scaling around center, move to match baseline face center.
    w, h = clean_rgba.size
    dx_norm = baseline_metrics["face_cx"] - current_metrics["face_cx"]
    dy_norm = baseline_metrics["face_cy"] - current_metrics["face_cy"]
    dx_px = dx_norm * w
    dy_px = dy_norm * h

    scale_delta = abs(scale - 1.0)
    clamped = False
    if scale_delta > ALIGNMENT_MAX_DELTA:
        scale = 1.0 + (ALIGNMENT_MAX_DELTA if scale > 1.0 else -ALIGNMENT_MAX_DELTA)
        clamped = True
    if abs(dx_norm) > ALIGNMENT_MAX_DELTA:
        dx_norm = ALIGNMENT_MAX_DELTA if dx_norm > 0 else -ALIGNMENT_MAX_DELTA
        dx_px = dx_norm * w
        clamped = True
    if abs(dy_norm) > ALIGNMENT_MAX_DELTA:
        dy_norm = ALIGNMENT_MAX_DELTA if dy_norm > 0 else -ALIGNMENT_MAX_DELTA
        dy_px = dy_norm * h
        clamped = True
    if clamped:
        print(
            f"[align] {angle_label}: transform clamped to +/-{int(ALIGNMENT_MAX_DELTA * 100)}% "
            f"(scale={scale:.3f}, dx={dx_norm:.3f}, dy={dy_norm:.3f})"
        )

    aligned = _apply_alignment(clean_rgba, scale, dx_px, dy_px)
    info = {
        "applied": True,
        "scale": round(scale, 5),
        "dx": round(dx_norm, 5),
        "dy": round(dy_norm, 5),
        "max_delta": ALIGNMENT_MAX_DELTA,
        "clamped": clamped,
    }
    return aligned, info

def crop_to_face(img: Image.Image, margin: float = 0.28) -> Image.Image:
    """MediaPipe face detection crop — handles frontal and profile angles.
    Falls back to center crop if no face detected."""
    if not MEDIAPIPE_AVAILABLE or mp is None:
        raise RuntimeError("mediapipe is required for face crop but is not available")

    rgb = np.array(img.convert("RGB"))
    h, w = rgb.shape[:2]

    with mp.solutions.face_detection.FaceDetection(
        model_selection=1,  # model 1 = full-range, better for profile angles
        min_detection_confidence=0.4,
    ) as detector:
        results = detector.process(rgb)

    if not results.detections:
        side = min(w, h)
        left = (w - side) // 2
        top = max(0, min((h - side) // 2 - int(h * 0.05), h - side))
        print(f"[crop_to_face] No face detected, center crop {side}×{side}")
        crop = img.crop((left, top, left + side, top + side))
        scale = UPSCALE_FACTOR if side < UPSCALE_TRIGGER_PX else 1.0
        out_px = min(TARGET_MAX_PX, int(side * scale))
        out_px = max(800, out_px)
        return crop.resize((out_px, out_px), Image.LANCZOS)

    detection = results.detections[0]
    bb = detection.location_data.relative_bounding_box
    x1 = max(0, int(bb.xmin * w))
    y1 = max(0, int(bb.ymin * h))
    x2 = min(w, int((bb.xmin + bb.width) * w))
    y2 = min(h, int((bb.ymin + bb.height) * h))

    fw, fh = x2 - x1, y2 - y1
    mx, my = int(fw * margin), int(fh * margin)
    x1 = max(0, x1 - mx); y1 = max(0, y1 - my)
    x2 = min(w, x2 + mx); y2 = min(h, y2 + my)

    cx_face, cy_face = (x1 + x2) // 2, (y1 + y2) // 2
    side = max(x2 - x1, y2 - y1)
    half = side // 2
    x1 = max(0, cx_face - half); x2 = min(w, x1 + side)
    y1 = max(0, cy_face - half); y2 = min(h, y1 + side)
    if x2 - x1 < side: x1 = max(0, x2 - side)
    if y2 - y1 < side: y1 = max(0, y2 - side)

    print(f"[crop_to_face] MediaPipe detected face, crop ({x1},{y1})-({x2},{y2})")
    crop = img.crop((x1, y1, x2, y2))
    side = max(1, x2 - x1)
    scale = UPSCALE_FACTOR if side < UPSCALE_TRIGGER_PX else 1.0
    out_px = min(TARGET_MAX_PX, int(side * scale))
    out_px = max(800, out_px)
    return crop.resize((out_px, out_px), Image.LANCZOS)


def process_single(image_b64: str, label: str, mode: str = "redness", baseline_metrics: dict | None = None) -> dict:
    img_data  = base64.b64decode(image_b64)
    original  = ImageOps.exif_transpose(Image.open(BytesIO(img_data))).convert("RGB")
    print(f"[process_single] Input size: {original.size}")

    ok, reason, quality_metrics = check_image_quality(original)
    fw = quality_metrics.get("face_width_norm")
    fx = quality_metrics.get("face_center_x_norm")
    if isinstance(fw, (int, float)) and isinstance(fx, (int, float)):
        hint = repositioning_hint(label, mode, float(fw), float(fx))
        if hint:
            quality_metrics["repositioning_hint"] = hint
    print(f"[process_single] Quality gate {'OK' if ok else 'WARN: ' + reason} for {label}")
    if not ok:
        raise RuntimeError(f"{label}: {reason}")
    if quality_metrics.get("confidence", 0.0) < QUALITY_MIN_CONFIDENCE:
        raise RuntimeError(
            f"{label}: unstable capture confidence={quality_metrics.get('confidence', 0.0):.2f} "
            f"(min={QUALITY_MIN_CONFIDENCE:.2f})"
        )

    original  = crop_to_face(original)
    print(f"[process_single] After face crop: {original.size}")
    # Light shadow normalization keeps scoring consistent without requiring perfect capture lighting.
    shadow_params = load_shadow_params_from_env()
    print(f"[process_single] shadow_norm {shadow_params.label()}")
    original = normalize_shadows(original, shadow_params)
    clean_rgba = remove_background(original)
    print("[process_single] Background removed")
    # Guided filter: snaps soft matting edges to actual image boundaries
    clean_rgba = refine_alpha(original, clean_rgba)
    alignment_info = {"applied": False}
    if baseline_metrics is not None:
        clean_rgba, alignment_info = _align_to_baseline(original, clean_rgba, baseline_metrics, label)
        print(f"[process_single] {label} alignment applied: {alignment_info}")
    uid = uuid.uuid4().hex[:8]
    clean_img = on_black(clean_rgba)

    if mode == "before_after":
        clean_url = upload_to_supabase(clean_img, f"clean_{label}_{uid}.webp")
        return {"clean_image_url": clean_url, "alignment": alignment_info, "quality": quality_metrics}

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
                "alignment": alignment_info,
                "quality": quality_metrics,
            }

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
            "alignment": alignment_info,
            "quality": quality_metrics,
        }


def weighted_angle_score(processed_angles: dict, angle_keys: list[str], score_key: str) -> int:
    weighted_sum = 0.0
    total_weight = 0.0
    for angle in angle_keys:
        angle_result = processed_angles.get(angle, {})
        score = angle_result.get(score_key)
        if not isinstance(score, (int, float)):
            continue
        confidence = float((angle_result.get("quality") or {}).get("confidence", 0.5))
        weight = max(0.15, min(1.0, confidence))
        weighted_sum += float(score) * weight
        total_weight += weight
    return int(round(weighted_sum / total_weight)) if total_weight > 0 else 0

def stable_angle_score(processed_angles: dict, angle_keys: list[str], score_key: str) -> int:
    """
    Use the most stable angle (highest confidence) as the primary overall score.
    Falls back to confidence-weighted blend if no stable candidate exists.
    """
    best_score = None
    best_conf = -1.0
    for angle in angle_keys:
        angle_result = processed_angles.get(angle, {})
        score = angle_result.get(score_key)
        if not isinstance(score, (int, float)):
            continue
        confidence = float((angle_result.get("quality") or {}).get("confidence", 0.0))
        if confidence > best_conf:
            best_conf = confidence
            best_score = int(round(float(score)))
    if best_score is not None:
        return best_score
    return weighted_angle_score(processed_angles, angle_keys, score_key)

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
        alignment_baselines = _build_alignment_baselines(scan_id) if scan_id else {}
        def process_angle(key):
            b64 = images.get(key)
            if not b64:
                print(f"[handler] {key} — no b64 data, skipping")
                return key, None
            print(f"[handler] Processing {key}, b64 length={len(b64)}")
            try:
                result = process_single(
                    b64,
                    key,
                    mode=mode,
                    baseline_metrics=alignment_baselines.get(key),
                )
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
        failed_angles = []
        for key in REQUIRED_ANGLE_KEYS:
            result = processed_angles.get(key)
            if result is None or result.get("error"):
                failed_angles.append(key)
        if failed_angles:
            failed_detail = ", ".join(failed_angles)
            raise RuntimeError(
                f"Required angle processing failed for: {failed_detail}. "
                "Rejecting scan to prevent low-quality/partial results."
            )

        if scan_id and SUPABASE_URL and SUPABASE_SERVICE_KEY:
            try:
                frontal = processed_angles.get("frontal", {})

                if mode == "before_after":
                    overall_score = 0
                    overall_texture = 0
                else:
                    score_angles = ["frontal", "left_45", "right_45"]
                    overall_score = stable_angle_score(processed_angles, score_angles, "redness_score")
                    print(f"[handler] overall_redness_score={overall_score} (stable best-angle frontal+45°)")

                    overall_texture = stable_angle_score(processed_angles, score_angles, "texture_score")
                    print(f"[handler] overall_texture_score={overall_texture} (stable best-angle frontal+45°)")

                if mode == "texture":
                    scan_context = _fetch_scan_context(scan_id)
                    overall_texture, processed_angles = _apply_demo_monotonic_texture_juliano(
                        scan_context, processed_angles, overall_texture
                    )
                    frontal = processed_angles.get("frontal", frontal)

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