import base64
import os
import uuid
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import requests
from PIL import Image, ImageFilter, ImageOps

import clipseg_cheek

# Register the HEIF/HEIC opener with Pillow so Image.open() auto-detects HEIC
# bytes (the format the iOS app uploads). Pillow's auto-detect reads the file
# header, so no other code in this module needs to know the input format.
from pillow_heif import register_heif_opener

register_heif_opener()

try:
    import runpod
except ImportError:  # Allows local helper tests without RunPod installed.
    runpod = None


SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "scans")

ANGLE_KEYS = ["frontal", "left_45", "left_90", "right_45", "right_90"]
# Redness scoring: right profile only (single source of truth for severity).
ANALYSIS_ANGLES = ("right_90",)
PRIMARY_ANALYSIS_ANGLE = "right_90"
PORTRAIT_WIDTH = 2160
PORTRAIT_HEIGHT = 2700
ANALYSIS_CROP_SIZE = 1000

# ---------------------------------------------------------------------------
# MODNet portrait matting (the simple, fast, May-4 golden pipeline)
# ---------------------------------------------------------------------------
# Single-image ONNX model, no recurrent state, no aux inputs. Each angle is
# independent. Inference at 512 px long edge runs ~250 ms per angle on a
# typical RunPod CPU pod, and the matte is then refined in two passes:
#   1. guided filter against the source luma  -> snaps to real image edges
#   2. studio finish (Adobe-Express style)    -> closes pinholes, kills
#      outer halo glow, smoothstep edge curve, sub-pixel feather
MODNET_MODEL_PATH = os.environ.get(
    "MODNET_MODEL_PATH", "/root/.modnet/modnet_photographic.onnx"
)
MODNET_INPUT_SIZE = int(os.environ.get("MODNET_INPUT_SIZE", "512"))
# Clarity/sharpen strength for the final clean image. 1.15 matches May-4:
# subtle midtone clarity + UnsharpMask makes pores/skin texture pop without
# halos around facial features.
SHARPEN_AMOUNT = float(os.environ.get("SHARPEN_AMOUNT", "1.15"))


def _init_matting():
    model_path = Path(MODNET_MODEL_PATH)
    if not model_path.exists() or model_path.stat().st_size == 0:
        print(f"[matting] MODNet not found at {model_path}; background removal DISABLED")
        return None
    try:
        opts = ort.SessionOptions()
        # Saturate the CPU on serial inference. We never run MODNet from
        # multiple threads simultaneously, so giving ORT all cores is safe.
        opts.intra_op_num_threads = max(1, (os.cpu_count() or 4))
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(
            str(model_path), opts, providers=["CPUExecutionProvider"]
        )
        print(
            f"[matting] MODNet session initialized providers={sess.get_providers()} "
            f"inputs={[i.name for i in sess.get_inputs()]}"
        )
        return sess
    except Exception as exc:  # noqa: BLE001
        print(f"[matting] Init failed, background removal DISABLED: {exc}")
        return None


MATTING_SESSION = _init_matting()

# ---------------------------------------------------------------------------
# Cheek ROI: CLIPSeg text prompt @ 352px → ear removal → BONE tight crop
# ---------------------------------------------------------------------------
FACE_LANDMARKER_PATH = os.environ.get(
    "FACE_LANDMARKER_PATH", "/root/.mediapipe/face_landmarker.task"
)
LANDMARK_LONG_EDGE = int(os.environ.get("LANDMARK_LONG_EDGE", "960"))
CHEEK_TIGHT_PADDING = int(os.environ.get("CHEEK_TIGHT_PADDING", "24"))
CLIPSEG_CHEEK_PROMPT = os.environ.get(
    "CLIPSEG_CHEEK_PROMPT", "cheek and jaw skin texture"
)
CLIPSEG_EAR_PROMPT = os.environ.get("CLIPSEG_EAR_PROMPT", "ear")
CLIPSEG_EAR_THRESHOLD = float(os.environ.get("CLIPSEG_EAR_THRESHOLD", "0.35"))
# Subtract eye/nose/mouth blobs (comma-separated CLIPSeg prompts).
CLIPSEG_EXCLUDE_PROMPTS = os.environ.get("CLIPSEG_EXCLUDE_PROMPTS", "eye,nose,mouth")
CLIPSEG_EXCLUDE_THRESHOLD = float(os.environ.get("CLIPSEG_EXCLUDE_THRESHOLD", "0.38"))
# Profile ear strip: keep left fraction of cheek mask span (right_90).
PROFILE_EAR_X_KEEP = float(os.environ.get("PROFILE_EAR_X_KEEP", "0.68"))
# Sharp angular polygon from largest fragment (lower = more corners).
POLYGON_EPSILON_FRAC = float(os.environ.get("POLYGON_EPSILON_FRAC", "0.014"))

# Warm CLIPSeg weights at import (long first cold start moved to image build).
clipseg_cheek._load()


def _init_face_landmarker():
    model_path = Path(FACE_LANDMARKER_PATH)
    if not model_path.exists() or model_path.stat().st_size == 0:
        print(f"[landmarks] model not found at {model_path}; cheek ROI uses alpha fallback")
        return None
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core import base_options as mp_base

        opts = vision.FaceLandmarkerOptions(
            base_options=mp_base.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
        )
        landmarker = vision.FaceLandmarker.create_from_options(opts)
        print("[landmarks] FaceLandmarker session OK")
        return landmarker
    except Exception as exc:  # noqa: BLE001
        print(f"[landmarks] init failed: {exc}")
        return None


FACE_LANDMARKER = _init_face_landmarker()
_MP_IMAGE_FORMAT = None


def _mp_image(rgb: np.ndarray):
    global _MP_IMAGE_FORMAT
    import mediapipe as mp

    if _MP_IMAGE_FORMAT is None:
        _MP_IMAGE_FORMAT = mp.ImageFormat.SRGB
    return mp.Image(image_format=_MP_IMAGE_FORMAT, data=np.ascontiguousarray(rgb))


def detect_face_landmarks(rgb: np.ndarray):
    """478 landmarks in full-image pixel coords, or None."""
    if FACE_LANDMARKER is None:
        return None
    h, w = rgb.shape[:2]
    scale = LANDMARK_LONG_EDGE / float(max(h, w))
    if scale < 1.0:
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        small = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    else:
        small = rgb
    try:
        result = FACE_LANDMARKER.detect(_mp_image(small))
    except Exception as exc:  # noqa: BLE001
        print(f"[landmarks] detect failed: {exc}")
        return None
    if not result.face_landmarks:
        return None
    return result.face_landmarks[0]


def _eroded_alpha_u8(alpha: np.ndarray, h: int, w: int) -> np.ndarray:
    erode_px = max(12, int(min(h, w) * 0.04))
    return cv2.erode(
        alpha.astype(np.uint8),
        np.ones((erode_px, erode_px), np.uint8),
        iterations=1,
    )


def remove_ear_from_mask(
    mask: np.ndarray,
    clean_rgb: np.ndarray,
    landmarks,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Step 2: subtract ear CLIPSeg + profile geometry + chin cap."""
    h, w = height, width
    out = mask.copy()

    # Text prompt: remove ear pixels CLIPSeg finds.
    ear_mask = clipseg_cheek.predict_mask(
        clean_rgb, CLIPSEG_EAR_PROMPT, threshold=CLIPSEG_EAR_THRESHOLD
    )
    if ear_mask is not None:
        out &= ~ear_mask

    # right_90 profile: ear sits on the high-x side of the face span.
    if np.any(out):
        cols = np.where(np.any(out, axis=0))[0]
        if cols.size:
            x_cut = int(cols[0] + (cols[-1] - cols[0]) * PROFILE_EAR_X_KEEP)
            out &= np.arange(w, dtype=np.int32) <= x_cut

    if landmarks is not None and len(landmarks) > 152:
        chin_y = int(landmarks[152].y * h) + int(0.04 * h)
        out[chin_y:, :] = False
        out_u8 = (out.astype(np.uint8) * 255)
        for ear_idx in (454, 361, 288, 234):
            if ear_idx < len(landmarks):
                ex = int(landmarks[ear_idx].x * w)
                ey = int(landmarks[ear_idx].y * h)
                r = max(12, int(0.05 * min(w, h)))
                cv2.circle(out_u8, (ex, ey), r, 0, -1)
        out = out_u8 > 127

    return out.astype(bool)


def _exclude_eyes_nose_mouth(
    mask: np.ndarray,
    clean_rgb: np.ndarray,
    landmarks,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Remove eyes, nose, mouth via CLIPSeg subtract + landmark caps."""
    h, w = height, width
    out = mask.copy()
    for prompt in [p.strip() for p in CLIPSEG_EXCLUDE_PROMPTS.split(",") if p.strip()]:
        ex = clipseg_cheek.predict_mask(
            clean_rgb, prompt, threshold=CLIPSEG_EXCLUDE_THRESHOLD
        )
        if ex is not None:
            out &= ~ex

    if landmarks is not None:
        eye_ids = (33, 133, 159, 145, 263, 362, 386, 374, 249, 390)
        eye_ys = [landmarks[i].y * h for i in eye_ids if i < len(landmarks)]
        if eye_ys:
            out[: int(max(eye_ys) + 0.02 * h), :] = False
        if len(landmarks) > 1:
            nx = int(landmarks[1].x * w)
            band = int(0.07 * w)
            out[:, max(0, nx - band) : min(w, nx + band)] = False
    return out.astype(bool)


def _keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the biggest connected region (drop all smaller fragments)."""
    u8 = (mask.astype(np.uint8) * 255)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(u8, connectivity=8)
    if n_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    if areas.size == 0:
        return mask
    best = 1 + int(np.argmax(areas))
    return labels == best


def _angular_polygon_mask(mask: np.ndarray) -> np.ndarray:
    """Single irregular polygon with sharp edges (torn-fragment look)."""
    h, w = mask.shape[:2]
    u8 = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    eps = max(2.0, POLYGON_EPSILON_FRAC * peri)
    approx = cv2.approxPolyDP(cnt, eps, True)
    if len(approx) < 3:
        return mask
    poly = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(poly, [approx], 255)
    return poly > 0


def extract_cheek_tight_bone(
    bone_rgb: np.ndarray,
    clean_rgb: np.ndarray,
    alpha: np.ndarray | None,
    landmarks,
) -> tuple[Image.Image, str, int, dict]:
    """
    Gemini-style fragment: CLIPSeg cheek+jaw → exclude features → ear cut →
    largest blob only → sharp angular polygon → BONE duotone tight crop.
    """
    h, w = bone_rgb.shape[:2]
    timing: dict = {}

    t0 = datetime.now(timezone.utc)
    mask = clipseg_cheek.predict_mask(clean_rgb, CLIPSEG_CHEEK_PROMPT)
    timing["clipseg_ms"] = int((datetime.now(timezone.utc) - t0).total_seconds() * 1000)
    method = "clipseg_angular_fragment_bone"

    if mask is None or not np.any(mask):
        method = "clipseg_fallback_alpha"
        if alpha is None:
            return Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8)), method, 0, timing
        eroded = _eroded_alpha_u8(alpha, h, w)
        mask = eroded > 100
        cols = np.where(np.any(mask, axis=0))[0]
        if cols.size:
            x_cut = int(cols[0] + (cols[-1] - cols[0]) * PROFILE_EAR_X_KEEP)
            mask &= np.arange(w) <= x_cut
    else:
        t1 = datetime.now(timezone.utc)
        mask = _exclude_eyes_nose_mouth(mask, clean_rgb, landmarks, width=w, height=h)
        timing["exclude_features_ms"] = int(
            (datetime.now(timezone.utc) - t1).total_seconds() * 1000
        )
        t2 = datetime.now(timezone.utc)
        mask = remove_ear_from_mask(mask, clean_rgb, landmarks, width=w, height=h)
        timing["ear_removal_ms"] = int(
            (datetime.now(timezone.utc) - t2).total_seconds() * 1000
        )

    if alpha is not None:
        mask &= _eroded_alpha_u8(alpha, h, w) > 100

    mask_u8 = (mask.astype(np.uint8) * 255)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = mask_u8 > 127

    t3 = datetime.now(timezone.utc)
    mask = _keep_largest_component(mask)
    mask = _angular_polygon_mask(mask)
    timing["fragment_ms"] = int((datetime.now(timezone.utc) - t3).total_seconds() * 1000)

    if not np.any(mask):
        return Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8)), method, 0, timing

    ys, xs = np.where(mask)
    pad = CHEEK_TIGHT_PADDING
    x1 = max(0, int(xs.min()) - pad)
    y1 = max(0, int(ys.min()) - pad)
    x2 = min(w, int(xs.max()) + pad + 1)
    y2 = min(h, int(ys.max()) + pad + 1)

    crop = bone_rgb[y1:y2, x1:x2].copy()
    mask_crop = mask[y1:y2, x1:x2]
    crop[~mask_crop] = 0
    pixels = int(mask_crop.sum())
    return Image.fromarray(crop, mode="RGB"), method, pixels, timing


def _modnet_target_size(width: int, height: int) -> tuple[int, int]:
    """Resize to MODNET_INPUT_SIZE long edge, with both sides multiples of
    32 so MODNet's 5-stage downsample inside the network halves cleanly."""
    scale = MODNET_INPUT_SIZE / float(max(width, height))
    tw = max(32, int(round(width * scale)))
    th = max(32, int(round(height * scale)))
    tw -= tw % 32
    th -= th % 32
    return tw, th


def run_matting(img_rgb: Image.Image):
    """Run MODNet on a single image. Returns uint8 alpha at source resolution,
    or None if matting is disabled.

    MODNet's ONNX export takes one input (the RGB image as NCHW float32 in
    [-1, 1]) and returns the alpha matte. No recurrent state, no aux inputs.
    """
    if MATTING_SESSION is None:
        return None

    rgb = np.array(img_rgb.convert("RGB"))
    h, w = rgb.shape[:2]
    tw, th = _modnet_target_size(w, h)
    resized = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA)

    # NCHW float32 in [-1, 1] — MODNet's normalization.
    src = (resized.astype(np.float32) - 127.5) / 127.5
    src = np.ascontiguousarray(np.transpose(src, (2, 0, 1))[None, ...])

    input_name = MATTING_SESSION.get_inputs()[0].name
    output_name = MATTING_SESSION.get_outputs()[0].name
    outputs = MATTING_SESSION.run([output_name], {input_name: src})
    alpha = outputs[0][0, 0]  # (th, tw) float in [0, 1]
    alpha_full = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(alpha_full * 255.0, 0, 255).astype(np.uint8)


def _guided_filter_alpha(
    guide_gray: np.ndarray, alpha: np.ndarray, radius: int = 6, eps: float = 1e-3
) -> np.ndarray:
    """Edge-aware refinement: snaps soft matte boundaries to real image edges.

    Radius 6 matches the May-4 MODNet pipeline — wide enough to let the
    bilinear-upsampled matte snap cleanly to source-image edges without
    over-smoothing fine hair detail.
    """

    def box(I: np.ndarray, r: int) -> np.ndarray:
        return cv2.boxFilter(I.astype(np.float32), -1, (2 * r + 1, 2 * r + 1))

    I = guide_gray.astype(np.float32) / 255.0
    p = alpha.astype(np.float32) / 255.0

    mean_I = box(I, radius)
    mean_p = box(p, radius)
    cov_Ip = box(I * p, radius) - mean_I * mean_p
    var_I = box(I * I, radius) - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    q = box(a, radius) * I + box(b, radius)
    return np.clip(q * 255.0, 0, 255).astype(np.uint8)


def _composite_on_black(rgb: np.ndarray, alpha: np.ndarray) -> Image.Image:
    """Multiply RGB by alpha to get a clean image on a black background."""
    a = alpha.astype(np.float32) / 255.0
    composed = (rgb.astype(np.float32) * a[..., None]).astype(np.uint8)
    return Image.fromarray(composed, mode="RGB")


def _studio_finish_alpha(alpha: np.ndarray) -> np.ndarray:
    """Second-pass cleanup on the alpha matte — Adobe Express "auto" style.

    The matting model gives a usable matte, but for a photo-studio
    pure-black-background look we also want:
      - no pinholes inside the foreground (close 1-2 px holes)
      - no specks floating outside the subject (open small islands)
      - no outer halo glow (kill very low alpha values)
      - a tight but smooth edge transition (smoothstep curve + soft feather)
    All ops are O(N) numpy/cv2 — total cost is ~20-40 ms at 2160x2700.
    """
    # 1. Morphological cleanup. 3x3 kernel preserves thin hair strands while
    #    closing 1-2 px pinholes inside the FG and removing matching specks
    #    outside it.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel, iterations=1)
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_OPEN, kernel, iterations=1)

    # 2. Contrast curve over the soft edge band. Anything below 20/255 is
    #    treated as background (kills outer haze), anything above 235/255
    #    stays full FG. The middle band gets a smoothstep so the transition
    #    tightens without becoming a hard cut — strands stay strands.
    af = alpha.astype(np.float32) / 255.0
    lo, hi = 20.0 / 255.0, 235.0 / 255.0
    af = np.where(af < lo, 0.0, af)
    af = np.clip((af - lo) / (hi - lo), 0.0, 1.0)
    af = af * af * (3.0 - 2.0 * af)  # smoothstep

    # 3. Sub-pixel feather so the final composite doesn't show pixel
    #    staircase on close inspection. sigma=0.8 is enough.
    alpha = (af * 255.0).astype(np.uint8)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=0.8)
    return alpha


# ---------------------------------------------------------------------------
# Editor refine pipeline (mode="refine")
# ---------------------------------------------------------------------------
# The initial scan optimizes for speed: MODNet at 512 px + a one-pass studio
# finish on the alpha matte. The editor refine job has a 30-second user
# budget and operates on the *already-processed* clean image. It does two
# things the initial scan doesn't:
#
#   1. Edge halo cleanup — detect light pixels at hair edges that survived
#      the first pass (color spill from the original background bleeding
#      through soft-alpha regions) and push them toward black. This is what
#      removes the "~0.1% white halo".
#   2. Standardization — find the face with OpenCV Haar cascades, translate
#      and scale the subject so the face lands at a canonical position on a
#      2160x2700 black canvas. Same face size / framing on every refine, so
#      cross-session comparisons line up.


# Canonical placement targets used by `standardize_to_canonical`. Face
# height = 45 % of frame height, face top = 20 % from frame top, face
# horizontally centered. Picked to leave forehead room without crowding
# the chin, matching the framing of a passport / studio portrait.
STD_FACE_HEIGHT_FRAC = float(os.environ.get("STD_FACE_HEIGHT_FRAC", "0.45"))
STD_FACE_TOP_FRAC = float(os.environ.get("STD_FACE_TOP_FRAC", "0.20"))


def refine_studio_quality(img_rgb: Image.Image) -> Image.Image:
    """Second-pass halo cleanup on an already-processed black-bg image.

    The processed image is `original_rgb * alpha` composited onto pure
    black. Hair edges with soft alpha + light source-background pixels
    leave a faint "white halo" — we detect those edge-band pixels and
    push them toward black without touching solid foreground.

    Pure numpy/OpenCV — ~80-150 ms per 2160x2700 angle.
    """
    arr = np.array(img_rgb.convert("RGB")).astype(np.float32)

    # 1. Re-estimate alpha from the displayed pixel value. Pure-black BG
    #    means `displayed.max() == 0` there; non-zero pixels are foreground
    #    weighted by the (lost) original alpha. We treat the max channel
    #    intensity / 255 as a proxy for alpha.
    alpha = arr.max(axis=2) / 255.0

    bulk_fg = alpha > 0.65
    if not np.any(bulk_fg):
        return img_rgb  # nothing meaningful to refine

    # 2. Find the median brightness of the bulk foreground. Edge pixels
    #    that are *brighter* than this are almost certainly background
    #    spill, not natural FG content.
    hsv = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    v = hsv[..., 2]
    fg_v_median = float(np.median(v[bulk_fg]))

    # 3. Identify edge band — alpha 0.05–0.65, the transition zone.
    edge_band = (alpha > 0.05) & (alpha < 0.65)
    if np.any(edge_band):
        too_bright = edge_band & (v > fg_v_median * 1.30)
        if np.any(too_bright):
            ratio = (fg_v_median / np.maximum(v, 1.0)).astype(np.float32)
            # Cap ratio so we never *brighten* a pixel.
            ratio = np.clip(ratio, 0.0, 1.0)
            scale = np.where(too_bright, ratio, 1.0).astype(np.float32)
            arr = arr * scale[..., None]

    # 4. Tighten the silhouette: re-apply a smoothstep on alpha (using the
    #    *refined* pixel intensities so any halo darkening propagates) and
    #    re-multiply. This pushes the very low alpha values to zero (kills
    #    any residual outer glow).
    alpha_new = arr.max(axis=2) / 255.0
    lo, hi = 0.05, 0.85
    af = np.clip((alpha_new - lo) / (hi - lo), 0.0, 1.0)
    af = af * af * (3.0 - 2.0 * af)
    arr = arr * af[..., None]

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")


# Haar cascades for face detection. Built into opencv-python-headless, so
# no extra Docker layer needed. Frontal cascade is more accurate but only
# works on near-frontal poses; the profile cascade handles the 90° angles.
try:
    _HAAR_DIR = cv2.data.haarcascades
    FACE_FRONTAL = cv2.CascadeClassifier(_HAAR_DIR + "haarcascade_frontalface_default.xml")
    FACE_PROFILE = cv2.CascadeClassifier(_HAAR_DIR + "haarcascade_profileface.xml")
    if FACE_FRONTAL.empty() or FACE_PROFILE.empty():
        print("[face] WARNING: Haar cascades empty; standardization will pass through")
        FACE_FRONTAL = FACE_PROFILE = None  # type: ignore
    else:
        print("[face] Haar cascades loaded (frontal + profile)")
except Exception as exc:  # noqa: BLE001
    FACE_FRONTAL = FACE_PROFILE = None  # type: ignore
    print(f"[face] Haar cascade init failed: {exc}")


def detect_face_bbox_fast(
    img_rgb: Image.Image, angle_label: str
) -> tuple[int, int, int, int] | None:
    """Haar on a downscaled frame — same bbox logic, ~4× faster on 2160px portraits."""
    w, h = img_rgb.size
    target = 1080
    if max(w, h) > target:
        scale = target / float(max(w, h))
        sw, sh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        small = img_rgb.resize((sw, sh), Image.Resampling.BILINEAR)
        bbox = detect_face_bbox(small, angle_label)
        if bbox is not None:
            x, y, bw, bh = bbox
            inv = 1.0 / scale
            return (
                int(round(x * inv)),
                int(round(y * inv)),
                max(1, int(round(bw * inv))),
                max(1, int(round(bh * inv))),
            )
    return detect_face_bbox(img_rgb, angle_label)


def detect_face_bbox(
    img_rgb: Image.Image, angle_label: str
) -> tuple[int, int, int, int] | None:
    """Return (x, y, w, h) of the largest detected face, or None.

    For 90° profile angles we try the profile cascade first; for everything
    else (frontal, 45°) we try the frontal cascade first. Either cascade
    can serve as a fallback. The face is searched against the grayscale
    version of the image.
    """
    if FACE_FRONTAL is None or FACE_PROFILE is None:
        return None

    gray = cv2.cvtColor(np.array(img_rgb.convert("RGB")), cv2.COLOR_RGB2GRAY)
    is_profile = "90" in angle_label
    primary = FACE_PROFILE if is_profile else FACE_FRONTAL
    fallback = FACE_FRONTAL if is_profile else FACE_PROFILE

    # Right 90° profiles: slightly looser Haar so we avoid center_fallback on cheek.
    min_neighbors = 4 if angle_label == "right_90" else 5
    min_size = (120, 120) if angle_label == "right_90" else (150, 150)
    for cascade in (primary, fallback):
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=min_neighbors,
            minSize=min_size,
        )
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            return int(x), int(y), int(w), int(h)
    return None


def standardize_to_canonical(
    img_rgb: Image.Image, face_bbox: tuple[int, int, int, int]
) -> Image.Image:
    """Translate + scale the cleaned subject onto a 2160x2700 black canvas
    so the face lands at a canonical position. Same framing every refine.

    Anchor: face is centered horizontally, face-top sits at
    STD_FACE_TOP_FRAC of the frame, face height = STD_FACE_HEIGHT_FRAC of
    the frame. We never crop the input image; if the scaled subject would
    fall outside the canvas, the canvas just shows black there.
    """
    fx, fy, fw, fh = face_bbox
    if fh <= 0:
        return img_rgb

    target_face_h = STD_FACE_HEIGHT_FRAC * PORTRAIT_HEIGHT
    scale = target_face_h / float(fh)

    new_w = max(1, int(round(img_rgb.width * scale)))
    new_h = max(1, int(round(img_rgb.height * scale)))
    scaled = img_rgb.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Where the face center should end up in the canvas.
    target_face_cx = PORTRAIT_WIDTH / 2.0
    target_face_top_y = STD_FACE_TOP_FRAC * PORTRAIT_HEIGHT
    target_face_cy = target_face_top_y + target_face_h / 2.0

    # Face center in the scaled image.
    scaled_face_cx = (fx + fw / 2.0) * scale
    scaled_face_cy = (fy + fh / 2.0) * scale

    paste_x = int(round(target_face_cx - scaled_face_cx))
    paste_y = int(round(target_face_cy - scaled_face_cy))

    canvas = Image.new("RGB", (PORTRAIT_WIDTH, PORTRAIT_HEIGHT), (0, 0, 0))
    canvas.paste(scaled, (paste_x, paste_y))
    return canvas


def _download_processed_url(url: str) -> Image.Image:
    """Download a previously-uploaded processed image from Supabase Storage.

    The bearer header works for both public and private buckets — public
    buckets just ignore it. Returns an EXIF-corrected RGB image.
    """
    headers = {}
    if SUPABASE_SERVICE_KEY:
        headers["Authorization"] = f"Bearer {SUPABASE_SERVICE_KEY}"
    resp = requests.get(url, headers=headers, timeout=90)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Refine download failed: {resp.status_code} {url[:120]}"
        )
    return ImageOps.exif_transpose(Image.open(BytesIO(resp.content))).convert("RGB")


def _clarity_and_sharpen(img: Image.Image) -> Image.Image:
    """May-4 finish: subtle Clarity (large-radius high-pass) + UnsharpMask.

    Clarity lifts midtone contrast so pores/texture pop without crunching
    edges; UnsharpMask then crispens fine lines. Strengths come from the
    May-4 golden pipeline and produce the same skin look across all angles.
    """
    arr = np.array(img.convert("RGB")).astype(np.float32)
    blur_large = cv2.GaussianBlur(arr, (0, 0), sigmaX=30)
    clarity_strength = 0.18 * SHARPEN_AMOUNT
    arr_clarity = np.clip(
        arr + (arr - blur_large) * clarity_strength, 0, 255
    ).astype(np.uint8)
    out = Image.fromarray(arr_clarity)
    unsharp_percent = max(100, int(100 * SHARPEN_AMOUNT))
    return out.filter(
        ImageFilter.UnsharpMask(radius=1.3, percent=unsharp_percent, threshold=1)
    )


def remove_background_and_finish(img_rgb: Image.Image):
    """Simple flow: MODNet -> guided filter -> studio finish -> composite -> sharpen.

    Returns (finished_rgb, alpha_uint8 or None). If matting is unavailable
    we degrade gracefully and only apply the Clarity/Sharpen finish on the
    untouched RGB image so the pipeline still produces output.

    Pipeline stages:
      1. MODNet            -> raw alpha matte at source resolution
      2. Guided filter     -> alpha snaps to real image edges (radius=6)
      3. Studio finish     -> Adobe-Express style alpha cleanup (morph,
                              smoothstep, sub-pixel feather)
      4. Composite-on-black-> RGB * alpha for a pure-black background
      5. Clarity + UnsharpMask for pore-level skin sharpness
    """
    alpha = run_matting(img_rgb)
    if alpha is None:
        return _clarity_and_sharpen(img_rgb), None

    rgb_arr = np.array(img_rgb.convert("RGB"))
    gray = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY)
    alpha = _guided_filter_alpha(gray, alpha, radius=6, eps=1e-3)
    alpha = _studio_finish_alpha(alpha)

    on_black = _composite_on_black(rgb_arr, alpha)
    finished = _clarity_and_sharpen(on_black)
    return finished, alpha


def decode_image(image_b64: str) -> Image.Image:
    if "," in image_b64 and image_b64.lstrip().startswith("data:"):
        image_b64 = image_b64.split(",", 1)[1]
    data = base64.b64decode(image_b64)
    return ImageOps.exif_transpose(Image.open(BytesIO(data))).convert("RGB")


def download_storage_image(path: str) -> Image.Image:
    url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{path}"
    resp = requests.get(url, headers={"Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"}, timeout=90)
    if resp.status_code != 200:
        raise RuntimeError(f"Storage download failed: {resp.status_code} {resp.text[:200]}")
    return ImageOps.exif_transpose(Image.open(BytesIO(resp.content))).convert("RGB")


def public_storage_url(path: str) -> str:
    return f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{path}"


def crop_to_aspect(img: Image.Image, target_width: int, target_height: int) -> Image.Image:
    target_aspect = target_width / target_height
    width, height = img.size
    source_aspect = width / height

    if source_aspect > target_aspect:
        crop_width = round(height * target_aspect)
        left = (width - crop_width) // 2
        box = (left, 0, left + crop_width, height)
    else:
        crop_height = round(width / target_aspect)
        top = (height - crop_height) // 2
        box = (0, top, width, top + crop_height)

    return img.crop(box)


def normalize_portrait(img: Image.Image) -> Image.Image:
    portrait = crop_to_aspect(img, PORTRAIT_WIDTH, PORTRAIT_HEIGHT)
    if portrait.size != (PORTRAIT_WIDTH, PORTRAIT_HEIGHT):
        portrait = portrait.resize((PORTRAIT_WIDTH, PORTRAIT_HEIGHT), Image.Resampling.LANCZOS)
    return portrait


def fixed_analysis_crop(img: Image.Image) -> Image.Image:
    """Dead-center 1000x1000 ROI — fallback when face detection fails."""
    left = (img.width - ANALYSIS_CROP_SIZE) // 2
    top = (img.height - ANALYSIS_CROP_SIZE) // 2
    return img.crop((left, top, left + ANALYSIS_CROP_SIZE, top + ANALYSIS_CROP_SIZE))


# Fractions of the Haar face box that cover the visible cheek (upper-mid profile).
_CHEEK_FRAC: dict[str, tuple[float, float, float, float]] = {
    # right_90: cheek mound — avoid far-right ear/hair (was 0.36–1.0 → redness 0)
    "right_90": (0.48, 0.26, 0.88, 0.68),
}

# Portrait fractions when Haar fails on right_90 (profile cheek sits center-right).
_RIGHT90_FALLBACK_FRAC = (0.50, 0.24, 0.86, 0.58)


def _clamp_box(l: int, t: int, r: int, b: int, w: int, h: int) -> tuple[int, int, int, int]:
    l = max(0, min(l, w - 1))
    t = max(0, min(t, h - 1))
    r = max(l + 1, min(r, w))
    b = max(t + 1, min(b, h))
    return l, t, r, b


def cheek_box_from_face(
    face_bbox: tuple[int, int, int, int], label: str, img_w: int, img_h: int
) -> tuple[int, int, int, int]:
    """Pixel box (l, t, r, b) for the visible cheek inside the face bbox."""
    fx, fy, fw, fh = face_bbox
    lf, tf, rf, bf = _CHEEK_FRAC.get(label, (0.2, 0.3, 0.8, 0.75))
    l = fx + int(fw * lf)
    r = fx + int(fw * rf)
    t = fy + int(fh * tf)
    b = fy + int(fh * bf)
    l, t, r, b = _clamp_box(l, t, r, b, img_w, img_h)

    # Square region centered on cheek so scoring stays stable (same output size).
    cx = (l + r) // 2
    cy = (t + b) // 2
    side = max(r - l, b - t, int(min(img_w, img_h) * 0.18))
    half = side // 2
    l, t, r, b = _clamp_box(cx - half, cy - half, cx + half, cy + half, img_w, img_h)
    return l, t, r, b


def _crop_box_to_analysis(
    img: Image.Image,
    alpha: np.ndarray | None,
    l: int,
    t: int,
    r: int,
    b: int,
    method: str,
) -> tuple[Image.Image, np.ndarray | None, dict]:
    w, h = img.size
    cheek = img.crop((l, t, r, b))
    crop = ImageOps.fit(
        cheek,
        (ANALYSIS_CROP_SIZE, ANALYSIS_CROP_SIZE),
        method=Image.Resampling.BILINEAR,
        centering=(0.5, 0.5),
    )
    alpha_crop = None
    if alpha is not None:
        patch = alpha[t:b, l:r]
        alpha_crop = np.array(
            Image.fromarray(patch).resize(
                (ANALYSIS_CROP_SIZE, ANALYSIS_CROP_SIZE),
                Image.Resampling.BILINEAR,
            )
        )
    return (
        crop,
        alpha_crop,
        {"x": l, "y": t, "width": r - l, "height": b - t, "method": method},
    )


def right_profile_fallback_box(img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """Fixed cheek region for right 90° when Haar misses (better than dead center)."""
    lf, tf, rf, bf = _RIGHT90_FALLBACK_FRAC
    l = int(img_w * lf)
    r = int(img_w * rf)
    t = int(img_h * tf)
    b = int(img_h * bf)
    l, t, r, b = _clamp_box(l, t, r, b, img_w, img_h)
    cx = (l + r) // 2
    cy = (t + b) // 2
    side = max(r - l, b - t, int(min(img_w, img_h) * 0.16))
    half = side // 2
    return _clamp_box(cx - half, cy - half, cx + half, cy + half, img_w, img_h)


def analysis_roi_crop(
    img: Image.Image, alpha: np.ndarray | None, label: str
) -> tuple[Image.Image, np.ndarray | None, dict]:
    """Cheek-focused 1000×1000 ROI for redness scoring (right_90 only)."""
    w, h = img.size
    bbox = detect_face_bbox_fast(img, label)
    if bbox is not None:
        l, t, r, b = cheek_box_from_face(bbox, label, w, h)
        return _crop_box_to_analysis(img, alpha, l, t, r, b, "cheek_bbox")

    if label == "right_90":
        l, t, r, b = right_profile_fallback_box(w, h)
        return _crop_box_to_analysis(img, alpha, l, t, r, b, "right_profile_fallback")

    crop = fixed_analysis_crop(img)
    left = (w - ANALYSIS_CROP_SIZE) // 2
    top = (h - ANALYSIS_CROP_SIZE) // 2
    alpha_crop = None
    if alpha is not None:
        alpha_crop = alpha[top : top + ANALYSIS_CROP_SIZE, left : left + ANALYSIS_CROP_SIZE]
    return (
        crop,
        alpha_crop,
        {
            "x": left,
            "y": top,
            "width": ANALYSIS_CROP_SIZE,
            "height": ANALYSIS_CROP_SIZE,
            "method": "center_fallback",
        },
    )


def jpeg_bytes(img: Image.Image, quality: int = 95) -> bytes:
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def webp_lossless_bytes(img: Image.Image) -> bytes:
    """Encode image as bit-exact lossless WebP (analysis-grade).

    Reserved for the right_90 ROI artifacts where pixel-exact fidelity matters
    for re-analysis. method=6 is the strongest (slowest) entropy coder.
    """
    buf = BytesIO()
    img.convert("RGB").save(
        buf,
        format="WEBP",
        lossless=True,
        quality=100,
        method=6,
    )
    return buf.getvalue()


def webp_visual_bytes(img: Image.Image, quality: int = 95) -> bytes:
    """Encode image as high-quality lossy WebP for display use.

    For the four non-analysis angles we only render them; we never re-run
    skin math on them. q=95 method=4 is visually indistinguishable from
    lossless on skin imagery and encodes ~5-10x faster than lossless.
    """
    buf = BytesIO()
    img.convert("RGB").save(
        buf,
        format="WEBP",
        quality=quality,
        method=4,
    )
    return buf.getvalue()


def upload_bytes(data: bytes, filename: str, content_type: str) -> str:
    path = f"processed/{filename}"
    url = f"{SUPABASE_URL}/storage/v1/object/{SUPABASE_BUCKET}/{path}"
    resp = requests.put(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "Content-Type": content_type,
            "x-upsert": "true",
        },
        timeout=120,
    )
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Storage upload failed: {resp.status_code} {resp.text[:200]}")
    return public_storage_url(path)


def upload_jpeg(img: Image.Image, filename: str, quality: int = 95) -> str:
    return upload_bytes(jpeg_bytes(img, quality=quality), filename, "image/jpeg")


def upload_webp_lossless(img: Image.Image, filename: str) -> str:
    return upload_bytes(webp_lossless_bytes(img), filename, "image/webp")


def upload_webp_visual(img: Image.Image, filename: str, quality: int = 95) -> str:
    return upload_bytes(webp_visual_bytes(img, quality=quality), filename, "image/webp")


def upload_png(img: Image.Image, filename: str) -> str:
    buf = BytesIO()
    img.convert("RGB").save(buf, format="PNG", optimize=True)
    return upload_bytes(buf.getvalue(), filename, "image/png")


def make_analysis_map(
    img: Image.Image, mode: str = "redness", alpha: np.ndarray | None = None
) -> Image.Image:
    """Build the false-color visia map. When `alpha` is provided we mask the
    output to the foreground only (matches May-4: visia maps sit on black so
    the eye isn't drawn to background lighting noise)."""
    arr = np.array(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)

    if mode == "texture":
        inv = 255 - contrast
        rgb = cv2.cvtColor(inv, cv2.COLOR_GRAY2RGB)
    else:
        bone = cv2.applyColorMap(contrast, cv2.COLORMAP_BONE)
        rgb = cv2.cvtColor(bone, cv2.COLOR_BGR2RGB)

    if alpha is not None:
        a = alpha.astype(np.float32) / 255.0
        rgb = (rgb.astype(np.float32) * a[..., None]).astype(np.uint8)
    return Image.fromarray(rgb)


def compute_quality(clean: Image.Image, label: str = "analysis") -> dict:
    gray = cv2.cvtColor(np.array(clean.convert("RGB")), cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    mean = float(gray.mean())
    exposure = max(0, 100 - abs(mean - 132.0) * 1.4)
    contrast_score = min(100, contrast * 2.5)
    quality = int(round(0.5 * exposure + 0.5 * contrast_score))
    warnings = []
    if blur < 80:
        warnings.append(f"{label} crop may be blurry")
    if mean < 80:
        warnings.append(f"{label} crop is underexposed")
    if mean > 190:
        warnings.append(f"{label} crop is overexposed")
    if contrast < 25:
        warnings.append(f"{label} crop has low contrast")
    return {"quality_score": quality, "quality_warnings": warnings}


def load_angle_image(images: dict, image_paths: dict, label: str) -> tuple[Image.Image | None, str | None]:
    if image_paths.get(label):
        path = image_paths[label]
        return download_storage_image(path), public_storage_url(path)
    if images.get(label):
        return decode_image(images[label]), None
    return None, None


def update_supabase_scan(scan_id: str, processed_angles: dict, primary: dict) -> None:
    url = f"{SUPABASE_URL}/rest/v1/scans?id=eq.{scan_id}"
    body = {
        "status": "done",
        "processed_angles": processed_angles,
        "clean_image_url": primary.get("clean_image_url"),
        "redness_image_url": primary.get("visia_image_url"),
        "image_url": primary.get("visia_image_url"),
        "redness_severity": None,
    }
    resp = requests.patch(
        url,
        json=body,
        headers={
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "apikey": SUPABASE_SERVICE_KEY,
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        },
        timeout=60,
    )
    if resp.status_code not in (200, 201, 204):
        raise RuntimeError(f"DB update failed: {resp.status_code} {resp.text[:200]}")


def update_supabase_scan_studio(scan_id: str, studio_angles: dict) -> None:
    """Write the editor-refine results back to the scan row.

    We use a dedicated `studio_angles` JSON field plus a top-level
    `studio_image_url` (mirroring how `processed_angles` + `clean_image_url`
    work for the initial pass) so the app can render the editor view
    without having to inspect a nested dict.
    """
    # NOTE: we deliberately do NOT touch `status` here — the initial scan
    # already set it to "done" and the app polls that field. The refine
    # job announces itself only through the new `studio_*` fields, which
    # the app polls separately.
    url = f"{SUPABASE_URL}/rest/v1/scans?id=eq.{scan_id}"
    right90 = studio_angles.get(PRIMARY_ANALYSIS_ANGLE, {})
    body = {
        "studio_angles": studio_angles,
        "studio_image_url": right90.get("studio_image_url"),
    }
    resp = requests.patch(
        url,
        json=body,
        headers={
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "apikey": SUPABASE_SERVICE_KEY,
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        },
        timeout=60,
    )
    if resp.status_code not in (200, 201, 204):
        raise RuntimeError(f"Studio DB update failed: {resp.status_code} {resp.text[:200]}")


def _refine_one_angle(
    label: str, source_url: str, uid: str, *, halo_only: bool = True
) -> tuple[str, dict]:
    """Download a processed clean image and run editor refine.

    `halo_only=True` (default): edge halo cleanup only — no auto zoom or
    canonical re-framing; the clinician nudges position manually in the app.
    """
    if not source_url:
        return label, {"studio_image_url": None, "error": "missing source url"}

    img = _download_processed_url(source_url)
    cleaned = refine_studio_quality(img)

    if halo_only:
        out = cleaned
        framing = "halo_only"
    else:
        bbox = detect_face_bbox(cleaned, label)
        if bbox is None:
            out = cleaned
            framing = "passthrough"
        else:
            out = standardize_to_canonical(cleaned, bbox)
            framing = f"face@{bbox}"

    studio_url = upload_webp_lossless(out, f"studio_{label}_{uid}.webp")
    print(f"[refine] {label} OK framing={framing}")
    return label, {
        "studio_image_url": studio_url,
        "source_clean_url": source_url,
        "framing": framing,
    }


def apply_horizontal_shift(img_rgb: Image.Image, offset_x: int) -> Image.Image:
    """Shift subject left/right on the same canvas (black background)."""
    return apply_refine_transform(img_rgb, int(offset_x), 0, 1.0)


def apply_refine_transform(
    img_rgb: Image.Image, offset_x: int, offset_y: int, scale: float
) -> Image.Image:
    """Scale around center, then translate on a fixed black canvas."""
    scale = max(0.5, min(2.0, float(scale or 1.0)))
    w, h = img_rgb.size
    working = img_rgb.convert("RGB")

    if abs(scale - 1.0) > 1e-3:
        nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        scaled = working.resize((nw, nh), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (w, h), (0, 0, 0))
        canvas.paste(scaled, ((w - nw) // 2, (h - nh) // 2))
        working = canvas

    arr = np.array(working)
    out = np.zeros_like(arr)
    ox, oy = int(offset_x), int(offset_y)

    if oy >= 0:
        y_dst, y_src = slice(oy, h), slice(0, h - oy)
    else:
        y_dst, y_src = slice(0, h + oy), slice(-oy, h)

    if ox >= 0:
        x_dst, x_src = slice(ox, w), slice(0, w - ox)
    else:
        x_dst, x_src = slice(0, w + ox), slice(-ox, w)

    out[y_dst, x_dst] = arr[y_src, x_src]
    return Image.fromarray(out, mode="RGB")


def commit_refined_angle(
    scan_id: str,
    label: str,
    refined_url: str,
    offset_x: int,
    offset_y: int,
    scale: float,
    processed_angles: dict,
) -> dict:
    """Bake manual offset into the refined image and replace processed_angles[label]."""
    img = _download_processed_url(refined_url)
    shifted = apply_refine_transform(img, int(offset_x), int(offset_y), float(scale))
    uid = uuid.uuid4().hex[:10]
    if label in ANALYSIS_ANGLES:
        clean_url = upload_webp_lossless(shifted, f"clean_{label}_{uid}.webp")
    else:
        clean_url = upload_webp_visual(shifted, f"clean_{label}_{uid}.webp", quality=95)

    prev = dict(processed_angles.get(label) or {})
    prev["clean_image_url"] = clean_url
    prev["refined_at"] = datetime.now(timezone.utc).isoformat()
    prev["refine_offset_x"] = int(offset_x)
    prev["refine_offset_y"] = int(offset_y)
    prev["refine_scale"] = float(scale)
    return prev


def process_refine(scan_id: str, clean_urls: dict, *, halo_only: bool = True) -> dict:
    """Editor refine entry point: edge halo cleanup + canonical placement.

    `clean_urls` is `{angle_label: clean_image_url}` from the initial scan.
    All per-angle work is independent and dominated by HTTP I/O, so we run
    it through a thread pool — same pattern as the initial Phase 2.
    """
    if not clean_urls:
        return {}

    uid = uuid.uuid4().hex[:10]
    items = [(label, url) for label, url in clean_urls.items() if url]

    studio: dict = {}
    with ThreadPoolExecutor(max_workers=min(len(items), 5)) as pool:
        futures = [
            pool.submit(_refine_one_angle, label, url, uid, halo_only=halo_only)
            for label, url in items
        ]
        for fut in futures:
            label, data = fut.result()
            studio[label] = data
    return studio


def merge_supabase_scan_studio(scan_id: str, studio_patch: dict) -> None:
    """Merge per-angle studio output without wiping other angles."""
    url = f"{SUPABASE_URL}/rest/v1/scans?id=eq.{scan_id}&select=studio_angles"
    resp = requests.get(
        url,
        headers={
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "apikey": SUPABASE_SERVICE_KEY,
        },
        timeout=30,
    )
    existing: dict = {}
    if resp.status_code == 200:
        rows = resp.json()
        if rows:
            existing = dict(rows[0].get("studio_angles") or {})
    existing.update(studio_patch)
    right90 = existing.get(PRIMARY_ANALYSIS_ANGLE, {})
    body = {
        "studio_angles": existing,
        "studio_image_url": right90.get("studio_image_url"),
    }
    patch = requests.patch(
        f"{SUPABASE_URL}/rest/v1/scans?id=eq.{scan_id}",
        json=body,
        headers={
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "apikey": SUPABASE_SERVICE_KEY,
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        },
        timeout=60,
    )
    if patch.status_code not in (200, 201, 204):
        raise RuntimeError(f"Studio merge failed: {patch.status_code} {patch.text[:200]}")


def update_supabase_processed_angle(
    scan_id: str, label: str, angle_data: dict, processed_angles: dict
) -> None:
    """Replace one angle in processed_angles after manual refine save."""
    merged = dict(processed_angles)
    merged[label] = angle_data
    primary = merged.get(PRIMARY_ANALYSIS_ANGLE, {})
    body = {
        "processed_angles": merged,
        "clean_image_url": primary.get("clean_image_url"),
        "redness_image_url": primary.get("visia_image_url")
        or primary.get("redness_image_url"),
        "redness_severity": None,
        "studio_angles": None,
        "studio_image_url": None,
    }
    resp = requests.patch(
        f"{SUPABASE_URL}/rest/v1/scans?id=eq.{scan_id}",
        json=body,
        headers={
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "apikey": SUPABASE_SERVICE_KEY,
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        },
        timeout=60,
    )
    if resp.status_code not in (200, 201, 204):
        raise RuntimeError(f"Processed angle update failed: {resp.status_code} {resp.text[:200]}")


def _encode_and_upload(prepared: dict, mode: str, uid: str) -> tuple[str, dict]:
    """I/O- and CPU-bound work that runs in parallel across angles.

    `prepared` carries everything matting/normalize already produced; this
    stage only does false-color rendering, WebP/JPEG encoding, and storage
    uploads — all libwebp/libjpeg/requests calls drop the GIL so a
    ThreadPoolExecutor gives real parallelism here.
    """
    label = prepared["label"]
    clean = prepared["clean"]
    alpha = prepared["alpha"]
    original_url = prepared["original_url"]

    visia = make_analysis_map(clean, mode, alpha=alpha)

    if label in ANALYSIS_ANGLES:
        clean_url = upload_webp_lossless(clean, f"clean_{label}_{uid}.webp")
    else:
        # Display-only angles: q=95 WebP is visually indistinguishable on
        # skin imagery and 5-10x faster to encode than lossless.
        clean_url = upload_webp_visual(clean, f"clean_{label}_{uid}.webp", quality=95)

    visia_url = upload_jpeg(visia, f"visia_{label}_{uid}.jpg", quality=92)

    angle_data: dict = {
        "original_image_url": original_url,
        "clean_image_url": clean_url,
        "visia_image_url": visia_url,
        "background_removed": alpha is not None,
    }

    if label in ANALYSIS_ANGLES:
        rgb = np.array(clean.convert("RGB"))
        bone_rgb = np.array(visia.convert("RGB"))
        t_lm = datetime.now(timezone.utc)
        landmarks = detect_face_landmarks(rgb)
        cheek_preview, cheek_method, cheek_pixels, cheek_timing = extract_cheek_tight_bone(
            bone_rgb, rgb, alpha, landmarks
        )
        cheek_url = upload_png(cheek_preview, f"cheek_{label}_{uid}.png")
        lm_ms = int(
            (datetime.now(timezone.utc) - t_lm).total_seconds() * 1000
        )
        angle_data.update(
            {
                "analysis_step": "cheek_roi",
                "cheek_roi_method": cheek_method,
                "cheek_roi_prompt": CLIPSEG_CHEEK_PROMPT,
                "cheek_roi_image_url": cheek_url,
                "cheek_pixel_count": cheek_pixels,
                "landmark_ms": lm_ms,
                **cheek_timing,
                **compute_quality(clean, label),
            }
        )
        print(
            f"[handler] {label} cheek_roi method={cheek_method} "
            f"prompt={CLIPSEG_CHEEK_PROMPT!r} pixels={cheek_pixels} "
            f"timing={cheek_timing} landmark_ms={lm_ms} "
            f"clipseg_ready={clipseg_cheek.clipseg_ready()}"
        )

    print(f"[handler] {label} OK")
    return label, angle_data


def process_images(images: dict, image_paths: dict, mode: str = "redness") -> dict:
    """Two-phase pipeline:

    Phase 1 (sequential): per angle — download original, normalize to the
        portrait frame, run MODNet matting + guided-filter refine + studio
        finish + Clarity/UnsharpMask. MODNet on CPU already saturates all
        cores via ORT's intra-op pool; running multiple inferences
        concurrently would thrash, so we serialize this phase.

    Phase 2 (parallel): build the false-color visia map, encode WebP/JPEG,
        and upload to Supabase Storage. All of these release the GIL during
        native libwebp/libjpeg/requests calls, so threads actually overlap.
    """
    uid = uuid.uuid4().hex[:10]
    prepared_angles: list[dict] = []

    for label in ANGLE_KEYS:
        original, original_url = load_angle_image(images, image_paths, label)
        if original is None:
            continue
        portrait = normalize_portrait(original)
        clean, alpha = remove_background_and_finish(portrait)
        prepared_angles.append(
            {
                "label": label,
                "original_url": original_url,
                "clean": clean,
                "alpha": alpha,
            }
        )
        print(
            f"[handler] {label} matting "
            + ("OK" if alpha is not None else "SKIPPED (no MODNet model)")
        )

    processed: dict = {}
    if not prepared_angles:
        return processed

    with ThreadPoolExecutor(max_workers=len(prepared_angles)) as pool:
        futures = [pool.submit(_encode_and_upload, item, mode, uid) for item in prepared_angles]
        for fut in futures:
            label, data = fut.result()
            processed[label] = data

    return processed


def handler(job):
    job_input = job.get("input", {})
    scan_id = job_input.get("scan_id")
    mode = job_input.get("mode", "redness")

    # ---- Editor refine mode -------------------------------------------------
    # Triggered by the "Open in Editor" button. Takes the already-processed
    # clean_image_urls from the initial scan and runs studio-grade halo
    # cleanup + canonical face placement on each angle.
    if mode == "refine":
        clean_urls = job_input.get("clean_image_urls") or {}
        if not scan_id or not clean_urls:
            return {"error": "refine mode requires scan_id and clean_image_urls"}
        if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY are required")

        halo_only = job_input.get("halo_only", True)
        merge_studio = job_input.get("merge_studio", True)
        print(
            f"[handler] mode=refine scan_id={scan_id} "
            f"angles={list(clean_urls.keys())} halo_only={halo_only}"
        )
        studio_angles = process_refine(scan_id, clean_urls, halo_only=halo_only)
        if merge_studio:
            merge_supabase_scan_studio(scan_id, studio_angles)
        else:
            update_supabase_scan_studio(scan_id, studio_angles)
        print(f"[handler] refine DB updated OK for scan_id={scan_id}")
        return {
            "status": "studio_done",
            "scan_id": scan_id,
            "studio_angles": studio_angles,
        }

    if mode == "commit_refine":
        label = job_input.get("angle")
        refined_url = job_input.get("refined_url")
        offset_x = int(job_input.get("offset_x") or 0)
        offset_y = int(job_input.get("offset_y") or 0)
        scale = float(job_input.get("scale") or 1.0)
        processed_angles = job_input.get("processed_angles") or {}
        if not scan_id or not label or not refined_url:
            return {"error": "commit_refine requires scan_id, angle, refined_url"}
        if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
            raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY are required")
        angle_data = commit_refined_angle(
            scan_id, label, refined_url, offset_x, offset_y, scale, processed_angles
        )
        update_supabase_processed_angle(scan_id, label, angle_data, processed_angles)
        print(f"[handler] commit_refine OK scan_id={scan_id} angle={label}")
        return {
            "status": "committed",
            "scan_id": scan_id,
            "angle": label,
            "clean_image_url": angle_data.get("clean_image_url"),
        }

    # ---- Initial scan mode (default) ---------------------------------------
    images = job_input.get("images") or {}
    image_paths = job_input.get("image_paths") or {}

    print(
        f"[handler] mode={mode} scan_id={scan_id} "
        f"image_paths={list(image_paths.keys())} images={list(images.keys())}"
    )

    if not isinstance(images, dict) or not isinstance(image_paths, dict):
        return {"error": "input.images and input.image_paths must be objects keyed by angle"}
    if PRIMARY_ANALYSIS_ANGLE not in images and PRIMARY_ANALYSIS_ANGLE not in image_paths:
        return {"error": f"Missing required {PRIMARY_ANALYSIS_ANGLE} image"}
    if (scan_id or image_paths) and (not SUPABASE_URL or not SUPABASE_SERVICE_KEY):
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY are required")

    processed_angles = process_images(images, image_paths, mode)
    primary = processed_angles.get(PRIMARY_ANALYSIS_ANGLE, {})

    if scan_id:
        update_supabase_scan(scan_id, processed_angles, primary)
        print(
            f"[handler] DB updated OK for scan_id={scan_id} "
            f"prep_only angles={list(ANALYSIS_ANGLES)}"
        )

    return {
        "status": "done",
        "scan_id": scan_id,
        "analysis_angles": list(ANALYSIS_ANGLES),
        "analysis_step": "prep_only",
        "processed_angles": processed_angles,
    }


if __name__ == "__main__":
    if runpod is None:
        raise RuntimeError("runpod package is required to start the worker")
    runpod.serverless.start({"handler": handler})
