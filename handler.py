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

# Register the HEIF/HEIC opener with Pillow so Image.open() auto-detects HEIC
# bytes (the format the iOS app uploads). Pillow's auto-detect reads the file
# header, so no other code in this module needs to know the input format.
from pillow_heif import register_heif_opener

register_heif_opener()

try:
    import runpod
except ImportError:  # Allows local helper tests without RunPod installed.
    runpod = None

from gemini_fragment import run_gemini_refine, run_gemini_remove_face_shadows


SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "scans")

ANGLE_KEYS = ["frontal", "left_45", "left_90", "right_45", "right_90"]
# Redness scoring: both 90° profiles (edge fn averages Gemini scores).
ANALYSIS_ANGLES = ("left_90", "right_90")
PRIMARY_ANALYSIS_ANGLE = "right_90"
PROFILE_90_ANGLES = ("right_90", "left_90")


def resolve_primary_profile_angle(images: dict, image_paths: dict) -> str | None:
    """Prefer right_90 for scan row thumbnails; allow left_90-only uploads."""
    for label in PROFILE_90_ANGLES:
        if image_paths.get(label) or images.get(label):
            return label
    return None
# Inverted duotone analysis map (texture, pigmentation, acne scars, redness).
DUOTONE_MODES = frozenset({"texture", "pigmentation", "acne_scars", "acne", "redness"})
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
GEMINI_DESHADOW_ENABLED = os.environ.get("GEMINI_DESHADOW", "true").lower() not in (
    "0",
    "false",
    "no",
)


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


def _matte_and_composite(img_rgb: Image.Image, *, enhance: bool) -> tuple[Image.Image, np.ndarray | None]:
    """MODNet matte + guided filter + studio finish + black composite.

    When enhance=True, apply clarity/sharpen after compositing on black.
    """
    alpha = run_matting(img_rgb)
    if alpha is None:
        return (_clarity_and_sharpen(img_rgb) if enhance else img_rgb), None

    rgb_arr = np.array(img_rgb.convert("RGB"))
    gray = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY)
    alpha = _guided_filter_alpha(gray, alpha, radius=6, eps=1e-3)
    alpha = _studio_finish_alpha(alpha)

    on_black = _composite_on_black(rgb_arr, alpha)
    finished = _clarity_and_sharpen(on_black) if enhance else on_black
    return finished, alpha


def remove_background_and_finish(img_rgb: Image.Image):
    """MODNet + composite + clarity/sharpen (SHARPEN_AMOUNT)."""
    return _matte_and_composite(img_rgb, enhance=True)


def apply_face_shadow_removal(clean: Image.Image, label: str) -> Image.Image:
    """Gemini pass after matting: remove cast shadows on the face, then continue pipeline."""
    if not GEMINI_DESHADOW_ENABLED:
        return clean
    deshadowed, err = run_gemini_remove_face_shadows(clean)
    if deshadowed is not None:
        print(f"[handler] {label} face deshadow OK")
        return deshadowed
    print(f"[handler] {label} face deshadow skipped: {err}")
    return clean


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

    if mode in DUOTONE_MODES:
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


def update_supabase_scan(
    scan_id: str, processed_angles: dict, primary: dict, mode: str = "redness"
) -> None:
    url = f"{SUPABASE_URL}/rest/v1/scans?id=eq.{scan_id}"
    body = {
        "status": "done" if mode == "before_after" else "visia_ready",
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


def update_supabase_scan_partial(
    scan_id: str, processed_angles: dict, mode: str = "before_after"
) -> None:
    """Expose clean images as soon as each before/after angle is ready."""
    primary = processed_angles.get(PRIMARY_ANALYSIS_ANGLE, {})
    if not primary:
        primary = next(
            (row for row in processed_angles.values() if row.get("clean_image_url")),
            {},
        )
    update_supabase_scan(scan_id, processed_angles, primary, mode)


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
    gemini_refined, gemini_error = run_gemini_refine(img)
    if gemini_refined is not None:
        cleaned = gemini_refined
        refine_method = "gemini_refine"
    else:
        print(f"[refine] {label} Gemini fallback: {gemini_error}")
        cleaned = refine_studio_quality(img)
        refine_method = "local_refine"

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
    print(f"[refine] {label} OK method={refine_method} framing={framing}")
    return label, {
        "studio_image_url": studio_url,
        "source_clean_url": source_url,
        "framing": framing,
        "refine_method": refine_method,
        "refine_error": gemini_error if gemini_refined is None else None,
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
    scan_mode: str = "redness",
) -> dict:
    """Bake manual offset into the refined image and replace only the clean image."""
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


def _encode_and_upload(prepared: dict, mode: str, uid: str, modes: list[str] | None = None) -> tuple[str, dict]:
    """I/O- and CPU-bound work that runs in parallel across angles.

    `prepared` carries everything matting/normalize already produced; this
    stage only does false-color rendering, WebP encoding, and storage
    uploads — all libwebp/requests calls drop the GIL so a
    ThreadPoolExecutor gives real parallelism here.

    When `modes` has multiple analysis targets, each angle stores a
    `visia_maps` dict keyed by mode plus `visia_image_url` for the primary.
    """
    label = prepared["label"]
    clean = prepared["clean"]
    alpha = prepared["alpha"]
    original_url = prepared["original_url"]
    analysis_modes = [m for m in (modes or [mode]) if m != "before_after"]
    primary_mode = analysis_modes[0] if analysis_modes else mode

    visia_maps: dict[str, str | None] = {}
    primary_visia_url: str | None = None

    if mode == "before_after" or not analysis_modes:
        visia = None
    elif len(analysis_modes) == 1:
        visia = make_analysis_map(clean, analysis_modes[0], alpha=alpha)
        if visia is not None:
            visia_maps[analysis_modes[0]] = upload_jpeg(
                visia, f"visia_{label}_{analysis_modes[0]}_{uid}.jpg", quality=92
            )
        primary_visia_url = visia_maps.get(analysis_modes[0])
    else:
        for m in analysis_modes:
            visia = make_analysis_map(clean, m, alpha=alpha)
            visia_maps[m] = (
                upload_jpeg(visia, f"visia_{label}_{m}_{uid}.jpg", quality=92)
                if visia is not None
                else None
            )
        primary_visia_url = visia_maps.get(primary_mode)

    if label in ANALYSIS_ANGLES:
        visia_url = primary_visia_url
        clean_url = upload_webp_lossless(clean, f"clean_{label}_{uid}.webp")
    else:
        clean_url = upload_webp_visual(clean, f"clean_{label}_{uid}.webp", quality=95)
        visia_url = primary_visia_url

    angle_data: dict = {
        "original_image_url": original_url,
        "clean_image_url": clean_url,
        "visia_image_url": visia_url,
        "background_removed": alpha is not None,
    }
    if visia_maps:
        angle_data["visia_maps"] = visia_maps

    if label in ANALYSIS_ANGLES and visia_url is not None:
        angle_data.update(
            {
                "analysis_step": "visia_ready",
                **compute_quality(clean, label),
            }
        )
        print(f"[handler] {label} visia_ready visia_url={visia_url}")

    print(f"[handler] {label} OK")
    return label, angle_data


def _angles_for_mode(processed_angles: dict, mode: str) -> dict:
    """Copy processed angles with visia_image_url set for a specific mode."""
    out: dict = {}
    for label, row in processed_angles.items():
        data = dict(row)
        visia_maps = data.get("visia_maps") or {}
        if isinstance(visia_maps, dict) and mode in visia_maps:
            data["visia_image_url"] = visia_maps.get(mode)
        out[label] = data
    return out


def sync_linked_scans(
    linked_scan_ids: dict,
    processed_angles: dict,
    primary_label: str,
) -> None:
    """Write mode-specific visia maps to linked capture rows."""
    if not linked_scan_ids or not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        return
    for mode, linked_id in linked_scan_ids.items():
        angles = _angles_for_mode(processed_angles, mode)
        primary = angles.get(primary_label) or next(
            (row for row in angles.values() if row.get("clean_image_url")),
            {},
        )
        update_supabase_scan(linked_id, angles, primary, mode)
        print(f"[handler] synced linked scan mode={mode} id={linked_id}")


def process_images(
    images: dict,
    image_paths: dict,
    mode: str = "redness",
    scan_id: str | None = None,
    modes: list[str] | None = None,
) -> dict:
    """Two-phase pipeline:

    Phase 1 (sequential): per angle — download original, normalize to the
        portrait frame, run MODNet matting + guided-filter refine + studio
        finish + Clarity/UnsharpMask. MODNet on CPU already saturates all
        cores via ORT's intra-op pool; running multiple inferences
        concurrently would thrash, so we serialize this phase.

    Phase 2 (parallel): build the false-color visia map, encode WebP,
        and upload to Supabase Storage. All of these release the GIL during
        native libwebp/requests calls, so threads actually overlap.
    """
    uid = uuid.uuid4().hex[:10]
    prepared_angles: list[dict] = []
    processed: dict = {}
    analysis_modes = [m for m in (modes or [mode]) if m != "before_after"]
    effective_mode = mode if mode == "before_after" or not analysis_modes else analysis_modes[0]

    if mode == "before_after" and scan_id:
        for label in ANGLE_KEYS:
            original, original_url = load_angle_image(images, image_paths, label)
            if original is None:
                continue
            portrait = normalize_portrait(original)
            clean, alpha = remove_background_and_finish(portrait)
            clean = apply_face_shadow_removal(clean, label)
            print(
                f"[handler] {label} matting "
                + ("OK" if alpha is not None else "SKIPPED (no MODNet model)")
            )
            encoded_label, data = _encode_and_upload(
                {
                    "label": label,
                    "original_url": original_url,
                    "clean": clean,
                    "alpha": alpha,
                },
                mode,
                uid,
            )
            processed[encoded_label] = data
            update_supabase_scan_partial(scan_id, processed, mode)
        return processed

    for label in ANGLE_KEYS:
        original, original_url = load_angle_image(images, image_paths, label)
        if original is None:
            continue
        portrait = normalize_portrait(original)
        clean, alpha = remove_background_and_finish(portrait)
        clean = apply_face_shadow_removal(clean, label)
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

    if not prepared_angles:
        return processed

    with ThreadPoolExecutor(max_workers=len(prepared_angles)) as pool:
        futures = [
            pool.submit(_encode_and_upload, item, effective_mode, uid, analysis_modes or [mode])
            for item in prepared_angles
        ]
        for fut in futures:
            label, data = fut.result()
            processed[label] = data

    return processed


def handler(job):
    job_input = job.get("input", {})
    scan_id = job_input.get("scan_id")
    mode = job_input.get("mode", "redness")
    modes_raw = job_input.get("modes")
    modes = (
        [str(m) for m in modes_raw if str(m) != "before_after"]
        if isinstance(modes_raw, list) and modes_raw
        else None
    )
    linked_scan_ids = job_input.get("linked_scan_ids") or {}

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
        scan_mode = str(job_input.get("scan_mode") or "redness")
        angle_data = commit_refined_angle(
            scan_id,
            label,
            refined_url,
            offset_x,
            offset_y,
            scale,
            processed_angles,
            scan_mode=scan_mode,
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
        f"[handler] mode={mode} modes={modes or [mode]} scan_id={scan_id} "
        f"image_paths={list(image_paths.keys())} images={list(images.keys())}"
    )

    if not isinstance(images, dict) or not isinstance(image_paths, dict):
        return {"error": "input.images and input.image_paths must be objects keyed by angle"}
    primary_label = resolve_primary_profile_angle(images, image_paths)
    if not primary_label:
        return {"error": "Missing left_90 or right_90 profile image"}
    if (scan_id or image_paths) and (not SUPABASE_URL or not SUPABASE_SERVICE_KEY):
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY are required")

    processed_angles = process_images(
        images, image_paths, mode, scan_id=scan_id, modes=modes
    )
    primary = processed_angles.get(primary_label, {})

    if scan_id:
        update_supabase_scan(scan_id, processed_angles, primary, mode)
        if isinstance(linked_scan_ids, dict) and linked_scan_ids:
            sync_linked_scans(linked_scan_ids, processed_angles, primary_label)
        print(
            f"[handler] DB {'done' if mode == 'before_after' else 'visia_ready'} scan_id={scan_id} "
            f"(scoring via check-job poll)"
        )

    return {
        "status": "visia_ready",
        "scan_id": scan_id,
        "analysis_angles": list(ANALYSIS_ANGLES),
        "analysis_step": primary.get("analysis_step", "visia_ready"),
        "processed_angles": processed_angles,
    }


if __name__ == "__main__":
    if runpod is None:
        raise RuntimeError("runpod package is required to start the worker")
    runpod.serverless.start({"handler": handler})
