import base64
import os
import uuid
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


SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
SUPABASE_BUCKET = os.environ.get("SUPABASE_BUCKET", "scans")

ANGLE_KEYS = ["frontal", "left_45", "left_90", "right_45", "right_90"]
ANALYSIS_ANGLE = "right_90"
PORTRAIT_WIDTH = 2160
PORTRAIT_HEIGHT = 2700
ANALYSIS_CROP_SIZE = 1000

# ---------------------------------------------------------------------------
# Robust Video Matting (RVM) portrait matting
# ---------------------------------------------------------------------------
# Upgrade over MODNet specifically targeting hair-edge quality. Same MIT
# license and ONNX deployment story but produces noticeably cleaner hair
# strands and ear/jaw transitions. We feed each angle independently — RVM
# is a video model but accepts a single frame when the recurrent state is
# zeroed and not threaded between calls.
RVM_MODEL_PATH = os.environ.get(
    "RVM_MODEL_PATH", "/root/.rvm/rvm_mobilenetv3_fp32.onnx"
)
# Long edge we resize each angle to before inference. 720 keeps CPU
# inference under ~700 ms per angle on a typical RunPod CPU pod while
# still resolving individual hair strands. The alpha matte is bilinearly
# upsampled back to the source resolution and then snapped to image edges
# with the guided filter below.
RVM_LONG_SIDE = int(os.environ.get("RVM_LONG_SIDE", "720"))
# Internal downsample ratio fed to RVM as one of its inputs. The RVM docs
# specify the internal downsampled resolution should sit between 256 and
# 512 px for clean hair matting. For our 576x720 (after aspect-preserving
# resize of a portrait) ratio 0.5 lands the internal feature map at
# 288x360 — well inside the sweet spot. 0.375 (the 1280x720 example value
# from the docs) would put it at 216x270, below 256.
RVM_DOWNSAMPLE = float(os.environ.get("RVM_DOWNSAMPLE", "0.5"))
# Sharpen strength for the final clean image. 1.15 matches the May-4 look:
# subtle Clarity pass + UnsharpMask makes pores/skin texture pop without
# halos around facial features.
SHARPEN_AMOUNT = float(os.environ.get("SHARPEN_AMOUNT", "1.15"))


def _init_matting():
    model_path = Path(RVM_MODEL_PATH)
    if not model_path.exists() or model_path.stat().st_size == 0:
        print(f"[matting] Model not found at {model_path}; background removal DISABLED")
        return None
    try:
        opts = ort.SessionOptions()
        # Saturate the CPU on serial inference; we never run RVM from
        # multiple threads simultaneously, so giving ORT all cores is safe.
        opts.intra_op_num_threads = max(1, (os.cpu_count() or 4))
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(
            str(model_path), opts, providers=["CPUExecutionProvider"]
        )
        print(
            f"[matting] RVM session initialized providers={sess.get_providers()} "
            f"inputs={[i.name for i in sess.get_inputs()]}"
        )
        return sess
    except Exception as exc:  # noqa: BLE001
        print(f"[matting] Init failed, background removal DISABLED: {exc}")
        return None


MATTING_SESSION = _init_matting()


def run_matting(img_rgb: Image.Image):
    """Run RVM on a single image. Returns (fgr_uint8_HWC, pha_uint8_HW) at
    the input resolution, or None if matting is disabled.

    RVM's ONNX export is built for video, so the I/O signature is:
        inputs:  src (NCHW float32 [0,1]), r1i..r4i (recurrent states),
                 downsample_ratio (scalar float32)
        outputs: fgr, pha, r1o..r4o
    For still images we pass (1,1,1,1) zero tensors for the recurrent
    states and discard the returned states. This matches the official
    repo's recommended single-frame usage.

    We keep `fgr` (color-decontaminated foreground) in addition to `pha`
    because RVM was trained to produce a foreground RGB with the
    background's color contribution removed at semi-transparent hair
    pixels. Compositing `fgr * pha` (vs `original_rgb * pha`) eliminates
    background color spill into hair strands — a quality win at zero
    additional inference cost.
    """
    if MATTING_SESSION is None:
        return None

    rgb = np.array(img_rgb.convert("RGB"))
    h, w = rgb.shape[:2]
    scale = RVM_LONG_SIDE / float(max(w, h))
    tw = max(64, int(round(w * scale)))
    th = max(64, int(round(h * scale)))
    # Round to multiples of 4 — RVM's internal feature pyramid expects the
    # input dims to halve cleanly a couple of times.
    tw -= tw % 4
    th -= th % 4
    resized = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA)

    # NCHW float32 in [0, 1] — RVM does NOT use MODNet's [-1, 1] norm.
    src = resized.astype(np.float32) / 255.0
    src = np.transpose(src, (2, 0, 1))[None, ...]

    # Zero recurrent state for single-frame inference. The (1,1,1,1) shape
    # is what the upstream ONNX inference example uses.
    rec = np.zeros((1, 1, 1, 1), dtype=np.float32)
    downsample = np.array([RVM_DOWNSAMPLE], dtype=np.float32)

    inputs = {
        "src": src,
        "r1i": rec,
        "r2i": rec,
        "r3i": rec,
        "r4i": rec,
        "downsample_ratio": downsample,
    }
    # Outputs in repo order: [fgr, pha, r1o, r2o, r3o, r4o].
    outputs = MATTING_SESSION.run(None, inputs)
    fgr_chw = outputs[0][0]  # (3, th, tw) float [0,1]
    pha = outputs[1][0, 0]  # (th, tw) float [0,1]

    # Resize both back to the source resolution. fgr needs HWC layout for
    # cv2.resize's multi-channel handling. We force contiguity after the
    # transpose so cv2's C-side doesn't pay for an internal copy.
    fgr_hwc = np.ascontiguousarray(np.transpose(fgr_chw, (1, 2, 0)))
    fgr_full = cv2.resize(fgr_hwc, (w, h), interpolation=cv2.INTER_LINEAR)
    pha_full = cv2.resize(pha, (w, h), interpolation=cv2.INTER_LINEAR)

    return (
        np.clip(fgr_full * 255.0, 0, 255).astype(np.uint8),
        np.clip(pha_full * 255.0, 0, 255).astype(np.uint8),
    )


def _guided_filter_alpha(
    guide_gray: np.ndarray, alpha: np.ndarray, radius: int = 4, eps: float = 1e-3
) -> np.ndarray:
    """Edge-aware refinement: snaps soft matte boundaries to real image edges.

    RVM already produces sharper hair edges than MODNet, so we run guided
    filtering with a smaller radius (4 vs 6) — just enough to snap the
    bilinear-upsampled matte to source-image edges without over-smoothing
    the strand detail RVM resolved.
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
    """Run RVM + guided-filter refinement + Clarity/Sharpen finish.

    Returns (finished_rgb, alpha_uint8 or None). If matting is unavailable
    we degrade gracefully and only apply the Clarity/Sharpen finish on the
    untouched RGB image so the pipeline still produces output.

    Crucially, we composite using RVM's `fgr` output (color-decontaminated
    foreground) rather than the original RGB. At semi-transparent hair
    pixels RVM has already subtracted off the background's color
    contribution — so `fgr * pha` produces cleaner hair strands on black
    than `original_rgb * pha` does, with no extra inference work.
    """
    matting = run_matting(img_rgb)
    if matting is None:
        return _clarity_and_sharpen(img_rgb), None

    fgr, alpha = matting

    # Refine the upsampled matte against the source luma so the alpha
    # snaps to real image edges. Use the original RGB (not fgr) as the
    # guide because hair strands are most visible against the actual
    # background luminance.
    rgb_arr = np.array(img_rgb.convert("RGB"))
    gray = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY)
    alpha = _guided_filter_alpha(gray, alpha, radius=4, eps=1e-3)

    on_black = _composite_on_black(fgr, alpha)
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
    """Fallback when face detection is unavailable: dead-center 1000x1000."""
    left = (img.width - ANALYSIS_CROP_SIZE) // 2
    top = (img.height - ANALYSIS_CROP_SIZE) // 2
    return img.crop((left, top, left + ANALYSIS_CROP_SIZE, top + ANALYSIS_CROP_SIZE))


# ---------------------------------------------------------------------------
# Face-aware analysis crop (anchors the ROI to the actual face, not the frame)
# ---------------------------------------------------------------------------
# A fixed center crop was a constant source of variance for the right_90
# metrics — a few cm of head shift would push the cheek partly outside the
# ROI and pull hair / background in. Anchoring to the detected face bbox
# eliminates that variance: the same skin pixels land in the ROI on every
# capture regardless of where the user framed their head.
#
# MediaPipe Face Detection model_selection=1 ("full range") handles distances
# up to ~5m and is more robust to side-profile poses than model_selection=0.
# We also drop min_detection_confidence to 0.3 so strict 90° profiles, which
# the model is less confident on, still detect.
try:
    import mediapipe as mp  # type: ignore

    FACE_DETECTOR = mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.3,
    )
    print(
        f"[face] MediaPipe Face Detection initialized "
        f"(model=full_range, conf=0.3, mp={mp.__version__})"
    )
except Exception as exc:  # noqa: BLE001
    FACE_DETECTOR = None
    print(f"[face] MediaPipe unavailable; analysis crop falls back to fixed center: {exc}")


def _detect_face_center(portrait_rgb: Image.Image) -> tuple[int, int] | None:
    """Return the pixel-space (cx, cy) of the detected face bbox center,
    or None when no face is found / MediaPipe is unavailable."""
    if FACE_DETECTOR is None:
        return None
    try:
        rgb = np.array(portrait_rgb.convert("RGB"))
        # MediaPipe expects RGB input; we already have RGB.
        results = FACE_DETECTOR.process(rgb)
        if not results.detections:
            return None
        # Pick the highest-score detection. For a single subject this is
        # almost always the only one.
        det = max(
            results.detections,
            key=lambda d: d.score[0] if d.score else 0.0,
        )
        bbox = det.location_data.relative_bounding_box
        w, h = portrait_rgb.size
        cx = int((bbox.xmin + bbox.width / 2.0) * w)
        cy = int((bbox.ymin + bbox.height / 2.0) * h)
        return cx, cy
    except Exception as exc:  # noqa: BLE001
        print(f"[face] detection error, falling back to center: {exc}")
        return None


def compute_analysis_crop_box(
    portrait: Image.Image, alpha: np.ndarray | None
) -> tuple[tuple[int, int, int, int], float, bool]:
    """Decide the 1000x1000 analysis crop coordinates for this portrait.

    Returns (box, alpha_coverage, used_face_detection):
      box: (left, top, right, bottom) clamped inside the portrait
      alpha_coverage: fraction of the box covered by alpha > 128 (face
        framing quality metric). 1.0 when alpha is None.
      used_face_detection: True if MediaPipe gave us the anchor; False if
        we fell back to fixed center.

    The box is always clamped so it never extends past the portrait edges,
    which means callers can crop directly without a bounds check.
    """
    w, h = portrait.size
    half = ANALYSIS_CROP_SIZE // 2

    center = _detect_face_center(portrait)
    used_face = center is not None
    if center is None:
        center = (w // 2, h // 2)

    cx, cy = center
    left = max(0, min(w - ANALYSIS_CROP_SIZE, cx - half))
    top = max(0, min(h - ANALYSIS_CROP_SIZE, cy - half))
    box = (left, top, left + ANALYSIS_CROP_SIZE, top + ANALYSIS_CROP_SIZE)

    coverage = 1.0
    if alpha is not None:
        alpha_crop = alpha[top : top + ANALYSIS_CROP_SIZE, left : left + ANALYSIS_CROP_SIZE]
        coverage = float((alpha_crop > 128).sum()) / float(
            ANALYSIS_CROP_SIZE * ANALYSIS_CROP_SIZE
        )

    return box, coverage, used_face


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


def skin_mask(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lightness = lab[..., 0] * (100.0 / 255.0)
    return (lightness > 18.0) & (lightness < 98.0)


def _face_mask(rgb: np.ndarray, alpha_crop: np.ndarray | None) -> np.ndarray:
    """Combine brightness gating with the RVM alpha matte so scoring only
    sees actual face pixels — no background, no hair edge."""
    mask = skin_mask(rgb)
    if alpha_crop is not None:
        # alpha > 200 -> definite foreground (avoids matte halo pixels).
        mask &= alpha_crop > 200
    return mask


def compute_redness_score(crop: Image.Image, alpha_crop: np.ndarray | None = None) -> int:
    rgb = np.array(crop.convert("RGB"))
    mask = _face_mask(rgb, alpha_crop)
    if not np.any(mask):
        return 0

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    a_star = lab[..., 1] - 128.0
    redness = np.clip((a_star - 8.0) / 26.0, 0.0, 1.0)
    return int(round(float(redness[mask].mean()) * 100))


def compute_white_score(crop: Image.Image, alpha_crop: np.ndarray | None = None) -> int:
    rgb = np.array(crop.convert("RGB"))
    mask = _face_mask(rgb, alpha_crop)
    if not np.any(mask):
        return 0

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    lightness = lab[..., 0] * (100.0 / 255.0)
    a_star = lab[..., 1] - 128.0
    b_star = lab[..., 2] - 128.0
    chroma = np.sqrt(a_star * a_star + b_star * b_star)
    white = (lightness > 72.0) & (chroma < 16.0) & mask
    return int(round(float(white.sum()) / float(mask.sum()) * 100))


def compute_quality(crop: Image.Image) -> dict:
    gray = cv2.cvtColor(np.array(crop.convert("RGB")), cv2.COLOR_RGB2GRAY)
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    mean = float(gray.mean())
    exposure = max(0, 100 - abs(mean - 132.0) * 1.4)
    contrast_score = min(100, contrast * 2.5)
    quality = int(round(0.5 * exposure + 0.5 * contrast_score))
    warnings = []
    if blur < 80:
        warnings.append("Right_90 crop may be blurry")
    if mean < 80:
        warnings.append("Right_90 crop is underexposed")
    if mean > 190:
        warnings.append("Right_90 crop is overexposed")
    if contrast < 25:
        warnings.append("Right_90 crop has low contrast")
    return {"quality_score": quality, "quality_warnings": warnings}


def load_angle_image(images: dict, image_paths: dict, label: str) -> tuple[Image.Image | None, str | None]:
    if image_paths.get(label):
        path = image_paths[label]
        return download_storage_image(path), public_storage_url(path)
    if images.get(label):
        return decode_image(images[label]), None
    return None, None


def update_supabase_scan(scan_id: str, processed_angles: dict, right90: dict) -> None:
    url = f"{SUPABASE_URL}/rest/v1/scans?id=eq.{scan_id}"
    body = {
        "status": "done",
        "processed_angles": processed_angles,
        "clean_image_url": right90.get("crop_image_url") or right90.get("clean_image_url"),
        "redness_image_url": right90.get("visia_image_url"),
        "image_url": right90.get("visia_image_url"),
        "redness_severity": right90.get("redness_score", 0),
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

    if label == ANALYSIS_ANGLE:
        # Analysis-grade: bit-exact lossless WebP for the full clean frame
        # and the ROI crop, so any future re-analysis sees identical pixels.
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

    if label == ANALYSIS_ANGLE:
        # Crop box and framing coverage were pre-computed in Phase 1 so
        # MediaPipe inference doesn't race with the parallel encode pool.
        crop_box: tuple[int, int, int, int] = prepared["analysis_crop_box"]
        framing_coverage: float = prepared["analysis_framing_coverage"]
        face_anchored: bool = prepared["analysis_face_anchored"]
        left, top, right, bottom = crop_box

        crop = clean.crop(crop_box)
        alpha_crop = None
        if alpha is not None:
            alpha_crop = alpha[top:bottom, left:right]

        crop_url = upload_webp_lossless(crop, f"right90_crop_{uid}.webp")
        angle_data.update(
            {
                "crop_image_url": crop_url,
                "crop_box": {
                    "x": left,
                    "y": top,
                    "width": ANALYSIS_CROP_SIZE,
                    "height": ANALYSIS_CROP_SIZE,
                },
                "face_anchored": face_anchored,
                "framing_coverage": round(framing_coverage, 3),
                "redness_score": compute_redness_score(crop, alpha_crop),
                "white_score": compute_white_score(crop, alpha_crop),
                **compute_quality(crop),
            }
        )
        # Surface a soft warning if the ROI overlaps the face poorly —
        # the app can prompt for a retake without us rejecting the scan.
        if framing_coverage < 0.45:
            angle_data.setdefault("quality_warnings", []).append(
                "Right_90 framing low: face does not fill the analysis ROI."
            )

    print(
        f"[handler] {label} OK"
        + (f" redness={angle_data.get('redness_score')}" if label == ANALYSIS_ANGLE else "")
    )
    return label, angle_data


def process_images(images: dict, image_paths: dict, mode: str = "redness") -> dict:
    """Two-phase pipeline:

    Phase 1 (sequential): per angle — download original, normalize to the
        portrait frame, run RVM matting + guided-filter refine + Clarity
        / UnsharpMask finish. RVM on CPU already saturates all cores via
        ORT's intra-op pool; running multiple inferences concurrently would
        thrash, so we serialize this phase.

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

        item: dict = {
            "label": label,
            "original_url": original_url,
            "clean": clean,
            "alpha": alpha,
        }

        # Face-anchored analysis crop only matters for right_90; running
        # MediaPipe on the other angles would just burn ~30 ms each for
        # nothing. Doing it sequentially in Phase 1 also sidesteps
        # MediaPipe's non-thread-safe `process()`.
        if label == ANALYSIS_ANGLE:
            crop_box, coverage, face_anchored = compute_analysis_crop_box(clean, alpha)
            item["analysis_crop_box"] = crop_box
            item["analysis_framing_coverage"] = coverage
            item["analysis_face_anchored"] = face_anchored
            print(
                f"[handler] {label} crop "
                f"box={crop_box} cov={coverage:.2f} "
                f"{'face-anchored' if face_anchored else 'center-fallback'}"
            )

        prepared_angles.append(item)
        print(
            f"[handler] {label} matting "
            + ("OK" if alpha is not None else "SKIPPED (no RVM model)")
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
    images = job_input.get("images") or {}
    image_paths = job_input.get("image_paths") or {}
    mode = job_input.get("mode", "redness")

    print(f"[handler] scan_id={scan_id} image_paths={list(image_paths.keys())} images={list(images.keys())}")

    if not isinstance(images, dict) or not isinstance(image_paths, dict):
        return {"error": "input.images and input.image_paths must be objects keyed by angle"}
    if ANALYSIS_ANGLE not in images and ANALYSIS_ANGLE not in image_paths:
        return {"error": f"Missing required {ANALYSIS_ANGLE} image"}
    if (scan_id or image_paths) and (not SUPABASE_URL or not SUPABASE_SERVICE_KEY):
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_KEY are required")

    processed_angles = process_images(images, image_paths, mode)
    right90 = processed_angles.get(ANALYSIS_ANGLE, {})

    if scan_id:
        update_supabase_scan(scan_id, processed_angles, right90)
        print(f"[handler] DB updated OK for scan_id={scan_id}")

    return {
        "status": "done",
        "scan_id": scan_id,
        "analysis_angle": ANALYSIS_ANGLE,
        "processed_angles": processed_angles,
    }


if __name__ == "__main__":
    if runpod is None:
        raise RuntimeError("runpod package is required to start the worker")
    runpod.serverless.start({"handler": handler})
