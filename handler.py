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
# MODNet portrait matting (May-4 golden pipeline)
# ---------------------------------------------------------------------------
# We bake `modnet_photographic.onnx` into the image at /root/.modnet/ (see
# Dockerfile). ORT CPU inference at 512px input takes ~400-600 ms per angle
# on a typical RunPod CPU pod; we run angles sequentially so ORT's internal
# thread pool can saturate all cores without contention.
MODNET_MODEL_PATH = os.environ.get(
    "MODNET_MODEL_PATH", "/root/.modnet/modnet_photographic.onnx"
)
MODNET_INPUT_SIZE = int(os.environ.get("MODNET_INPUT_SIZE", "512"))
# Sharpen strength for the final clean image. 1.15 matches the May-4 look:
# subtle Clarity pass + UnsharpMask makes pores/skin texture pop without
# halos around facial features.
SHARPEN_AMOUNT = float(os.environ.get("SHARPEN_AMOUNT", "1.15"))


def _init_modnet():
    model_path = Path(MODNET_MODEL_PATH)
    if not model_path.exists() or model_path.stat().st_size == 0:
        print(f"[modnet] Model not found at {model_path}; background removal DISABLED")
        return None
    try:
        opts = ort.SessionOptions()
        # Saturate the CPU on serial inference; we never run MODNet from
        # multiple threads simultaneously.
        opts.intra_op_num_threads = max(1, (os.cpu_count() or 4))
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess = ort.InferenceSession(
            str(model_path), opts, providers=["CPUExecutionProvider"]
        )
        print(f"[modnet] Session initialized providers={sess.get_providers()}")
        return sess
    except Exception as exc:  # noqa: BLE001
        print(f"[modnet] Init failed, background removal DISABLED: {exc}")
        return None


MODNET_SESSION = _init_modnet()


def _modnet_target_size(width: int, height: int, session) -> tuple[int, int]:
    """Pick the input size MODNet will actually run at.

    Some MODNet ONNX exports lock the input to 512x512; others expose dynamic
    H/W. When dynamic, pick a 32-multiple keeping the source aspect ratio with
    the long side near MODNET_INPUT_SIZE — the matte upsamples cleanly back
    to the original resolution via bilinear.
    """
    input_shape = session.get_inputs()[0].shape
    if (
        len(input_shape) == 4
        and isinstance(input_shape[2], int)
        and isinstance(input_shape[3], int)
    ):
        return int(input_shape[3]), int(input_shape[2])

    if width >= height:
        new_w = MODNET_INPUT_SIZE
        new_h = int(round(height / width * MODNET_INPUT_SIZE))
    else:
        new_h = MODNET_INPUT_SIZE
        new_w = int(round(width / height * MODNET_INPUT_SIZE))
    new_w = max(32, new_w - (new_w % 32))
    new_h = max(32, new_h - (new_h % 32))
    return new_w, new_h


def run_modnet(img_rgb: Image.Image):
    """Return uint8 alpha matte at the input resolution, or None if disabled."""
    if MODNET_SESSION is None:
        return None
    rgb = np.array(img_rgb.convert("RGB"))
    h, w = rgb.shape[:2]
    tw, th = _modnet_target_size(w, h, MODNET_SESSION)
    resized = cv2.resize(rgb, (tw, th), interpolation=cv2.INTER_AREA)
    # MODNet normalization: (x/255 - 0.5) / 0.5 = (x/127.5) - 1, NCHW float32.
    tensor = ((resized.astype(np.float32) / 255.0) - 0.5) / 0.5
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...]

    input_name = MODNET_SESSION.get_inputs()[0].name
    output_names = [o.name for o in MODNET_SESSION.get_outputs()]
    matte = MODNET_SESSION.run(output_names, {input_name: tensor})[0]
    matte = np.squeeze(matte)
    matte = cv2.resize(matte, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(matte * 255.0, 0, 255).astype(np.uint8)


def _guided_filter_alpha(
    guide_gray: np.ndarray, alpha: np.ndarray, radius: int = 6, eps: float = 1e-3
) -> np.ndarray:
    """Edge-aware refinement: snaps soft matte boundaries to real image edges.

    Eliminates the soft halo / fringing that pure MODNet output exhibits
    around hair and ears. Implementation is the standard guided filter
    (He et al.) applied with the source luma as the guide.
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
    """Run MODNet + guided-filter refinement + Clarity/Sharpen finish.

    Returns (finished_rgb, alpha_uint8 or None). If MODNet is unavailable
    we degrade gracefully and only apply the Clarity/Sharpen finish on the
    untouched RGB image so the pipeline still produces output.
    """
    alpha = run_modnet(img_rgb)
    if alpha is None:
        return _clarity_and_sharpen(img_rgb), None

    rgb_arr = np.array(img_rgb.convert("RGB"))
    gray = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2GRAY)
    alpha = _guided_filter_alpha(gray, alpha, radius=6, eps=1e-3)

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
    left = (img.width - ANALYSIS_CROP_SIZE) // 2
    top = (img.height - ANALYSIS_CROP_SIZE) // 2
    return img.crop((left, top, left + ANALYSIS_CROP_SIZE, top + ANALYSIS_CROP_SIZE))


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
    """Combine brightness gating with MODNet alpha so scoring only sees
    actual face pixels — no background, no hair edge."""
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

    `prepared` carries everything MODNet/normalize already produced; this
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
        crop = fixed_analysis_crop(clean)
        alpha_crop = None
        if alpha is not None:
            left = (clean.width - ANALYSIS_CROP_SIZE) // 2
            top = (clean.height - ANALYSIS_CROP_SIZE) // 2
            alpha_crop = alpha[top : top + ANALYSIS_CROP_SIZE, left : left + ANALYSIS_CROP_SIZE]
        crop_url = upload_webp_lossless(crop, f"right90_crop_{uid}.webp")
        angle_data.update(
            {
                "crop_image_url": crop_url,
                "crop_box": {
                    "x": (clean.width - ANALYSIS_CROP_SIZE) // 2,
                    "y": (clean.height - ANALYSIS_CROP_SIZE) // 2,
                    "width": ANALYSIS_CROP_SIZE,
                    "height": ANALYSIS_CROP_SIZE,
                },
                "redness_score": compute_redness_score(crop, alpha_crop),
                "white_score": compute_white_score(crop, alpha_crop),
                **compute_quality(crop),
            }
        )

    print(
        f"[handler] {label} OK"
        + (f" redness={angle_data.get('redness_score')}" if label == ANALYSIS_ANGLE else "")
    )
    return label, angle_data


def process_images(images: dict, image_paths: dict, mode: str = "redness") -> dict:
    """Two-phase pipeline:

    Phase 1 (sequential): per angle — download original, normalize to the
        portrait frame, run MODNet matting + Clarity/UnsharpMask finish.
        MODNet on CPU already saturates all cores via ORT's intra-op pool;
        running multiple inferences concurrently would thrash, so serialize.

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
            + ("OK" if alpha is not None else "SKIPPED (no MODNet)")
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
