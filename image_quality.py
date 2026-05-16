"""Capture quality gates (blur, face, shadow ratio, ship confidence)."""
from __future__ import annotations

import os

import cv2
import numpy as np
from PIL import Image

try:
    import mediapipe as mp

    MEDIAPIPE_AVAILABLE = True
except Exception:
    mp = None
    MEDIAPIPE_AVAILABLE = False

BLUR_MIN_SCORE = float(os.environ.get("BLUR_MIN_SCORE", "42"))
QUALITY_MIN_CONFIDENCE = float(os.environ.get("QUALITY_MIN_CONFIDENCE", "0.28"))
SHADOW_RATIO_WARN = float(os.environ.get("SHADOW_RATIO_WARN", "0.12"))


def centre_gray_metrics(rgb: np.ndarray) -> dict:
    """Same centre crop as check_image_quality — blur, shadow_ratio, exposure."""
    h, w = rgb.shape[:2]
    cx, cy = w // 2, h // 2
    crop_size = min(w, h) // 2
    centre = rgb[
        cy - crop_size // 2 : cy + crop_size // 2,
        cx - crop_size // 2 : cx + crop_size // 2,
    ]
    gray = cv2.cvtColor(centre, cv2.COLOR_RGB2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    shadow_ratio = float(np.mean(gray < 42))
    exposure = float(np.mean(gray) / 255.0)
    return {
        "blur_score": blur_score,
        "shadow_ratio": shadow_ratio,
        "exposure": exposure,
    }


def confidence_from_metrics(metrics: dict, face_score: float) -> float:
    blur_score = float(metrics["blur_score"])
    shadow_ratio = float(metrics["shadow_ratio"])
    exposure = float(metrics["exposure"])
    blur_norm = float(np.clip((blur_score - BLUR_MIN_SCORE * 0.8) / 125.0, 0.0, 1.0))
    face_norm = float(np.clip((face_score - 0.35) / 0.65, 0.0, 1.0))
    shadow_penalty = float(np.clip((shadow_ratio - SHADOW_RATIO_WARN) / 0.55, 0.0, 1.0))
    exposure_penalty = float(np.clip(abs(exposure - 0.48) / 0.35, 0.0, 1.0))
    confidence = float(np.clip(0.58 * blur_norm + 0.30 * face_norm + 0.12 * (1.0 - shadow_penalty), 0.0, 1.0))
    return float(confidence * (1.0 - 0.25 * exposure_penalty))


def gate_decision(metrics: dict, face_score: float) -> tuple[bool, str, float]:
    """Returns (ok, reason, confidence) using production thresholds."""
    blur_score = float(metrics["blur_score"])
    if blur_score < BLUR_MIN_SCORE * 0.78:
        return False, f"Image too blurry (score={blur_score:.0f}, min={BLUR_MIN_SCORE:.0f})", 0.0
    if face_score <= 0.0:
        return False, "No face detected in image", 0.0
    confidence = confidence_from_metrics(metrics, face_score)
    if confidence < QUALITY_MIN_CONFIDENCE:
        return (
            False,
            f"unstable capture confidence={confidence:.2f} (min={QUALITY_MIN_CONFIDENCE:.2f})",
            confidence,
        )
    return True, "ok", confidence


def detect_face_detection(rgb: np.ndarray):
    """Largest face detection or None."""
    if not MEDIAPIPE_AVAILABLE or mp is None:
        return None
    h, w = rgb.shape[:2]
    with mp.solutions.face_detection.FaceDetection(
        model_selection=1,
        min_detection_confidence=0.35,
    ) as detector:
        results = detector.process(rgb)
    if not results.detections:
        return None
    best = results.detections[0]
    best_area = 0.0
    for det in results.detections:
        bb = det.location_data.relative_bounding_box
        area = float(bb.width) * float(bb.height)
        if area > best_area:
            best_area = area
            best = det
    bb = best.location_data.relative_bounding_box
    xmin = float(bb.xmin)
    ymin = float(bb.ymin)
    bw = float(bb.width)
    bh = float(bb.height)
    return {
        "face_score": float(best.score[0]) if best.score else 0.0,
        "face_width_norm": bw,
        "face_height_norm": bh,
        "face_center_x_norm": xmin + bw / 2.0,
        "face_center_y_norm": ymin + bh / 2.0,
        "image_width": w,
        "image_height": h,
    }


def detect_face_score(rgb: np.ndarray) -> float:
    det = detect_face_detection(rgb)
    return float(det["face_score"]) if det else 0.0


def check_image_quality(img: Image.Image, *, quiet: bool = False) -> tuple[bool, str, dict]:
    """Returns (ok, reason, metrics). May 4 MVP: warn on issues but never block processing."""
    rgb = np.array(img.convert("RGB"))
    metrics = centre_gray_metrics(rgb)
    blur_score = metrics["blur_score"]
    if not quiet:
        print(f"[quality_gate] blur_score={blur_score:.1f}")

    if blur_score < BLUR_MIN_SCORE * 0.78:
        if not quiet:
            print(f"[quality_gate] WARN blurry score={blur_score:.0f} — processing anyway")
        return True, f"warn: blurry ({blur_score:.0f})", {
            **metrics,
            "face_score": 0.0,
            "confidence": 0.35,
        }

    det = detect_face_detection(rgb)
    if not det or det["face_score"] <= 0.0:
        if not quiet:
            print("[quality_gate] WARN no face / mediapipe unavailable — processing anyway")
        return True, "warn: no face detect", {
            **metrics,
            "face_score": 0.0,
            "confidence": 0.4,
        }

    face_score = float(det["face_score"])
    confidence = confidence_from_metrics(metrics, face_score)
    out = {
        **metrics,
        "face_score": face_score,
        "confidence": confidence,
        "face_width_norm": det["face_width_norm"],
        "face_height_norm": det["face_height_norm"],
        "face_center_x_norm": det["face_center_x_norm"],
        "face_center_y_norm": det["face_center_y_norm"],
    }
    ok, reason, _ = gate_decision(metrics, face_score)
    if not ok:
        if not quiet:
            print(f"[quality_gate] WARN {reason} — processing anyway (May 4 MVP)")
        return True, f"warn: {reason}", out
    if not quiet:
        print(
            f"[quality_gate] OK — blur={blur_score:.0f}, face={face_score:.2f}, confidence={confidence:.2f}"
        )
    return True, "ok", out
