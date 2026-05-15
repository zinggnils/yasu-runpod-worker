"""Standard face framing targets (aligned with yasu-web CameraCapture distance bands)."""
from __future__ import annotations

from typing import Optional

# Web: redness close-up vs texture; profile steps use same size band per mode.
MODE_FACE_SIZE = {
    "redness": {"min": 0.52, "max": 0.72, "ideal": 0.62},
    "texture": {"min": 0.38, "max": 0.55, "ideal": 0.46},
    "pigmentation": {"min": 0.38, "max": 0.55, "ideal": 0.46},
}

DEFAULT_MODE = "redness"

# Normalized face-center X targets per angle index (0=frontal … 4=right 90).
ANGLE_TARGET_CENTER_X = {
    0: 0.50,
    1: 0.58,
    2: 0.64,
    3: 0.42,
    4: 0.36,
}

ANGLE_LABELS = {
    0: "frontal",
    1: "left_45",
    2: "left_90",
    3: "right_45",
    4: "right_90",
}

LABEL_TO_INDEX = {v: k for k, v in ANGLE_LABELS.items()}

CENTER_X_TOLERANCE = 0.09
FRONTAL_RETAKE_CONFIDENCE = 0.45


def mode_key(mode: Optional[str]) -> str:
    m = (mode or DEFAULT_MODE).lower()
    return m if m in MODE_FACE_SIZE else DEFAULT_MODE


def size_band(mode: Optional[str]) -> dict:
    return MODE_FACE_SIZE[mode_key(mode)]


def angle_index(label: str) -> int:
    return LABEL_TO_INDEX.get(label, 0)


def repositioning_hint(
    label: str,
    mode: Optional[str],
    face_width_norm: float,
    face_center_x_norm: float,
) -> str:
    """Short operator hint for zoom / turn (empty if within standard band)."""
    band = size_band(mode)
    idx = angle_index(label)
    target_x = ANGLE_TARGET_CENTER_X.get(idx, 0.5)

    if face_width_norm < band["min"] - 0.03:
        return "Move closer" if mode_key(mode) == "redness" else "Move closer to the guide"
    if face_width_norm > band["max"] + 0.03:
        return "Move back"

    dx = face_center_x_norm - target_x
    if idx == 0:
        if abs(dx) > CENTER_X_TOLERANCE:
            return "Center your face in the oval"
        return ""

    # Left profiles: face center should sit right of frame center.
    if idx in (1, 2):
        if dx < -CENTER_X_TOLERANCE:
            return "Turn more to your left"
        if dx > CENTER_X_TOLERANCE + 0.04:
            return "Turn slightly back toward camera"
        return ""

    # Right profiles.
    if idx in (3, 4):
        if dx > CENTER_X_TOLERANCE:
            return "Turn more to your right"
        if dx < -(CENTER_X_TOLERANCE + 0.04):
            return "Turn slightly back toward camera"
        return ""

    return ""
