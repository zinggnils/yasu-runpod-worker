#!/usr/bin/env python3
"""
Manual calibration runner — one portrait per angle, local artifacts only.

Examples:
  cd ~/yasu-runpod-worker
  python scripts/calibrate_angle.py --image ~/Desktop/left45.heic --angle left_45
  python scripts/calibrate_angle.py --image ~/Desktop/right90.jpg --angle right_90 --mask ~/Desktop/cheek_mask.png

Outputs under ./calibration_out/<angle>_<timestamp>/:
  01_original.jpg
  02_portrait.jpg          (2160x2700 normalize)
  03_clean.jpg             (MODNet + studio, or portrait if no model)
  04_visia_redness.jpg     (false-color map)
  05_center_crop_1000.jpg  (current worker ROI)
  06_alpha.png             (if matting ran)
  scores.json              (redness/white/quality + mask override if --mask)
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import handler  # noqa: E402


def load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    """White pixels = cheek region to score."""
    m = Image.open(path).convert("L").resize(size, Image.Resampling.NEAREST)
    return np.array(m) > 127


def score_with_mask(clean: Image.Image, alpha: np.ndarray | None, cheek: np.ndarray) -> dict:
    rgb = np.array(clean.convert("RGB"))
    mask = cheek.copy()
    if alpha is not None:
        mask &= alpha > 200
    mask &= handler.skin_mask(rgb)
    if not np.any(mask):
        return {"redness_score": 0, "white_score": 0, "pixels": 0}

    lab = __import__("cv2").cvtColor(rgb, __import__("cv2").COLOR_RGB2LAB).astype(np.float32)
    a_star = lab[..., 1] - 128.0
    b_star = lab[..., 2] - 128.0
    lightness = lab[..., 0] * (100.0 / 255.0)
    chroma = np.sqrt(a_star * a_star + b_star * b_star)

    redness_raw = np.clip((a_star - 8.0) / 26.0, 0.0, 1.0)
    white = (lightness > 72.0) & (chroma < 16.0) & mask

    return {
        "redness_score": int(round(float(redness_raw[mask].mean()) * 100)),
        "white_score": int(round(float(white.sum()) / float(mask.sum()) * 100)),
        "pixels": int(mask.sum()),
        "a_star_mean": round(float(a_star[mask].mean()), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one angle through the worker pipeline locally.")
    parser.add_argument("--image", required=True, type=Path, help="HEIC/JPG/PNG input")
    parser.add_argument("--angle", required=True, choices=["left_45", "right_90", "frontal"])
    parser.add_argument(
        "--mask",
        type=Path,
        help="Optional cheek mask PNG (white=skin to score). Use after Photoshop/Remove BG.",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "calibration_out")
    args = parser.parse_args()

    if not args.image.exists():
        raise SystemExit(f"Missing image: {args.image}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out / f"{args.angle}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    original = Image.open(args.image)
    original.convert("RGB").save(out_dir / "01_original.jpg", quality=95)

    portrait = handler.normalize_portrait(original)
    portrait.save(out_dir / "02_portrait.jpg", quality=95)

    clean, alpha = handler.remove_background_and_finish(portrait)
    clean.save(out_dir / "03_clean.jpg", quality=95)
    if alpha is not None:
        Image.fromarray(alpha).save(out_dir / "06_alpha.png")

    visia = handler.make_analysis_map(clean, "redness", alpha=alpha)
    visia.save(out_dir / "04_visia_redness.jpg", quality=92)

    crop = handler.fixed_analysis_crop(clean)
    left = (clean.width - handler.ANALYSIS_CROP_SIZE) // 2
    top = (clean.height - handler.ANALYSIS_CROP_SIZE) // 2
    alpha_crop = None
    if alpha is not None:
        alpha_crop = alpha[top : top + handler.ANALYSIS_CROP_SIZE, left : left + handler.ANALYSIS_CROP_SIZE]
    crop.save(out_dir / "05_center_crop_1000.jpg", quality=95)

    overlay = handler.compute_redness_overlay(clean, alpha)
    redness_on_clean = handler.apply_redness_overlay(clean, overlay)
    redness_on_clean.save(out_dir / "08_redness_overlay.png")

    bbox = handler.detect_face_bbox(clean, args.angle)
    scores = {
        "angle": args.angle,
        "ita_full_frame": {
            "redness_score": handler.compute_redness_score_ita(clean, alpha),
            "white_score": handler.compute_white_score(clean, alpha),
            **handler.compute_quality(clean, args.angle),
            "scoring_method": "ita_full_frame",
        },
        "center_crop_legacy": {
            "redness_score": handler.compute_redness_score_ita(crop, alpha_crop),
            "white_score": handler.compute_white_score(crop, alpha_crop),
            **handler.compute_quality(crop, args.angle),
            "crop_box": {"x": left, "y": top, "width": 1000, "height": 1000},
        },
        "face_bbox": list(bbox) if bbox else None,
        "matting": alpha is not None,
        "tuning": {
            "formula": "ITA: median a* baseline on eroded skin, mean top erythema pixels",
            "env": "Adjust via handler constants / future env vars",
        },
    }

    if args.mask:
        cheek = load_mask(args.mask, (clean.width, clean.height))
        preview = np.array(clean.convert("RGB"))
        preview[~cheek] = (preview[~cheek] * 0.25).astype(np.uint8)
        Image.fromarray(preview).save(out_dir / "07_cheek_mask_preview.jpg", quality=95)
        scores["custom_cheek_mask"] = score_with_mask(clean, alpha, cheek)

    (out_dir / "scores.json").write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores, indent=2))
    print(f"\nWrote artifacts to {out_dir}")


if __name__ == "__main__":
    main()
