#!/usr/bin/env python3
"""
Manual prep runner — one portrait per angle, local artifacts only.

Examples:
  cd ~/yasu-runpod-worker
  python scripts/calibrate_angle.py --image ~/Desktop/right90.jpg --angle right_90

Outputs under ./calibration_out/<angle>_<timestamp>/:
  01_original.jpg
  02_portrait.jpg
  03_clean.jpg
  04_visia_redness.jpg
  05_center_crop_1000.jpg
  06_alpha.png
  prep.json
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run prep-only pipeline for one angle")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--angle", default="right_90", choices=handler.ANGLE_KEYS)
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "calibration_out" / f"{args.angle}_{ts}"
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
    crop.save(out_dir / "05_center_crop_1000.jpg", quality=95)

    rgb = np.array(clean.convert("RGB"))
    landmarks = handler.detect_face_landmarks(rgb)
    cheek_mask, cheek_method = handler.build_right_90_cheek_mask(
        rgb, alpha, landmarks
    )
    handler.render_cheek_cutout(clean, cheek_mask).save(
        out_dir / "07_cheek_cutout.png", quality=95
    )
    Image.fromarray((cheek_mask.astype(np.uint8) * 255)).save(
        out_dir / "08_cheek_mask.png"
    )

    prep = {
        "angle": args.angle,
        "analysis_step": "cheek_roi",
        "cheek_roi_method": cheek_method,
        "cheek_pixel_count": int(cheek_mask.sum()),
        "landmarks_detected": landmarks is not None,
        "matting": alpha is not None,
        **handler.compute_quality(clean, args.angle),
        "face_bbox": list(handler.detect_face_bbox(clean, args.angle) or ()),
    }
    (out_dir / "prep.json").write_text(json.dumps(prep, indent=2))
    print(json.dumps(prep, indent=2))
    print(f"\nWrote artifacts to {out_dir}")


if __name__ == "__main__":
    main()
