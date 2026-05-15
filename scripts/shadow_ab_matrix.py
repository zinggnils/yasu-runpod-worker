#!/usr/bin/env python3
"""
A/B matrix for normalize_shadows on real captures.

Quality gates in production run on the *raw* image (before shadow norm). This script
reports raw vs post-norm metrics so you can tune CLAHE/lift without letting bad
frames "sneak through" if gates were ever moved post-norm.

Usage:
  python scripts/shadow_ab_matrix.py /path/to/images [--out report.csv]
  SHADOW_PRESET=C_balanced python scripts/shadow_ab_matrix.py ./fixtures
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from image_quality import (  # noqa: E402
    BLUR_MIN_SCORE,
    QUALITY_MIN_CONFIDENCE,
    centre_gray_metrics,
    check_image_quality,
    gate_decision,
)
from shadow_norm import SHADOW_AB_PRESETS, ShadowNormParams, normalize_shadows  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}


def iter_images(folder: Path):
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() in IMAGE_EXTS and p.is_file():
            yield p


def load_rgb(path: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def row_for(path: Path, preset_name: str, params: ShadowNormParams) -> dict:
    img = load_rgb(path)
    raw_ok, raw_reason, raw_metrics = check_image_quality(img, quiet=True)
    raw_shadow = float(raw_metrics.get("shadow_ratio", 0))

    normed = normalize_shadows(img, params)
    post_metrics = centre_gray_metrics(np.array(normed.convert("RGB")))
    post_shadow = float(post_metrics["shadow_ratio"])
    face_score = float(raw_metrics.get("face_score", 0))

    # Hypothetical: gate applied after normalization (must not rescue rejects).
    post_ok, post_reason, post_conf = gate_decision(post_metrics, face_score)

    rescued = (not raw_ok) and post_ok
    shadow_delta = raw_shadow - post_shadow

    return {
        "file": path.name,
        "preset": preset_name,
        "params": params.label(),
        "raw_ok": raw_ok,
        "raw_reason": raw_reason,
        "raw_shadow_ratio": round(raw_shadow, 4),
        "raw_confidence": round(float(raw_metrics.get("confidence", 0)), 3),
        "post_shadow_ratio": round(post_shadow, 4),
        "shadow_delta": round(shadow_delta, 4),
        "post_ok_if_gate_after_norm": post_ok,
        "post_reason_if_gate_after_norm": post_reason,
        "post_confidence_if_gate_after_norm": round(post_conf, 3),
        "rescued_bad_frame": rescued,
    }


def main():
    parser = argparse.ArgumentParser(description="Shadow normalization A/B matrix")
    parser.add_argument("image_dir", type=Path, help="Folder of real captures (recursive)")
    parser.add_argument("--out", type=Path, default=None, help="CSV output path (default: stdout)")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress stderr summary (CSV still written)",
    )
    parser.add_argument(
        "--presets",
        nargs="*",
        default=list(SHADOW_AB_PRESETS.keys()),
        help=f"Preset keys (default: all). Available: {', '.join(SHADOW_AB_PRESETS)}",
    )
    args = parser.parse_args()

    if not args.image_dir.is_dir():
        raise SystemExit(f"Not a directory: {args.image_dir}")

    presets = []
    for key in args.presets:
        if key not in SHADOW_AB_PRESETS:
            raise SystemExit(f"Unknown preset {key!r}. Choose from: {list(SHADOW_AB_PRESETS)}")
        presets.append((key, SHADOW_AB_PRESETS[key]))

    paths = list(iter_images(args.image_dir))
    if not paths:
        raise SystemExit(f"No images under {args.image_dir}")

    rows: list[dict] = []
    for path in paths:
        for name, params in presets:
            rows.append(row_for(path, name, params))

    fieldnames = list(rows[0].keys())
    rescued = [r for r in rows if r["rescued_bad_frame"]]
    if not args.quiet:
        print(
            f"# images={len(paths)} presets={len(presets)} rows={len(rows)} "
            f"blur_min={BLUR_MIN_SCORE} conf_min={QUALITY_MIN_CONFIDENCE}",
            file=sys.stderr,
        )
        if rescued:
            print(
                f"# WARNING: {len(rescued)} row(s) would pass gate post-norm but fail raw:",
                file=sys.stderr,
            )
            for r in rescued[:20]:
                print(f"  {r['file']} [{r['preset']}]: {r['raw_reason']}", file=sys.stderr)
        elif rows:
            passed = sum(1 for r in rows if r["raw_ok"] and r["preset"] == presets[0][0])
            print(f"# raw_ok (baseline preset): {passed}/{len(paths)} images", file=sys.stderr)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {args.out} ({len(rows)} rows)", file=sys.stderr)
    else:
        w = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
