# Shadow tuning fixtures

Reference capture for side-shadow A/B (not committed — large file):

- `~/IMG_3897.png` — bad shadow (~28% centre `shadow_ratio` raw)

Run matrix + pick preset:

```bash
.venv/bin/python scripts/shadow_ab_matrix.py ~/IMG_3897.png --pick-best --out /tmp/shadow_3897.csv
```

Production defaults in `shadow_norm.py` match preset **`D_strong`** from that run (Dec 2025). Revert via env:

`SHADOW_CLAHE_CLIP=1.9` `SHADOW_DARK_THRESHOLD=95` `SHADOW_DARK_MULT=1.12` `SHADOW_DARK_LIFT=7.0`
