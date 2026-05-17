# YASU MVP (May 4 baseline)

## Pipeline

`decode` → `MODNet` background removal → `redness_score` → Supabase `scans` row

Angles: `frontal`, `left_45`, `left_90`, `right_45`, `right_90`

## This branch

`mvp-may4` @ `e14eec8b` — matches the good RunPod log (no MediaPipe, no face crop).

## RunPod deploy

1. Build/push from this branch (or tag `e14eec8b`).
2. Set env from `runpod.mvp.env` — **critical**: `ENABLE_FACE_CROP=false` (bad runs had face crop on).
3. Redeploy endpoint `rpkxri9favsg2y` (or update image).

## Good log signature

```
[handler] mode=redness
[process_single] Input size: (1920, 1008)
[run_modnet] Background matte generated
[process_single] redness_score=...
[handler] DB updated OK
```

## Repos (mvp-may4 branches)

| Repo | Commit |
|------|--------|
| yasu-runpod-worker | e14eec8b |
| yasu-web | 97e20fef |
| yasu-app | 0adaa770 |
| yasu-backend | 8a6905d + edge functions from main |

## Demo path

1. `cd yasu-web && npm run dev`
2. Mobile: Scans → New Scan → guided camera → Start Analysis
3. Open `/scans/{id}` for redness scores
