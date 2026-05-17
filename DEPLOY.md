# RunPod deployment

## Stable pipeline (May 4 golden path)

Default handler behavior matches scan `270526c8`:

- EXIF-corrected **full frame** → MODNet → redness score
- No face crop or quality gate unless enabled

Set on the RunPod endpoint (recommended):

```bash
ENABLE_FACE_CROP=false
ENABLE_QUALITY_GATE=false
TARGET_MAX_PX=1920
BG_REMOVAL_BACKEND=modnet
```

## GPU fleet

Restrict the endpoint to GPUs supported by the container PyTorch build (e.g. RTX A5000 / Ada, **not** Blackwell sm_120) until PyTorch is upgraded for sm_120.

Blackwell workers fail fitness checks with: `no kernel image is available for execution on the device`.

## Optional experiments

```bash
ENABLE_FACE_CROP=true    # MediaPipe 800×800 crop before MODNet (legacy MVP)
ENABLE_QUALITY_GATE=true # blur + face warn-only checks
```
