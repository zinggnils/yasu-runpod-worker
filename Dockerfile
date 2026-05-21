# CPU worker: MODNet (ONNX) + CLIPSeg cheek ROI (PyTorch CPU) + MediaPipe landmarks.
FROM python:3.11-slim

# libgl1: OpenCV/mediapipe wheels may link libGL.so.1 even in serverless (no GUI).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ca-certificates \
    curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# Install headless OpenCV first, then the rest. mediapipe can otherwise pull
# opencv-contrib-python which requires libGL at import time on slim images.
RUN pip install --upgrade pip && \
    pip install "opencv-python-headless==4.10.0.84" && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt && \
    pip uninstall -y opencv-python opencv-contrib-python opencv-contrib-python-headless 2>/dev/null || true && \
    pip install --force-reinstall --no-deps "opencv-python-headless==4.10.0.84"

ENV HF_HOME=/root/.cache/huggingface
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface
RUN python -c "from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation; \
    CLIPSegProcessor.from_pretrained('CIDAS/clipseg-rd64-refined'); \
    CLIPSegForImageSegmentation.from_pretrained('CIDAS/clipseg-rd64-refined')"

# Pre-download MODNet portrait matting weights into the image so cold-starts
# don't pay the ~25 MB download every time. MODNet is the simple, fast
# matting model the May-4 golden pipeline used. MIT-licensed.
# `curl -fL` makes the build fail loudly if GitHub Releases ever moves the
# asset rather than producing a 0-byte file that would silently disable
# matting at runtime.
RUN mkdir -p /root/.modnet && \
    curl -fL https://github.com/yakhyo/modnet/releases/download/weights/modnet_photographic.onnx \
      -o /root/.modnet/modnet_photographic.onnx

RUN mkdir -p /root/.mediapipe && \
    curl -fL https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task \
      -o /root/.mediapipe/face_landmarker.task

COPY clipseg_cheek.py handler.py .
CMD ["python", "-u", "handler.py"]
