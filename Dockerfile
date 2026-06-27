# Lightweight CPU worker: MODNet ONNX + Gemini API for cheek fragment on VISIA.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    ca-certificates \
    curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install "opencv-python-headless==4.10.0.84" && \
    pip install -r requirements.txt && \
    pip uninstall -y opencv-python opencv-contrib-python opencv-contrib-python-headless 2>/dev/null || true && \
    pip install --force-reinstall --no-deps "opencv-python-headless==4.10.0.84"

RUN mkdir -p /root/.modnet && \
    curl -fL https://github.com/yakhyo/modnet/releases/download/weights/modnet_photographic.onnx \
      -o /root/.modnet/modnet_photographic.onnx

COPY handler.py gemini_fragment.py shadow_enhance.py ./
CMD ["python", "-u", "handler.py"]
