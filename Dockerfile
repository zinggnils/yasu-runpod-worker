# CPU-only image: this worker does PIL/OpenCV processing + ONNX Runtime CPU
# inference for MODNet portrait matting. No CUDA/PyTorch needed.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgomp1 ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Pre-download MODNet portrait matting weights into the image so cold-starts
# don't pay the ~25 MB download every time. MODNet is the simple, fast
# matting model the May-4 golden pipeline used. MIT-licensed.
# `curl -fL` makes the build fail loudly if GitHub Releases ever moves the
# asset rather than producing a 0-byte file that would silently disable
# matting at runtime.
RUN mkdir -p /root/.modnet && \
    curl -fL https://github.com/yakhyo/modnet/releases/download/weights/modnet_photographic.onnx \
      -o /root/.modnet/modnet_photographic.onnx

RUN mkdir -p /root/.face-parse && \
    curl -fL https://github.com/yakhyo/face-parsing/releases/download/weights/resnet18.onnx \
      -o /root/.face-parse/resnet18.onnx

COPY handler.py .
CMD ["python", "-u", "handler.py"]
