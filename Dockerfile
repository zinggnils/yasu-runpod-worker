# CPU-only image: this worker does PIL/OpenCV processing + ONNX Runtime CPU
# inference for RVM portrait matting. No CUDA/PyTorch needed.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgomp1 ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Pre-download Robust Video Matting (RVM) MobileNetV3 weights into the image
# so cold-starts don't pay the ~14 MB download every time. RVM produces
# noticeably better hair-edge mattes than MODNet on still portraits while
# staying small enough for CPU inference. MIT-licensed.
# `curl -fL` makes the build fail loudly if GitHub Releases ever moves the
# asset rather than producing a 0-byte file that would silently disable
# matting at runtime.
RUN mkdir -p /root/.rvm && \
    curl -fL https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx \
      -o /root/.rvm/rvm_mobilenetv3_fp32.onnx

COPY handler.py .
CMD ["python", "-u", "handler.py"]
