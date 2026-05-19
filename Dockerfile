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
# don't pay the 25 MB download every time. SHA256 from yakhyo/modnet release
# matches the May-4 golden pipeline.
RUN mkdir -p /root/.modnet && \
    curl -L https://github.com/yakhyo/modnet/releases/download/weights/modnet_photographic.onnx \
      -o /root/.modnet/modnet_photographic.onnx && \
    echo "5069a5e306b9f5e9f4f2b0360264c9f8ea13b257c7c39943c7cf6a2ec3a102ae  /root/.modnet/modnet_photographic.onnx" \
      | sha256sum -c -

COPY handler.py .
CMD ["python", "-u", "handler.py"]
