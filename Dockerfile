# Offizielles Image nutzen, um 'pull access denied' zu vermeiden 
FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libcudnn8 curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Modelle vorab laden für Speed und Offline-Fallback
RUN mkdir -p /root/.u2net /root/.modnet && \
    curl -L https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx -o /root/.u2net/u2net.onnx && \
    curl -L https://github.com/yakhyo/modnet/releases/download/weights/modnet_photographic.onnx -o /root/.modnet/modnet_photographic.onnx && \
    echo "5069a5e306b9f5e9f4f2b0360264c9f8ea13b257c7c39943c7cf6a2ec3a102ae  /root/.modnet/modnet_photographic.onnx" | sha256sum -c -

COPY handler.py .
CMD ["python", "-u", "handler.py"]