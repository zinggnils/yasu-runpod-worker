# Wir nutzen ein Image, das CUDA und CUDNN bereits korrekt enthält
FROM dev-runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# System-Abhängigkeiten für OpenCV und CUDA-Support
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libcudnn8 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Modell vorab laden (verhindert den Timeout beim ersten Scan)
RUN mkdir -p /root/.u2net && \
    curl -L https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx -o /root/.u2net/u2net.onnx

COPY handler.py .
CMD ["python", "-u", "handler.py"]