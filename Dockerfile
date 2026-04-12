# Nutze ein optimiertes Python-GPU Image
FROM python:3.11-slim

# System-Abhängigkeiten für OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Requirements installieren
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code kopieren
COPY handler.py .

# RunPod erwartet den Start des Handlers
CMD ["python", "-u", "handler.py"]