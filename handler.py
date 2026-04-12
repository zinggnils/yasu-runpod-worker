import runpod
import cv2
import numpy as np
from rembg import remove
import base64
from PIL import Image
import io

def handler(job):
    try:
        # 1. Daten aus dem Request extrahieren
        job_input = job['input']
        image_b64 = job_input.get('image')
        
        if not image_b64:
            return {"error": "Kein Bild erhalten"}

        # 2. Base64 zu Bytes umwandeln
        img_bytes = base64.b64decode(image_b64)
        
        # 3. BACKGROUND REMOVAL (KI-gestützt auf GPU)
        # Hier nutzt er automatisch die NVIDIA Power
        no_bg_bytes = remove(img_bytes)

        # 4. CLINICAL FILTER / DUOTONE (Vorbereitung)
        # Hier laden wir später MediaPipe für die Face Detection nach
        # Für den Start schicken wir das ausgeschnittene Bild zurück
        
        # 5. Ergebnis zurück zu Base64
        result_b64 = base64.b64encode(no_bg_bytes).decode('utf-8')
        
        return {"image": result_b64, "status": "success"}

    except Exception as e:
        return {"error": str(e)}

# RunPod Loop starten
runpod.serverless.start({"handler": handler})