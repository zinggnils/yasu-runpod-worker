import runpod
import cv2
import numpy as np
from rembg import remove
import base64

def apply_clinical_duotone(img):
    # Umwandlung in Graustufen für Texturbetonung
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # CLAHE für klinischen Kontrast (Poren sichtbar machen)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_img = clahe.apply(gray)
    # DuoTone Mapping (Blau-Töne für medizinischen Look)
    colored = cv2.applyColorMap(contrast_img, cv2.COLORMAP_BONE)
    return colored

def handler(job):
    try:
        job_input = job['input']
        img_data = base64.b64decode(job_input['image'])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 1. Background Removal (High-End KI)
        no_bg_img = remove(img)
        
        # 2. Clinical Filter
        # Wir nehmen das Bild ohne Hintergrund und legen den Filter drüber
        clinical_img = apply_clinical_duotone(no_bg_img)
        
        # 3. Encoding
        _, buffer = cv2.imencode('.png', clinical_img)
        result_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return {"image": result_b64, "status": "success"}
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})