import runpod
import cv2
import numpy as np
import base64
from rembg import remove

def apply_clinical_duotone(img_rgba):
    # Konvertierung von RGBA zu BGR (OpenCV Standard)
    b, g, r, a = cv2.split(img_rgba)
    rgb_img = cv2.merge((b, g, r))
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_img = clahe.apply(gray)
    colored = cv2.applyColorMap(contrast_img, cv2.COLORMAP_BONE)
    # Transparenz wieder hinzufügen
    return cv2.merge((colored[:,:,0], colored[:,:,1], colored[:,:,2], a))

def handler(job):
    try:
        # Sicherstellen, dass Input da ist
        job_input = job.get('input', {})
        if 'image' not in job_input:
            return {"error": "Kein Bild empfangen"}

        # Bild dekodieren
        img_data = base64.b64decode(job_input['image'])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Bild konnte nicht gelesen werden"}

        # 1. Hintergrund entfernen
        no_bg = remove(img)
        
        # 2. Klinischer Filter
        final_img = apply_clinical_duotone(no_bg)
        
        # 3. Encoding
        _, buffer = cv2.imencode('.png', final_img)
        result_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return {"image": result_b64}
    except Exception as e:
        return {"error": f"Worker-Absturz: {str(e)}"}

runpod.serverless.start({"handler": handler})