import runpod
import cv2
import numpy as np
from rembg import remove
import base64

def apply_clinical_duotone(img_rgba):
    # rembg liefert RGBA. Wir trennen die Kanäle.
    # img_rgba[:,:,3] ist die Transparenz-Maske
    b, g, r, a = cv2.split(img_rgba)
    rgb_img = cv2.merge((b, g, r))

    # Umwandlung in Graustufen für Texturbetonung
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    
    # CLAHE für klinischen Kontrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_img = clahe.apply(gray)
    
    # DuoTone Mapping (BONE für medizinischen Look)
    colored = cv2.applyColorMap(contrast_img, cv2.COLORMAP_BONE)
    
    # Die Transparenz wieder hinzufügen, damit der Hintergrund leer bleibt
    final_rgba = cv2.merge((colored[:,:,0], colored[:,:,1], colored[:,:,2], a))
    return final_rgba

def handler(job):
    try:
        job_input = job['input']
        img_data = base64.b64decode(job_input['image'])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Bild konnte nicht dekodiert werden"}

        # 1. Background Removal
        # Wir nutzen session-less remove für Stabilität im Serverless
        no_bg_img = remove(img)
        
        # 2. Clinical Filter
        clinical_img = apply_clinical_duotone(no_bg_img)
        
        # 3. Encoding als PNG (wichtig für Transparenz)
        _, buffer = cv2.imencode('.png', clinical_img)
        result_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return {"image": result_b64, "status": "success"}
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})