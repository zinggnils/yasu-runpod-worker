import runpod
import cv2
import numpy as np
from rembg import remove, new_session
import base64

# Session einmal global laden für Speed
session = new_session()

def apply_clinical_duotone(img_rgba):
    b, g, r, a = cv2.split(img_rgba)
    rgb_img = cv2.merge((b, g, r))
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_img = clahe.apply(gray)
    colored = cv2.applyColorMap(contrast_img, cv2.COLORMAP_BONE)
    return cv2.merge((colored[:,:,0], colored[:,:,1], colored[:,:,2], a))

def handler(job):
    try:
        job_input = job['input']
        img_data = base64.b64decode(job_input['image'])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return {"error": "Bild-Dekodierung fehlgeschlagen"}

        # 1. Background Removal mit globaler Session
        no_bg_img = remove(img, session=session)
        
        # 2. Filter
        clinical_img = apply_clinical_duotone(no_bg_img)
        
        # 3. Encoding
        _, buffer = cv2.imencode('.png', clinical_img)
        result_b64 = base64.b64encode(buffer).decode('utf-8')
        
        return {"image": result_b64, "status": "success"}
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})