import runpod
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
from rembg import new_session, remove

# Session mit GPU-Support
session = new_session("u2net")

def to_b64(img_obj: Image.Image) -> str:
    buffered = BytesIO()
    img_obj.convert("RGB").save(buffered, format="JPEG", quality=95)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def apply_clinical_redness(img_np):
    """Der 'Blue Marker' Look vom 19.04."""
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    
    # Isoliere Rötungen im a-Kanal
    a_blurred = cv2.GaussianBlur(a, (5, 5), 0)
    min_val, max_val = np.percentile(a_blurred, [45, 98])
    redness_map = np.clip((a_blurred - min_val) / (max_val - min_val + 1e-5) * 255, 0, 255).astype(np.uint8)
    
    # Erstelle den bläulichen klinischen Look (COLORMAP_JET oder speziell gemischt)
    heatmap = cv2.applyColorMap(redness_map, cv2.COLORMAP_JET)
    return cv2.addWeighted(heatmap, 0.7, img_np, 0.3, 0)

def process_image(image_b64: str):
    # Original laden
    img_data = base64.b64decode(image_b64)
    original = Image.open(BytesIO(img_data)).convert("RGB")
    orig_np = np.array(original)

    # 1. Output: Clean (Hintergrund weg)
    clean_rgba = remove(original, session=session)
    clean_bg = Image.new("RGB", clean_rgba.size, (255, 255, 255))
    clean_bg.paste(clean_rgba, mask=clean_rgba.split()[3])

    # 2. Output: Redness Analysis (Der bläuliche Look aus dem Design)
    # Wir wenden die Analyse auf das freigestellte Gesicht an
    clean_np = np.array(clean_bg)
    redness_np = apply_clinical_redness(clean_np)
    redness_img = Image.fromarray(redness_np)

    # 3. Output: Skin Mask Toggle (Analyse ohne Hintergrundmaske)
    # Hier wird die Analyse auf das volle Originalbild angewendet
    full_redness_np = apply_clinical_redness(orig_np)
    full_redness_img = Image.fromarray(full_redness_np)

    return {
        "clean_image": to_b64(clean_bg),
        "redness_image": to_b64(redness_img),
        "full_analysis": to_b64(full_redness_img)
    }

def handler(job):
    job_input = job.get("input", {})
    image_b64 = job_input.get("image")
    if not image_b64: return {"error": "Kein Bild."}
    try:
        return process_image(image_b64)
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})