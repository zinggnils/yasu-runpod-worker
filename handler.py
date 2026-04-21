import runpod
import base64
from io import BytesIO
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import cv2
import onnxruntime as ort
from rembg import new_session, remove

# Sanity Checks beim Start
print(f"ORT version: {ort.__version__}")
print(f"Available providers: {ort.get_available_providers()}")

# Initialisiere rembg Session (Modell sollte durch Dockerfile bereits in /root/.u2net/ liegen)
try:
    session = new_session("u2net", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print("Rembg session initialized with CUDA/CPU.")
except Exception as e:
    print(f"Failed to initialize GPU session, falling back: {e}")
    session = new_session("u2net")

def to_b64(img_obj: Image.Image) -> str:
    """Konvertiert PIL Image in Base64 String (JPEG für Performance)"""
    buffered = BytesIO()
    img_obj.convert("RGB").save(buffered, format="JPEG", quality=90)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def analyze_redness_visia_style(clean_rgba: Image.Image) -> Image.Image:
    """
    Erstellt die medizinische Rötungsanalyse (VISIA-Style).
    Nutzt den LAB-Farbraum zur präzisen Hämoglobin-Isolierung.
    """
    # PIL zu OpenCV konvertieren
    img_np = np.array(clean_rgba.convert("RGB"))
    
    # LAB Farbraum Transformation
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # Fokus auf a-Kanal (Rotwerte)
    a_blurred = cv2.GaussianBlur(a_channel, (5, 5), 0)
    
    # Kontrastspreizung für die Visualisierung
    min_val = np.percentile(a_blurred, 45) # Basiswert Haut
    max_val = np.percentile(a_blurred, 98) # Spitzenwerte Rötung
    
    redness_map = np.clip((a_blurred - min_val) / (max_val - min_val + 1e-5) * 255, 0, 255).astype(np.uint8)

    # Heatmap generieren (Blau-Gelb-Rot Look)
    heatmap = cv2.applyColorMap(redness_map, cv2.COLORMAP_JET)
    
    # Überlagerung mit dem Original für Kontext
    result_np = cv2.addWeighted(heatmap, 0.75, img_np, 0.25, 0)

    return Image.fromarray(result_np)

def process_image(image_b64: str):
    """Verarbeitet das Bild und gibt 3 Varianten zurück"""
    # 1. Original Scan dekomprimieren & speichern
    img_data = base64.b64decode(image_b64)
    original_img = Image.open(BytesIO(img_data)).convert("RGB")

    # 2. Hintergrund entfernen (Clean)
    clean_rgba = remove(original_img, session=session).convert("RGBA")
    
    # Hintergrund für 'Clean' weiß füllen
    clean_bg = Image.new("RGB", clean_rgba.size, (255, 255, 255))
    clean_bg.paste(clean_rgba, mask=clean_rgba.split()[3])

    # 3. Rötungsanalyse (Redness)
    redness_img = analyze_redness_visia_style(clean_rgba)

    # Rückgabe aller drei Bilder
    return {
        "scan_image": to_b64(original_img),
        "clean_image": to_b64(clean_bg),
        "redness_image": to_b64(redness_img)
    }

def handler(job):
    job_input = job.get("input", {})
    image_b64 = job_input.get("image")
    
    if not image_b64:
        return {"error": "Kein Bild im Input gefunden (image_base64 erwartet)."}

    try:
        return process_image(image_b64)
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})