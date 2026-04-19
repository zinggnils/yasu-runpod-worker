import runpod
import base64
from io import BytesIO
from PIL import Image, ImageOps

def to_b64(img_obj):
    """Konvertiert ein PIL Image Objekt in einen Base64 String."""
    buffered = BytesIO()
    img_obj.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def process_image(image_b64):
    """Hier findet die eigentliche Bildverarbeitung statt."""
    # 1. Base64 zu PIL Image konvertieren
    img_data = base64.b64decode(image_b64)
    img = Image.open(BytesIO(img_data)).convert("RGB")

    # --- KI LOGIK PLATZHALTER ---
    # Hier kann dein Developer die echten Modelle einfügen.
    
    # Output 1: "Clean" (Simulation: Original kopieren)
    clean_img = img.copy() 
    
    # Output 2: "Redness" (Simulation: DuoTone Effekt via Filter)
    redness_img = ImageOps.colorize(ImageOps.grayscale(img), "#000055", "#ff0000")
    # --- KI LOGIK ENDE ---

    # Rückgabe beider Bilder als Base64 im JSON-Format
    return {
        "clean_image": to_b64(clean_img),
        "redness_image": to_b64(redness_img)
    }

def handler(job):
    """Einstiegspunkt für RunPod Serverless."""
    job_input = job['input']
    image_b64 = job_input.get('image')

    if not image_b64:
        return {"error": "No image provided"}

    try:
        # Führt die Verarbeitung aus (Variable korrekt übergeben)
        results = process_image(image_b64)
        return results 
    except Exception as e:
        return {"error": str(e)}

# Startet den RunPod Serverless Service
runpod.serverless.start({"handler": handler})