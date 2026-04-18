import runpod
import base64
from io import BytesIO
from PIL import Image, ImageOps

def to_b64(img_obj):
    buffered = BytesIO()
    img_obj.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def process_image(image_b64):
    # 1. Base64 zu PIL Image konvertieren
    img_data = base64.b64decode(image_b64)
    img = Image.open(BytesIO(img_data)).convert("RGB")

    # --- KI LOGIK ---
    # Hier kommen später deine echten Modelle rein.
    # Für den Test simulieren wir die zwei Outputs:
    
    # Output 1: "Clean" (hier könntest du Background Removal machen)
    clean_img = img.copy() 
    
    # Output 2: "Redness" (Simulation des DuoTone Effekts via Filter)
    redness_img = ImageOps.colorize(ImageOps.grayscale(img), "#000055", "#ff0000")
    # --- KI LOGIK ENDE ---

    return {
        "clean_image": to_b64(clean_img),
        "redness_image": to_b64(redness_img)
    }

def handler(job):
    job_input = job['input']
    image_b64 = job_input.get('image')

    if not image_b64:
        return {"error": "No image provided"}

    try:
        # Führt die Verarbeitung aus und liefert clean_image & redness_image
        return process_image(image_base64)
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})