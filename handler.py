import runpod
import base64
from io import BytesIO
from PIL import Image, ImageOps
import onnxruntime as ort
from rembg import new_session, remove

# --- SANITY CHECK (beim Start des Workers) ---
print(f"ORT version: {ort.__version__}")
print(f"Available providers: {ort.get_available_providers()}")

# Session global definieren (wird pro Worker wiederverwendet)
# Versucht erst CUDA (GPU), dann CPU
try:
    session = new_session("u2net", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    print("Rembg session initialized with CUDA/CPU.")
except Exception as e:
    print(f"Failed to initialize GPU session, falling back: {e}")
    session = new_session("u2net")

def to_b64(img_obj):
    buffered = BytesIO()
    img_obj.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def process_image(image_b64):
    img_data = base64.b64decode(image_b64)
    img = Image.open(BytesIO(img_data)).convert("RGB")

    # --- KI LOGIK ---
    # Hier nutzt dein Developer jetzt die session:
    clean_img = remove(img, session=session) 
    
    # Simulation des zweiten Bildes (Redness)
    redness_img = ImageOps.colorize(ImageOps.grayscale(img), "#000055", "#ff0000")
    # --- KI LOGIK ENDE ---

    return {
        "clean_image": to_b64(clean_img),
        "redness_image": to_b64(redness_img)
    }

def handler(job):
    job_input = job['input']
    image_b64 = job_input.get('image')
    if not image_b64: return {"error": "No image provided"}

    try:
        return process_image(image_b64)
    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})