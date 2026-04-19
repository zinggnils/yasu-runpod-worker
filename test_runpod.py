import requests
import base64
import time

# Deine Daten
API_KEY = "rpa_8TVPMOV730I23N94FIOQAPAH1NZ2MVIR3JNT1RWN1svcml"
ENDPOINT_ID = "rpkxri9favsg2y"
IMAGE_PATH = "/Users/nilszingg/test.png"

def test_gpu():
    with open(IMAGE_PATH, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")

    url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/run" # Wir nutzen /run (async)
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    payload = {"input": {"image": img_base64}}

    print("🚀 Sende Job an RunPod...")
    res = requests.post(url, json=payload, headers=headers)
    job_id = res.json().get("id")
    print(f"✅ Job ID: {job_id}. Warte auf Ergebnis (kann beim 1. Mal dauern)...")

    # Polling: Wir fragen alle 5 Sek nach, ob es fertig ist
    status_url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/status/{job_id}"
    while True:
        status_res = requests.get(status_url, headers=headers).json()
        status = status_res.get("status")
        print(f"⏳ Status: {status}")
        
        if status == "COMPLETED":
            print("🎉 ERFOLG!")
            print(f"Ergebnis (ersten 100 Zeichen): {str(status_res.get('output'))[:100]}")
            break
        elif status == "FAILED":
            print(f"❌ FEHLER: {status_res.get('error')}")
            break
        
        time.sleep(5)

if __name__ == "__main__":
    test_gpu()