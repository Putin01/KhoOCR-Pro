import requests
import os

HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/deepseek-ai/DeepSeek-OCR"

def ocr_with_deepseek(image_bytes):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    response = requests.post(API_URL, headers=headers, data=image_bytes)
    if response.status_code == 200:
        return response.json()
    return {"error": "OCR failed", "status": response.status_code}