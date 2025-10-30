from PIL import Image
from io import BytesIO
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model nhẹ, chạy trên CPU (Render Free)
model = AutoModelForCausalLM.from_pretrained(
    "openbmb/MiniCPM-V-2_6",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="cpu"
)
tokenizer = AutoTokenizer.from_pretrained(
    "openbmb/MiniCPM-V-2_6",
    trust_remote_code=True
)
model.eval()

def ocr_with_minicpm(image_bytes: bytes) -> dict:
    try:
        # Mở ảnh
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        
        # Prompt thông minh
        msgs = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Extract all text from this Vietnamese invoice in structured JSON format."}
                ]
            }
        ]
        
        # OCR
        result = model.chat(
            image=image,
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=True,
            temperature=0.7,
            max_new_tokens=512
        )
        
        return {"structured_text": result}
    
    except Exception as e:
        return {"error": str(e)}