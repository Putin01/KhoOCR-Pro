from fastapi import FastAPI
from api.routes_ocr import router as ocr_router

app = FastAPI(title="KhoOCR Pro")

app.include_router(ocr_router)

@app.get("/")
def home():
    return {"message": "KhoOCR Pro - DeepSeek OCR API"}
