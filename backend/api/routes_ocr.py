from fastapi import APIRouter, File, UploadFile
from services.ocr_engine import ocr_with_deepseek

router = APIRouter(prefix="/ocr", tags=["OCR"])

@router.post("/")
async def ocr_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    result = ocr_with_deepseek(image_bytes)
    return result