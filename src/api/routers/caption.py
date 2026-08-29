from fastapi import APIRouter, UploadFile, File
from src.api.schemas import CaptionResponse
from src.inference.caption_generate import generate_caption_from_bytes

router = APIRouter()


@router.post("/caption", response_model=CaptionResponse)
def generate_caption(image: UploadFile = File(...)):
    image_bytes = image.read()
    caption = generate_caption_from_bytes(image_bytes)
    return CaptionResponse(caption=caption)
