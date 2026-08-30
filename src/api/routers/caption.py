import asyncio
from fastapi import APIRouter, UploadFile, File, Depends, Request
from src.api.schemas import CaptionResponse
from src.model.image_captioner import ImageCaptioner

router = APIRouter()


def get_captioner(request: Request) -> ImageCaptioner:
    return request.app.state.captioner


@router.post("/caption", response_model=CaptionResponse)
async def generate_caption(image: UploadFile = File(...),
                           captioner: ImageCaptioner = Depends(get_captioner)):
    image_bytes = await image.read()
    caption = await asyncio.to_thread(captioner.generate_caption, image_bytes)
    return CaptionResponse(caption=caption)
