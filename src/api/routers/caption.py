import asyncio
import logging
from PIL import UnidentifiedImageError
from fastapi import APIRouter, UploadFile, File, Depends, Request, HTTPException
from src.api.schemas import CaptionResponse
from src.model.image_captioner import ImageCaptioner

logger = logging.getLogger(__name__)
router = APIRouter()

MAX_IMAGE_SIZE = 10 * 1024 * 1024
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
CHUNK_SIZE = 1024 * 1024


def get_captioner(request: Request) -> ImageCaptioner:
    return request.app.state.captioner


@router.post("/caption", response_model=CaptionResponse)
async def generate_caption(image: UploadFile = File(...),
                           captioner: ImageCaptioner = Depends(get_captioner)):

    if not image.content_type or image.content_type.lower() not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported file type: {image.content_type}. Allowed image types: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}"
        )
    buffer = bytearray()
    while chunk := await image.read(CHUNK_SIZE):
        buffer.extend(chunk)
        if len(buffer) >= MAX_IMAGE_SIZE:
            raise HTTPException(
                status_code=413,
                detail=f"File to large. Maximum allowed size is {MAX_IMAGE_SIZE // (1024 * 1024)}"
            )

    image_bytes = bytes(buffer)
    try:
        caption = await asyncio.to_thread(captioner.generate_caption, image_bytes)
        return CaptionResponse(caption=caption)
    except (UnidentifiedImageError, ValueError) as e:
        logger.warning(f"Invalid image uploaded: {e}")
        raise HTTPException(
            status_code=400,
            detail="Invalid image file or unsupported image format"
        )

    except Exception as e:
        logger.exception("Unexpected error during caption generation: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal error occurred while generating image caption"
        )
