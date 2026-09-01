import asyncio
import logging
from PIL import UnidentifiedImageError
from fastapi import APIRouter, UploadFile, File, Depends, Request, HTTPException
from src.api.schemas import CaptionResponse
from src.model.image_captioner import ImageCaptioner

logger = logging.getLogger(__name__)
router = APIRouter()


def get_captioner(request: Request) -> ImageCaptioner:
    return request.app.state.captioner


@router.post("/caption", response_model=CaptionResponse)
async def generate_caption(image: UploadFile = File(...),
                           captioner: ImageCaptioner = Depends(get_captioner)):
    image_bytes = await image.read()
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
