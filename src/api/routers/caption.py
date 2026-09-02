import time
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
        logger.warning("Usupported file type", extra={
                       "content_type": image.content_type, "filename": image.filename})
        raise HTTPException(
            status_code=400,
            detail=f"unsupported file type: {image.content_type}. Allowed image types: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}"
        )
    buffer = bytearray()
    while chunk := await image.read(CHUNK_SIZE):
        buffer.extend(chunk)
        if len(buffer) >= MAX_IMAGE_SIZE:
            logger.warning("Upload exceeded maximum allowed size",
                           extra={"bytes_read": len(
                               buffer), "max_size": MAX_IMAGE_SIZE}
                           )
            raise HTTPException(
                status_code=413,
                detail=f"File to large. Maximum allowed size is {MAX_IMAGE_SIZE // (1024 * 1024)}"
            )

    image_bytes = bytes(buffer)
    start_time = time.perf_counter()

    try:
        caption = await asyncio.to_thread(captioner.generate_caption, image_bytes)
        latency = round((time.perf_counter() - start_time) * 1000, 2)
        logger.info("Image caption generated successfully",
                    extra={"image_size": len(image_bytes),
                           "caption_length": len(caption),
                           "latency": latency})
        return CaptionResponse(caption=caption)

    except (UnidentifiedImageError, ValueError) as e:
        latency = round((time.perf_counter() - start_time) * 1000, 2)
        logger.warning("Image decoding failure",
                       extra={"error": str(e), "latency": latency})
        raise HTTPException(
            status_code=400,
            detail="Invalid image file or unsupported image format"
        )

    except Exception as e:
        latency = round((time.perf_counter() - start_time) * 1000, 2)
        logger.exception("Unexpected error during caption generation",
                         extra={"error": str(e), "latency": latency})
        raise HTTPException(
            status_code=500,
            detail="Internal error occurred while generating image caption"
        )
