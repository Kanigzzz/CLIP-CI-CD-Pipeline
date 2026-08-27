from src.model.image_captioner import ImageCaptioner

_captioner = ImageCaptioner()


def generate_caption_from_bytes(image_bytes: bytes) -> str:
    return _captioner.generate_caption(image_bytes)
