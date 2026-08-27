import torch
from io import BytesIO
from PIL import Image
from src.model.image_captioner import ImageCaptioner

_captioner = ImageCaptioner()


def generate_caption_from_bytes(image_bytes: bytes) -> str:
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    inputs = _captioner.processor(
        image, return_tensors="pt").to(_captioner.device)

    with torch.no_grad():
        out = _captioner.model.generate(**inputs, max_new_tokens=20)

    return _captioner.processor.decode(out[0], skip_special_tokens=True)
