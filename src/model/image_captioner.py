import torch
import yaml
from PIL import Image
from io import BytesIO
from pathlib import Path
from transformers import BlipProcessor, BlipForConditionalGeneration


_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / \
    "configs" / "config_image_captioner.yaml"
with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ImageCaptioner:
    def __init__(self):
        self.device = _get_device()

        self.processor = BlipProcessor.from_pretrained(cfg['data']['model'])
        self.model = BlipForConditionalGeneration.from_pretrained(
            cfg['data']['model'])
        self.model.to(self.device)
        self.model.eval()

    def generate_caption(self, images_bytes: bytes) -> str:
        raw_img = Image.open(BytesIO(images_bytes)).convert("RGB")
        inputs = self.processor(
            raw_img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=20)

        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption
