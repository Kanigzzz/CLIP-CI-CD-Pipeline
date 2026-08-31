from pathlib import Path
from optimum.onnxruntime import ORTModelForVision2Seq
from transformers import BlipProcessor

ROOT = Path(__file__).resolve().parent.parent

MODEL_ID = "Salesforce/blip-image-captioning-base"
OUTPUT_DIR = ROOT / "models" / "blip_onnx"


def export_blip():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = ORTModelForVision2Seq.from_pretrained(
        MODEL_ID,
        export=True
    )

    model.save_pretrained(str(OUTPUT_DIR))

    processor = BlipProcessor.from_pretrained(MODEL_ID)
    processor.save_pretrained(str(OUTPUT_DIR))


if __name__ == "__main__":
    export_blip()
