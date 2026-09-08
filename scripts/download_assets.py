import os
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")

MODEL_REPO = "Kamil123456789/clip-animals"
MODEL_REVISION = os.getenv(
    "MODEL_REVISION", "e3732e377768e0383caa96e3a1f43963bee9e79e")
DATA_REPO = "Kamil123456789/clip-animals-data"
DATA_REVISION = "061f7a988f864a5fb786b90acf1ab07136c3d5f7"


def download_model():
    dest = ROOT / "models" / "text_tower_quantized.onnx"
    rev_file = ROOT / "models" / ".model_revision"
    current_rev = rev_file.read_text().strip() if rev_file.exists() else None

    if dest.exists() and current_rev == MODEL_REVISION:
        print(
            f"Model already exists and revision matches ({MODEL_REVISION}), skipping")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading model...")
    hf_hub_download(
        repo_id=MODEL_REPO,
        revision=MODEL_REVISION,
        repo_type="model",
        filename="text_tower_quantized.onnx",
        local_dir=str(ROOT / "models")
    )

    rev_file.write_text(MODEL_REVISION)


def download_dataset():
    dest = ROOT / "data" / "database" / "animals_index.faiss"
    if dest.exists():
        print("Dataset already exists, skipping")
        return

    print("Dowloading dataset...")
    snapshot_download(
        repo_id=DATA_REPO,
        revision=DATA_REVISION,
        repo_type="dataset",
        local_dir=str(ROOT / "data" / "database")
    )

    wrong_animal = ROOT / "data" / "database" / "animals"
    correct_animals = ROOT / "data" / "animals"

    if wrong_animal.exists() and not correct_animals.exists():
        shutil.move(str(wrong_animal), str(correct_animals))


def download_blip():
    dest = ROOT / "models" / "blip"
    if dest.exists():
        print("BLIP model already")
        return

    print("Downloading BLIP model...")
    snapshot_download(
        repo_id="Salesforce/blip-image-captioning-base",
        local_dir=str(dest)
    )


if __name__ == "__main__":
    download_model()
    download_dataset()
    download_blip()
