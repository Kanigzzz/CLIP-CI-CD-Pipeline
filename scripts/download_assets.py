import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parent.parent

MODEL_REPO = "Kamil123456789/clip-animals"
MODEL_REVISION = "e3732e377768e0383caa96e3a1f43963bee9e79e"
DATA_REPO = "Kamil123456789/clip-animals-data"
DATA_REVISION = "061f7a988f864a5fb786b90acf1ab07136c3d5f7"


def download_model():
    dest = ROOT / "models" / "text_tower_quantized.onnx"
    if dest.exists():
        print("Model already exists, skipping")
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


if __name__ == "__main__":
    download_model()
    download_dataset()
