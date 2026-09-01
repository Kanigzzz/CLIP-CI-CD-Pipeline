import os
from pathlib import Path
from huggingface_hub import HfApi


ROOT = Path(__file__).resolve().parent.parent


def get_hf_token() -> str | None:
    token = os.getenv("HF_TOKEN")
    if not token:
        env_file = ROOT / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if line.startswith("HF_TOKEN"):
                        token = line.strip().split("=", 1)[1]
    return token


def upload_assets():
    token = get_hf_token()
    if not token:
        raise ValueError(
            "HF_TOKEN not found in environment variables or .env file.")
    api = HfApi(token=token)

    MODEL_REPO = "Kamil123456789/clip-animals"

    DATA_REPO = "Kamil123456789/clip-animals-data"

    api.create_repo(MODEL_REPO, repo_type="model", exist_ok=True)
    api.create_repo(DATA_REPO, repo_type="dataset", exist_ok=True)

    api.upload_file(
        path_or_fileobj="models/best_model.pth",
        path_in_repo="best_model.pth",
        repo_id=MODEL_REPO,
        repo_type="model"
    )

    api.upload_file(
        path_or_fileobj="models/text_tower_quantized.onnx",
        path_in_repo="text_tower_quantized.onnx",
        repo_id=MODEL_REPO,
        repo_type="model"
    )

    api.upload_file(
        path_or_fileobj="data/database/animals_index.faiss",
        path_in_repo="animals_index.faiss",
        repo_id=DATA_REPO,
        repo_type="dataset"
    )

    api.upload_file(
        path_or_fileobj="data/database/images_path.json",
        path_in_repo="images_path.json",
        repo_id=DATA_REPO,
        repo_type="dataset"
    )

    api.upload_folder(
        folder_path="data/animals",
        path_in_repo="animals",
        repo_id=DATA_REPO,
        repo_type="dataset"
    )


if __name__ == "__main__":
    upload_assets()
