from huggingface_hub import HfApi

api = HfApi()

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

