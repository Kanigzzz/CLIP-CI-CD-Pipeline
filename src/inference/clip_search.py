import json
import faiss
import torch
import torch.nn.functional as F
import yaml
from pathlib import Path
from transformers import AutoTokenizer

from src.model.clip_model import CLIPModel

_ROOT = Path(__file__).resolve().parent.parent.parent
with open(_ROOT / "configs" / "config_model_training.yaml") as f:
    _CFG = yaml.safe_load(f)


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class CLIPSearcher:
    def __init__(self):
        self.device = _get_device()

        self.model = CLIPModel(_CFG['model']['embedding_dim'])
        self.model.load_state_dict(
            torch.load(_ROOT / "models" / "best_model.pth",
                       map_location=self.device)
        )

        self.model.eval().to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            _CFG['model']['text_encoder'])
        self.index = faiss.read_index(
            str(_ROOT / "data" / "database" / "animals_index.faiss"))

        with open(_ROOT / "data" / "database" / "images_path.json") as f:
            self.image_paths = json.load(f)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        tokens = self.tokenizer(
            query,
            max_length=_CFG['model']['max_text_length'],
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            text_emb = self.model.text_tower(
                tokens['input_ids'], tokens['attention_mask'])
            text_emb = F.normalize(text_emb, p=2, dim=-1)

            scores, indices = self.index.search(text_emb.cpu().numpy(), top_k)

        return [
            {"path": self.image_paths[idx],
             "score": round(float(score), 4)}
            for idx, score in zip(indices[0], scores[0])
        ]
