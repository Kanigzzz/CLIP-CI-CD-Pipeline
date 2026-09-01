import json
import faiss
import onnxruntime as ort
import yaml
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer

_ROOT = Path(__file__).resolve().parent.parent.parent
with open(_ROOT / "configs" / "config_model_training.yaml") as f:
    _CFG = yaml.safe_load(f)


class CLIPSearcher:
    def __init__(self):
        onnx_model_path = _ROOT / "models" / "text_tower_quantized.onnx"
        self.session = ort.InferenceSession(str(onnx_model_path))

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
            return_tensors="np"
        )

        onnx_input = {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"]
        }

        onnx_output = self.session.run(['text_embedding'], onnx_input)
        raw_emb = onnx_output[0]

        norm = np.linalg.norm(raw_emb, ord=2, axis=-1, keepdims=True)
        text_emb = raw_emb / np.clip(norm, a_min=1e-12, a_max=None)

        scores, indices = self.index.search(text_emb.astype(np.float32), top_k)

        return [
            {"path": self.image_paths[idx],
             "score": round(float(score), 4)}
            for idx, score in zip(indices[0], scores[0])
        ]
