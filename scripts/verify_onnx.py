from pathlib import Path
import numpy as np
import onnxruntime as ort
import yaml
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from src.model.clip_model import CLIPModel

ROOT = Path(__file__).resolve().parent.parent

with open(ROOT / "configs" / "config_model_training.yaml") as f:
    cfg = yaml.safe_load(f)


def verify():
    text_query = "A photo of an anthelope"

    tokenizer = AutoTokenizer.from_pretrained(cfg['model']['text_encoder'])
    tokens = tokenizer(
        text_query,
        max_length=cfg['model']['max_text_length'],
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    full_model = CLIPModel(cfg['model']['embedding_dim'])
    full_model.load_state_dict(torch.load(
        ROOT / "models" / "best_model.pth", map_location="cpu"))
    full_model.eval()

    with torch.no_grad():
        embedding_text = full_model.text_tower(
            tokens['input_ids'], tokens['attention_mask'])
        embedding_text = F.normalize(embedding_text, p=2, dim=-1).numpy()

    onnx_path = ROOT / "models" / "text_tower.onnx"
    session = ort.InferenceSession(str(onnx_path))

    onnx_input = {
        "input_ids": tokens['input_ids'].numpy(),
        "attention_mask": tokens['attention_mask'].numpy()
    }

    onnx_output = session.run(['text_embedding'], onnx_input)
    raw_onnx_emb = onnx_output[0]

    norm = np.linalg.norm(raw_onnx_emb, ord=2, axis=-1, keepdims=True)
    onnx_emb = raw_onnx_emb / np.clip(norm, a_min=1e-12, a_max=None)

    max_diff = np.max(np.abs(embedding_text - onnx_emb))
    print(f"Maksymalna roznica miedzy Pytorchem a ONNX: {max_diff:.8f}")


if __name__ == "__main__":
    verify()
