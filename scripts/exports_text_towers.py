

import yaml
import torch
import onnx
from onnx.external_data_helper import load_external_data_for_model
from pathlib import Path
from src.model.clip_model import CLIPModel

ROOT = Path(__file__).resolve().parent.parent


with open(ROOT / "configs" / "config_model_training.yaml") as f:
    cfg = yaml.safe_load(f)


def export_text_tower():
    embedding_dim = cfg['model']['embedding_dim']
    max_length = cfg['model']['max_text_length']

    full_model = CLIPModel(embedding_dim)
    model_path = ROOT / "models" / "best_model.pth"
    full_model.load_state_dict(torch.load(model_path, map_location="cpu"))
    full_model.eval()

    text_tower = full_model.text_tower
    text_tower.eval()

    dummy_input_ids = torch.ones((1, max_length), dtype=torch.long)
    dummy_attention_mask = torch.ones((1, max_length), dtype=torch.long)

    output_onnx_file = ROOT / "models" / "text_tower.onnx"

    torch.onnx.export(
        text_tower,
        (dummy_input_ids, dummy_attention_mask),
        str(output_onnx_file),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input_ids', 'attention_mask'],
        output_names=['text_embedding'],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "text_embedding": {0: "batch_size"},

        }
    )

    onnx_model = onnx.load(str(output_onnx_file))
    load_external_data_for_model(onnx_model, str(output_onnx_file.parent))
    onnx.save(onnx_model, str(output_onnx_file))


if __name__ == "__main__":
    export_text_tower()
