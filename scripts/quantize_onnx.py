from pathlib import Path

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

ROOT = Path(__file__).resolve().parent.parent


def strip_initializer_value_info(model_path: Path, cleaned_path: Path) -> None:
    """Usuwa wpisy value_info opisujace wagi (initializery).

    Eksporter torch.onnx zapisuje value_info takze dla wag. quantize_dynamic
    zamienia Gemm(transB=1) na MatMul i przy okazji transponuje wage
    ([256, 768] -> [768, 256]), ale nie aktualizuje jej value_info. Kolejny
    przebieg shape inference widzi wtedy sprzecznosc ksztaltow i wywala
    InferenceError. Te wpisy sa nadmiarowe - ksztalty wag wynikaja wprost
    z samych initializerow.
    """
    model = onnx.load(str(model_path))

    initializer_names = {init.name for init in model.graph.initializer}
    kept = [vi for vi in model.graph.value_info if vi.name not in initializer_names]

    del model.graph.value_info[:]
    model.graph.value_info.extend(kept)

    onnx.save(model, str(cleaned_path))


def quantize():
    input_model = ROOT / "models" / "text_tower.onnx"
    cleaned_model = ROOT / "models" / "text_tower_cleaned.onnx"
    output_model = ROOT / "models" / "text_tower_quantized.onnx"

    strip_initializer_value_info(input_model, cleaned_model)

    quantize_dynamic(
        model_input=cleaned_model,
        model_output=output_model,
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["MatMul", "Gemm"],
    )

    cleaned_model.unlink()

    size_orig = input_model.stat().st_size / (1024 * 1024)
    quant_size = output_model.stat().st_size / (1024 * 1024)

    print(f"Rozmiar przed kwantyzacją : {size_orig:.2f} MB")
    print(f"Rozmiar po kwantyzacji    : {quant_size:.2f} MB")
    print(
        f"Oszczędność miejsca       : {((size_orig - quant_size) / size_orig) * 100:.1f}%")


if __name__ == "__main__":
    quantize()
