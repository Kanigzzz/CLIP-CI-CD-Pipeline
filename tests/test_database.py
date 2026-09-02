from pathlib import Path
from src.inference.clip_search import CLIPSearcher

ROOT = Path(__file__).resolve().parent.parent


def test_faiss_and_images_paths_aligment():
    searcher = CLIPSearcher()

    num_vectores = searcher.index.ntotal
    num_paths = len(searcher.image_paths)

    assert num_paths == num_vectores, (
        f"KRYTYCZNY BŁĄD INTEGRALNOŚCI: Liczba wektorów w FAISS ({num_vectores}) "
        f"nie zgadza się z liczbą ścieżek w JSON ({num_paths})!"
    )


def test_image_files_exist_on_disk():
    searcher = CLIPSearcher()

    for path in searcher.image_paths[:20]:

        clean_path = path.replace("../", "")
        full_path = ROOT / clean_path
        assert full_path.exists(
        ), f"Plik obrazu nie istnieje na dysku: {full_path}"
