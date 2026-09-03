from importlib.util import find_spec

_REQUIRED = ("faiss", "onnxruntime", "transformers")

collect_ignore_glob = [] if all(find_spec(m)
                                for m in _REQUIRED) else ["test_*.py"]
