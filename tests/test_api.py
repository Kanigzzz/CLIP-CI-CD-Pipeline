import io
import pytest
from PIL import Image
from src.api.routers.caption import get_captioner
from src.api.routers.search import get_searcher
from fastapi.testclient import TestClient
from src.api.main import app


class FakeCaptioner():
    def generate_caption(self, image_bytes: bytes) -> str:
        Image.open(io.BytesIO(image_bytes)).verify()
        return "test text"


class FakeSearcher():
    def __init__(self):
        self._result = [{"path": f"fake/{i}dog.jpg", "score": 0.99}
                        for i in range(100)]

    def search(self, query: str, top_k: int = 3):
        return self._result[:top_k]


class ImageFactoryClass:
    def create(self, type: str = "valid"):

        if type == "valid":
            img = Image.new("RGB", (64, 64), color="blue")
            img_byte = io.BytesIO()
            img.save(img_byte, format="JPEG")
            img_byte.seek(0)
            return ("test.jpg", img_byte, "image/jpg")

        if type == "corrupted":
            return ("corrupted.jpg", io.BytesIO(b"This is not and proper image"), "image/jpg")

        if type == "too_large":
            huge_data = b"0" * (11 * 1024 * 1024)
            return ("huge.jpg", io.BytesIO(huge_data), "image/jpg")


@pytest.fixture()
def image_factory():
    return ImageFactoryClass()


@pytest.fixture(scope="module")
def client():
    test_client = TestClient(app)

    app.dependency_overrides[get_captioner] = lambda: FakeCaptioner()
    app.dependency_overrides[get_searcher] = lambda: FakeSearcher()

    yield test_client

    app.dependency_overrides.clear()


def test_live_endpoint(client):
    response = client.get("/live")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "alive"


@pytest.mark.parametrize("query, top_k", [
    ("a photo of a cute animal", 1),
    ("a wild lion in Africa", 3),
    ("dog & cat #1! @special_chars?", 2),
    ("a very long description " * 30, 2),
])
def test_search_valid_querys(client, query, top_k):
    response = client.post(
        "/api/v1/search",
        json={
            "query": query,
            "top_k": top_k
        })

    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert len(data["results"]) <= top_k

    for item in data["results"]:
        assert "path" in item
        assert "score" in item
        assert isinstance(item["score"], float)


@pytest.mark.parametrize("top_k", [0, -1, -50, 110, 500])
def test_search_top_k_boundaries(client, top_k):
    response = client.post("/api/v1/search",
                           json={
                               "query": "cat",
                               "top_k": top_k
                           })
    assert response.status_code == 422


def test_missing_required_fields(client):
    response = client.post("/api/v1/search",
                           json={})
    assert response.status_code == 422


@pytest.mark.parametrize("query", [" ", "", "\t\n "])
def test_search_empty_queries(client, query):
    response = client.post("/api/v1/search",
                           json={
                               "query": query,
                               "top_k": 2
                           })
    assert response.status_code == 400


def test_caption_valid_image(client):
    img = Image.new("RGB", (64, 64), color="blue")
    img_byte = io.BytesIO()
    img.save(img_byte, format="JPEG")
    img_byte.seek(0)

    response = client.post(
        "/api/v1/caption",
        files={"image": ("test.jpg", img_byte, "image/jpeg")}
    )

    assert response.status_code == 200
    data = response.json()
    assert set(data) == {"caption"}
    assert isinstance(data["caption"], str)
    assert data["caption"].strip()


def test_caption_invalid_image(client):
    response = client.post(
        "/api/v1/caption",
        files={"image": ("test_plik.txt", b"Tekst", "text/plain")}
    )
    assert response.status_code == 400


def test_caption_too_large(client, image_factory):
    response = client.post(
        "/api/v1/caption",
        files={"image": image_factory.create("too_large")}
    )
    assert response.status_code == 413


def test_caption_corrupted_image(client, image_factory):
    response = client.post(
        "/api/v1/caption",
        files={"image": image_factory.create("corrupted")}
    )
    assert response.status_code == 400


def test_caption_missing_file(client):
    response = client.post("/api/v1/caption")
    assert response.status_code == 422


def test_caption_internal_error(client, image_factory):
    class CorruptedCaptioner:
        def generate_caption(self, image_bytes: bytes) -> str:
            raise Exception("Internal model error")

    from src.api.routers.caption import get_captioner
    app.dependency_overrides[get_captioner] = lambda: CorruptedCaptioner()

    response = client.post(
        "/api/v1/caption",
        files={"image": image_factory.create("valid")}
    )

    assert response.status_code == 500

    app.dependency_overrides[get_captioner] = lambda: FakeCaptioner()
