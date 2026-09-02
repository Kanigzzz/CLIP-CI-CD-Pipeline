import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from src.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "clip-api"


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
    assert len(data["results"]) == top_k

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
        files={"images": ("test.jpg", img_byte, "image/jpeg")}
    )

    assert response.status_code == 400
