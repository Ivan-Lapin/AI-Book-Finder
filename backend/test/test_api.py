from __future__ import annotations

import sys
import types
from pathlib import Path

# Lightweight stubs so tests can import app.main without real external deps.
qdrant_module = types.ModuleType("qdrant_client")


class DummyQdrantClient:
    def __init__(self, *args, **kwargs):
        pass


qdrant_module.QdrantClient = DummyQdrantClient
http_models = types.ModuleType("qdrant_client.http.models")
http_models.VectorParams = object
http_models.Distance = types.SimpleNamespace(COSINE="cosine")
http_models.PointStruct = object
http_module = types.ModuleType("qdrant_client.http")
http_module.models = http_models
sys.modules.setdefault("qdrant_client", qdrant_module)
sys.modules.setdefault("qdrant_client.http", http_module)
sys.modules.setdefault("qdrant_client.http.models", http_models)

st_module = types.ModuleType("sentence_transformers")


class DummySentenceTransformer:
    def __init__(self, *args, **kwargs):
        pass

    def get_sentence_embedding_dimension(self):
        return 384

    def encode(self, texts, **kwargs):
        return [[0.0] * 384 for _ in texts]


st_module.SentenceTransformer = DummySentenceTransformer
sys.modules.setdefault("sentence_transformers", st_module)

from fastapi.testclient import TestClient

from app import main
from app.models import Book


class FakeRepository:
    def __init__(self, indexed_books: int = 0) -> None:
        self._indexed_books = indexed_books

    def collection_exists(self) -> bool:
        return self._indexed_books > 0

    def indexed_books_count(self) -> int:
        return self._indexed_books

    def load_books_from_path(self, path: str | Path):
        return [
            Book(
                title="Доктор Живаго",
                category="Историческая проза",
                description="Любовная история на фоне революции в России",
                author="Борис Пастернак",
                year=1957,
            )
        ]

    def index_books(self, books):
        self._indexed_books = len(books)
        return len(books)


client = TestClient(main.app)


def test_healthz_ok():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_catalog_status_empty(monkeypatch):
    monkeypatch.setattr(main, "get_repository", lambda: FakeRepository(indexed_books=0))
    response = client.get("/catalog/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["catalog_ready"] is False
    assert payload["indexed_books"] == 0


def test_upload_invalid_json_returns_400():
    response = client.post(
        "/catalog/upload",
        files={"file": ("broken.json", b"{not json}", "application/json")},
    )
    assert response.status_code == 400
    assert "Некорректный JSON" in response.json()["detail"]


def test_search_returns_guidance_when_catalog_is_empty(monkeypatch):
    monkeypatch.setattr(main, "get_repository", lambda: FakeRepository(indexed_books=0))
    response = client.post("/search", json={"query": "историческая проза", "limit": 5})
    assert response.status_code == 200
    payload = response.json()
    assert payload["results"] == []
    assert "Каталог ещё не загружен" in payload["assistant_message"]
