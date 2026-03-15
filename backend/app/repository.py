from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Iterable, Sequence

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from .config import settings
from .embeddings import EmbeddingService
from .models import Book

logger = logging.getLogger(__name__)


class BookRepository:
    def __init__(self, client: QdrantClient, embedding_service: EmbeddingService) -> None:
        self.client = client
        self.embedding_service = embedding_service
        self.collection = settings.qdrant_collection

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            return

        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=rest.VectorParams(
                size=self.embedding_service.embedding_size,
                distance=rest.Distance.COSINE,
            ),
        )

    def index_books(self, books: Sequence[Book]) -> int:
        self.ensure_collection()
        documents = [self._build_document(book) for book in books]
        vectors = self.embedding_service.encode_passages(documents)
        points = []
        for book, vector, document in zip(books, vectors, documents, strict=True):
            points.append(
                rest.PointStruct(
                    id=self._book_id(book),
                    vector=vector,
                    payload={
                        **book.model_dump(),
                        "document": document,
                    },
                )
            )

        self.client.upsert(collection_name=self.collection, points=points, wait=True)
        return len(points)

    def search(self, query: str, limit: int = 5):
        logger.info("Repository search started")
        self.ensure_collection()
        logger.info("Collection ensured")

        vector = self.embedding_service.encode_queries([query])[0]
        logger.info("Query embedded")

        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=limit,
            with_payload=True,
        )
        logger.info("Qdrant query completed")

        return response.points

    def collection_info(self):
        self.ensure_collection()
        return self.client.get_collection(self.collection)

    @staticmethod
    def load_books_from_path(path: str | Path) -> list[Book]:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        books_data = data["books"] if isinstance(data, dict) and "books" in data else data
        return [Book.model_validate(item) for item in books_data]

    @staticmethod
    def _build_document(book: Book) -> str:
        return (
            f"Название: {book.title}. "
            f"Автор: {book.author}. "
            f"Жанр: {book.category}. "
            f"Описание: {book.description}. "
            f"Год: {book.year}. "
            f"Ключевые акценты: {book.title}. {book.category}."
        )

    @staticmethod
    def _book_id(book: Book) -> str:
        raw = f"{book.title}|{book.author}|{book.year}".encode("utf-8")
        return hashlib.md5(raw).hexdigest()
