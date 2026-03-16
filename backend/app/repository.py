from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

from pydantic import ValidationError
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from .config import settings
from .embeddings import EmbeddingService
from .models import Book


class BookRepository:
    def __init__(self, client: QdrantClient, embedding_service: EmbeddingService) -> None:
        self.client = client
        self.embedding_service = embedding_service
        self.collection = settings.qdrant_collection

    def collection_exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def ensure_collection(self) -> None:
        if self.collection_exists():
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
        self.ensure_collection()
        vector = self.embedding_service.encode_queries([query])[0]
        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=limit,
            with_payload=True,
        )
        return response.points

    def collection_info(self):
        self.ensure_collection()
        return self.client.get_collection(self.collection)

    def indexed_books_count(self) -> int:
        if not self.collection_exists():
            return 0
        info = self.client.get_collection(self.collection)
        return int(info.points_count or 0)

    @staticmethod
    def load_books_from_path(path: str | Path) -> list[Book]:
        with open(path, "r", encoding="utf-8") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Некорректный JSON: {exc.msg}") from exc

        if isinstance(data, dict) and "books" in data:
            books_data = data["books"]
        elif isinstance(data, list):
            books_data = data
        else:
            raise ValueError(
                "Ожидается JSON-массив книг или объект вида {'books': [...]}"
            )

        if not isinstance(books_data, list) or not books_data:
            raise ValueError("Список книг пуст или имеет некорректный формат")

        books: list[Book] = []
        for index, item in enumerate(books_data, start=1):
            if not isinstance(item, dict):
                raise ValueError(f"Книга #{index} должна быть JSON-объектом")
            try:
                books.append(Book.model_validate(item))
            except ValidationError as exc:
                raise ValueError(f"Ошибка в книге #{index}: {exc.errors()[0]['msg']}") from exc

        return books

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
