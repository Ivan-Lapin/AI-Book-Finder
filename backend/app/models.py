from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field


class Book(BaseModel):
    title: str
    category: str
    description: str
    author: str
    year: int


class IndexResponse(BaseModel):
    status: str = "ok"
    collection: str
    indexed_count: int
    total_books_in_payload: int
    message: str


class SearchRequest(BaseModel):
    query: str = Field(min_length=2)
    limit: int = Field(default=5, ge=1, le=10)


class BookResult(BaseModel):
    title: str
    author: str
    category: str
    year: int
    description: str
    score: float
    explanation: str
    matched_signals: List[str]


class SearchResponse(BaseModel):
    query: str
    normalized_query: str
    assistant_message: str
    results: List[BookResult]


class HealthResponse(BaseModel):
    status: str
    collection: str
    vector_size: Optional[int] = None
    indexed_vectors: Optional[int] = None
    model: str
    details: dict[str, Any] = Field(default_factory=dict)
