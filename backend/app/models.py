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
    year: int | str
    description: str
    score: float
    explanation: str
    matched_signals: List[str]


class SearchResponse(BaseModel):
    query: str
    normalized_query: str
    assistant_message: str
    interpretation: QueryInterpretation | None = None
    results: List[BookResult]


class HealthResponse(BaseModel):
    status: str
    collection: str
    vector_size: Optional[int] = None
    indexed_vectors: Optional[int] = None
    model: str
    details: dict[str, Any] = Field(default_factory=dict)
    
class QueryInterpretation(BaseModel):
    normalized_query: str
    query_type: str
    keywords: list[str] = Field(default_factory=list)
    genres: list[str] = Field(default_factory=list)
    themes: list[str] = Field(default_factory=list)
    moods: list[str] = Field(default_factory=list)


class CatalogStatusResponse(BaseModel):
    collection: str
    collection_exists: bool
    indexed_books: int
    model: str
    catalog_ready: bool
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
