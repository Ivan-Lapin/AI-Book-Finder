from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
import logging

import orjson
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient

from .config import settings
from .embeddings import get_embedding_service
from .models import (
    BookResult,
    CatalogStatusResponse,
    HealthResponse,
    IndexResponse,
    QueryInterpretation,
    SearchRequest,
    SearchResponse,
)
from .querying import (
    analyze_query,
    build_assistant_message,
    build_book_explanation,
    extract_signals,
    rerank_score,
)
from .repository import BookRepository

logger = logging.getLogger(__name__)

app = FastAPI(title=settings.app_name)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_repository() -> BookRepository:
    client = QdrantClient(url=settings.qdrant_url, check_compatibility=False)
    return BookRepository(client=client, embedding_service=get_embedding_service())


@app.on_event("startup")
def startup() -> None:
    os.makedirs(settings.upload_dir, exist_ok=True)


@app.get("/healthz", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    return HealthResponse(
        status="ok",
        collection=settings.qdrant_collection,
        model=settings.embedding_model_name,
        details={"qdrant_url": settings.qdrant_url},
    )


@app.get("/catalog/status", response_model=CatalogStatusResponse)
def catalog_status() -> CatalogStatusResponse:
    repository = get_repository()
    exists = repository.collection_exists()
    indexed_books = repository.indexed_books_count()
    catalog_ready = indexed_books > 0
    message = (
        "Каталог готов к поиску"
        if catalog_ready
        else "Каталог ещё не загружен: сначала проиндексируйте пример или загрузите свой JSON"
    )
    return CatalogStatusResponse(
        collection=settings.qdrant_collection,
        collection_exists=exists,
        indexed_books=indexed_books,
        model=settings.embedding_model_name,
        catalog_ready=catalog_ready,
        message=message,
        details={"default_catalog_path": settings.default_catalog_path},
    )


@app.post("/catalog/index-default", response_model=IndexResponse)
def index_default_catalog() -> IndexResponse:
    path = Path(settings.default_catalog_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Файл каталога по умолчанию не найден")

    repository = get_repository()
    try:
        books = repository.load_books_from_path(path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    indexed_count = repository.index_books(books)
    return IndexResponse(
        collection=settings.qdrant_collection,
        indexed_count=indexed_count,
        total_books_in_payload=len(books),
        message="Каталог по умолчанию успешно проиндексирован",
    )


@app.post("/catalog/upload", response_model=IndexResponse)
async def upload_catalog(file: UploadFile = File(...)) -> IndexResponse:
    if not file.filename or not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Поддерживаются только JSON-файлы с расширением .json")

    target_path = Path(settings.upload_dir) / file.filename
    content = await file.read()
    try:
        parsed = orjson.loads(content)
    except orjson.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Некорректный JSON: {exc}") from exc

    with open(target_path, "wb") as target_file:
        target_file.write(orjson.dumps(parsed))

    repository = get_repository()
    try:
        books = repository.load_books_from_path(target_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    indexed_count = repository.index_books(books)
    return IndexResponse(
        collection=settings.qdrant_collection,
        indexed_count=indexed_count,
        total_books_in_payload=len(books),
        message="Каталог загружен. Книги добавлены или обновлены в индексе.",
    )


@app.post("/search", response_model=SearchResponse)
def semantic_search(request: SearchRequest) -> SearchResponse:
    logger.info("Search started: %s", request.query)
    repository = get_repository()
    analysis = analyze_query(request.query)
    catalog_ready = repository.indexed_books_count() > 0

    if not catalog_ready:
        return SearchResponse(
            query=request.query,
            normalized_query=analysis.normalized,
            assistant_message=build_assistant_message(0, analysis, catalog_ready=False),
            interpretation=QueryInterpretation(
                normalized_query=analysis.normalized,
                query_type=analysis.query_type,
                keywords=analysis.keywords,
                moods=analysis.moods,
                genres=analysis.genres,
                themes=analysis.themes,
            ),
            results=[],
        )

    hits = repository.search(analysis.normalized, limit=max(request.limit * 10, 30))
    logger.info("Search candidates received: %s", len(hits))

    reranked: list[BookResult] = []
    for hit in hits:
        payload = hit.payload or {}
        document = payload.get("document", "")
        signals = extract_signals(analysis, document)
        final_score = rerank_score(float(hit.score or 0.0), signals, analysis)
        explanation = build_book_explanation(
            category=payload.get("category", "неизвестный жанр"),
            description=payload.get("description", ""),
            matched_signals=signals,
            semantic_score=float(hit.score or 0.0),
        )
        reranked.append(
            BookResult(
                title=payload.get("title", "Без названия"),
                author=payload.get("author", "Неизвестный автор"),
                category=payload.get("category", "Неизвестный жанр"),
                year=payload.get("year", 0),
                description=payload.get("description", ""),
                score=final_score,
                explanation=explanation,
                matched_signals=signals,
            )
        )

    results = sorted(reranked, key=lambda item: item.score, reverse=True)[: request.limit]

    return SearchResponse(
        query=request.query,
        normalized_query=analysis.normalized,
        assistant_message=build_assistant_message(len(results), analysis, catalog_ready=True),
        interpretation=QueryInterpretation(
            normalized_query=analysis.normalized,
            query_type=analysis.query_type,
            keywords=analysis.keywords,
            moods=analysis.moods,
            genres=analysis.genres,
            themes=analysis.themes,
        ),
        results=results,
    )
    
