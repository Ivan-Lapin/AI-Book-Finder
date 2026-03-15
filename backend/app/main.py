from __future__ import annotations

import os
import time
import requests
import threading
import logging
from functools import lru_cache
from pathlib import Path

import orjson
from app.config import settings
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from qdrant_client import QdrantClient

from .config import settings
from .embeddings import get_embedding_service
from .models import HealthResponse, IndexResponse, SearchRequest, SearchResponse, BookResult
from .querying import analyze_query, build_assistant_message, extract_signals, rerank_score
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

def warmup_embedding_model() -> None:
    try:
        logger.info("Embedding warmup started")
        repository = get_repository()
        repository.embedding_service.encode_queries(["query: тестовый запрос"])
        logger.info("Embedding warmup finished")
    except Exception as exc:
        logger.exception("Embedding warmup failed: %s", exc)

@lru_cache(maxsize=1)
def get_repository() -> BookRepository:
    client = QdrantClient(url=settings.qdrant_url)
    return BookRepository(client=client, embedding_service=get_embedding_service())

def wait_for_qdrant() -> None:
    deadline = time.time() + settings.qdrant_startup_timeout_sec

    while time.time() < deadline:
        try:
            response = requests.get(f"{settings.qdrant_url}/collections", timeout=3)
            if response.ok:
                return
        except requests.RequestException:
            pass

        time.sleep(settings.qdrant_startup_poll_interval_sec)

    raise RuntimeError(
        f"Qdrant is not available after {settings.qdrant_startup_timeout_sec} seconds"
    )

@app.on_event("startup")
def startup() -> None:
    os.makedirs(settings.upload_dir, exist_ok=True)
    threading.Thread(target=warmup_embedding_model, daemon=True).start()


@app.get("/healthz")
def healthcheck() -> dict:
    return {
        "status": "ok",
        "service": "backend",
    }
    
@app.get("/healthz/full", response_model=HealthResponse)
def full_healthcheck() -> HealthResponse:
    repository = get_repository()
    info = repository.collection_info()
    return HealthResponse(
        status="ok",
        collection=settings.qdrant_collection,
        vector_size=info.config.params.vectors.size,
        indexed_vectors=info.points_count,
        model=settings.embedding_model_name,
        details={
            "qdrant_url": settings.qdrant_url,
        },
    )


@app.post("/catalog/index-default", response_model=IndexResponse)
def index_default_catalog() -> IndexResponse:
    path = Path(settings.default_catalog_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Файл каталога по умолчанию не найден")

    wait_for_qdrant()
    repository = get_repository()
    books = repository.load_books_from_path(path)
    indexed_count = repository.index_books(books)
    return IndexResponse(
        collection=settings.qdrant_collection,
        indexed_count=indexed_count,
        total_books_in_payload=len(books),
        message="Каталог по умолчанию успешно проиндексирован",
    )


@app.post("/catalog/upload", response_model=IndexResponse)
async def upload_catalog(file: UploadFile = File(...)) -> IndexResponse:
    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Поддерживаются только JSON-файлы")

    target_path = Path(settings.upload_dir) / file.filename
    content = await file.read()
    try:
        parsed = orjson.loads(content)
    except orjson.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Некорректный JSON: {exc}") from exc

    with open(target_path, "wb") as target_file:
        target_file.write(orjson.dumps(parsed))

    wait_for_qdrant()
    repository = get_repository()
    books = repository.load_books_from_path(target_path)
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
    wait_for_qdrant()
    logger.info("Qdrant is reachable")
    repository = get_repository()
    logger.info("Repository acquired")
    analysis = analyze_query(request.query)
    logger.info("Query analyzed: %s", analysis.normalized)
    hits = repository.search(analysis.normalized, limit=max(request.limit * 3, 10))
    logger.info("Search finished, hits=%s", len(hits))
    reranked: list[BookResult] = []
    for hit in hits:
        payload = hit.payload or {}
        document = payload.get("document", "")
        signals = extract_signals(analysis, document)
        final_score = rerank_score(hit.score, signals, analysis)
        explanation = (
            f"Подходит по смыслу запроса: совпадают тема, жанровые или сюжетные сигналы"
            f" ({', '.join(signals)}); итоговый рейтинг учитывает семантическую близость и совпадение ключевых сигналов." if signals else
            "Подходит по общему семантическому смыслу запроса и описания книги; итоговый рейтинг учитывает семантическую близость."
        )
        reranked.append(
            BookResult(
                title=payload["title"],
                author=payload["author"],
                category=payload["category"],
                year=payload["year"],
                description=payload["description"],
                score=final_score,
                explanation=explanation,
                matched_signals=signals,
            )
        )

    results = sorted(reranked, key=lambda item: item.score, reverse=True)[: request.limit]

    return SearchResponse(
        query=request.query,
        normalized_query=analysis.normalized,
        assistant_message=build_assistant_message(len(results), analysis),
        results=results,
    )
