from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List

from sentence_transformers import SentenceTransformer

from .config import settings


class EmbeddingService:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.embedding_size = self.model.get_sentence_embedding_dimension()

    def encode_queries(self, texts: Iterable[str]) -> List[list[float]]:
        prepared = [self._prepare_query(text) for text in texts]
        vectors = self.model.encode(
            prepared,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]

    def encode_passages(self, texts: Iterable[str]) -> List[list[float]]:
        prepared = [self._prepare_passage(text) for text in texts]
        vectors = self.model.encode(
            prepared,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return [vector.tolist() for vector in vectors]

    @staticmethod
    def _prepare_query(text: str) -> str:
        return f"query: {text.strip()}"

    @staticmethod
    def _prepare_passage(text: str) -> str:
        return f"passage: {text.strip()}"


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService(settings.embedding_model_name)
