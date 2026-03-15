from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class QueryAnalysis:
    original: str
    normalized: str
    keywords: list[str]
    moods: list[str]


STOPWORDS = {
    "книга", "книгу", "книги", "роман", "романы", "что", "где", "про", "для",
    "мне", "хочу", "нужна", "нужно", "найди", "посоветуй", "какой", "какую",
    "с", "и", "в", "на", "о", "об", "по", "или", "а", "не", "очень",
}

MOOD_ALIASES = {
    "неожиданн": "неожиданная развязка",
    "твист": "неожиданная развязка",
    "мрачн": "мрачная атмосфера",
    "светл": "светлая атмосфера",
    "любов": "любовная линия",
    "полит": "политические интриги",
    "маг": "магия",
    "детектив": "расследование",
    "тайн": "тайны",
    "приключ": "приключение",
    "подрост": "подростковые темы",
    "фантаст": "фантастический мир",
}


def analyze_query(query: str) -> QueryAnalysis:
    normalized = " ".join(query.lower().strip().split())
    raw_tokens = re.findall(r"[а-яa-z0-9-]+", normalized, flags=re.IGNORECASE)
    keywords = [token for token in raw_tokens if token not in STOPWORDS and len(token) > 2]

    moods: list[str] = []
    for keyword in keywords:
        for stem, mood in MOOD_ALIASES.items():
            if stem in keyword and mood not in moods:
                moods.append(mood)

    return QueryAnalysis(
        original=query,
        normalized=normalized,
        keywords=keywords,
        moods=moods,
    )


def build_assistant_message(result_count: int, analysis: QueryAnalysis) -> str:
    if result_count == 0:
        return (
            "Я не нашёл достаточно близких совпадений. Попробуйте уточнить жанр, настроение, тему или тип героя."
        )

    focus_parts = analysis.moods or analysis.keywords[:3]
    focus = ", ".join(focus_parts) if focus_parts else analysis.normalized
    return (
        f"Подобрал {result_count} наиболее релевантных книг. "
        f"При поиске я ориентировался на смысл запроса и сигналы: {focus}."
    )


def extract_signals(query_analysis: QueryAnalysis, book_payload_text: str) -> list[str]:
    text = book_payload_text.lower()
    matched_keywords = [kw for kw in query_analysis.keywords if kw in text]
    matched_moods = [m for m in query_analysis.moods if any(part in text for part in m.split())]
    combined = matched_keywords + matched_moods
    unique: list[str] = []
    for item in combined:
        if item not in unique:
            unique.append(item)
    return unique[:4]


def rerank_score(semantic_score: float, matched_signals: list[str], query_analysis: QueryAnalysis) -> float:
    keyword_overlap = len(matched_signals)
    max_possible = max(len(query_analysis.keywords) + len(query_analysis.moods), 1)
    overlap_score = min(keyword_overlap / max_possible, 1.0)
    return round(semantic_score * 0.85 + overlap_score * 0.15, 4)
