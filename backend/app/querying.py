from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class QueryAnalysis:
    original: str
    normalized: str
    keywords: list[str]
    moods: list[str]
    genres: list[str]
    themes: list[str]
    query_type: str


STOPWORDS = {
    "книга", "книгу", "книги", "роман", "романы", "произведение", "произведения",
    "что", "где", "про", "для", "мне", "хочу", "нужна", "нужно", "найди",
    "посоветуй", "какой", "какую", "подбери", "хотел", "хотела", "есть", "с",
    "и", "в", "во", "на", "о", "об", "по", "или", "а", "не", "очень",
    "где", "действия", "происходят", "просиходят", "вообще", "что-то", "сюжет",
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
    "приключ": "приключения",
    "подрост": "подростковые темы",
    "фантаст": "фантастический мир",
}

GENRE_ALIASES = {
    "проза": "проза",
    "истор": "историческая проза",
    "антиутоп": "антиутопия",
    "фэнт": "фэнтези",
    "фантаст": "фантастика",
    "детектив": "детектив",
    "триллер": "триллер",
    "роман": "роман",
}

THEME_ALIASES = {
    "револю": "революция",
    "росси": "Россия",
    "войн": "война",
    "любов": "любовь",
    "взрослен": "взросление",
    "общест": "общество",
    "власт": "власть",
    "тоталитар": "тоталитаризм",
    "убий": "убийство",
    "расслед": "расследование",
}


def _dedupe(items: list[str]) -> list[str]:
    unique: list[str] = []
    for item in items:
        if item not in unique:
            unique.append(item)
    return unique


def analyze_query(query: str) -> QueryAnalysis:
    normalized = " ".join(query.lower().strip().split())
    raw_tokens = re.findall(r"[а-яa-z0-9-]+", normalized, flags=re.IGNORECASE)
    keywords = [token for token in raw_tokens if token not in STOPWORDS and len(token) > 2]

    moods: list[str] = []
    genres: list[str] = []
    themes: list[str] = []

    for keyword in keywords:
        for stem, mood in MOOD_ALIASES.items():
            if stem in keyword:
                moods.append(mood)
        for stem, genre in GENRE_ALIASES.items():
            if stem in keyword:
                genres.append(genre)
        for stem, theme in THEME_ALIASES.items():
            if stem in keyword:
                themes.append(theme)

    moods = _dedupe(moods)
    genres = _dedupe(genres)
    themes = _dedupe(themes)

    query_type = "общий"
    if genres and themes:
        query_type = "жанр + тема"
    elif genres:
        query_type = "жанровый"
    elif themes:
        query_type = "тематический"
    elif moods:
        query_type = "по настроению"

    return QueryAnalysis(
        original=query,
        normalized=normalized,
        keywords=keywords,
        moods=moods,
        genres=genres,
        themes=themes,
        query_type=query_type,
    )


def build_assistant_message(result_count: int, analysis: QueryAnalysis, *, catalog_ready: bool) -> str:
    if not catalog_ready:
        return (
            "Каталог ещё не загружен или в индексе пока нет книг. "
            "Сначала загрузите JSON-файл с книгами или проиндексируйте пример из проекта."
        )

    if result_count == 0:
        hints: list[str] = []
        if not analysis.genres:
            hints.append("добавьте жанр")
        if not analysis.themes:
            hints.append("уточните тему")
        if not analysis.moods:
            hints.append("опишите настроение или тип героя")
        hint_text = ", ".join(hints[:2]) if hints else "уточните жанр, тему или настроение"
        return (
            "Я не нашёл достаточно близких совпадений. "
            f"Попробуйте переформулировать запрос: {hint_text}."
        )

    focus_parts = analysis.themes + analysis.genres + analysis.moods + analysis.keywords[:3]
    focus = ", ".join(_dedupe(focus_parts)[:4]) if focus_parts else analysis.normalized
    return (
        f"Подобрал {result_count} наиболее релевантных книг. "
        f"Я ориентировался на смысл запроса и сигналы: {focus}."
    )


def extract_signals(query_analysis: QueryAnalysis, book_payload_text: str) -> list[str]:
    text = book_payload_text.lower()
    matched_keywords = [kw for kw in query_analysis.keywords if kw in text]
    matched_moods = [m for m in query_analysis.moods if any(part in text for part in m.split())]
    matched_genres = [genre for genre in query_analysis.genres if genre.lower() in text]
    matched_themes = [theme for theme in query_analysis.themes if theme.lower() in text]
    return _dedupe(matched_themes + matched_genres + matched_moods + matched_keywords)[:5]


def rerank_score(semantic_score: float, matched_signals: list[str], query_analysis: QueryAnalysis) -> float:
    keyword_overlap = len(matched_signals)
    max_possible = max(
        len(query_analysis.keywords) + len(query_analysis.moods) + len(query_analysis.genres) + len(query_analysis.themes),
        1,
    )
    overlap_score = min(keyword_overlap / max_possible, 1.0)
    return round(semantic_score * 0.9 + overlap_score * 0.1, 4)


def build_book_explanation(*, category: str, description: str, matched_signals: list[str], semantic_score: float) -> str:
    if matched_signals:
        signals_text = ", ".join(matched_signals[:3])
        return (
            f"Книга выглядит релевантной по теме запроса: совпали сигналы {signals_text}. "
            f"Дополнительно результат поддержан семантической близостью описания и жанра «{category}»."
        )

    if semantic_score >= 0.75:
        return (
            "Описание книги семантически очень близко запросу: система сопоставила общий сюжетный и тематический контекст, "
            f"даже без явных точных совпадений слов. Жанр книги — «{category}»."
        )

    return (
        "Книга подобрана по общему смысловому сходству запроса и описания. "
        f"Это более мягкое совпадение по теме и контексту в жанре «{category}»."
    )
