"""One-request locale-aware BM25 retrieval with bounded local lexical reranking."""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, cast

from turbopuffer.types.namespace_multi_query_params import Query

from tpuf_optimize.strategy import TPufStrategy


class SerialNamespace(Protocol):
    """The single-query namespace surface used by this experiment."""

    def query(self, **kwargs: Any) -> Any: ...


class SingleRequestLexicalStrategy:
    """The cdb2d64 one-request title + passage BM25 candidate strategy."""

    MAX_TOP_K = 100
    TITLE_WEIGHT = 1.0
    PASSAGE_WEIGHT = 0.5

    def __init__(self) -> None:
        self._base = TPufStrategy()

    def query(self, query_text: str, locale: str, top_k: int) -> Query:
        """Build one bounded serial query and return local-reranking fields."""
        if not query_text.strip():
            raise ValueError("query_text must not be empty")
        if not locale.strip():
            raise ValueError("locale must not be empty")
        if top_k < 1 or top_k > self.MAX_TOP_K:
            raise ValueError(f"top_k must be between 1 and {self.MAX_TOP_K}")

        locale_analyzer = self._base.LOCALE_ANALYZERS.get(locale)
        suffix = (
            locale_analyzer[0]
            if locale_analyzer is not None
            else self._base.DEFAULT_ANALYZER
        )
        return cast(
            Query,
            {
                "rank_by": (
                    "Sum",
                    (
                        (
                            "Product",
                            self.TITLE_WEIGHT,
                            (f"title__{suffix}", "BM25", query_text),
                        ),
                        (
                            "Product",
                            self.PASSAGE_WEIGHT,
                            (f"passage__{suffix}", "BM25", query_text),
                        ),
                    ),
                ),
                "filters": ("locale", "Eq", locale),
                "top_k": top_k,
                "include_attributes": ["title", "description", "brand", "locale"],
            },
        )


def _tokens(text: object) -> list[str]:
    """Normalize punctuation, accents, and CJK characters for exact evidence."""
    normalized = unicodedata.normalize("NFKD", str(text or "")).casefold()
    normalized = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Mn"
    )
    tokens: list[str] = []
    current: list[str] = []
    for character in normalized:
        codepoint = ord(character)
        is_cjk = (
            0x3040 <= codepoint <= 0x30FF
            or 0x3400 <= codepoint <= 0x9FFF
            or 0xF900 <= codepoint <= 0xFAFF
        )
        if is_cjk:
            if current:
                tokens.append("".join(current))
                current.clear()
            tokens.append(character)
        elif character.isalnum():
            current.append(character)
        elif current:
            tokens.append("".join(current))
            current.clear()
    if current:
        tokens.append("".join(current))
    return tokens


def _coverage(query_terms: Sequence[str], field_terms: Sequence[str]) -> float:
    """Fraction of distinct query terms found in a field."""
    unique_query_terms = set(query_terms)
    if not unique_query_terms:
        return 0.0
    return len(unique_query_terms.intersection(field_terms)) / len(unique_query_terms)


def _ordered_coverage(query_terms: Sequence[str], title_terms: Sequence[str]) -> float:
    """Fraction of adjacent query pairs found as adjacent title tokens."""
    if len(query_terms) < 2:
        return 0.0
    pairs = tuple(zip(query_terms, query_terms[1:]))
    title_pairs = set(zip(title_terms, title_terms[1:]))
    return sum(pair in title_pairs for pair in pairs) / len(pairs)


def _as_mapping(row: object) -> Mapping[str, Any]:
    if isinstance(row, Mapping):
        return row
    model_dump = getattr(row, "model_dump", None)
    if callable(model_dump):
        return cast(Mapping[str, Any], model_dump())
    return {}


def local_lexical_evidence(query_text: str, row: object) -> float:
    """Score exact query-term evidence in title and other returned fields."""
    query_terms = _tokens(query_text)
    if not query_terms:
        return 0.0
    attributes = _as_mapping(row)
    title_terms = _tokens(attributes.get("title"))
    description_terms = _tokens(attributes.get("description"))
    brand_terms = _tokens(attributes.get("brand"))

    exact_title_phrase = float(
        len(query_terms) > 1
        and any(
            title_terms[start:start + len(query_terms)] == query_terms
            for start in range(len(title_terms) - len(query_terms) + 1)
        )
    )
    return (
        0.45 * _coverage(query_terms, title_terms)
        + 0.20 * _ordered_coverage(query_terms, title_terms)
        + 0.15 * exact_title_phrase
        + 0.10 * _coverage(query_terms, brand_terms)
        + 0.10 * _coverage(query_terms, description_terms)
    )


def rerank_candidates(
    query_text: str, rows: Sequence[object], boost: float
) -> list[object]:
    """Rerank only retrieved candidates, preserving BM25 rank as the anchor."""
    if not 0.0 <= boost <= 1.0:
        raise ValueError("boost must be between 0 and 1")
    if not rows or boost == 0.0:
        return list(rows)

    count = len(rows)
    scored = [
        (
            (1.0 - boost) * ((count - index) / count)
            + boost * local_lexical_evidence(query_text, row),
            index,
            row,
        )
        for index, row in enumerate(rows)
    ]
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [row for _, _, row in scored]


def search_one(
    namespace: SerialNamespace,
    query_text: str,
    locale: str,
    top_k: int = 100,
    boost: float = 0.35,
) -> list[object]:
    """Fetch one bounded BM25 candidate list and rerank it locally once."""
    strategy = SingleRequestLexicalStrategy()
    query = strategy.query(query_text, locale, top_k)
    response = namespace.query(**query)
    return rerank_candidates(query_text, response.rows or [], boost)
