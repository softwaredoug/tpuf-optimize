"""A single weighted BM25 query against a Turbopuffer namespace."""

from __future__ import annotations

from typing import Protocol

from turbopuffer.types import Filter, NamespaceQueryResponse, RankBy, Row


TITLE_WEIGHT_BY_LOCALE = {
    "us": 64.0,
    "jp": 8.0,
    "es": 32.0,
}
DEFAULT_TITLE_WEIGHT = 16.0
DESCRIPTION_WEIGHT = 1.0


class QueryNamespace(Protocol):
    """The namespace query surface used by this module."""

    def query(
        self,
        *,
        rank_by: RankBy,
        filters: Filter,
        top_k: int,
        include_attributes: list[str],
    ) -> NamespaceQueryResponse: ...


def rank_by(query_text: str, locale: str | None = None) -> RankBy:
    """Build a locale-tuned title/description weighted BM25 score."""
    title_weight = TITLE_WEIGHT_BY_LOCALE.get(locale or "", DEFAULT_TITLE_WEIGHT)
    return (
        "Sum",
        (
            ("Product", title_weight, ("title", "BM25", query_text)),
            ("Product", DESCRIPTION_WEIGHT, ("description", "BM25", query_text)),
        ),
    )


def query(
    namespace: QueryNamespace,
    query_text: str,
    locale: str,
    top_k: int = 10,
) -> list[Row]:
    """Issue one weighted product-text query, restricted to its locale."""
    if not query_text.strip():
        raise ValueError("query_text must not be empty")
    if not locale.strip():
        raise ValueError("locale must not be empty")
    if top_k < 1:
        raise ValueError("top_k must be greater than zero")

    response = namespace.query(
        rank_by=rank_by(query_text, locale),
        filters=("locale", "Eq", locale),
        top_k=top_k,
        include_attributes=["title", "description", "locale"],
    )
    return response.rows or []
