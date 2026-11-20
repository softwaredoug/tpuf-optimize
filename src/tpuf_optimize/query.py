"""A single weighted BM25 query against a Turbopuffer namespace."""

from __future__ import annotations

from typing import Protocol

from turbopuffer.types import Filter, NamespaceQueryResponse, RankBy, Row


TITLE_WEIGHT = 2.0
DESCRIPTION_WEIGHT = 1.0
BRAND_WEIGHT = 0.5
BULLET_POINT_WEIGHT = 0.5
CATEGORY_WEIGHT = 0.25
COLOR_WEIGHT = 0.25


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


def rank_by(query_text: str) -> RankBy:
    """Build a weighted BM25 score across product text attributes."""
    return (
        "Sum",
        (
            ("Product", TITLE_WEIGHT, ("title", "BM25", query_text)),
            ("Product", DESCRIPTION_WEIGHT, ("description", "BM25", query_text)),
            ("Product", BRAND_WEIGHT, ("brand", "BM25", query_text)),
            ("Product", BULLET_POINT_WEIGHT, ("bullet_point", "BM25", query_text)),
            ("Product", CATEGORY_WEIGHT, ("category", "BM25", query_text)),
            ("Product", COLOR_WEIGHT, ("color", "BM25", query_text)),
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
        rank_by=rank_by(query_text),
        filters=("locale", "Eq", locale),
        top_k=top_k,
        include_attributes=["title", "description", "locale"],
    )
    return response.rows or []
