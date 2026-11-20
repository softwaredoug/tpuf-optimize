"""A single weighted BM25 query against a Turbopuffer namespace."""

from __future__ import annotations

from typing import Protocol

from turbopuffer.types import Filter, NamespaceQueryResponse, RankBy, Row


TITLE_WEIGHT = 6.0
PHRASE_WEIGHT = 4.0


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
    """Build title BM25 ranking with a soft exact-phrase title boost."""
    return (
        "Sum",
        (
            ("Product", TITLE_WEIGHT, ("title", "BM25", query_text)),
            (
                "Product",
                PHRASE_WEIGHT,
                ("title", "ContainsTokenSequence", query_text),
            ),
        ),
    )


def query(
    namespace: QueryNamespace,
    query_text: str,
    locale: str,
    top_k: int = 10,
) -> list[Row]:
    """Issue one phrase-boosted title query, restricted to its locale."""
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
