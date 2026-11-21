"""ESCI product indexing and retrieval strategy for Turbopuffer."""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any, cast

import pandas as pd
from turbopuffer.types import AttributeSchemaParam, RankBy, RowParam
from turbopuffer.types.namespace_multi_query_params import Query


class TPufStrategy:
    """Own the schema, document mapping, and query behavior being optimized."""

    TITLE_WEIGHT = 2.0
    DESCRIPTION_WEIGHT = 1.0
    COMBINED_WEIGHT = 0.5
    COMBINED_TEXT = "combined_text"

    TEXT_COLUMNS = {
        "title": "title",
        "description": "description",
        "brand": "product_brand",
        "bullet_point": "product_bullet_point",
        "color": "product_color",
        "category": "product_category",
    }
    DOCUMENT_COLUMNS = {**TEXT_COLUMNS, "locale": "product_locale"}

    def __init__(self, combined_text: bool = False) -> None:
        self.combined_text = combined_text

    @property
    def schema(self) -> dict[str, AttributeSchemaParam]:
        """Schema for the strategy's full-text indexed attributes."""
        schema: dict[str, AttributeSchemaParam] = {
            name: {"type": "string", "full_text_search": True}
            for name in self.TEXT_COLUMNS
        }
        if self.combined_text:
            schema[self.COMBINED_TEXT] = {"type": "string", "full_text_search": True}
        return schema

    @staticmethod
    def _as_text(value: object) -> str | None:
        if value is None or value is pd.NA or value is pd.NaT:
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        text = str(value).strip()
        return text or None

    def documents(
        self, corpus: pd.DataFrame, limit: int | None = None
    ) -> Iterator[RowParam]:
        """Convert ESCI corpus records into Turbopuffer documents."""
        id_column = "doc_id" if "doc_id" in corpus.columns else "product_id"
        selected_columns = [id_column]
        selected_columns.extend(
            source for source in self.DOCUMENT_COLUMNS.values()
            if source in corpus.columns
        )
        output_names = {
            source: name
            for name, source in self.DOCUMENT_COLUMNS.items()
            if source in corpus.columns
        }

        emitted = 0
        seen_document_ids: set[str] = set()
        for values in corpus[selected_columns].itertuples(index=False, name=None):
            if limit is not None and emitted >= limit:
                break
            document_id = self._as_text(values[0])
            if document_id is None or document_id in seen_document_ids:
                continue
            seen_document_ids.add(document_id)

            document: dict[str, Any] = {"id": document_id}
            for source, value in zip(selected_columns[1:], values[1:]):
                text = self._as_text(value)
                if text is not None:
                    document[output_names[source]] = text
            if self.combined_text:
                combined_parts = [
                    part
                    for field in (
                        "title",
                        "title",
                        "description",
                        "brand",
                        "bullet_point",
                        "category",
                        "color",
                    )
                    if (part := document.get(field))
                ]
                if combined_parts:
                    document[self.COMBINED_TEXT] = " ".join(combined_parts)
            yield cast(RowParam, document)
            emitted += 1

    def rank_by(self, query_text: str) -> RankBy:
        """Build a BM25 score with title weighted twice as much as description."""
        fields = [
            ("Product", self.TITLE_WEIGHT, ("title", "BM25", query_text)),
            (
                "Product",
                self.DESCRIPTION_WEIGHT,
                ("description", "BM25", query_text),
            ),
        ]
        if self.combined_text:
            fields.append(
                (
                    "Product",
                    self.COMBINED_WEIGHT,
                    (self.COMBINED_TEXT, "BM25", query_text),
                )
            )
        return cast(RankBy, ("Sum", tuple(fields)))

    def query(self, query_text: str, locale: str, top_k: int) -> Query:
        """Build one locale-filtered query for Turbopuffer's multi-query API."""
        if not query_text.strip():
            raise ValueError("query_text must not be empty")
        if not locale.strip():
            raise ValueError("locale must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be greater than zero")
        return {
            "rank_by": self.rank_by(query_text),
            "filters": ("locale", "Eq", locale),
            "top_k": top_k,
            "include_attributes": ["title", "description", "locale"],
        }
