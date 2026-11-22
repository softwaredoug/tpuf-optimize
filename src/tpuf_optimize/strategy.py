"""ESCI field variants and retrieval schema for Turbopuffer."""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterator, Sequence
from typing import Any, cast

import pandas as pd
from turbopuffer.types import (
    AttributeSchemaParam,
    DistanceMetric,
    RankBy,
    Row,
    RowParam,
)
from turbopuffer.types.namespace_multi_query_params import Query


class TPufStrategy:
    """Own the fields, analyzers, embeddings, and queries under experimentation."""

    TITLE_WEIGHT = 2.0
    FUZZY_TITLE_WEIGHT = 0.35
    FUZZY_BRAND_WEIGHT = 0.15
    FIELD_FUSION_CHANNELS = {
        "passage_fold": (("passage__word_v4__fold", 0.5),),
    }

    def __init__(self, field_fusion: str = "none") -> None:
        """Select an optional lexical analyzer channel."""
        if field_fusion != "none" and field_fusion not in self.FIELD_FUSION_CHANNELS:
            raise ValueError(
                f"Unknown field fusion {field_fusion!r}; expected 'none' or one of "
                f"{', '.join(self.FIELD_FUSION_CHANNELS)}"
            )
        self.field_fusion = field_fusion

    EMBEDDING_MODEL = "voyage/voyage-4"
    EMBEDDING_DIMS = 1024
    EMBEDDING_TYPE = "[1024]f16"
    MAX_BATCH_SIZE = 256

    TEXT_COLUMNS = {
        "title": "title",
        "description": "description",
        "brand": "product_brand",
        "bullet_point": "product_bullet_point",
        "color": "product_color",
        "category": "product_category",
    }
    DOCUMENT_COLUMNS = {**TEXT_COLUMNS, "locale": "product_locale"}
    DOCUMENT_COLUMNS_INVERSE = {
        source: name for name, source in DOCUMENT_COLUMNS.items()
    }
    TEXT_FIELDS = (*TEXT_COLUMNS, "passage")

    # Analyzer suffixes follow:
    # logical_field__tokenizer__language__options.
    LOCALE_ANALYZERS: dict[str, tuple[str, dict[str, object]]] = {
        "us": (
            "word_v4__en__stem_stop",
            {
                "tokenizer": "word_v4",
                "language": "english",
                "stemming": True,
                "remove_stopwords": True,
            },
        ),
        "es": (
            "word_v4__es__stem_stop_fold",
            {
                "tokenizer": "word_v4",
                "language": "spanish",
                "stemming": True,
                "remove_stopwords": True,
                "ascii_folding": True,
            },
        ),
        "jp": ("kuromoji__ja", {"tokenizer": "kuromoji"}),
    }
    DEFAULT_ANALYZER = "word_v4__default"
    DEFAULT_FTS_CONFIG: dict[str, object] = {"tokenizer": "word_v4"}

    # Keep extra tokenizers and BM25 variants concentrated on title and brand.
    EXTRA_ANALYZERS: dict[str, tuple[str, dict[str, object], str | None]] = {
        "title__word_v2__default": ("title", {"tokenizer": "word_v2"}, None),
        "brand__word_v2__default": ("brand", {"tokenizer": "word_v2"}, None),
        "title__whitespace__default": (
            "title",
            {"tokenizer": "whitespace"},
            None,
        ),
        "brand__whitespace__default": (
            "brand",
            {"tokenizer": "whitespace"},
            None,
        ),
        "title__word_v4__fold": (
            "title",
            {"tokenizer": "word_v4", "ascii_folding": True},
            None,
        ),
        "title__word_v4__b025": (
            "title",
            {"tokenizer": "word_v4", "b": 0.25},
            None,
        ),
        "description__word_v4__b025": (
            "description",
            {"tokenizer": "word_v4", "b": 0.25},
            None,
        ),
        "passage__word_v2__default": ("passage", {"tokenizer": "word_v2"}, None),
        "passage__whitespace__default": (
            "passage",
            {"tokenizer": "whitespace"},
            None,
        ),
        "passage__word_v4__fold": (
            "passage",
            {"tokenizer": "word_v4", "ascii_folding": True},
            None,
        ),
        "passage__word_v4__b025": (
            "passage",
            {"tokenizer": "word_v4", "b": 0.25},
            None,
        ),
        "title__word_v4__fuzzy": (
            "title",
            {"tokenizer": "word_v4"},
            "fuzzy",
        ),
        "brand__word_v4__fuzzy": (
            "brand",
            {"tokenizer": "word_v4"},
            "fuzzy",
        ),
        "passage__word_v4__fuzzy": (
            "passage",
            {"tokenizer": "word_v4"},
            "fuzzy",
        ),
    }
    PRETOKENIZED_FIELDS = ("title", "brand", "passage")
    FUZZY_QUERY_MIN_CHARS = 6
    LEXICAL_RERANK_MAX_BOOST = 0.18

    EMBEDDING_FIELDS = {
        "passage__embed__voyage_4": ("embedding__voyage_4__passage", "passage"),
    }

    @property
    def distance_metric(self) -> DistanceMetric:
        """Distance metric used by the strategy's dense-vector attributes."""
        return "cosine_distance"

    @staticmethod
    def _fts_schema(config: dict[str, object]) -> AttributeSchemaParam:
        return {
            "type": "string",
            "full_text_search": config,  # type: ignore[typeddict-item]
        }

    @property
    def schema(self) -> dict[str, AttributeSchemaParam]:
        """All base, analyzer-specific, fuzzy, and embedding attributes."""
        schema: dict[str, AttributeSchemaParam] = {
            "locale": {"type": "string", "filterable": True}
        }
        for field in self.TEXT_FIELDS:
            # Raw source fields are returned to callers but are not separately indexed.
            schema[field] = {"type": "string", "filterable": False}
            schema[f"{field}__{self.DEFAULT_ANALYZER}"] = self._fts_schema(
                self.DEFAULT_FTS_CONFIG
            )
            for suffix, config in self.LOCALE_ANALYZERS.values():
                schema[f"{field}__{suffix}"] = self._fts_schema(config)

        for name, (_field, config, extra_index) in self.EXTRA_ANALYZERS.items():
            field_schema: dict[str, object] = {
                "type": "string",
                "full_text_search": config,
            }
            if extra_index == "fuzzy":
                field_schema["fuzzy"] = True
            schema[name] = cast(AttributeSchemaParam, field_schema)

        for field in self.PRETOKENIZED_FIELDS:
            schema[f"{field}__pretokenized_array__simple"] = {
                "type": "[]string",
                "full_text_search": {"tokenizer": "pre_tokenized_array"},
            }

        for source_name, (vector_name, _view) in self.EMBEDDING_FIELDS.items():
            schema[source_name] = {
                "type": "string",
                "filterable": False,
                "embed": {
                    "model": self.EMBEDDING_MODEL,
                    "attribute": vector_name,
                    "dims": self.EMBEDDING_DIMS,
                },
            }
            schema[vector_name] = {
                "type": self.EMBEDDING_TYPE,
                "ann": True,
            }
        return schema

    @staticmethod
    def _as_text(value: object) -> str | None:
        if value is None or value is pd.NA or value is pd.NaT:
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def pretokenize(text: str) -> list[str]:
        """Simple Unicode tokenizer for the pre_tokenized_array experiment."""
        normalized = unicodedata.normalize("NFKC", text).casefold()
        tokens: list[str] = []
        current: list[str] = []

        def flush() -> None:
            if current:
                tokens.append("".join(current))
                current.clear()

        for character in normalized:
            codepoint = ord(character)
            is_cjk = (
                0x3040 <= codepoint <= 0x30FF
                or 0x3400 <= codepoint <= 0x9FFF
                or 0xF900 <= codepoint <= 0xFAFF
            )
            if is_cjk:
                flush()
                tokens.append(character)
            elif character.isalnum():
                current.append(character)
            else:
                flush()
        flush()
        return tokens

    def documents(
        self, corpus: pd.DataFrame, limit: int | None = None
    ) -> Iterator[RowParam]:
        """Expand each ESCI product into raw, analyzer-specific, and embed fields."""
        id_column = "doc_id" if "doc_id" in corpus.columns else "product_id"
        selected_columns = [id_column]
        selected_columns.extend(
            source
            for source in self.DOCUMENT_COLUMNS.values()
            if source in corpus.columns
        )

        emitted = 0
        for values in corpus[selected_columns].itertuples(index=False, name=None):
            if limit is not None and emitted >= limit:
                break
            document_id = self._as_text(values[0])
            if document_id is None:
                continue

            document: dict[str, Any] = {"id": document_id}
            for source, value in zip(selected_columns[1:], values[1:]):
                text = self._as_text(value)
                if text is not None:
                    document[self.DOCUMENT_COLUMNS_INVERSE[source]] = text

            title = str(document.get("title", ""))
            description = str(document.get("description", ""))
            passage = (
                f"## {title}\n\n{description}".rstrip()
                if title
                else description
            )
            if passage:
                document["passage"] = passage

            locale = str(document.get("locale", ""))
            for field in self.TEXT_FIELDS:
                text = str(document.get(field, ""))
                if text:
                    document[f"{field}__{self.DEFAULT_ANALYZER}"] = text
                    locale_analyzer = self.LOCALE_ANALYZERS.get(locale)
                    if locale_analyzer is not None:
                        suffix, _ = locale_analyzer
                        document[f"{field}__{suffix}"] = text

            for name, (field, _config, _extra_index) in self.EXTRA_ANALYZERS.items():
                document[name] = str(document.get(field, ""))

            for field in self.PRETOKENIZED_FIELDS:
                document[f"{field}__pretokenized_array__simple"] = self.pretokenize(
                    str(document.get(field, ""))
                )

            for source_name, (_vector_name, source_field) in self.EMBEDDING_FIELDS.items():
                document[source_name] = str(document.get(source_field, ""))

            yield cast(RowParam, document)
            emitted += 1

    def rank_by(self, query_text: str, locale: str) -> RankBy:
        """Rank the combined passage, boosting title and typo-tolerant matches."""
        locale_analyzer = self.LOCALE_ANALYZERS.get(locale)
        suffix = (
            locale_analyzer[0]
            if locale_analyzer is not None
            else self.DEFAULT_ANALYZER
        )
        clauses: list[Any] = [
            (f"passage__{suffix}", "BM25", query_text),
            (
                "Product",
                self.TITLE_WEIGHT,
                (f"title__{suffix}", "BM25", query_text),
            ),
        ]

        # Fuzzy filters return a binary score. Apply a small boost for long terms
        # to recover likely typos without replacing the more discriminative BM25.
        if locale != "jp":
            fuzzy_options = {
                "max_edit_distance": [
                    {"min_query_chars": self.FUZZY_QUERY_MIN_CHARS, "distance": 1}
                ],
                "case_sensitive": False,
            }
            for token in re.findall(r"[^\W_]+", query_text, flags=re.UNICODE):
                if len(token) < self.FUZZY_QUERY_MIN_CHARS:
                    continue
                clauses.extend(
                    (
                        (
                            "Product",
                            self.FUZZY_TITLE_WEIGHT,
                            (
                                "title__word_v4__fuzzy",
                                "Fuzzy",
                                token,
                                fuzzy_options,
                            ),
                        ),
                        (
                            "Product",
                            self.FUZZY_BRAND_WEIGHT,
                            (
                                "brand__word_v4__fuzzy",
                                "Fuzzy",
                                token,
                                fuzzy_options,
                            ),
                        ),
                    )
                )

        for field, weight in self.FIELD_FUSION_CHANNELS.get(
            self.field_fusion, ()
        ):
            clauses.append(("Product", weight, (field, "BM25", query_text)))

        return cast(RankBy, ("Sum", tuple(clauses)))

    def query(self, query_text: str, locale: str, top_k: int) -> Query:
        """Build one locale-filtered query for Turbopuffer's multi-query API."""
        if not query_text.strip():
            raise ValueError("query_text must not be empty")
        if not locale.strip():
            raise ValueError("locale must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be greater than zero")
        return {
            "rank_by": self.rank_by(query_text, locale),
            "filters": ("locale", "Eq", locale),
            "top_k": top_k,
            "include_attributes": ["title", "description", "brand", "locale"],
        }

    def lexical_candidate_queries(
        self, query_text: str, locale: str, top_k: int
    ) -> tuple[Query, ...]:
        """Build selective lexical retrieval arms for post-retrieval union fusion.

        The regular passage/title/fuzzy query remains an independent channel.
        These arms retrieve all-token title matches and exact phrases found in
        the indexed passage or brand, rather than adding those signals as
        additive rank-by boosts.
        """
        if not query_text.strip():
            raise ValueError("query_text must not be empty")
        if not locale.strip():
            raise ValueError("locale must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be greater than zero")

        locale_analyzer = self.LOCALE_ANALYZERS.get(locale)
        suffix = (
            locale_analyzer[0]
            if locale_analyzer is not None
            else self.DEFAULT_ANALYZER
        )
        return (
            self._filtered_candidate_query(
                f"title__{suffix}",
                "ContainsAllTokens",
                query_text,
                locale,
                top_k,
            ),
            self._filtered_candidate_query(
                f"passage__{suffix}",
                "ContainsTokenSequence",
                query_text,
                locale,
                top_k,
            ),
            self._filtered_candidate_query(
                "brand__word_v4__default",
                "ContainsTokenSequence",
                query_text,
                locale,
                top_k,
            ),
        )

    @staticmethod
    def _filtered_candidate_query(
        field: str,
        filter_operator: str,
        query_text: str,
        locale: str,
        top_k: int,
    ) -> Query:
        return {
            "rank_by": (field, "BM25", query_text),
            "filters": (
                "And",
                (
                    ("locale", "Eq", locale),
                    (field, filter_operator, query_text),
                ),
            ),
            "top_k": top_k,
            "include_attributes": ["title", "description", "brand", "locale"],
        }

    @staticmethod
    def _position_tokens(text: str) -> list[str]:
        """Tokenize text consistently for a lightweight lexical reranker."""
        normalized = unicodedata.normalize("NFKC", text).casefold()
        return re.findall(
            r"[^\W_]+|[\u3040-\u30ff\u3400-\u9fff\uf900-\ufaff]",
            normalized,
            flags=re.UNICODE,
        )

    @classmethod
    def _field_match_quality(cls, query_tokens: Sequence[str], text: str) -> float:
        """Score query coverage, order, compactness, phrase, and early matches."""
        if not query_tokens or not text:
            return 0.0

        field_tokens = cls._position_tokens(text)
        if not field_tokens:
            return 0.0

        unique_query = list(dict.fromkeys(query_tokens))
        positions = {
            token: [index for index, value in enumerate(field_tokens) if value == token]
            for token in unique_query
        }
        matched = [token for token in unique_query if positions[token]]
        coverage = len(matched) / len(unique_query)

        # Count query terms encountered in their original order. Unmatched terms
        # are skipped, so partial coverage still receives proportional credit.
        ordered_count = 0
        cursor = 0
        for token in query_tokens:
            try:
                cursor = field_tokens.index(token, cursor) + 1
            except ValueError:
                continue
            ordered_count += 1
        ordered = ordered_count / len(query_tokens)

        # Find the narrowest field-token window containing all distinct terms
        # that occur in the field; scale by coverage when some terms are absent.
        located = sorted(
            (position, token)
            for token in matched
            for position in positions[token]
        )
        left = 0
        found: dict[str, int] = {}
        compactness = 0.0
        for right, (position, token) in enumerate(located):
            found[token] = found.get(token, 0) + 1
            while len(found) == len(matched) and left <= right:
                width = position - located[left][0] + 1
                compactness = max(compactness, len(matched) / width)
                left_token = located[left][1]
                found[left_token] -= 1
                if not found[left_token]:
                    del found[left_token]
                left += 1
        compactness *= coverage

        earliest = min((positions[token][0] for token in matched), default=0)
        early = (
            1.0 - earliest / max(len(field_tokens), 1)
            if matched
            else 0.0
        )
        normalized_text = " ".join(field_tokens)
        phrase = " ".join(query_tokens) in normalized_text

        return (
            0.52 * coverage
            + 0.16 * ordered
            + 0.13 * compactness
            + 0.11 * early
            + 0.08 * float(phrase)
        )

    @classmethod
    def lexical_position_score(
        cls, query_text: str, title: str | None, description: str | None,
        brand: str | None,
    ) -> float:
        """Return a bounded lexical feature score; absent fields are neutral."""
        query_tokens = cls._position_tokens(query_text)
        if not query_tokens:
            return 0.0
        fields = [
            (title, 0.70),
            (description, 0.22),
            (brand, 0.08),
        ]
        available_weight = sum(weight for text, weight in fields if text)
        if not available_weight:
            return 0.0
        return sum(
            weight * cls._field_match_quality(query_tokens, text)
            for text, weight in fields
            if text
        ) / available_weight

    @classmethod
    def lexical_cross_field_coverage(
        cls, query_text: str, title: str | None, description: str | None,
        brand: str | None,
    ) -> float:
        """Measure query terms split across fields beyond the best single field."""
        query_terms = set(cls._position_tokens(query_text))
        if not query_terms:
            return 0.0
        field_terms = [
            set(cls._position_tokens(text))
            for text in (title, description, brand)
            if text
        ]
        if not field_terms:
            return 0.0
        union = set().union(*field_terms)
        union_coverage = len(query_terms & union) / len(query_terms)
        best_single_coverage = max(
            len(query_terms & terms) / len(query_terms) for terms in field_terms
        )
        return max(0.0, union_coverage - best_single_coverage)

    @classmethod
    def rerank_lexically(
        cls,
        query_text: str,
        rows: list[Row],
        max_boost: float = LEXICAL_RERANK_MAX_BOOST,
        cross_field_boost: float = 0.0,
    ) -> list[Row]:
        """Apply bounded position and optional cross-field lexical adjustments."""
        if not 0.0 <= max_boost <= 1.0:
            raise ValueError("max_boost must be between 0 and 1")
        if not 0.0 <= cross_field_boost <= 1.0:
            raise ValueError("cross_field_boost must be between 0 and 1")
        if len(rows) < 2:
            return rows
        denominator = max(len(rows) - 1, 1)

        def adjusted(item: tuple[int, Row]) -> tuple[float, int]:
            original_rank, row = item
            attributes = (
                cast(dict[str, object], row)
                if isinstance(row, dict)
                else cast(dict[str, object], row.model_dump())
            )
            title = cast(str | None, attributes.get("title"))
            description = cast(str | None, attributes.get("description"))
            brand = cast(str | None, attributes.get("brand"))
            score = cls.lexical_position_score(
                query_text,
                title,
                description,
                brand,
            )
            cross_field_score = (
                cls.lexical_cross_field_coverage(
                    query_text, title, description, brand
                )
                if cross_field_boost
                else 0.0
            )
            parent_utility = 1.0 - (original_rank / denominator)
            return (
                parent_utility + max_boost * score
                + cross_field_boost * cross_field_score,
                -original_rank,
            )

        return [
            row
            for _rank, row in sorted(
                enumerate(rows), key=adjusted, reverse=True
            )
        ]
