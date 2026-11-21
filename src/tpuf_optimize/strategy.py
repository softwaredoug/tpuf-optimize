"""ESCI field variants and retrieval schema for Turbopuffer."""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Iterator
from typing import Any, cast

import pandas as pd
from turbopuffer.types import AttributeSchemaParam, DistanceMetric, RankBy, RowParam
from turbopuffer.types.namespace_multi_query_params import Query


class TPufStrategy:
    """Own the fields, analyzers, embeddings, and queries under experimentation."""

    TITLE_WEIGHT = 2.0
    DESCRIPTION_WEIGHT = 1.0
    MAX_BINARY_PRESENCE_TERMS = 16
    _QUERY_WORD = re.compile(r"[^\W_]+", re.UNICODE)

    # These locales remove stopwords in their locale-specific indexed fields.
    # Keep the client-side predicate list intentionally conservative: the
    # token predicates are scored independently, so stopwords must not receive
    # a score merely because the index analyzer discarded them.
    _STOPWORDS = {
        "us": frozenset(
            """
            a an and are as at be by for from in is it of on or that the to
            with
            """.split()
        ),
        "es": frozenset(
            """
            a al algo algunas algunos ante antes como con contra cual cuando
            de del desde donde durante e el ella ellas ellos en entre era es
            esa esas ese eso esos esta estaba estado estas este esto estos
            fue ha hasta hay la las le les lo los mas me mi muy ni no nos o
            para pero por que se sin sobre su sus te tu un una uno unos y ya
            """.split()
        ),
    }

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
        """Build the locale-appropriate weighted title/description BM25 query."""
        locale_analyzer = self.LOCALE_ANALYZERS.get(locale)
        suffix = (
            locale_analyzer[0]
            if locale_analyzer is not None
            else self.DEFAULT_ANALYZER
        )
        return (
            "Sum",
            (
                (
                    "Product",
                    self.TITLE_WEIGHT,
                    (f"title__{suffix}", "BM25", query_text),
                ),
                (
                    "Product",
                    self.DESCRIPTION_WEIGHT,
                    (f"description__{suffix}", "BM25", query_text),
                ),
            ),
        )

    @classmethod
    def binary_presence_terms(cls, query_text: str, locale: str) -> list[str]:
        """Return bounded, distinct query terms for the binary-presence arm.

        Japanese uses dictionary-based Kuromoji segmentation in the index.
        This module has no equivalent client-side dictionary, so it declines
        that arm rather than splitting Japanese into misleading codepoints.
        """
        if locale == "jp":
            return []

        normalized = unicodedata.normalize("NFKC", query_text).casefold()
        stopwords = cls._STOPWORDS.get(locale, frozenset())
        terms: list[str] = []
        seen: set[str] = set()
        for match in cls._QUERY_WORD.finditer(normalized):
            term = match.group()
            if term in stopwords or term in seen:
                continue
            seen.add(term)
            terms.append(term)
            if len(terms) == cls.MAX_BINARY_PRESENCE_TERMS:
                break
        return terms

    def binary_presence_query(
        self, query_text: str, locale: str, top_k: int
    ) -> Query | None:
        """Build whole-namespace lexical candidate retrieval without BM25.

        Each distinct query term contributes max(2 * title_presence,
        passage_presence), then the term contributions are summed. This is a
        retrieval query, not a candidate-ID-constrained reranker.

        Return ``None`` when client-side analysis cannot safely construct
        predicates (empty/stopword-only input or Japanese Kuromoji input); the
        caller should then use its normal retrieval arm. This avoids issuing
        a query whose clauses all score zero.
        """
        if not query_text.strip():
            raise ValueError("query_text must not be empty")
        if not locale.strip():
            raise ValueError("locale must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be greater than zero")

        terms = self.binary_presence_terms(query_text, locale)
        if not terms:
            return None

        locale_analyzer = self.LOCALE_ANALYZERS.get(locale)
        suffix = (
            locale_analyzer[0]
            if locale_analyzer is not None
            else self.DEFAULT_ANALYZER
        )
        title_field = f"title__{suffix}"
        passage_field = f"passage__{suffix}"
        clauses: list[RankBy] = []
        for term in terms:
            clauses.append(
                (
                    "Max",
                    [
                        (
                            "Product",
                            self.TITLE_WEIGHT,
                            (title_field, "ContainsAllTokens", term),
                        ),
                        (passage_field, "ContainsAllTokens", term),
                    ],
                )
            )

        return {
            "rank_by": ("Sum", clauses),
            "filters": ("locale", "Eq", locale),
            "top_k": top_k,
            "include_attributes": ["title", "description", "locale"],
        }

    @staticmethod
    def sort_binary_presence_rows(rows: list[Any]) -> list[Any]:
        """Stably order returned rows by score, then document ID.

        Composite rank expressions do not accept an attribute tie-breaker.
        This makes ties deterministic within the returned candidate pool;
        callers needing deterministic cutoff ties should overfetch before
        truncating locally.
        """
        return sorted(
            rows,
            key=lambda row: (
                -float(
                    row["$dist"]
                    if hasattr(row, "__getitem__")
                    else getattr(row, "$dist")
                ),
                str(row.id),
            ),
        )

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
            "include_attributes": ["title", "description", "locale"],
        }
