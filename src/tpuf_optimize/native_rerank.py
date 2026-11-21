"""Frozen two-arm ESCI candidate retrieval and native reranker evaluation."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from statistics import mean, stdev
from typing import Any, cast

from turbopuffer.types import Row
from turbopuffer.types.namespace_multi_query_params import Query

from tpuf_optimize.strategy import TPufStrategy

NATIVE_RERANK_MODELS = {
    "cohere": "cohere/rerank-v3.5",
    "voyage": "voyage/rerank-3",
}
RERANK_CANDIDATE_DEPTH = 100
RERANK_TOP_K = 10
RRF_RANK_CONSTANT = 10
RRF_BASE_WEIGHTS = (2.0, 1.0)
RRF_IDENTIFIER_WEIGHTS = (1.0, 2.0)
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_IDENTIFIER_RE = re.compile(r"(?=.*[a-z])(?=.*\d)[a-z0-9_-]{5,}", re.IGNORECASE)
_IDENTITY_FIELDS = ("title", "brand", "color", "category")


def normalize_locale(locale: str) -> str:
    """Map the ESCI locale names returned by Referee to indexed locale values."""
    try:
        return {
            "en-US": "us",
            "es-ES": "es",
            "ja-JP": "jp",
            "us": "us",
            "es": "es",
            "jp": "jp",
        }[locale]
    except KeyError as error:
        raise ValueError(f"Unsupported ESCI locale: {locale!r}") from error


def candidate_queries(
    strategy: TPufStrategy, query_text: str, locale: str
) -> tuple[Query, Query]:
    """Build the exact Voyage-4 ANN and locale-aware weighted BM25 arms."""
    if not query_text.strip():
        raise ValueError("query_text must not be empty")
    indexed_locale = normalize_locale(locale)
    include_attributes = [
        "passage",
        "title",
        "description",
        "brand",
        "color",
        "category",
        "locale",
    ]
    lexical = strategy.query(query_text, indexed_locale, RERANK_CANDIDATE_DEPTH)
    lexical["include_attributes"] = include_attributes
    dense: Query = {
        "rank_by": (
            "passage__embed__voyage_4",
            "ANN",
            ("Embed", query_text),
        ),
        "filters": ("locale", "Eq", indexed_locale),
        "top_k": RERANK_CANDIDATE_DEPTH,
        "include_attributes": include_attributes,
    }
    return dense, lexical


def _row_value(candidate: Mapping[str, object], key: str) -> str:
    value = candidate.get(key)
    return "" if value is None else str(value)


def _normalized_tokens(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(text)]


def _direct_attribute_match(query: str, candidate: Mapping[str, object]) -> bool:
    query_tokens = _normalized_tokens(query)
    if not query_tokens:
        return False
    fields = " ".join(_row_value(candidate, name) for name in _IDENTITY_FIELDS)
    candidate_tokens = set(_normalized_tokens(fields))
    return all(token in candidate_tokens for token in query_tokens)


def identifier_gated_weights(
    query: str,
    dense: Sequence[Mapping[str, object]],
    lexical: Sequence[Mapping[str, object]],
) -> tuple[float, float]:
    """Reproduce the identifier gate used by commit 7e22661."""
    identifiers = [
        token
        for token in _normalized_tokens(query)
        if _IDENTIFIER_RE.fullmatch(token)
    ]
    if not lexical:
        return RRF_BASE_WEIGHTS

    lexical_top = lexical[0]
    lexical_id = _row_value(lexical_top, "id").casefold()
    lexical_text = " ".join(
        (lexical_id, *(_row_value(lexical_top, field) for field in _IDENTITY_FIELDS))
    ).casefold()
    lexical_hit = any(identifier in lexical_text for identifier in identifiers)
    lexical_hit = lexical_hit or _direct_attribute_match(query, lexical_top)
    if not lexical_hit:
        return RRF_BASE_WEIGHTS

    for candidate in dense[:3]:
        dense_id = _row_value(candidate, "id")
        dense_text = " ".join(
            (dense_id, *(_row_value(candidate, field) for field in _IDENTITY_FIELDS))
        ).casefold()
        dense_hit = any(identifier in dense_text for identifier in identifiers)
        dense_hit = dense_hit or _direct_attribute_match(query, candidate)
        if dense_id == _row_value(lexical_top, "id") or dense_hit:
            return RRF_BASE_WEIGHTS
    return RRF_IDENTIFIER_WEIGHTS


def local_identifier_gated_rrf(
    query: str, dense: list[Row], lexical: list[Row], top_k: int = RERANK_TOP_K
) -> list[tuple[str, float]]:
    """Locally RRF the two arms with the 7e22661 lexical identity gate."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    dense_rows = [cast(Mapping[str, object], row.model_dump()) for row in dense]
    lexical_rows = [cast(Mapping[str, object], row.model_dump()) for row in lexical]
    dense_weight, lexical_weight = identifier_gated_weights(
        query, dense_rows, lexical_rows
    )
    scores: dict[str, float] = {}
    for weight, rows in (
        (dense_weight, dense),
        (lexical_weight, lexical),
    ):
        for rank, row in enumerate(rows, start=1):
            identifier = str(row.id)
            scores[identifier] = scores.get(identifier, 0.0) + (
                weight / (RRF_RANK_CONSTANT + rank)
            )
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]


def native_rerank_by(model: str, query_text: str) -> dict[str, str]:
    """Return the outer rerank_by object for a single two-arm multi-query."""
    if model not in NATIVE_RERANK_MODELS.values():
        raise ValueError(f"Unsupported native reranker: {model}")
    return {
        "model": model,
        "attribute": "passage",
        "query": query_text,
    }


def paired_delta_summary(
    baseline_scores: Mapping[str, float],
    variant_scores: Mapping[str, float],
) -> dict[str, float | int]:
    """Summarize paired NDCG differences with a 95% normal-approximation CI."""
    if baseline_scores.keys() != variant_scores.keys():
        raise ValueError("Paired scores must contain the same query IDs")
    deltas = [variant_scores[key] - baseline_scores[key] for key in baseline_scores]
    if not deltas:
        raise ValueError("At least one paired score is required")
    delta = mean(deltas)
    standard_error = stdev(deltas) / math.sqrt(len(deltas)) if len(deltas) > 1 else 0.0
    # N=1000: the 1.96 normal critical value is within 0.003 of t_999.
    margin = 1.96 * standard_error
    return {
        "n": len(deltas),
        "mean_delta": delta,
        "ci95_low": delta - margin,
        "ci95_high": delta + margin,
    }


def result_rows(rows: Sequence[Row]) -> list[tuple[str, float]]:
    """Extract unique ranked rows, rejecting malformed scores or duplicate IDs."""
    results: list[tuple[str, float]] = []
    seen: set[str] = set()
    for row in rows[:RERANK_TOP_K]:
        document_id = str(row.id)
        score = row["$dist"]
        if not document_id or document_id == "-1" or document_id in seen:
            raise ValueError("Turbopuffer returned an invalid or duplicate document ID")
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError("Turbopuffer returned a missing or invalid score")
        seen.add(document_id)
        results.append((document_id, float(score)))
    return results


def passage_chars(rows: Sequence[Row]) -> int:
    """Count retrieved passage characters for a clearly labeled rough token proxy."""
    return sum(
        len(str(row.model_dump().get("passage") or ""))
        for row in rows
    )


def response_performance(response: Any) -> dict[str, float | int | None]:
    """Extract server timings when present in the SDK response."""
    performance = getattr(response, "performance", None)
    if performance is None:
        return {"server_total_ms": None, "query_execution_ms": None}
    values = performance.model_dump()
    return {
        key: cast(float | int | None, values.get(key))
        for key in ("server_total_ms", "query_execution_ms")
    }


def build_submission_rows(
    query_id: str, results: Sequence[tuple[str, float]]
) -> list[dict[str, str | int | float]]:
    """Build Referee result objects, using sentinel only for actual empty results."""
    if not results:
        return [{"query_id": query_id, "doc_id": "-1", "rank": -1, "score": -1}]
    return [
        {
            "query_id": query_id,
            "doc_id": document_id,
            "rank": rank,
            "score": score,
        }
        for rank, (document_id, score) in enumerate(results, start=1)
    ]


def private_task_record(payload: Mapping[str, object]) -> str:
    """Serialize task data for private storage; caller owns file permissions."""
    return json.dumps(payload, ensure_ascii=False, allow_nan=False)
