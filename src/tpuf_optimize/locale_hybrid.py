"""Offline rank-fusion helpers for locale-weight experiments.

This module deliberately works on saved dense/lexical result lists. It does not
issue Turbopuffer queries and it only uses the fields present in those lists.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


GLOBAL_DENSE_WEIGHT = 2.0
GLOBAL_LEXICAL_WEIGHT = 1.0
RRF_K = 10
ARM_LIMIT = 100
RESULT_LIMIT = 10
MIN_LOCALE_SUPPORT = 20
SHRINKAGE_PRIOR_GROUPS = 2.0

# A compact, predeclared set of alternatives; do not expand it per locale.
WEIGHT_CANDIDATES: tuple[tuple[float, float], ...] = (
    (2.0, 1.0),
    (3.0, 1.0),
    (1.0, 1.0),
    (1.0, 2.0),
    (1.0, 3.0),
)


@dataclass(frozen=True)
class ScoredDocument:
    """A result document with a fused score and deterministic tie order."""

    doc_id: str
    score: float


def weighted_rrf(
    dense: Sequence[Mapping[str, Any]],
    lexical: Sequence[Mapping[str, Any]],
    *,
    dense_weight: float = GLOBAL_DENSE_WEIGHT,
    lexical_weight: float = GLOBAL_LEXICAL_WEIGHT,
    k: int = RRF_K,
    arm_limit: int = ARM_LIMIT,
    result_limit: int = RESULT_LIMIT,
) -> list[ScoredDocument]:
    """Fuse two ranked arms with weighted reciprocal-rank fusion.

    Input order is rank order. Documents first encountered in the dense arm
    retain precedence on equal fused scores, followed by lexical-only
    documents, matching the stable dense-first baseline behavior.
    """
    if dense_weight < 0 or lexical_weight < 0:
        raise ValueError("weights must be non-negative")
    if dense_weight == 0 and lexical_weight == 0:
        raise ValueError("at least one weight must be positive")
    if k < 0 or arm_limit < 1 or result_limit < 1:
        raise ValueError("k must be non-negative and limits must be positive")

    scores: dict[str, float] = {}
    tie_order: list[str] = []
    for rows, weight in ((dense, dense_weight), (lexical, lexical_weight)):
        for rank, row in enumerate(rows[:arm_limit], start=1):
            doc_id = str(row["doc_id"])
            if doc_id not in scores:
                scores[doc_id] = 0.0
                tie_order.append(doc_id)
            scores[doc_id] += weight / (k + rank)

    tie_position = {doc_id: index for index, doc_id in enumerate(tie_order)}
    ordered = sorted(
        scores,
        key=lambda doc_id: (-scores[doc_id], tie_position[doc_id]),
    )
    return [
        ScoredDocument(doc_id, scores[doc_id])
        for doc_id in ordered[:result_limit]
    ]


def shrink_locale_weights(
    local_weights: tuple[float, float],
    support: int,
    *,
    global_weights: tuple[float, float] = (
        GLOBAL_DENSE_WEIGHT,
        GLOBAL_LEXICAL_WEIGHT,
    ),
    min_support: int = MIN_LOCALE_SUPPORT,
    prior_groups: float = SHRINKAGE_PRIOR_GROUPS,
) -> tuple[float, float]:
    """Shrink a fitted locale pair toward global weights, with fallback.

    ``support`` counts independent training-fit query groups, not candidate
    documents. Below ``min_support`` the global parent is used unchanged.
    The global pair acts as ``prior_groups`` pseudo-observations.
    """
    if support < 0 or min_support < 0 or prior_groups < 0:
        raise ValueError("support and shrinkage parameters must be non-negative")
    if support < min_support:
        return global_weights
    fraction = support / (support + prior_groups)
    return tuple(
        global_weight + fraction * (local_weight - global_weight)
        for local_weight, global_weight in zip(local_weights, global_weights)
    )  # type: ignore[return-value]


def grouped_validation_split(
    query_texts: Mapping[str, str],
    *,
    validation_fraction: float = 0.2,
    salt: str = "locale-hybrid-v1",
) -> tuple[set[str], set[str]]:
    """Deterministically split query IDs while keeping repeated queries grouped."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    fit: set[str] = set()
    validation: set[str] = set()
    for query_id, query_text in query_texts.items():
        normalized = re.sub(r"\s+", " ", query_text.casefold()).strip()
        group = normalized or query_id
        digest = hashlib.sha256(f"{salt}\0{group}".encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "big") / 2**64
        (validation if bucket < validation_fraction else fit).add(query_id)
    return fit, validation


@dataclass(frozen=True)
class MetricSummary:
    """Mean query metric and a standard-error estimate."""

    count: int
    mean: float | None
    standard_error: float | None


def summarize(values: Sequence[float]) -> MetricSummary:
    """Summarize query-level metrics with the usual sample standard error."""
    count = len(values)
    if count == 0:
        return MetricSummary(0, None, None)
    mean = sum(values) / count
    if count < 2:
        return MetricSummary(count, mean, None)
    variance = sum((value - mean) ** 2 for value in values) / (count - 1)
    return MetricSummary(count, mean, math.sqrt(variance / count))


def paired_metric_summaries(
    parent: Mapping[str, float],
    candidate: Mapping[str, float],
    locales: Mapping[str, str],
) -> dict[str, MetricSummary]:
    """Return paired NDCG deltas overall and per locale, with uncertainty."""
    shared = parent.keys() & candidate.keys() & locales.keys()
    deltas = {query_id: candidate[query_id] - parent[query_id] for query_id in shared}
    result = {"overall": summarize(list(deltas.values()))}
    for locale in sorted({locales[query_id] for query_id in shared}):
        result[locale] = summarize(
            [delta for query_id, delta in deltas.items() if locales[query_id] == locale]
        )
    return result
