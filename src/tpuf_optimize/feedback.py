"""Conservative, candidate-only pseudo-relevance feedback helpers.

This module deliberately keeps feedback planning separate from Turbopuffer
calls. A caller may issue at most one lexical query from ``followup_query``;
the original query remains its anchor. The offline probe only reranks documents
already present in the dense/lexical candidate cache, so it does not estimate
the recall benefit of a live follow-up retrieval.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

Candidate = Mapping[str, object]
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_IDENTIFIER_RE = re.compile(r"(?=.*\d)[A-Za-z0-9][A-Za-z0-9._/-]*")
_ATTRIBUTE_FIELDS = ("brand", "color", "category")
_TEXT_FIELDS = ("title", "brand", "color", "category")


@dataclass(frozen=True)
class FeedbackPlan:
    """Expansion decision, with an unchanged original query as its anchor."""

    original_query: str
    locale: str
    terms: tuple[str, ...]

    @property
    def followup_query(self) -> str | None:
        """One lexical follow-up query, or ``None`` when feedback is gated."""
        if not self.terms:
            return None
        return f"{self.original_query.strip()} {' '.join(self.terms)}"


def _tokens(value: object) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(str(value or ""))]


def _identifier_query(query: str) -> bool:
    """Conservatively suppress product-code-like queries."""
    stripped = query.strip()
    if not stripped:
        return True
    raw_tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9._/-]*", stripped)
    return any(
        len(token) >= 3 and _IDENTIFIER_RE.fullmatch(token)
        for token in raw_tokens
    )


def _ranked_by_id(
    candidates: Sequence[Candidate], depth: int
) -> dict[str, tuple[int, Candidate]]:
    ranked: dict[str, tuple[int, Candidate]] = {}
    for rank, candidate in enumerate(candidates[:depth], start=1):
        doc_id = str(candidate.get("doc_id", ""))
        if doc_id and doc_id not in ranked:
            ranked[doc_id] = (rank, candidate)
    return ranked


def plan_feedback(
    query: str,
    locale: str,
    dense: Sequence[Candidate],
    lexical: Sequence[Candidate],
    *,
    depth: int = 20,
    min_support: int = 3,
    max_terms: int = 2,
) -> FeedbackPlan:
    """Find up to two non-query terms supported across joint top candidates.

    Support is counted once per distinct document in the intersection of the
    top ``depth`` Voyage and BM25 results. Attribute values mentioned in the
    query gate out candidates that explicitly contradict the requested value.
    Terms present in at least 75% of the remaining candidates are treated as
    generic. Neither rank list gets an extra vote for terms: each document
    contributes at most once.
    """
    if not query.strip() or not locale.strip():
        raise ValueError("query and locale must not be empty")
    if depth < 1 or min_support < 2 or not 1 <= max_terms <= 2:
        raise ValueError(
            "depth must be positive, min_support at least two, "
            "and max_terms between one and two"
        )
    if _identifier_query(query):
        return FeedbackPlan(query, locale, ())

    dense_by_id = _ranked_by_id(dense, depth)
    lexical_by_id = _ranked_by_id(lexical, depth)
    joint_ids = dense_by_id.keys() & lexical_by_id.keys()
    query_tokens = set(_tokens(query))

    # Infer only attributes evidenced by these candidates, not an external
    # vocabulary. A populated value that disagrees with a query-mentioned
    # value marks that candidate as contradictory for feedback support.
    requested_attributes: dict[str, set[str]] = {}
    for field in _ATTRIBUTE_FIELDS:
        observed_values = [
            _tokens(candidate.get(field, ""))
            for doc_id in joint_ids
            for candidate in (dense_by_id[doc_id][1],)
        ]
        requested = {
            token
            for value_tokens in observed_values
            for token in value_tokens
            if token in query_tokens
        }
        if requested:
            requested_attributes[field] = requested

    usable: list[tuple[str, Candidate, int, int]] = []
    for doc_id in joint_ids:
        dense_rank, candidate = dense_by_id[doc_id]
        lexical_rank, _ = lexical_by_id[doc_id]
        contradicted = False
        for field, requested in requested_attributes.items():
            value_tokens = set(_tokens(candidate.get(field, "")))
            if value_tokens and not value_tokens.intersection(requested):
                contradicted = True
                break
        if not contradicted:
            usable.append((doc_id, candidate, dense_rank, lexical_rank))

    if len(usable) < min_support:
        return FeedbackPlan(query, locale, ())

    document_frequency: Counter[str] = Counter()
    rank_support: Counter[str] = Counter()
    for _doc_id, candidate, dense_rank, lexical_rank in usable:
        candidate_tokens = {
            token
            for field in _TEXT_FIELDS
            for token in _tokens(candidate.get(field, ""))
            if len(token) >= 3 and token.isalpha() and token not in query_tokens
        }
        for token in candidate_tokens:
            document_frequency[token] += 1
            # One rank contribution per document, not one per retrieval arm.
            mean_rank = (dense_rank + lexical_rank) / 2
            rank_support[token] += round(1000 / (10 + mean_rank))

    generic_cutoff = math.ceil(0.75 * len(usable))
    eligible = [
        token
        for token, support in document_frequency.items()
        if support >= min_support and support < generic_cutoff
    ]
    eligible.sort(
        key=lambda token: (
            -document_frequency[token],
            -rank_support[token],
            token,
        )
    )
    return FeedbackPlan(query, locale, tuple(eligible[:max_terms]))


def weighted_rrf(
    dense: Sequence[Candidate],
    lexical: Sequence[Candidate],
    *,
    dense_weight: float = 2.0,
    lexical_weight: float = 1.0,
    rank_constant: int = 10,
    top_k: int = 10,
) -> list[tuple[str, float, Candidate]]:
    """Reproduce the pinned dense-first weighted-RRF ordering for cached arms."""
    weights = (dense_weight, lexical_weight)
    if (
        any(not math.isfinite(weight) or weight < 0 for weight in weights)
        or not any(weight > 0 for weight in weights)
    ):
        raise ValueError("RRF weights must be non-negative and not both zero")
    if rank_constant < 0 or top_k < 1:
        raise ValueError("rank_constant must be non-negative and top_k positive")

    scores: dict[str, float] = {}
    documents: dict[str, Candidate] = {}
    for candidates, weight in ((dense, dense_weight), (lexical, lexical_weight)):
        for rank, candidate in enumerate(candidates, start=1):
            doc_id = str(candidate.get("doc_id", ""))
            if not doc_id:
                continue
            documents.setdefault(doc_id, candidate)
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (
                rank_constant + rank
            )
    # Python's stable sort preserves first-seen dense order for score ties.
    ordered = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]
    return [(doc_id, scores[doc_id], documents[doc_id]) for doc_id in ordered]


def bounded_followup_fusion(
    baseline: Sequence[tuple[str, float, Candidate]],
    followup: Sequence[Candidate],
    *,
    top_k: int = 10,
    max_boost_fraction: float = 0.10,
    rank_constant: int = 10,
) -> list[tuple[str, float, Candidate]]:
    """Add one lexical follow-up list with a hard-capped contribution.

    The baseline remains the primary score (normally the pinned 2:1 dense /
    original-BM25 RRF). A follow-up result can add at most ``max_boost_fraction``
    of the best baseline score, with lower ranks contributing less. The single
    follow-up list is counted once; callers must not issue or fuse per-term
    lists separately.
    """
    if not 0 <= max_boost_fraction <= 0.10:
        raise ValueError("max_boost_fraction must be between zero and 0.10")
    if top_k < 1 or rank_constant < 0:
        raise ValueError("top_k must be positive and rank_constant non-negative")
    if not baseline:
        return []

    scores = {doc_id: score for doc_id, score, _candidate in baseline}
    documents = {
        doc_id: candidate for doc_id, _score, candidate in baseline
    }
    cap = baseline[0][1] * max_boost_fraction
    seen_followup: set[str] = set()
    for rank, candidate in enumerate(followup, start=1):
        doc_id = str(candidate.get("doc_id", ""))
        if not doc_id or doc_id in seen_followup:
            continue
        seen_followup.add(doc_id)
        documents.setdefault(doc_id, candidate)
        contribution = cap * (rank_constant + 1) / (rank_constant + rank)
        scores[doc_id] = scores.get(doc_id, 0.0) + contribution

    ordered = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]
    return [(doc_id, scores[doc_id], documents[doc_id]) for doc_id in ordered]


def offline_feedback_rerank(
    query: str,
    locale: str,
    dense: Sequence[Candidate],
    lexical: Sequence[Candidate],
    *,
    top_k: int = 10,
    depth: int = 20,
    min_support: int = 3,
    max_terms: int = 2,
    max_boost_fraction: float = 0.10,
) -> tuple[FeedbackPlan, list[tuple[str, float, Candidate]]]:
    """Probe a capped term-match boost over the existing candidate union only.

    This is intentionally *not* a simulation of querying Turbopuffer with
    ``followup_query``. A matched candidate receives at most 10% of the
    strongest baseline RRF score, regardless of whether it matches one or both
    expansion terms. Thus correlated terms or the two original retrieval arms
    cannot multiply the feedback vote.
    """
    if not 0 <= max_boost_fraction <= 0.10:
        raise ValueError("max_boost_fraction must be between zero and 0.10")
    plan = plan_feedback(
        query,
        locale,
        dense,
        lexical,
        depth=depth,
        min_support=min_support,
        max_terms=max_terms,
    )
    baseline = weighted_rrf(
        dense, lexical, rank_constant=10, top_k=max(top_k, len(dense) + len(lexical))
    )
    if not plan.terms or not baseline:
        return plan, baseline[:top_k]

    cap = baseline[0][1] * max_boost_fraction
    terms = set(plan.terms)
    reranked: list[tuple[str, float, Candidate]] = []
    for doc_id, score, candidate in baseline:
        document_tokens = {
            token
            for field in _TEXT_FIELDS
            for token in _tokens(candidate.get(field, ""))
        }
        boost = cap if terms.intersection(document_tokens) else 0.0
        reranked.append((doc_id, score + boost, candidate))
    reranked.sort(key=lambda result: result[1], reverse=True)
    return plan, reranked[:top_k]
