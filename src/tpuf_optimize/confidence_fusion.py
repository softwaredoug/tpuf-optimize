"""Query-local confidence gates for dense/lexical reciprocal-rank fusion."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from math import isfinite
from statistics import median
from typing import Literal

Candidate = Mapping[str, object]
Gate = Literal["margin", "identifier"]
Weights = tuple[float, float]

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_IDENTIFIER_RE = re.compile(r"(?=.*[a-z])(?=.*\d)[a-z0-9_-]{5,}", re.IGNORECASE)
_TEXT_FIELDS = ("title", "brand", "color", "category")


def _value(candidate: Candidate, *keys: str) -> str:
    for key in keys:
        value = candidate.get(key)
        if value is not None:
            return str(value)
    return ""


def _numeric(candidate: Candidate, *keys: str) -> float | None:
    for key in keys:
        value = candidate.get(key)
        if isinstance(value, (int, float)) and isfinite(value):
            return float(value)
    return None


def _normalized_tokens(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_RE.findall(text)]


def _score_gap(results: Sequence[Candidate]) -> float:
    """Return a scale-free top-versus-tail score separation in [0, 1]."""
    scores: list[float] = []
    for result in results:
        score = _numeric(result, "score", "$dist")
        if score is not None:
            scores.append(score)
    if len(scores) < 3:
        return 0.0

    # Cached lists are ordered best-first. Compare their top score with the
    # median score in the bottom quartile, relative to the score magnitude.
    tail_size = max(1, len(scores) // 4)
    top_score = scores[0]
    tail_midpoint = median(scores[-tail_size:])
    scale = max(abs(top_score), abs(tail_midpoint))
    if scale <= 0:
        return 0.0
    return min(1.0, max(0.0, (top_score - tail_midpoint) / scale))


def _top_overlap(
    dense: Sequence[Candidate], lexical: Sequence[Candidate], depth: int = 10
) -> float:
    """Jaccard overlap of the top results, independent of score scales."""
    dense_ids = {
        _value(item, "doc_id", "id") for item in dense[:depth]
    } - {""}
    lexical_ids = {
        _value(item, "doc_id", "id") for item in lexical[:depth]
    } - {""}
    union = dense_ids | lexical_ids
    if not union:
        return 0.0
    return len(dense_ids & lexical_ids) / len(union)


def _direct_attribute_match(query: str, candidate: Candidate) -> bool:
    """Check if all query terms occur in concise product identity fields."""
    query_tokens = _normalized_tokens(query)
    if not query_tokens:
        return False
    attributes = " ".join(_value(candidate, field) for field in _TEXT_FIELDS)
    attribute_tokens = set(_normalized_tokens(attributes))
    return all(token in attribute_tokens for token in query_tokens)


def _identifier_lead(
    query: str, dense: Sequence[Candidate], lexical: Sequence[Candidate]
) -> bool:
    """Detect an exact query-to-product hit unique to the lexical top results."""
    query_tokens = _normalized_tokens(query)
    identifiers = [
        token for token in query_tokens if _IDENTIFIER_RE.fullmatch(token)
    ]
    if not lexical:
        return False

    lexical_top = lexical[0]
    lexical_id = _value(lexical_top, "doc_id", "id").casefold()
    lexical_text = " ".join(
        (lexical_id, *(_value(lexical_top, field) for field in _TEXT_FIELDS))
    ).casefold()
    lexical_hit = (
        any(identifier in lexical_text for identifier in identifiers)
        or _direct_attribute_match(query, lexical_top)
    )
    if not lexical_hit:
        return False

    # A matching dense top result is corroboration, not a lexical-only lead.
    for candidate in dense[:3]:
        candidate_id = _value(candidate, "doc_id", "id")
        candidate_text = " ".join(
            (candidate_id, *(_value(candidate, field) for field in _TEXT_FIELDS))
        ).casefold()
        if candidate_id == lexical_id or (
            any(identifier in candidate_text for identifier in identifiers)
            or _direct_attribute_match(query, candidate)
        ):
            return False
    return True


def confidence_aware_weights(
    query: str,
    dense: Sequence[Candidate],
    lexical: Sequence[Candidate],
    *,
    gate: Gate = "margin",
    baseline: Weights = (2.0, 1.0),
    overlap_threshold: float = 0.3,
    confidence_margin: float = 0.2,
) -> Weights:
    """Select a conservative per-query dense/lexical RRF weighting.

    ``margin`` moves from the fixed 2:1 baseline only when one list has a
    clearly larger top-to-tail score gap and the lists have little overlap.
    ``identifier`` uses a lexical-heavy 1:2 weight only when the lexical top
    resolves an exact identifier or direct attribute match absent from the
    dense top three.
    """
    if gate not in ("margin", "identifier"):
        raise ValueError(f"unknown confidence gate: {gate}")
    if len(baseline) != 2 or any(
        not isfinite(weight) or weight < 0 for weight in baseline
    ):
        raise ValueError("baseline must contain two non-negative weights")
    if sum(baseline) <= 0:
        raise ValueError("baseline must have a positive weight")
    if not 0 <= overlap_threshold <= 1:
        raise ValueError("overlap_threshold must be between zero and one")
    if not 0 <= confidence_margin <= 1:
        raise ValueError("confidence_margin must be between zero and one")

    if gate == "identifier":
        return (1.0, 2.0) if _identifier_lead(query, dense, lexical) else baseline

    if _top_overlap(dense, lexical) >= overlap_threshold:
        return baseline
    dense_gap = _score_gap(dense)
    lexical_gap = _score_gap(lexical)
    if dense_gap - lexical_gap >= confidence_margin:
        return (3.0, 1.0)
    if lexical_gap - dense_gap >= confidence_margin:
        return (1.0, 2.0)
    return baseline
