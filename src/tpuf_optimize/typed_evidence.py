"""Bounded reranking from typed product evidence and hybrid-channel agreement.

The reranker consumes the existing dense and lexical candidate lists; it does
not require a new index. It intentionally reads only title and explicit
brand/color metadata, rather than treating a passage that contains the title as
independent evidence.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?(?![\w.])")
_UNIT_RE = re.compile(
    r"(?<!\w)(ounces?|oz|pounds?|lbs?|kilograms?|kg|grams?|g|milligrams?|mg|"
    r"milliliters?|millilitres?|ml|liters?|litres?|l|inches?|in|"
    r"centimeters?|centimetres?|cm|millimeters?|millimetres?|mm|"
    r"feet|ft|watts?|w|volts?|v|packs?|count|ct|pieces?|pcs?)(?!\w)",
    re.IGNORECASE,
)
_MODEL_RE = re.compile(
    r"(?<!\w)(?=[a-z0-9-]*[a-z])(?=[a-z0-9-]*\d)"
    r"[a-z0-9]+(?:-[a-z0-9]+)*(?!\w)",
    re.IGNORECASE,
)
_UNIT_ALIASES = {
    "ounce": "oz", "ounces": "oz", "pound": "lb", "pounds": "lb", "lbs": "lb",
    "kilogram": "kg", "kilograms": "kg", "gram": "g", "grams": "g",
    "milligram": "mg", "milligrams": "mg", "milliliter": "ml",
    "milliliters": "ml", "millilitre": "ml", "millilitres": "ml",
    "liter": "l", "liters": "l", "litre": "l", "litres": "l",
    "inch": "in", "inches": "in", "centimeter": "cm", "centimeters": "cm",
    "centimetre": "cm", "centimetres": "cm", "millimeter": "mm",
    "millimeters": "mm", "millimetre": "mm", "millimetres": "mm",
    "watt": "w", "watts": "w", "volt": "v", "volts": "v",
    "packs": "pack", "count": "ct", "pieces": "pc", "pcs": "pc",
}
_STOPWORDS = frozenset(
    "a an and for in of the to with by from on at set pack size inc llc ltd "
    "corp company incorporated".split()
)
_COLOR_ALIASES = {
    "grey": "gray", "navy": "blue", "teal": "blue", "aqua": "blue",
    "burgundy": "red", "maroon": "red", "ivory": "white", "cream": "white",
    "tan": "brown", "beige": "brown", "multicolored": "multi",
    "multicolor": "multi", "multiple": "multi",
}
_COLOR_TERMS = frozenset(
    "black white gray silver red orange yellow green blue purple pink brown "
    "gold clear transparent multi".split()
)


class EvidenceState(IntEnum):
    """Three-valued evidence state for an explicitly requested product slot."""

    CONTRADICTION = -1
    UNKNOWN = 0
    SUPPORT = 1


@dataclass(frozen=True)
class TypedEvidenceConfig:
    """Weights for the bounded non-RRF contribution."""

    brand: float = 0.28
    color: float = 0.32
    model: float = 0.44
    number: float = 0.26
    unit: float = 0.26
    typed_support_interaction: float = 0.16
    typed_conflict_interaction: float = -0.08
    title_coverage: float = 0.20
    title_proximity: float = 0.14
    channel_agreement: float = 0.14
    dense_gap: float = 0.05
    lexical_gap: float = 0.05
    gap_agreement: float = 0.07
    max_adjustment: float = 0.02


@dataclass(frozen=True)
class EvidenceFeatures:
    """Interpretable per-candidate features; slot states are -1, 0, or +1."""

    brand: EvidenceState
    color: EvidenceState
    model: EvidenceState
    number: EvidenceState
    unit: EvidenceState
    typed_support_interaction: float
    typed_conflict_interaction: float
    title_coverage: float
    title_proximity: float
    channel_agreement: float
    dense_gap: float
    lexical_gap: float
    gap_agreement: float


def _normalize(text: object) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).casefold()


def _tokens(text: object) -> list[str]:
    return _TOKEN_RE.findall(_normalize(text))


def _numeric_tokens(text: str) -> set[str]:
    result: set[str] = set()
    for match in _NUMBER_RE.finditer(_normalize(text).replace(",", ".")):
        try:
            result.add(f"{float(match.group()):g}")
        except ValueError:
            result.add(match.group())
    return result


def _unit_tokens(text: str) -> set[str]:
    return {
        _UNIT_ALIASES.get(match.group().casefold(), match.group().casefold())
        for match in _UNIT_RE.finditer(_normalize(text))
    }


def _model_tokens(text: str) -> set[str]:
    return {match.group().casefold() for match in _MODEL_RE.finditer(_normalize(text))}


def _state(query_values: set[str], document_values: set[str]) -> EvidenceState:
    if not query_values:
        return EvidenceState.UNKNOWN
    if document_values & query_values:
        return EvidenceState.SUPPORT
    if document_values:
        return EvidenceState.CONTRADICTION
    return EvidenceState.UNKNOWN


def _brand_query_values(
    query_tokens: set[str], candidates: Sequence[Mapping[str, Any]]
) -> set[str]:
    # Infer a brand request only from brands actually present in the retrieved
    # metadata and non-generic query terms. This avoids a brand lexicon and
    # makes absent brand metadata unknown rather than a negative.
    values: set[str] = set()
    for candidate in candidates:
        brand_tokens = {
            token for token in _tokens(candidate.get("brand", ""))
            if token not in _STOPWORDS
        }
        overlap = brand_tokens & query_tokens
        if brand_tokens and (
            brand_tokens <= query_tokens
            or len(overlap) >= 2
            or any(len(token) >= 6 for token in overlap)
        ):
            values.update(overlap)
    return values


def _color_query_values(query_tokens: set[str]) -> set[str]:
    return {
        _COLOR_ALIASES.get(token, token)
        for token in query_tokens
        if token in _COLOR_TERMS or token in _COLOR_ALIASES
    }


def _color_values(value: object) -> set[str]:
    return {
        _COLOR_ALIASES.get(token, token)
        for token in _tokens(value)
        if token in _COLOR_TERMS or token in _COLOR_ALIASES
    }


def _ordered_proximity(query_tokens: Sequence[str], title_tokens: Sequence[str]) -> float:
    distinct = list(dict.fromkeys(query_tokens))
    if len(distinct) < 2:
        return 0.0
    positions: dict[str, list[int]] = {}
    for index, token in enumerate(title_tokens):
        positions.setdefault(token, []).append(index)
    # Greedily align distinct query terms in query order, then normalize the
    # span. This rewards local phrase evidence without counting repeated terms.
    best = 0.0
    for start in positions.get(distinct[0], []):
        previous = start
        aligned = 1
        for token in distinct[1:]:
            next_position = next(
                (position for position in positions.get(token, ()) if position > previous),
                None,
            )
            if next_position is None:
                break
            aligned += 1
            previous = next_position
        fraction = aligned / len(distinct)
        span_quality = 1.0 / max(1, previous - start)
        best = max(best, fraction * (0.5 + 0.5 * span_quality))
    return min(1.0, best)


def _normalized_gap(
    rows: Sequence[Mapping[str, Any]], score_key: str, *, lower_is_better: bool
) -> dict[str, float]:
    scores: list[tuple[str, float]] = []
    for row in rows:
        raw_id = row.get("doc_id", row.get("id"))
        score = row.get(score_key, row.get("score", row.get("$dist")))
        if (
            raw_id is not None
            and isinstance(score, (int, float))
            and math.isfinite(float(score))
        ):
            scores.append((str(raw_id), float(score)))
    if not scores:
        return {}
    values = [value for _, value in scores]
    low, high = min(values), max(values)
    span = high - low
    if span <= 1e-12:
        return {doc_id: 1.0 for doc_id, _ in scores}
    return {
        doc_id: 1.0 - ((value - low) / span if lower_is_better else (high - value) / span)
        for doc_id, value in scores
    }


def _rrf(
    dense: Sequence[Mapping[str, Any]],
    lexical: Sequence[Mapping[str, Any]],
    dense_weight: float,
    lexical_weight: float,
    rank_constant: int,
) -> tuple[dict[str, float], dict[str, dict[str, Any]], dict[str, int], dict[str, int]]:
    scores: dict[str, float] = {}
    documents: dict[str, dict[str, Any]] = {}
    dense_ranks: dict[str, int] = {}
    lexical_ranks: dict[str, int] = {}
    # Dense-first insertion and stable sorting exactly match cc5fc8a's RRF
    # insertion and tie order.
    for rows, weight, rank_map in (
        (dense, dense_weight, dense_ranks),
        (lexical, lexical_weight, lexical_ranks),
    ):
        for rank, row in enumerate(rows, 1):
            doc_id = str(row.get("doc_id", row.get("id", "")))
            if not doc_id:
                continue
            rank_map[doc_id] = rank
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (rank_constant + rank)
            current = documents.setdefault(doc_id, {"doc_id": doc_id})
            for key, value in row.items():
                if value is None or value == "":
                    continue
                if key not in current or current[key] in (None, "") or rows is lexical:
                    current[key] = value
    return scores, documents, dense_ranks, lexical_ranks


def candidate_features(
    query_text: str,
    candidate: Mapping[str, Any],
    all_candidates: Sequence[Mapping[str, Any]],
    dense_ranks: Mapping[str, int],
    lexical_ranks: Mapping[str, int],
    dense_gaps: Mapping[str, float],
    lexical_gaps: Mapping[str, float],
) -> EvidenceFeatures:
    """Compute typed slot, title, and channel evidence for one candidate."""
    query_tokens_ordered = _tokens(query_text)
    query_tokens = set(query_tokens_ordered)
    title_tokens = _tokens(candidate.get("title", ""))
    title_set = set(title_tokens)

    query_brand = _brand_query_values(query_tokens, all_candidates)
    known_brands = {
        token
        for document in all_candidates
        for token in _tokens(document.get("brand", ""))
        if token not in _STOPWORDS
    }
    candidate_brand = {
        token for token in _tokens(candidate.get("brand", ""))
        if token not in _STOPWORDS
    }
    brand_values = candidate_brand & known_brands if candidate_brand else set()

    query_numbers = _numeric_tokens(query_text)
    query_units = _unit_tokens(query_text)
    title_numbers = _numeric_tokens(str(candidate.get("title", "")))
    title_units = _unit_tokens(str(candidate.get("title", "")))
    query_models = _model_tokens(query_text)
    title_models = _model_tokens(str(candidate.get("title", "")))
    slot_states = (
        _state(query_brand, brand_values),
        _state(_color_query_values(query_tokens), _color_values(candidate.get("color", ""))),
        _state(query_models, title_models),
        _state(query_numbers, title_numbers),
        _state(query_units, title_units),
    )
    observed_slots = [state for state in slot_states if state is not EvidenceState.UNKNOWN]
    pair_count = len(observed_slots) * (len(observed_slots) - 1) / 2
    support_pairs = sum(
        left is EvidenceState.SUPPORT and right is EvidenceState.SUPPORT
        for index, left in enumerate(observed_slots)
        for right in observed_slots[index + 1:]
    )
    conflict_pairs = sum(
        {left, right} == {EvidenceState.SUPPORT, EvidenceState.CONTRADICTION}
        for index, left in enumerate(observed_slots)
        for right in observed_slots[index + 1:]
    )

    doc_id = str(candidate.get("doc_id", candidate.get("id", "")))
    dense_rank = dense_ranks.get(doc_id)
    lexical_rank = lexical_ranks.get(doc_id)
    if dense_rank is not None and lexical_rank is not None:
        rank_gap = abs(dense_rank - lexical_rank) / max(dense_rank, lexical_rank, 1)
        channel_agreement = 1.0 - rank_gap
        gap_agreement = 1.0 - abs(
            dense_gaps.get(doc_id, 0.0) - lexical_gaps.get(doc_id, 0.0)
        )
    else:
        channel_agreement = 0.0
        gap_agreement = 0.0

    return EvidenceFeatures(
        brand=slot_states[0],
        color=slot_states[1],
        model=slot_states[2],
        number=slot_states[3],
        unit=slot_states[4],
        typed_support_interaction=(
            support_pairs / pair_count if pair_count else 0.0
        ),
        typed_conflict_interaction=(
            conflict_pairs / pair_count if pair_count else 0.0
        ),
        title_coverage=len(set(query_tokens_ordered) & title_set)
        / max(1, len(set(query_tokens_ordered))),
        title_proximity=_ordered_proximity(query_tokens_ordered, title_tokens),
        channel_agreement=channel_agreement,
        dense_gap=dense_gaps.get(doc_id, 0.0),
        lexical_gap=lexical_gaps.get(doc_id, 0.0),
        gap_agreement=gap_agreement,
    )


def _evidence_score(features: EvidenceFeatures, config: TypedEvidenceConfig) -> float:
    return (
        config.brand * int(features.brand)
        + config.color * int(features.color)
        + config.model * int(features.model)
        + config.number * int(features.number)
        + config.unit * int(features.unit)
        + config.typed_support_interaction * features.typed_support_interaction
        + config.typed_conflict_interaction * features.typed_conflict_interaction
        + config.title_coverage * features.title_coverage
        + config.title_proximity * features.title_proximity
        + config.channel_agreement * features.channel_agreement
        + config.dense_gap * features.dense_gap
        + config.lexical_gap * features.lexical_gap
        + config.gap_agreement * features.gap_agreement
    )


def rerank_candidates(
    query_text: str,
    dense: Sequence[Mapping[str, Any]],
    lexical: Sequence[Mapping[str, Any]],
    *,
    top_k: int = 10,
    dense_weight: float = 2.0,
    lexical_weight: float = 1.0,
    rank_constant: int = 10,
    config: TypedEvidenceConfig = TypedEvidenceConfig(),
) -> list[dict[str, Any]]:
    """Fuse 100-per-arm candidates, apply capped typed evidence, and rank.

    Lower dense scores are better (distance); higher lexical scores are better
    (BM25). Metadata is merged from the candidate rows and all ranking is
    deterministic, preserving dense-first ordering on exact ties.
    """
    if not query_text.strip():
        raise ValueError("query_text must not be empty")
    if top_k < 1 or rank_constant < 0:
        raise ValueError("top_k must be positive and rank_constant nonnegative")
    if (
        dense_weight < 0 or lexical_weight < 0
        or not math.isfinite(dense_weight) or not math.isfinite(lexical_weight)
        or dense_weight + lexical_weight == 0
    ):
        raise ValueError("fusion weights must be finite, nonnegative, and nonzero")
    if config.max_adjustment < 0 or not math.isfinite(config.max_adjustment):
        raise ValueError("max_adjustment must be finite and nonnegative")

    base_scores, documents, dense_ranks, lexical_ranks = _rrf(
        dense, lexical, dense_weight, lexical_weight, rank_constant
    )
    dense_gaps = _normalized_gap(dense, "score", lower_is_better=True)
    lexical_gaps = _normalized_gap(lexical, "score", lower_is_better=False)
    all_candidates = list(documents.values())
    results: list[dict[str, Any]] = []
    for doc_id, base_score in base_scores.items():
        document = documents[doc_id]
        features = candidate_features(
            query_text, document, all_candidates, dense_ranks, lexical_ranks,
            dense_gaps, lexical_gaps,
        )
        raw = _evidence_score(features, config)
        adjustment = config.max_adjustment * math.tanh(raw)
        result = dict(document)
        result.update(
            rrf_score=base_score,
            evidence_score=raw,
            rerank_adjustment=adjustment,
            score=base_score + adjustment,
        )
        results.append(result)
    results.sort(key=lambda item: -float(item["score"]))
    return results[:top_k]
