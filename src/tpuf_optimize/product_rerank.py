"""Deterministic, bounded reranking for ESCI hybrid retrieval candidates.

The candidate pool is the union of dense and lexical results. Reciprocal-rank
fusion (dense: lexical = 2:1) remains the dominant signal; title and explicit
product-specification evidence only make a small, bounded adjustment.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from typing import Any

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?(?![\w.])")
_UNIT_RE = re.compile(
    r"(?<!\w)(?P<unit>"
    r"ounces?|oz|pounds?|lbs?|kilograms?|kg|grams?|g|milligrams?|mg|"
    r"milliliters?|millilitres?|ml|liters?|litres?|l|"
    r"inches?|in|centimeters?|centimetres?|cm|millimeters?|millimetres?|mm|"
    r"feet|ft|watts?|w|volts?|v|packs?|count|ct|pieces?|pcs?"
    r")(?!\w)",
    re.IGNORECASE,
)
_MODEL_CODE_RE = re.compile(
    r"(?<![\w])(?=[a-z0-9-]*[a-z])(?=[a-z0-9-]*\d)"
    r"[a-z0-9]+(?:-[a-z0-9]+)*(?![\w])",
    re.IGNORECASE,
)
_UNIT_ALIASES = {
    "ounce": "oz",
    "ounces": "oz",
    "pound": "lb",
    "pounds": "lb",
    "lbs": "lb",
    "kilogram": "kg",
    "kilograms": "kg",
    "gram": "g",
    "grams": "g",
    "milligram": "mg",
    "milligrams": "mg",
    "milliliter": "ml",
    "milliliters": "ml",
    "millilitre": "ml",
    "millilitres": "ml",
    "liter": "l",
    "liters": "l",
    "litre": "l",
    "litres": "l",
    "inch": "in",
    "inches": "in",
    "centimeter": "cm",
    "centimeters": "cm",
    "centimetre": "cm",
    "centimetres": "cm",
    "millimeter": "mm",
    "millimeters": "mm",
    "millimetre": "mm",
    "millimetres": "mm",
    "watt": "w",
    "watts": "w",
    "volt": "v",
    "volts": "v",
    "packs": "pack",
    "count": "ct",
    "pieces": "pc",
    "pcs": "pc",
}


def _normalize(text: object) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).casefold()


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(_normalize(text))


def _is_cjk(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3040 <= codepoint <= 0x30FF
        or 0x3400 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
    )


def _compact_cjk(text: str) -> str:
    """Return the normalized CJK sequence, excluding whitespace/punctuation."""
    return "".join(
        character for character in _normalize(text) if _is_cjk(character)
    )


def _numeric_specs(text: str) -> list[tuple[str, str | None]]:
    normalized = _normalize(text).replace(",", ".")
    units = list(_UNIT_RE.finditer(normalized))
    specs: list[tuple[str, str | None]] = []
    for match in _NUMBER_RE.finditer(normalized):
        following = next((unit for unit in units if unit.start() >= match.end()), None)
        unit_name: str | None = None
        if following is not None and following.start() - match.end() <= 2:
            raw_unit = following.group("unit").casefold()
            unit_name = _UNIT_ALIASES.get(raw_unit, raw_unit)
        number = match.group().replace(",", ".")
        try:
            number = f"{float(number):g}"
        except ValueError:
            pass
        specs.append((number, unit_name))
    return specs


def _model_codes(text: str) -> set[str]:
    return {match.group().casefold() for match in _MODEL_CODE_RE.finditer(_normalize(text))}


def _rrf(
    dense: Sequence[Mapping[str, Any]],
    lexical: Sequence[Mapping[str, Any]],
    *,
    lexical_weight: float,
    dense_weight: float,
    rank_constant: int,
) -> tuple[dict[str, float], dict[str, dict[str, Any]]]:
    scores: dict[str, float] = {}
    documents: dict[str, dict[str, Any]] = {}
    for rows, weight in ((lexical, lexical_weight), (dense, dense_weight)):
        seen: set[str] = set()
        for rank, row in enumerate(rows, start=1):
            doc_id = str(row.get("doc_id", row.get("id", "")))
            if not doc_id or doc_id in seen:
                continue
            seen.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (
                rank_constant + rank
            )
            current = documents.setdefault(doc_id, {"doc_id": doc_id})
            # Prefer lexical metadata, then fill missing/blank values from dense.
            for key, value in row.items():
                if value is None or value == "":
                    continue
                if key not in current or current[key] in ("", None):
                    current[key] = value
                elif rows is lexical:
                    current[key] = value
    return scores, documents


def _title_evidence(query: str, title: str, locale: str) -> float:
    query_tokens = _tokens(query)
    title_tokens = _tokens(title)
    if not query_tokens:
        return 0.0
    title_set = set(title_tokens)
    coverage = sum(token in title_set for token in set(query_tokens)) / len(
        set(query_tokens)
    )

    # Ordered coverage measures whether the query terms appear in title order.
    title_positions: dict[str, list[int]] = {}
    for position, token in enumerate(title_tokens):
        title_positions.setdefault(token, []).append(position)
    previous = -1
    ordered = 0
    for token in query_tokens:
        next_position = next(
            (position for position in title_positions.get(token, ()) if position > previous),
            None,
        )
        if next_position is not None:
            ordered += 1
            previous = next_position
    ordered_fraction = ordered / len(query_tokens)

    if len(query_tokens) > 1:
        adjacent = 0.0
        for left, right in zip(query_tokens, query_tokens[1:]):
            positions = title_positions.get(left, ())
            right_positions = title_positions.get(right, ())
            if any(
                0 < right_position - left_position <= 3
                for left_position in positions
                for right_position in right_positions
            ):
                adjacent += 1
        proximity = adjacent / (len(query_tokens) - 1)
    else:
        proximity = coverage

    # Japanese has no whitespace-delimited words. Give multi-character phrase
    # containment a small, separate signal; never reward a single common glyph.
    cjk_score = 0.0
    if locale.casefold() in {"jp", "ja", "ja-jp"}:
        query_cjk = _compact_cjk(query)
        title_cjk = _compact_cjk(title)
        if len(query_cjk) >= 2 and title_cjk:
            cjk_coverage = sum(character in title_cjk for character in query_cjk) / len(
                query_cjk
            )
            phrase_match = query_cjk in title_cjk
            cjk_score = 0.65 * cjk_coverage + (0.35 if phrase_match else 0.0)

    latin_score = 0.55 * coverage + 0.25 * ordered_fraction + 0.20 * proximity
    if cjk_score:
        return max(latin_score, 0.7 * latin_score + 0.3 * cjk_score)
    return latin_score


def _attribute_evidence(
    query: str,
    candidate: Mapping[str, Any],
    all_documents: Sequence[Mapping[str, Any]],
) -> float:
    query_tokens = set(_tokens(query))
    evidence = 0.0
    for field, weight in (("brand", 0.045), ("color", 0.035)):
        values = [
            set(_tokens(document.get(field, "")))
            for document in all_documents
            if document.get(field)
        ]
        explicit_terms = set().union(*values) & query_tokens if values else set()
        if not explicit_terms:
            continue
        candidate_value = set(_tokens(candidate.get(field, "")))
        if not candidate_value:
            # Unknown attributes are neutral; missing fields must not demote.
            continue
        if candidate_value & explicit_terms:
            evidence += weight
        elif candidate_value & set().union(*values):
            evidence -= weight * 0.35
    return evidence


def _specification_evidence(query: str, title: str) -> float:
    query_specs = _numeric_specs(query)
    title_specs = _numeric_specs(title)
    query_codes = _model_codes(query)
    title_codes = _model_codes(title)
    evidence = 0.0

    if query_specs:
        matched = sum(spec in title_specs for spec in query_specs)
        if matched:
            evidence += 0.20 * matched / len(query_specs)
        elif title_specs:
            # Only contradict an explicit title specification. No penalty when
            # the title simply omits a requested quantity or size.
            evidence -= 0.30
    if query_codes:
        matched_codes = query_codes & title_codes
        if matched_codes:
            evidence += 0.65 * len(matched_codes) / len(query_codes)
        elif title_codes:
            evidence -= 0.70
    return evidence


def rerank_candidates(
    query_text: str,
    locale: str,
    dense: Sequence[Mapping[str, Any]],
    lexical: Sequence[Mapping[str, Any]],
    *,
    top_k: int | None = None,
    lexical_weight: float = 1.0,
    dense_weight: float = 2.0,
    rank_constant: int = 60,
    max_adjustment: float = 0.012,
) -> list[dict[str, Any]]:
    """Fuse a dense/lexical union and apply bounded product-title evidence.

    The input sequences are ordered by their retrieval ranks. Each returned
    Dense and lexical ranks are fused with weights 2:1 respectively by default.
    Each result contains merged attributes, deterministic fused ``score``,
    and its ``rrf_score`` and ``rerank_adjustment`` for diagnostics. Set
    ``max_adjustment=0`` to return the unmodified weighted-RRF ordering.
    """
    if not query_text.strip():
        raise ValueError("query_text must not be empty")
    if rank_constant < 1:
        raise ValueError("rank_constant must be positive")
    if lexical_weight < 0 or dense_weight < 0 or lexical_weight + dense_weight == 0:
        raise ValueError("fusion weights must be nonnegative and not both zero")
    if max_adjustment < 0 or not math.isfinite(max_adjustment):
        raise ValueError("max_adjustment must be finite and nonnegative")

    base_scores, documents = _rrf(
        dense,
        lexical,
        lexical_weight=lexical_weight,
        dense_weight=dense_weight,
        rank_constant=rank_constant,
    )
    all_documents = list(documents.values())
    results: list[dict[str, Any]] = []
    for doc_id, base_score in base_scores.items():
        document = documents[doc_id]
        title = str(document.get("title", ""))
        title_signal = _title_evidence(query_text, title, locale)
        spec_signal = _specification_evidence(query_text, title)
        attribute_signal = _attribute_evidence(query_text, document, all_documents)
        # The title signal helps separate exact/near-exact matches; explicit
        # numeric/code agreement is more diagnostic, and brand/color stay soft.
        raw_adjustment = (
            0.45 * title_signal
            + 0.35 * spec_signal
            + attribute_signal
        )
        adjustment = max(
            -max_adjustment,
            min(max_adjustment, max_adjustment * raw_adjustment),
        )
        result = dict(document)
        result["rrf_score"] = base_score
        result["rerank_adjustment"] = adjustment
        result["score"] = base_score + adjustment
        results.append(result)

    results.sort(key=lambda result: (-result["score"], str(result["doc_id"])))
    return results if top_k is None else results[: max(0, top_k)]
