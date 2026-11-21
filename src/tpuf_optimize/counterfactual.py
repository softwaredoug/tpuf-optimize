"""Conservative, candidate-matched leave-one-token-out search diagnostics."""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

_TOKEN = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", re.UNICODE)
_STRUCTURED_FIELDS = ("brand", "color", "category")
_ALL_TEXT_FIELDS = (
    "title",
    "description",
    "brand",
    "color",
    "category",
)
_RRF_CONSTANT = 60
_PROBE_WEIGHT = 0.25


@dataclass(frozen=True)
class TokenProbe:
    """One selected token and its query with that token removed."""

    token: str
    ablated_query: str
    reason: str


def _get(row: object, key: str) -> object | None:
    if isinstance(row, Mapping):
        return row.get(key)
    extra = getattr(row, "model_extra", None)
    if isinstance(extra, Mapping) and key in extra:
        return extra[key]
    try:
        return row[key]  # type: ignore[index]
    except (KeyError, TypeError, AttributeError):
        return getattr(row, key, None)


def _tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return [match.group(0) for match in _TOKEN.finditer(normalized)]


def _row_text(row: object, fields: Sequence[str] = _ALL_TEXT_FIELDS) -> str:
    return " ".join(
        str(value)
        for field in fields
        if (value := _get(row, field)) not in (None, "")
    )


def select_probe_token(query: str, candidates: Sequence[object]) -> TokenProbe | None:
    """Choose at most one plausible identifier or partially-covered attribute.

    Selection is deliberately conservative: a query needs at least two terms,
    and an attribute probe must occur in some, but not all, candidate results.
    Numeric or mixed alphanumeric identifiers are eligible without that
    coverage check. In either case, the probe only removes one query token.
    """
    matches = list(_TOKEN.finditer(query))
    if len(matches) < 2 or len(candidates) < 2:
        return None

    normalized_candidates = [
        set(_tokens(_row_text(candidate))) for candidate in candidates
    ]
    structured_candidates = [
        set(_tokens(_row_text(candidate, _STRUCTURED_FIELDS)))
        for candidate in candidates
    ]

    eligible: list[tuple[int, int, TokenProbe]] = []
    for index, match in enumerate(matches):
        token = unicodedata.normalize("NFKC", match.group(0)).casefold()
        if len(token) < 2:
            continue
        remainder = (query[: match.start()] + " " + query[match.end():]).strip()
        if not remainder:
            continue

        identifier = (
            any(character.isdigit() for character in token)
            and any(character.isalpha() for character in token)
        ) or token.isdigit()
        coverage = sum(token in candidate for candidate in normalized_candidates)
        if identifier:
            reason = "identifier"
            priority = 0
        else:
            structured_coverage = sum(
                token in candidate for candidate in structured_candidates
            )
            if not 0 < structured_coverage < len(candidates):
                if not 0 < coverage < len(candidates):
                    continue
                reason = "incomplete_candidate_coverage"
            else:
                reason = "attribute_coverage"
            # Prefer the more distinctive, less universally represented term.
            priority = 1

        eligible.append(
            (
                priority,
                index,
                TokenProbe(token=match.group(0), ablated_query=remainder, reason=reason),
            )
        )

    if not eligible:
        return None
    return min(eligible, key=lambda choice: (choice[0], choice[1]))[2]


def rerank_with_probe(
    original_rows: Sequence[object],
    probe_rows: Sequence[object],
    token: str,
) -> list[object]:
    """Apply a small rank-only boost for candidates helped by token ablation.

    The probe must contain computed leave-one-out scores for the exact original
    candidate IDs. Original results remain the candidate universe, and only
    candidates that lack the token and gain rank can receive a soft boost.
    """
    if not original_rows or not probe_rows:
        return list(original_rows)

    normalized_token = _tokens(token)
    if not normalized_token:
        return list(original_rows)
    target = normalized_token[0]

    original_rank = {
        str(_get(row, "id")): rank
        for rank, row in enumerate(original_rows, start=1)
    }
    original_lexical_scores: dict[str, float] = {}
    probe_scores: dict[str, float] = {}
    for row in probe_rows:
        doc_id = str(_get(row, "id"))
        original_value = _get(row, "original_lexical_score")
        probe_value = _get(row, "counterfactual_score")
        if isinstance(original_value, (int, float)) and math.isfinite(
            float(original_value)
        ):
            original_lexical_scores[doc_id] = float(original_value)
        if isinstance(probe_value, (int, float)) and math.isfinite(
            float(probe_value)
        ):
            probe_scores[doc_id] = float(probe_value)

    ordered_ids = [
        doc_id
        for doc_id in original_rank
        if doc_id in probe_scores and doc_id in original_lexical_scores
    ]
    ordered_original_ids = sorted(
        ordered_ids,
        key=lambda doc_id: (
            -original_lexical_scores[doc_id],
            original_rank[doc_id],
        ),
    )
    original_lexical_rank = {
        doc_id: rank for rank, doc_id in enumerate(ordered_original_ids, start=1)
    }
    ordered_probe_ids = sorted(
        ordered_ids,
        key=lambda doc_id: (-probe_scores[doc_id], original_rank[doc_id]),
    )
    probe_rank = {
        doc_id: rank for rank, doc_id in enumerate(ordered_probe_ids, start=1)
    }

    def treatment_score(row: object) -> float:
        doc_id = str(_get(row, "id"))
        rank = original_rank[doc_id]
        score = 1 / (_RRF_CONSTANT + rank)
        has_token = target in set(_tokens(_row_text(row)))
        counterfactual_rank = probe_rank.get(doc_id)
        lexical_rank = original_lexical_rank.get(doc_id)
        if (
            not has_token
            and counterfactual_rank is not None
            and lexical_rank is not None
            and counterfactual_rank < lexical_rank
        ):
            score += _PROBE_WEIGHT / (_RRF_CONSTANT + counterfactual_rank)
        return score

    return sorted(original_rows, key=treatment_score, reverse=True)
