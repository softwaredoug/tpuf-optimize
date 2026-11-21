"""Offline, query-grouped pairwise reranking experiments.

The module consumes cached retrieval candidates; it deliberately has no
Turbopuffer or Referee client. Query and document identifiers are used only to
join examples and make deterministic output, never as model features.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

RRF_K = 10
FEATURE_NAMES = (
    "dense_present",
    "lexical_present",
    "dense_rank_rrf",
    "lexical_rank_rrf",
    "rrf_2_1",
    "dense_score_z",
    "lexical_score_z",
    "title_query_coverage",
    "title_token_precision",
    "title_exact_phrase",
    "title_length_log",
)
COMPUTED_FEATURE_NAMES = (
    "title_bm25_z",
    "title_bm25_available",
    "passage_bm25_z",
    "passage_bm25_available",
    "brand_bm25_z",
    "brand_bm25_available",
    "title_full_coverage",
    "title_full_coverage_available",
)
PASSAGE_ARM_FEATURE_NAMES = (
    "passage_arm_present",
    "passage_arm_rank_rrf",
    "passage_arm_score_z",
)
ENHANCED_FEATURE_NAMES = FEATURE_NAMES + COMPUTED_FEATURE_NAMES
TYPED_FEATURE_NAMES = (
    "typed_product_match",
    "typed_support",
    "typed_contradiction",
    "typed_unknown",
)
TYPED_ENHANCED_FEATURE_NAMES = ENHANCED_FEATURE_NAMES + TYPED_FEATURE_NAMES
TYPED_NO_UNKNOWN_FEATURE_NAMES = TYPED_FEATURE_NAMES[:-1]
TYPED_ENHANCED_NO_UNKNOWN_FEATURE_NAMES = (
    ENHANCED_FEATURE_NAMES + TYPED_NO_UNKNOWN_FEATURE_NAMES
)
TYPED_COMPUTED_NO_PASSAGE_FEATURE_NAMES = (
    FEATURE_NAMES
    + tuple(
        name
        for name in COMPUTED_FEATURE_NAMES
        if not name.startswith("passage_bm25_")
    )
    + TYPED_NO_UNKNOWN_FEATURE_NAMES
)
PASSAGE_TYPED_COMPUTED_FEATURE_NAMES = (
    TYPED_COMPUTED_NO_PASSAGE_FEATURE_NAMES + PASSAGE_ARM_FEATURE_NAMES
)
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_MODEL_RE = re.compile(
    r"(?<!\w)(?=[a-z0-9-]*[a-z])(?=[a-z0-9-]*\d)"
    r"[a-z0-9]+(?:-[a-z0-9]+)*(?!\w)",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?(?!\w)")
_UNIT_RE = re.compile(
    r"(?<!\w)(ounces?|oz|pounds?|lbs?|kilograms?|kg|grams?|g|milligrams?|mg|"
    r"milliliters?|millilitres?|ml|liters?|litres?|l|inches?|in|"
    r"centimeters?|centimetres?|cm|millimeters?|millimetres?|mm|"
    r"feet|ft|watts?|w|volts?|v|packs?|count|ct|pieces?|pcs?)(?!\w)",
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


@dataclass(frozen=True)
class Candidate:
    """One deduplicated document in a query's dense/lexical candidate union."""

    doc_id: str
    title: str
    dense_score: float | None
    lexical_score: float | None
    dense_rank: int | None
    lexical_rank: int | None
    title_bm25: float | None = None
    passage_bm25: float | None = None
    brand_bm25: float | None = None
    title_full_coverage: bool | None = None
    brand: str = ""
    color: str = ""
    passage_rank: int | None = None
    passage_score: float | None = None


@dataclass(frozen=True)
class QueryCandidates:
    """Retrieval candidates and text for one query group."""

    query_id: str
    query_text: str
    locale: str
    candidates: tuple[Candidate, ...]


@dataclass(frozen=True)
class LinearRanker:
    """Small pairwise logistic ranker with an L2-regularized weight vector."""

    weights: tuple[float, ...]
    feature_names: tuple[str, ...] = FEATURE_NAMES

    def score(self, features: Sequence[float]) -> float:
        if len(features) != len(self.weights):
            raise ValueError("feature vector width does not match model")
        return sum(weight * value for weight, value in zip(self.weights, features))

    def to_json(self) -> dict[str, Any]:
        return {"feature_names": list(self.feature_names), "weights": list(self.weights)}

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> LinearRanker:
        """Load a weight-only artifact, rejecting schema/feature mismatches."""
        names = payload.get("feature_names")
        raw_weights = payload.get("weights")
        valid_names = (
            list(FEATURE_NAMES),
            list(ENHANCED_FEATURE_NAMES),
            list(TYPED_ENHANCED_FEATURE_NAMES),
            list(TYPED_ENHANCED_NO_UNKNOWN_FEATURE_NAMES),
            list(FEATURE_NAMES + TYPED_NO_UNKNOWN_FEATURE_NAMES),
            list(PASSAGE_TYPED_COMPUTED_FEATURE_NAMES),
        )
        if names not in valid_names or not isinstance(raw_weights, list):
            raise ValueError("ranker artifact has an incompatible feature schema")
        expected_width = len(names)
        weights = tuple(_number(value) for value in raw_weights)
        if len(weights) != expected_width or any(
            value is None for value in weights
        ):
            raise ValueError("ranker artifact has invalid weights")
        return cls(
            tuple(value for value in weights if value is not None),
            tuple(names),
        )


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _candidate_list(value: object, label: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{label} candidates must be a list")
    if any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"{label} candidates must contain objects")
    return value


def parse_query_record(record: Mapping[str, Any]) -> QueryCandidates:
    """Parse one cached candidate record and preserve source ranks and scores."""
    required = ("query_id", "query_text", "locale")
    if any(not isinstance(record.get(key), str) for key in required):
        raise ValueError(f"candidate record requires string fields {required}")
    query_id = str(record["query_id"])
    query_text = str(record["query_text"])
    locale = str(record["locale"])
    merged: dict[str, dict[str, Any]] = {}
    for source in ("dense", "lexical"):
        for rank, item in enumerate(_candidate_list(record.get(source), source), 1):
            raw_id = item.get("doc_id")
            if raw_id is None or str(raw_id) == "":
                raise ValueError(f"{source} candidate is missing doc_id")
            doc_id = str(raw_id)
            entry = merged.setdefault(doc_id, {"doc_id": doc_id})
            # Dense wins if metadata differs between retrieval sources.
            title = item.get("title")
            if isinstance(title, str) and (source == "dense" or not entry.get("title")):
                entry["title"] = title
            score = _number(item.get("score"))
            if score is not None:
                entry[f"{source}_score"] = score
            entry[f"{source}_rank"] = rank
            raw_features = item.get("features")
            if isinstance(raw_features, Mapping):
                for name in ("title_bm25", "passage_bm25", "brand_bm25"):
                    feature_value = _number(raw_features.get(name))
                    # Preserve zero as present and permit the other arm to
                    # fill a missing/unsupported value, never overwrite real data.
                    if feature_value is not None and entry.get(name) is None:
                        entry[name] = feature_value
                coverage = raw_features.get("title_full_coverage")
                if isinstance(coverage, bool) and entry.get(
                    "title_full_coverage"
                ) is None:
                    entry["title_full_coverage"] = coverage
            for name in ("brand", "color"):
                value = item.get(name)
                if isinstance(value, str) and value and not entry.get(name):
                    entry[name] = value

    candidates = tuple(
        Candidate(
            doc_id=doc_id,
            title=str(item.get("title", "")),
            dense_score=item.get("dense_score"),
            lexical_score=item.get("lexical_score"),
            dense_rank=item.get("dense_rank"),
            lexical_rank=item.get("lexical_rank"),
            title_bm25=item.get("title_bm25"),
            passage_bm25=item.get("passage_bm25"),
            brand_bm25=item.get("brand_bm25"),
            title_full_coverage=item.get("title_full_coverage"),
            brand=str(item.get("brand", "")),
            color=str(item.get("color", "")),
        )
        for doc_id, item in merged.items()
    )
    return QueryCandidates(query_id, query_text, locale, candidates)


def read_candidates(path: Path) -> list[QueryCandidates]:
    """Read JSONL candidate records, rejecting duplicate query groups."""
    groups: list[QueryCandidates] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(record, Mapping):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            group = parse_query_record(record)
            if group.query_id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate query_id")
            seen.add(group.query_id)
            groups.append(group)
    return groups


def read_grades(path: Path) -> dict[str, dict[str, float]]:
    """Read query-grouped grades; grade zero is retained but never a negative."""
    grades: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(record, Mapping) or not isinstance(
                record.get("query_id"), str
            ):
                raise ValueError(f"{path}:{line_number}: expected query_id")
            query_id = str(record["query_id"])
            if query_id in grades:
                raise ValueError(f"{path}:{line_number}: duplicate query_id")
            raw_grades = record.get("grades", {})
            parsed: dict[str, float] = {}
            if isinstance(raw_grades, Mapping):
                for doc_id, value in raw_grades.items():
                    grade = _number(value)
                    if grade is not None and grade >= 0:
                        parsed[str(doc_id)] = grade
            elif isinstance(raw_grades, list):
                for item in raw_grades:
                    if not isinstance(item, Mapping):
                        continue
                    grade = _number(item.get("grade"))
                    if item.get("doc_id") is not None and grade is not None and grade >= 0:
                        parsed[str(item["doc_id"])] = grade
            else:
                raise ValueError(f"{path}:{line_number}: grades must be object or list")
            grades[query_id] = parsed
    return grades


def read_targets(path: Path) -> dict[str, dict[str, float]]:
    """Read Referee's top judged documents for NDCG ideal-DCG only.

    Target judgments must not be supplied to feature extraction or model fitting.
    """
    targets: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(record, Mapping) or not isinstance(
                record.get("query_id"), str
            ):
                raise ValueError(f"{path}:{line_number}: expected query_id")
            query_id = str(record["query_id"])
            if query_id in targets:
                raise ValueError(f"{path}:{line_number}: duplicate query_id")
            raw_targets = record.get("targets", [])
            if not isinstance(raw_targets, list):
                raise ValueError(f"{path}:{line_number}: targets must be a list")
            parsed: dict[str, float] = {}
            for item in raw_targets:
                if not isinstance(item, Mapping):
                    continue
                grade = _number(item.get("grade"))
                if item.get("doc_id") is not None and grade is not None and grade >= 0:
                    parsed[str(item["doc_id"])] = grade
            targets[query_id] = parsed
    return targets


def rrf_score(candidate: Candidate, dense_weight: float = 2.0) -> float:
    """Match the repository comparator: 2:1 weights and one-based RRF k=10."""
    dense = (
        dense_weight / (RRF_K + candidate.dense_rank)
        if candidate.dense_rank is not None
        else 0.0
    )
    lexical = (
        1.0 / (RRF_K + candidate.lexical_rank)
        if candidate.lexical_rank is not None
        else 0.0
    )
    return dense + lexical


def rrf_ranking(
    group: QueryCandidates, dense_weight: float = 2.0
) -> list[tuple[Candidate, float]]:
    """Return weighted RRF, preserving dense-first insertion on exact ties.

    Candidate mapping order is dense ranked-list order followed by lexical-only
    documents, matching ``TPufStrategy.reciprocal_rank_fusion``'s stable sort.
    """
    return sorted(
        ((candidate, rrf_score(candidate, dense_weight)) for candidate in group.candidates),
        key=lambda pair: pair[1],
        reverse=True,
    )


def _z_scores(values: Sequence[float | None]) -> list[float]:
    present = [value for value in values if value is not None]
    if len(present) < 2:
        return [0.0] * len(values)
    mean = sum(present) / len(present)
    variance = sum((value - mean) ** 2 for value in present) / len(present)
    scale = math.sqrt(variance)
    if scale <= 1e-12:
        return [0.0] * len(values)
    return [((value - mean) / scale if value is not None else 0.0) for value in values]


def _normalized_tokens(text: object) -> list[str]:
    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    return _TOKEN_RE.findall(normalized)


def _normalized_numbers(text: str) -> set[str]:
    result: set[str] = set()
    for match in _NUMBER_RE.finditer(text.replace(",", ".")):
        try:
            result.add(f"{float(match.group()):g}")
        except ValueError:
            result.add(match.group())
    return result


def _normalized_units(text: str) -> set[str]:
    return {
        _UNIT_ALIASES.get(match.group().casefold(), match.group().casefold())
        for match in _UNIT_RE.finditer(text)
    }


def _typed_product_features(group: QueryCandidates) -> dict[str, tuple[float, ...]]:
    """Compute compact three-state product evidence from existing cache fields.

    Slot inference follows the typed-evidence experiment: brands must be
    suggested by the query and retrieved explicit brand metadata; colors are
    normalized aliases; identifiers, numbers and units are read from title.
    Missing evidence is neutral (unknown), never a contradiction.
    """
    query = unicodedata.normalize("NFKC", group.query_text).casefold()
    query_tokens = set(_TOKEN_RE.findall(query))
    candidates = group.candidates
    brand_query_values: set[str] = set()
    for candidate in candidates:
        brand_tokens = {
            token for token in _normalized_tokens(candidate.brand)
            if token not in _STOPWORDS
        }
        overlap = brand_tokens & query_tokens
        if brand_tokens and (
            brand_tokens <= query_tokens
            or len(overlap) >= 2
            or any(len(token) >= 6 for token in overlap)
        ):
            brand_query_values.update(overlap)
    color_query_values = {
        _COLOR_ALIASES.get(token, token)
        for token in query_tokens
        if token in _COLOR_TERMS or token in _COLOR_ALIASES
    }
    model_query_values = {match.group().casefold() for match in _MODEL_RE.finditer(query)}
    number_query_values = _normalized_numbers(query)
    unit_query_values = _normalized_units(query)

    result: dict[str, tuple[float, ...]] = {}
    for candidate in candidates:
        title = unicodedata.normalize("NFKC", candidate.title).casefold()
        title_tokens = set(_TOKEN_RE.findall(title))
        brand_values = {
            token for token in _normalized_tokens(candidate.brand)
            if token not in _STOPWORDS
        }
        color_values = {
            _COLOR_ALIASES.get(token, token)
            for token in (
                set(_normalized_tokens(candidate.color)) | title_tokens
            )
            if token in _COLOR_TERMS or token in _COLOR_ALIASES
        }
        document_slots = (
            (brand_query_values, brand_values),
            (color_query_values, color_values),
            (
                model_query_values,
                {match.group().casefold() for match in _MODEL_RE.finditer(title)},
            ),
            (number_query_values, _normalized_numbers(title)),
            (unit_query_values, _normalized_units(title)),
        )
        states: list[int] = []
        identifier_states: list[int] = []
        for slot_index, (requested, available) in enumerate(document_slots):
            if not requested:
                continue
            state = 1 if requested & available else (-1 if available else 0)
            states.append(state)
            if slot_index >= 2:
                identifier_states.append(state)
        denominator = max(len(states), 1)
        result[candidate.doc_id] = (
            sum(state == 1 for state in identifier_states)
            / max(len(identifier_states), 1),
            sum(state == 1 for state in states) / denominator if states else 0.0,
            sum(state == -1 for state in states) / denominator if states else 0.0,
            sum(state == 0 for state in states) / denominator if states else 0.0,
        )
    return result


def feature_vectors(
    group: QueryCandidates,
    *,
    include_computed: bool = False,
    include_typed: bool = False,
    include_typed_unknown: bool = True,
    include_passage_arm: bool = False,
) -> dict[str, tuple[float, ...]]:
    """Build per-query runtime features; no identifier or query identity enters."""
    if include_passage_arm and not (
        include_computed and include_typed and not include_typed_unknown
    ):
        raise ValueError(
            "the passage arm requires typed/computed features without unknown state"
        )
    candidates = group.candidates
    dense_z = _z_scores([candidate.dense_score for candidate in candidates])
    lexical_z = _z_scores([candidate.lexical_score for candidate in candidates])
    computed_fields = (
        ("title_bm25", "brand_bm25")
        if include_passage_arm
        else ("title_bm25", "passage_bm25", "brand_bm25")
    )
    computed_values = [
        _z_scores([getattr(candidate, field) for candidate in candidates])
        for field in computed_fields
    ]
    passage_score_z = (
        _z_scores([candidate.passage_score for candidate in candidates])
        if include_passage_arm
        else []
    )
    typed_values = _typed_product_features(group) if include_typed else {}
    query_tokens = set(_TOKEN_RE.findall(group.query_text.casefold()))
    normalized_query = " ".join(_TOKEN_RE.findall(group.query_text.casefold()))
    denominator = max(len(query_tokens), 1)
    vectors: dict[str, tuple[float, ...]] = {}
    for index, candidate in enumerate(candidates):
        title_tokens = _TOKEN_RE.findall(candidate.title.casefold())
        title_set = set(title_tokens)
        overlap = len(query_tokens.intersection(title_set))
        base = (
            float(candidate.dense_rank is not None),
            float(candidate.lexical_rank is not None),
            1.0 / (RRF_K + candidate.dense_rank)
            if candidate.dense_rank is not None
            else 0.0,
            1.0 / (RRF_K + candidate.lexical_rank)
            if candidate.lexical_rank is not None
            else 0.0,
            rrf_score(candidate),
            dense_z[index],
            lexical_z[index],
            overlap / denominator,
            overlap / max(len(title_set), 1),
            float(bool(normalized_query) and normalized_query in " ".join(title_tokens)),
            math.log1p(len(title_tokens)),
        )
        if include_computed:
            enriched = tuple(
                value
                for field_index, field in enumerate(computed_fields)
                for value in (
                    computed_values[field_index][index],
                    float(getattr(candidate, field) is not None),
                )
            ) + (
                float(candidate.title_full_coverage is True),
                float(candidate.title_full_coverage is not None),
            )
            base = base + enriched
        if include_typed:
            typed = typed_values[candidate.doc_id]
            base = base + (
                typed if include_typed_unknown else typed[:-1]
            )
        if include_passage_arm:
            passage_rank = candidate.passage_rank
            base = base + (
                float(passage_rank is not None),
                1.0 / (RRF_K + passage_rank) if passage_rank is not None else 0.0,
                passage_score_z[index],
            )
        vectors[candidate.doc_id] = base
    return vectors


def split_query_groups(
    groups: Sequence[QueryCandidates], validation_fraction: float, seed: int
) -> tuple[list[QueryCandidates], list[QueryCandidates]]:
    """Deterministically split by query group, never separating group members."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    if len(groups) < 2:
        raise ValueError("at least two query groups are required")
    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)
    validation_size = min(
        len(groups) - 1, max(1, round(len(groups) * validation_fraction))
    )
    validation = shuffled[:validation_size]
    validation_ids = {group.query_id for group in validation}
    training = [group for group in groups if group.query_id not in validation_ids]
    return training, validation


def _gain(grade: float) -> float:
    return 2.0**grade - 1.0


def _pair_examples(
    groups: Iterable[QueryCandidates],
    grades: Mapping[str, Mapping[str, float]],
    *,
    zero_sample_per_query: int = 0,
    zero_pair_weight: float = 0.1,
    seed: int = 17,
    include_computed: bool = False,
    include_typed: bool = False,
    include_typed_unknown: bool = True,
    include_passage_arm: bool = False,
) -> list[tuple[tuple[float, ...], float]]:
    """Create query-local preference pairs; optionally sample weak zero negatives.

    By default grade zero remains ambiguous and is ignored. The opt-in sample
    policy is an experiment, not a claim that unjudged documents are irrelevant.
    Sampling randomness is seeded from the query-group's stable input position;
    identifiers never enter the feature vectors or selection seed.
    """
    if zero_sample_per_query < 0 or zero_pair_weight < 0:
        raise ValueError("zero sampling parameters must be nonnegative")
    examples: list[tuple[tuple[float, ...], float]] = []
    for group_index, group in enumerate(groups):
        labels = grades.get(group.query_id, {})
        vectors = feature_vectors(
            group,
            include_computed=include_computed,
            include_typed=include_typed,
            include_typed_unknown=include_typed_unknown,
            include_passage_arm=include_passage_arm,
        )
        observed = [
            (candidate.doc_id, labels[candidate.doc_id])
            for candidate in group.candidates
            if candidate.doc_id in labels and labels[candidate.doc_id] > 0
        ]
        for left_index, (left_id, left_grade) in enumerate(observed):
            for right_id, right_grade in observed[left_index + 1:]:
                if left_grade == right_grade:
                    continue
                if left_grade > right_grade:
                    preferred_id, other_id = left_id, right_id
                    preferred_grade, other_grade = left_grade, right_grade
                else:
                    preferred_id, other_id = right_id, left_id
                    preferred_grade, other_grade = right_grade, left_grade
                delta_gain = _gain(preferred_grade) - _gain(other_grade)
                examples.append(
                    (
                        tuple(
                            a - b
                            for a, b in zip(vectors[preferred_id], vectors[other_id])
                        ),
                        delta_gain,
                    )
                )
        if zero_sample_per_query:
            zero_ids = [
                candidate.doc_id
                for candidate in group.candidates
                if labels.get(candidate.doc_id) == 0
            ]
            if zero_ids and observed:
                query_seed = seed + group_index
                sampled_ids = random.Random(query_seed).sample(
                    zero_ids, min(zero_sample_per_query, len(zero_ids))
                )
                for positive_id, positive_grade in observed:
                    for zero_id in sampled_ids:
                        examples.append(
                            (
                                tuple(
                                    a - b
                                    for a, b in zip(
                                        vectors[positive_id], vectors[zero_id]
                                    )
                                ),
                                zero_pair_weight * _gain(positive_grade),
                            )
                        )
    return examples


def fit_ranker(
    groups: Sequence[QueryCandidates],
    grades: Mapping[str, Mapping[str, float]],
    *,
    epochs: int = 500,
    learning_rate: float = 0.1,
    l2: float = 0.01,
    zero_sample_per_query: int = 0,
    zero_pair_weight: float = 0.1,
    seed: int = 17,
    include_computed: bool = False,
    include_typed: bool = False,
    include_typed_unknown: bool = True,
    include_passage_arm: bool = False,
) -> LinearRanker:
    """Fit a deterministic full-batch pairwise logistic linear ranker."""
    if epochs < 1 or learning_rate <= 0 or l2 < 0:
        raise ValueError("epochs and learning_rate must be positive; l2 nonnegative")
    examples = _pair_examples(
        groups,
        grades,
        zero_sample_per_query=zero_sample_per_query,
        zero_pair_weight=zero_pair_weight,
        seed=seed,
        include_computed=include_computed,
        include_typed=include_typed,
        include_typed_unknown=include_typed_unknown,
        include_passage_arm=include_passage_arm,
    )
    if not examples:
        raise ValueError("training data contains no unequal positive-grade pairs")
    if include_passage_arm:
        feature_names = PASSAGE_TYPED_COMPUTED_FEATURE_NAMES
    else:
        feature_names = (
            (ENHANCED_FEATURE_NAMES if include_computed else FEATURE_NAMES)
            + (
                (
                    TYPED_FEATURE_NAMES
                    if include_typed_unknown
                    else TYPED_NO_UNKNOWN_FEATURE_NAMES
                )
                if include_typed
                else ()
            )
        )
    weights = [0.0] * len(feature_names)
    for _ in range(epochs):
        gradient = [l2 * weight for weight in weights]
        for difference, pair_weight in examples:
            margin = sum(weight * value for weight, value in zip(weights, difference))
            # Stable logistic derivative: d/dm log(1 + exp(-m)).
            if margin >= 0:
                probability = math.exp(-margin) / (1.0 + math.exp(-margin))
            else:
                probability = 1.0 / (1.0 + math.exp(margin))
            for index, value in enumerate(difference):
                gradient[index] -= pair_weight * probability * value / len(examples)
        for index, derivative in enumerate(gradient):
            weights[index] -= learning_rate * derivative
    return LinearRanker(tuple(weights), feature_names)


def model_ranking(
    group: QueryCandidates, model: LinearRanker
) -> list[tuple[Candidate, float]]:
    """Apply a trained ranker, with deterministic identifier-only tie breaking."""
    valid_schemas = (
        FEATURE_NAMES,
        ENHANCED_FEATURE_NAMES,
        FEATURE_NAMES + TYPED_FEATURE_NAMES,
        FEATURE_NAMES + TYPED_NO_UNKNOWN_FEATURE_NAMES,
        TYPED_ENHANCED_FEATURE_NAMES,
        TYPED_ENHANCED_NO_UNKNOWN_FEATURE_NAMES,
        PASSAGE_TYPED_COMPUTED_FEATURE_NAMES,
    )
    if model.feature_names not in valid_schemas:
        raise ValueError("model has an incompatible feature schema")
    include_computed = model.feature_names in (
        ENHANCED_FEATURE_NAMES,
        TYPED_ENHANCED_FEATURE_NAMES,
        TYPED_ENHANCED_NO_UNKNOWN_FEATURE_NAMES,
        PASSAGE_TYPED_COMPUTED_FEATURE_NAMES,
    )
    include_typed = model.feature_names in (
        FEATURE_NAMES + TYPED_FEATURE_NAMES,
        FEATURE_NAMES + TYPED_NO_UNKNOWN_FEATURE_NAMES,
        TYPED_ENHANCED_FEATURE_NAMES,
        TYPED_ENHANCED_NO_UNKNOWN_FEATURE_NAMES,
        PASSAGE_TYPED_COMPUTED_FEATURE_NAMES,
    )
    include_typed_unknown = (
        model.feature_names
        not in (
            FEATURE_NAMES + TYPED_NO_UNKNOWN_FEATURE_NAMES,
            TYPED_ENHANCED_NO_UNKNOWN_FEATURE_NAMES,
            PASSAGE_TYPED_COMPUTED_FEATURE_NAMES,
        )
    )
    features = feature_vectors(
        group,
        include_computed=include_computed,
        include_typed=include_typed,
        include_typed_unknown=include_typed_unknown,
        include_passage_arm=(
            model.feature_names == PASSAGE_TYPED_COMPUTED_FEATURE_NAMES
        ),
    )
    return sorted(
        (
            (candidate, model.score(features[candidate.doc_id]))
            for candidate in group.candidates
        ),
        key=lambda pair: (-pair[1], pair[0].doc_id),
    )


def ndcg(
    ranking: Sequence[tuple[Candidate, float]],
    labels: Mapping[str, float],
    cutoff: int = 10,
    *,
    ideal_grades: Mapping[str, float] | None = None,
) -> float | None:
    """NDCG using candidate labels for DCG and optional complete targets for IDCG.

    Grade-zero/unknown documents add no gain. Referee's top judged targets, when
    provided, are used only for ideal DCG—not for features, fitting or rankings.
    """
    if cutoff < 1:
        raise ValueError("cutoff must be positive")
    judged = {doc_id: grade for doc_id, grade in labels.items() if grade > 0}
    ideal_labels = labels if ideal_grades is None else ideal_grades
    ideal = sorted(
        (_gain(grade) for grade in ideal_labels.values() if grade > 0),
        reverse=True,
    )[:cutoff]
    ideal_dcg = sum(gain / math.log2(rank + 2) for rank, gain in enumerate(ideal))
    if ideal_dcg <= 0:
        return None
    dcg = sum(
        _gain(judged[candidate.doc_id]) / math.log2(rank + 2)
        for rank, (candidate, _) in enumerate(ranking[:cutoff])
        if candidate.doc_id in judged
    )
    return dcg / ideal_dcg


def evaluate(
    groups: Sequence[QueryCandidates],
    grades: Mapping[str, Mapping[str, float]],
    model: LinearRanker,
    cutoff: int = 10,
    ideal_grades: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, float | int | None]:
    """Compare model and 2:1 RRF on the exact same query candidate sets."""
    baseline_values: list[float] = []
    model_values: list[float] = []
    eligible = 0
    for group in groups:
        labels = grades.get(group.query_id, {})
        ideal = (
            ideal_grades.get(group.query_id, labels)
            if ideal_grades is not None
            else labels
        )
        baseline = ndcg(rrf_ranking(group), labels, cutoff, ideal_grades=ideal)
        reranked = ndcg(
            model_ranking(group, model), labels, cutoff, ideal_grades=ideal
        )
        if baseline is None or reranked is None:
            continue
        eligible += 1
        baseline_values.append(baseline)
        model_values.append(reranked)
    return {
        "queries": len(groups),
        "evaluated_queries": eligible,
        "rrf_2_1_ndcg": (
            sum(baseline_values) / len(baseline_values) if baseline_values else None
        ),
        "pairwise_linear_ndcg": (
            sum(model_values) / len(model_values) if model_values else None
        ),
    }


def compare_ablations(
    groups: Sequence[QueryCandidates],
    grades: Mapping[str, Mapping[str, float]],
    frozen_model: LinearRanker,
    *,
    validation_fraction: float = 0.25,
    seed: int = 17,
    epochs: int = 500,
    learning_rate: float = 0.1,
    l2: float = 0.01,
    cutoff: int = 10,
    ideal_grades: Mapping[str, Mapping[str, float]] | None = None,
) -> tuple[dict[str, dict[str, float | int | None]], dict[str, Any]]:
    """Compare frozen/base-refit/enriched models on one query-disjoint fold."""
    if frozen_model.feature_names != FEATURE_NAMES:
        raise ValueError("the frozen comparison model must use adf9768 base features")
    training, validation = split_query_groups(groups, validation_fraction, seed)
    base_refit = fit_ranker(
        training, grades, epochs=epochs, learning_rate=learning_rate, l2=l2
    )
    enhanced_refit = fit_ranker(
        training,
        grades,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        include_computed=True,
    )
    models = {
        "frozen_adf9768": frozen_model,
        "same_fold_base_refit": base_refit,
        "same_fold_computed_refit": enhanced_refit,
    }
    reports = {
        name: evaluate(
            validation, grades, model, cutoff, ideal_grades=ideal_grades
        )
        for name, model in models.items()
    }
    train_ids = {group.query_id for group in training}
    validation_ids = {group.query_id for group in validation}
    return reports, {
        "fold": {
            "train_queries": len(training),
            "validation_queries": len(validation),
            "query_disjoint": not train_ids.intersection(validation_ids),
            "identical_validation_candidates": True,
        },
        "models": models,
    }


def candidate_oracle_ndcg(
    groups: Sequence[QueryCandidates],
    grades: Mapping[str, Mapping[str, float]],
    cutoff: int = 10,
    ideal_grades: Mapping[str, Mapping[str, float]] | None = None,
) -> float | None:
    """Training-only candidate-set oracle for diagnostics; never a system ranker."""
    values: list[float] = []
    for group in groups:
        labels = grades.get(group.query_id, {})
        ideal = (
            ideal_grades.get(group.query_id, labels)
            if ideal_grades is not None
            else labels
        )
        oracle = sorted(
            group.candidates,
            key=lambda candidate: (
                -_gain(labels.get(candidate.doc_id, 0.0)),
                candidate.doc_id,
            ),
        )
        value = ndcg(
            [(candidate, 0.0) for candidate in oracle],
            labels,
            cutoff,
            ideal_grades=ideal,
        )
        if value is not None:
            values.append(value)
    return sum(values) / len(values) if values else None


def _write_csv(
    path: Path,
    groups: Sequence[QueryCandidates],
    model: LinearRanker,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("query_id", "doc_id", "rank", "score"))
        for group in groups:
            for rank, (candidate, score) in enumerate(model_ranking(group, model), 1):
                writer.writerow((group.query_id, candidate.doc_id, rank, score))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--grades", type=Path, required=True)
    parser.add_argument(
        "--targets",
        type=Path,
        help="Optional Referee judged targets used only for NDCG ideal DCG.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument(
        "--zero-sample-per-query",
        type=int,
        default=0,
        help="Experimental sampled grade-zero pairs per training query (default: off).",
    )
    parser.add_argument(
        "--zero-pair-weight",
        type=float,
        default=0.1,
        help="Relative weight for opt-in grade-zero pair constraints.",
    )
    parser.add_argument("--cutoff", type=int, default=10)
    return parser


def _ablation_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run frozen/refit computed-feature ablation."
    )
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--grades", type=Path, required=True)
    parser.add_argument("--targets", type=Path)
    parser.add_argument(
        "--frozen-model",
        type=Path,
        default=Path("models/esci_pairwise_ranker_adf9768.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--cutoff", type=int, default=10)
    return parser


def ablation_main(argv: list[str] | None = None) -> int:
    """Evaluate three rankers on the same query-disjoint validation candidates."""
    parser = _ablation_parser()
    args = parser.parse_args(argv)
    try:
        groups = read_candidates(args.candidates)
        grades = read_grades(args.grades)
        targets = read_targets(args.targets) if args.targets is not None else None
        frozen_payload = json.loads(args.frozen_model.read_text(encoding="utf-8"))
        if not isinstance(frozen_payload, Mapping):
            raise ValueError("frozen ranker artifact must be a JSON object")
        frozen = LinearRanker.from_json(frozen_payload)
        reports, detail = compare_ablations(
            groups,
            grades,
            frozen,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
            cutoff=args.cutoff,
            ideal_grades=targets,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        _training, validation = split_query_groups(
            groups, args.validation_fraction, args.seed
        )
        for name, model in detail["models"].items():
            _write_csv(args.output_dir / f"{name}.csv", validation, model)
            (args.output_dir / f"{name}.json").write_text(
                json.dumps(model.to_json(), indent=2) + "\n", encoding="utf-8"
            )
        report = {"fold": detail["fold"], "validation": reports}
        (args.output_dir / "ablation_report.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(report, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Train on query-disjoint groups, evaluate held-out groups, save rankings."""
    args = _parser().parse_args(argv)
    try:
        groups = read_candidates(args.candidates)
        grades = read_grades(args.grades)
        targets = read_targets(args.targets) if args.targets is not None else None
        train_groups, validation_groups = split_query_groups(
            groups, args.validation_fraction, args.seed
        )
        model = fit_ranker(
            train_groups,
            grades,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
            zero_sample_per_query=args.zero_sample_per_query,
            zero_pair_weight=args.zero_pair_weight,
            seed=args.seed,
        )
        report = {
            "train_queries": len(train_groups),
            "validation_queries": len(validation_groups),
            "train_candidate_oracle_ndcg": candidate_oracle_ndcg(
                train_groups,
                grades,
                args.cutoff,
                ideal_grades=targets,
            ),
            "validation": evaluate(
                validation_groups,
                grades,
                model,
                args.cutoff,
                ideal_grades=targets,
            ),
        }
        _write_csv(args.output, validation_groups, model)
        if args.model_output is not None:
            args.model_output.write_text(
                json.dumps(model.to_json(), indent=2) + "\n", encoding="utf-8"
            )
    except (OSError, ValueError) as error:
        _parser().error(str(error))
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
