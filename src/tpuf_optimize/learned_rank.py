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
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True)
class Candidate:
    """One deduplicated document in a query's dense/lexical candidate union."""

    doc_id: str
    title: str
    dense_score: float | None
    lexical_score: float | None
    dense_rank: int | None
    lexical_rank: int | None


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

    candidates = tuple(
        Candidate(
            doc_id=doc_id,
            title=str(item.get("title", "")),
            dense_score=item.get("dense_score"),
            lexical_score=item.get("lexical_score"),
            dense_rank=item.get("dense_rank"),
            lexical_rank=item.get("lexical_rank"),
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


def feature_vectors(group: QueryCandidates) -> dict[str, tuple[float, ...]]:
    """Build per-query runtime features; no identifier or query identity enters."""
    candidates = group.candidates
    dense_z = _z_scores([candidate.dense_score for candidate in candidates])
    lexical_z = _z_scores([candidate.lexical_score for candidate in candidates])
    query_tokens = set(_TOKEN_RE.findall(group.query_text.casefold()))
    normalized_query = " ".join(_TOKEN_RE.findall(group.query_text.casefold()))
    denominator = max(len(query_tokens), 1)
    vectors: dict[str, tuple[float, ...]] = {}
    for index, candidate in enumerate(candidates):
        title_tokens = _TOKEN_RE.findall(candidate.title.casefold())
        title_set = set(title_tokens)
        overlap = len(query_tokens.intersection(title_set))
        vectors[candidate.doc_id] = (
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
    groups: Iterable[QueryCandidates], grades: Mapping[str, Mapping[str, float]]
) -> list[tuple[tuple[float, ...], float]]:
    """Create query-local preference differences, ignoring ambiguous grade 0."""
    examples: list[tuple[tuple[float, ...], float]] = []
    for group in groups:
        labels = grades.get(group.query_id, {})
        vectors = feature_vectors(group)
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
    return examples


def fit_ranker(
    groups: Sequence[QueryCandidates],
    grades: Mapping[str, Mapping[str, float]],
    *,
    epochs: int = 500,
    learning_rate: float = 0.1,
    l2: float = 0.01,
) -> LinearRanker:
    """Fit a deterministic full-batch pairwise logistic linear ranker."""
    if epochs < 1 or learning_rate <= 0 or l2 < 0:
        raise ValueError("epochs and learning_rate must be positive; l2 nonnegative")
    examples = _pair_examples(groups, grades)
    if not examples:
        raise ValueError("training data contains no unequal positive-grade pairs")
    weights = [0.0] * len(FEATURE_NAMES)
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
    return LinearRanker(tuple(weights))


def model_ranking(
    group: QueryCandidates, model: LinearRanker
) -> list[tuple[Candidate, float]]:
    """Apply a trained ranker, with deterministic identifier-only tie breaking."""
    features = feature_vectors(group)
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
) -> float | None:
    """NDCG over available judgments; unknown documents have zero observed gain.

    Grade-zero judgments are ambiguous and excluded from both gains and the
    ideal list. Unjudged/zero-grade candidates still occupy their retrieval rank.
    """
    if cutoff < 1:
        raise ValueError("cutoff must be positive")
    judged = {doc_id: grade for doc_id, grade in labels.items() if grade > 0}
    ideal = sorted((_gain(grade) for grade in judged.values()), reverse=True)[:cutoff]
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
) -> dict[str, float | int | None]:
    """Compare model and 2:1 RRF on the exact same query candidate sets."""
    baseline_values: list[float] = []
    model_values: list[float] = []
    eligible = 0
    for group in groups:
        labels = grades.get(group.query_id, {})
        baseline = ndcg(rrf_ranking(group), labels, cutoff)
        reranked = ndcg(model_ranking(group, model), labels, cutoff)
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


def candidate_oracle_ndcg(
    groups: Sequence[QueryCandidates],
    grades: Mapping[str, Mapping[str, float]],
    cutoff: int = 10,
) -> float | None:
    """Training-only candidate-set oracle for diagnostics; never a system ranker."""
    values: list[float] = []
    for group in groups:
        labels = grades.get(group.query_id, {})
        oracle = sorted(
            group.candidates,
            key=lambda candidate: (
                -_gain(labels.get(candidate.doc_id, 0.0)),
                candidate.doc_id,
            ),
        )
        value = ndcg([(candidate, 0.0) for candidate in oracle], labels, cutoff)
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-output", type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--cutoff", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Train on query-disjoint groups, evaluate held-out groups, save rankings."""
    args = _parser().parse_args(argv)
    try:
        groups = read_candidates(args.candidates)
        grades = read_grades(args.grades)
        train_groups, validation_groups = split_query_groups(
            groups, args.validation_fraction, args.seed
        )
        model = fit_ranker(
            train_groups,
            grades,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            l2=args.l2,
        )
        report = {
            "train_queries": len(train_groups),
            "validation_queries": len(validation_groups),
            "train_candidate_oracle_ndcg": candidate_oracle_ndcg(
                train_groups, grades, args.cutoff
            ),
            "validation": evaluate(validation_groups, grades, model, args.cutoff),
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
