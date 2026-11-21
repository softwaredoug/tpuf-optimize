"""Offline passage-arm expansion and training-only diagnostics.

This module reads saved retrieval and Referee artifacts only. It contains no
Turbopuffer or Referee client and never changes the cached dense/lexical scores
or order. Any future metadata collection is exposed as a query builder so it
can be run separately in an explicitly approved Turbopuffer slot.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol, cast

from turbopuffer.types.namespace_multi_query_params import Query

from tpuf_optimize.computed_feature_collector import (
    ComputedFeatureQueryStrategy,
    _title_full_coverage,
)
from tpuf_optimize.computed_feature_rank import (
    TYPED_ENHANCED_NO_UNKNOWN_FEATURE_NAMES,
    Candidate,
    LinearRanker,
    QueryCandidates,
    _gain,
    candidate_oracle_ndcg,
    evaluate,
    fit_ranker,
    ndcg,
    read_candidates,
    read_grades,
    read_targets,
    split_query_groups,
)
from tpuf_optimize.typed_computed_experiment import (
    _fit_only_tune,
    _latency_summary,
    _paired_bootstrap,
    _per_query_ndcg,
)


PASSAGE_FORMULATION_COMMIT = "cf42f0e"
PASSAGE_CANDIDATE_DEPTH = 100
_METADATA_FEATURES = ("title_bm25", "passage_bm25", "brand_bm25")


class QueryNamespace(Protocol):
    """The serial namespace query method used for bounded metadata reads."""

    def query(self, **kwargs: Any) -> Any: ...


def read_passage_arm(path: Path) -> dict[str, tuple[tuple[str, int, float], ...]]:
    """Read passage BM25 result rows, validating per-query ranks/sentinels."""
    result: dict[str, list[tuple[str, int, float]]] = {}
    seen: dict[str, set[str]] = {}
    sentinels: set[str] = set()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"query_id", "doc_id", "rank", "score"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("passage CSV requires query_id, doc_id, rank, score")
        for line_number, row in enumerate(reader, start=2):
            query_id = row.get("query_id", "")
            doc_id = row.get("doc_id", "")
            if not query_id or not doc_id:
                raise ValueError(f"{path}:{line_number}: empty query_id/doc_id")
            try:
                rank = int(row["rank"])
                score = float(row["score"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"{path}:{line_number}: invalid rank/score") from error
            if not math.isfinite(score):
                raise ValueError(f"{path}:{line_number}: score must be finite")
            if doc_id == "-1":
                if (
                    rank != -1
                    or score != -1.0
                    or query_id in sentinels
                    or result.get(query_id)
                ):
                    raise ValueError(f"{path}:{line_number}: invalid sentinel row")
                sentinels.add(query_id)
                result.setdefault(query_id, [])
                continue
            if rank < 1 or query_id in sentinels:
                raise ValueError(f"{path}:{line_number}: invalid result rank")
            query_seen = seen.setdefault(query_id, set())
            if doc_id in query_seen:
                raise ValueError(f"{path}:{line_number}: duplicate passage candidate")
            query_seen.add(doc_id)
            result.setdefault(query_id, []).append((doc_id, rank, score))

    normalized: dict[str, tuple[tuple[str, int, float], ...]] = {}
    for query_id, hits in result.items():
        ordered = sorted(hits, key=lambda hit: hit[1])
        if ordered and [hit[1] for hit in ordered] != list(
            range(1, len(ordered) + 1)
        ):
            raise ValueError("passage ranks must be contiguous from one per query")
        if len(ordered) > PASSAGE_CANDIDATE_DEPTH:
            raise ValueError("passage candidate list exceeds the bounded depth of 100")
        normalized[query_id] = tuple(ordered)
    return normalized


def attach_passage_arm(
    groups: Sequence[QueryCandidates],
    passage: Mapping[str, Sequence[tuple[str, int, float]]],
) -> tuple[list[QueryCandidates], dict[str, set[str]]]:
    """Join passage candidates by exact query/doc ID, preserving original arms."""
    group_by_id = {group.query_id: group for group in groups}
    if len(group_by_id) != len(groups) or set(group_by_id) != set(passage):
        raise ValueError("passage and source candidate query IDs must match exactly")

    expanded: list[QueryCandidates] = []
    novel_by_query: dict[str, set[str]] = {}
    for group in groups:
        hits = passage[group.query_id]
        passage_by_id: dict[str, tuple[int, float]] = {}
        for doc_id, rank, score in hits:
            if doc_id == "-1" or rank < 1 or not math.isfinite(score):
                raise ValueError("invalid passage candidate")
            if doc_id in passage_by_id:
                raise ValueError("duplicate passage candidate")
            passage_by_id[doc_id] = (rank, score)

        original_by_id = {candidate.doc_id: candidate for candidate in group.candidates}
        if len(original_by_id) != len(group.candidates):
            raise ValueError("source candidate group contains duplicate doc IDs")
        novel = set(passage_by_id) - set(original_by_id)
        novel_by_query[group.query_id] = novel
        merged: list[Candidate] = []
        for candidate in group.candidates:
            hit = passage_by_id.get(candidate.doc_id)
            merged.append(
                candidate
                if hit is None
                else replace(candidate, passage_rank=hit[0], passage_score=hit[1])
            )
        for doc_id, rank, score in hits:
            if doc_id not in original_by_id:
                merged.append(
                    Candidate(
                        doc_id=doc_id,
                        title="",
                        dense_score=None,
                        lexical_score=None,
                        dense_rank=None,
                        lexical_rank=None,
                        passage_rank=rank,
                        passage_score=score,
                    )
                )

        # Guard the invariant explicitly: both original ranked arms are retained.
        for before, after in zip(group.candidates, merged):
            if (
                before.doc_id != after.doc_id
                or before.dense_score != after.dense_score
                or before.lexical_score != after.lexical_score
                or before.dense_rank != after.dense_rank
                or before.lexical_rank != after.lexical_rank
            ):
                raise ValueError("passage join altered original ranks or scores")
        expanded.append(
            QueryCandidates(
                group.query_id, group.query_text, group.locale, tuple(merged)
            )
        )
    return expanded, novel_by_query


def passage_metadata_query(
    query_text: str, locale: str, candidate_ids: Sequence[str]
) -> Query:
    """Build one bounded, ID-filtered read; never performs the read itself."""
    unique_ids = list(dict.fromkeys(candidate_ids))
    if not unique_ids or len(unique_ids) > PASSAGE_CANDIDATE_DEPTH:
        raise ValueError("metadata query must contain between 1 and 100 unique IDs")
    if any(not isinstance(doc_id, str) or not doc_id or doc_id == "-1" for doc_id in unique_ids):
        raise ValueError("metadata query IDs must be non-empty document IDs")
    if not query_text.strip() or not locale.strip():
        raise ValueError("query_text and locale must not be empty")
    strategy = ComputedFeatureQueryStrategy("dense")
    normalized_locale = strategy.normalize_locale(locale)
    return cast(
        Query,
        {
            "filters": (
                "And",
                (
                    ("locale", "Eq", normalized_locale),
                    ("id", "In", unique_ids),
                ),
            ),
            "rank_by": ("id", "asc"),
            "top_k": len(unique_ids),
            "include_attributes": ["title", "brand", "color", "locale"],
            "compute_attributes": strategy.computed_attributes(
                query_text, normalized_locale
            ),
        },
    )


def collect_passage_metadata_checkpoint(
    namespace: QueryNamespace,
    groups: Sequence[QueryCandidates],
    novel_by_query: Mapping[str, set[str]],
    checkpoint_path: Path,
    output_path: Path,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Collect exact passage-only metadata with one durable bounded read/query.

    This is intentionally not called by the offline experiment runner. Invoke
    it only during an explicitly approved Turbopuffer slot. The checkpoint is
    append-only and records a query only after returned IDs/locale are verified.
    """
    if checkpoint_path.resolve() == output_path.resolve():
        raise ValueError("metadata checkpoint and completed output must differ")
    group_by_id = {group.query_id: group for group in groups}
    eligible = {query_id for query_id, ids in novel_by_query.items() if ids}
    if set(group_by_id) != set(novel_by_query):
        raise ValueError("metadata requests and source query IDs must match exactly")

    completed: dict[str, dict[str, dict[str, Any]]] = {}
    if checkpoint_path.exists():
        with checkpoint_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"{checkpoint_path}:{line_number}: invalid checkpoint JSON"
                    ) from error
                if not isinstance(record, Mapping):
                    raise ValueError("metadata checkpoint rows must be objects")
                query_id = record.get("query_id")
                if not isinstance(query_id, str) or query_id not in eligible:
                    raise ValueError("metadata checkpoint has an unexpected query ID")
                group = group_by_id[query_id]
                expected_ids = novel_by_query[query_id]
                if (
                    record.get("query_text") != group.query_text
                    or record.get("locale")
                    != ComputedFeatureQueryStrategy.normalize_locale(group.locale)
                    or set(record.get("candidate_ids", [])) != expected_ids
                ):
                    raise ValueError("metadata checkpoint provenance/IDs mismatch")
                docs = record.get("documents")
                if not isinstance(docs, Mapping) or set(docs) != expected_ids:
                    raise ValueError("metadata checkpoint has incomplete ID coverage")
                if any(not isinstance(value, Mapping) for value in docs.values()):
                    raise ValueError("metadata checkpoint documents must be objects")
                if query_id in completed:
                    raise ValueError("metadata checkpoint has duplicate query records")
                completed[query_id] = {
                    str(doc_id): cast(dict[str, Any], value)
                    for doc_id, value in docs.items()
                }

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.touch(exist_ok=True)
    for query_id in sorted(eligible):
        if query_id in completed:
            continue
        group = group_by_id[query_id]
        candidate_ids = sorted(novel_by_query[query_id])
        query = passage_metadata_query(
            group.query_text, group.locale, candidate_ids
        )
        response = namespace.query(**query)
        rows = getattr(response, "rows", None) or []
        by_id: dict[str, dict[str, Any]] = {}
        expected_locale = ComputedFeatureQueryStrategy.normalize_locale(
            group.locale
        )
        for row in rows:
            values = row.model_dump()
            raw_id = values.get("id")
            if raw_id is None:
                raise ValueError("metadata response row has no document ID")
            doc_id = str(raw_id)
            if doc_id in by_id:
                raise ValueError("metadata response duplicated a document ID")
            if values.get("locale") != expected_locale:
                raise ValueError("metadata response locale does not match query")
            features: dict[str, float | None] = {}
            for name in _METADATA_FEATURES:
                value = values.get(name)
                if value is None:
                    features[name] = None
                elif (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise ValueError(f"invalid computed feature {name}")
                else:
                    features[name] = float(value)
            raw_title = values.get("title")
            raw_brand = values.get("brand")
            raw_color = values.get("color")
            if any(
                value is not None and not isinstance(value, str)
                for value in (raw_title, raw_brand, raw_color)
            ):
                raise ValueError("metadata response text attributes must be strings")
            title = raw_title if isinstance(raw_title, str) else ""
            brand = raw_brand if isinstance(raw_brand, str) else ""
            color = raw_color if isinstance(raw_color, str) else ""
            features["title_full_coverage"] = _title_full_coverage(
                group.query_text, raw_title
            )
            by_id[doc_id] = {
                "title": title,
                "brand": brand,
                "color": color,
                "locale": values["locale"],
                "features": features,
            }
        if set(by_id) != novel_by_query[query_id]:
            raise ValueError("ID-filtered metadata read did not exactly cover IDs")
        record = {
            "query_id": query_id,
            "query_text": group.query_text,
            "locale": expected_locale,
            "candidate_ids": candidate_ids,
            "documents": by_id,
        }
        with checkpoint_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        completed[query_id] = by_id

    if set(completed) != eligible:
        raise ValueError("metadata collection did not cover all eligible queries")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(completed, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, output_path)
    return completed


def enrich_passage_only_candidates(
    groups: Sequence[QueryCandidates],
    novel_by_query: Mapping[str, set[str]],
    metadata: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[QueryCandidates]:
    """Attach exact ID-filtered metadata without touching original candidates."""
    eligible_queries = {
        query_id for query_id, doc_ids in novel_by_query.items() if doc_ids
    }
    if set(metadata) != eligible_queries:
        raise ValueError("metadata query coverage does not match passage-only IDs")
    enriched: list[QueryCandidates] = []
    for group in groups:
        novel = novel_by_query[group.query_id]
        docs = metadata.get(group.query_id, {})
        if set(docs) != novel:
            raise ValueError("ID-filtered metadata did not exactly cover novel IDs")
        by_id: dict[str, Candidate] = {}
        for doc_id in novel:
            item = docs[doc_id]
            doc_locale = item.get("locale")
            expected_locale = ComputedFeatureQueryStrategy.normalize_locale(
                group.locale
            )
            if doc_locale is not None and doc_locale != expected_locale:
                raise ValueError("ID-filtered metadata locale mismatch")
            raw_features = item.get("features", {})
            if not isinstance(raw_features, Mapping):
                raise ValueError("metadata features must be an object")
            feature_values: dict[str, float | None] = {}
            for name in _METADATA_FEATURES:
                value = raw_features.get(name)
                if value is None:
                    feature_values[name] = None
                elif (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise ValueError(f"invalid computed feature {name}")
                else:
                    feature_values[name] = float(value)
            coverage = raw_features.get("title_full_coverage")
            if coverage is not None and not isinstance(coverage, bool):
                raise ValueError("title_full_coverage must be boolean or missing")
            title = item.get("title", "")
            brand = item.get("brand", "")
            color = item.get("color", "")
            if not all(isinstance(value, str) for value in (title, brand, color)):
                raise ValueError("metadata title, brand, and color must be strings")
            by_id[doc_id] = Candidate(
                doc_id=doc_id,
                title=title,
                dense_score=None,
                lexical_score=None,
                dense_rank=None,
                lexical_rank=None,
                title_bm25=feature_values["title_bm25"],
                passage_bm25=feature_values["passage_bm25"],
                brand_bm25=feature_values["brand_bm25"],
                title_full_coverage=coverage,
                brand=brand,
                color=color,
                passage_rank=next(
                    candidate.passage_rank
                    for candidate in group.candidates
                    if candidate.doc_id == doc_id
                ),
                passage_score=next(
                    candidate.passage_score
                    for candidate in group.candidates
                    if candidate.doc_id == doc_id
                ),
            )
        candidates = tuple(
            by_id[candidate.doc_id]
            if candidate.doc_id in novel
            else candidate
            for candidate in group.candidates
        )
        enriched.append(
            QueryCandidates(
                group.query_id, group.query_text, group.locale, candidates
            )
        )
    return enriched


def _validate_task_provenance(
    groups: Sequence[QueryCandidates], task_path: Path
) -> None:
    task = json.loads(task_path.read_text(encoding="utf-8"))
    raw_queries = task.get("queries")
    if not isinstance(raw_queries, list):
        raise ValueError("training task artifact has no query list")
    task_queries = {
        item.get("query_id"): item for item in raw_queries if isinstance(item, Mapping)
    }
    by_id = {group.query_id: group for group in groups}
    if set(task_queries) != set(by_id):
        raise ValueError("Referee training task IDs do not match cached queries")
    locale_aliases = {"en-US": "us", "es-ES": "es", "ja-JP": "jp"}
    for query_id, group in by_id.items():
        query = task_queries[query_id]
        if query.get("query_text") != group.query_text:
            raise ValueError("Referee query text does not match candidate source")
        metadata = query.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            raise ValueError("Referee query metadata must be an object")
        locale = metadata.get("locale", metadata.get("product_locale"))
        if locale is not None and locale not in (
            group.locale,
            locale_aliases.get(group.locale),
        ):
            raise ValueError("Referee locale does not match candidate source")


def _read_novel_positive_judgments(
    path: Path,
    novel_by_query: Mapping[str, set[str]],
) -> dict[str, dict[str, float]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    eligible_queries = {
        query_id for query_id, doc_ids in novel_by_query.items() if doc_ids
    }
    if not isinstance(raw, Mapping) or set(raw) != eligible_queries:
        raise ValueError("novel-judgment query IDs do not match candidate additions")
    judgments: dict[str, dict[str, float]] = {}
    for query_id, expected_ids in novel_by_query.items():
        if not expected_ids:
            judgments[query_id] = {}
            continue
        raw_query = raw[query_id]
        if not isinstance(raw_query, Mapping) or not set(raw_query).issubset(expected_ids):
            raise ValueError("novel judgments contain unexpected candidate IDs")
        parsed: dict[str, float] = {}
        for doc_id, grade in raw_query.items():
            if (
                isinstance(grade, bool)
                or not isinstance(grade, (int, float))
                or not math.isfinite(float(grade))
                or grade <= 0
            ):
                raise ValueError(
                    "only positive Referee judgments are training labels here"
                )
            parsed[str(doc_id)] = float(grade)
        judgments[query_id] = parsed
    return judgments


def _validate_debug_manifest(
    path: Path,
    *,
    eligible_queries: int,
    novel_pairs: int,
    positive_judgments: int,
) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "token_scope": "esci_breed_training_only",
        "eligible_queries": eligible_queries,
        "novel_candidate_pairs_requested": novel_pairs,
        "novel_candidate_pairs_returned": novel_pairs,
        "positive_judgments": positive_judgments,
        "zero_or_unjudged_ambiguous_ignored": novel_pairs - positive_judgments,
        "all_query_texts_verified": True,
        "all_candidate_results_verified": True,
        "unjudged_documents_used_as_training_labels": False,
    }
    if not isinstance(manifest, Mapping) or any(
        manifest.get(name) != value for name, value in expected.items()
    ):
        raise ValueError("Referee debug aggregate manifest does not match inputs")
    return dict(manifest)


def _paired_target_recovery(
    groups: Sequence[QueryCandidates],
    targets: Mapping[str, Mapping[str, float]],
    novel_by_query: Mapping[str, set[str]],
) -> dict[str, int | float]:
    total_targets = 0
    original_hits = 0
    expanded_hits = 0
    queries_original_hit = 0
    queries_expanded_hit = 0
    novel_target_hits = 0
    for group in groups:
        target_ids = set(targets.get(group.query_id, {}))
        original_ids = {
            candidate.doc_id
            for candidate in group.candidates
            if candidate.doc_id not in novel_by_query[group.query_id]
        }
        expanded_ids = {candidate.doc_id for candidate in group.candidates}
        original_count = len(target_ids & original_ids)
        expanded_count = len(target_ids & expanded_ids)
        total_targets += len(target_ids)
        original_hits += original_count
        expanded_hits += expanded_count
        novel_target_hits += len(target_ids & novel_by_query[group.query_id])
        queries_original_hit += bool(original_count)
        queries_expanded_hit += bool(expanded_count)
    return {
        "judged_top_targets": total_targets,
        "original_union_target_hits": original_hits,
        "expanded_pool_target_hits": expanded_hits,
        "original_union_target_recall": (
            original_hits / total_targets if total_targets else 0.0
        ),
        "expanded_pool_target_recall": (
            expanded_hits / total_targets if total_targets else 0.0
        ),
        "queries_with_target_in_original_union": queries_original_hit,
        "queries_with_target_in_expanded_pool": queries_expanded_hit,
        "targets_recovered_only_by_passage_arm": novel_target_hits,
    }


def _rank_score_provenance(
    original: Sequence[QueryCandidates],
    passage: Mapping[str, Sequence[tuple[str, int, float]]],
) -> dict[str, float | int]:
    """Compare cached passage BM25 scores to the saved passage-arm scores."""
    taus: list[float] = []
    compared = 0
    exact_scores = 0
    absolute_errors: list[float] = []
    for group in original:
        cached = {candidate.doc_id: candidate.passage_bm25 for candidate in group.candidates}
        shared: list[tuple[int, float, float]] = []
        for doc_id, rank, score in passage[group.query_id]:
            cached_score = cached.get(doc_id)
            if cached_score is not None:
                shared.append((rank, score, cached_score))
        if len(shared) > 1:
            concordant = discordant = 0
            shared.sort(key=lambda value: value[0])
            for index, left in enumerate(shared):
                for right in shared[index + 1:]:
                    delta_source = left[1] - right[1]
                    delta_cached = float(left[2]) - float(right[2])
                    if abs(delta_source) <= 1e-12 or abs(delta_cached) <= 1e-12:
                        continue
                    if delta_source * delta_cached > 0:
                        concordant += 1
                    else:
                        discordant += 1
            pairs = concordant + discordant
            if pairs:
                taus.append((concordant - discordant) / pairs)
        for _rank, score, cached_score in shared:
            cached_value = float(cached_score)
            compared += 1
            absolute_errors.append(abs(score - cached_value))
            exact_scores += math.isclose(
                score, cached_value, rel_tol=1e-5, abs_tol=1e-5
            )
    return {
        "shared_candidate_scores": compared,
        "exact_score_matches_at_1e-5": exact_scores,
        "median_absolute_score_difference": (
            statistics.median(absolute_errors) if absolute_errors else 0.0
        ),
        "p95_absolute_score_difference": (
            sorted(absolute_errors)[
                min(
                    len(absolute_errors) - 1,
                    math.ceil(0.95 * len(absolute_errors)) - 1,
                )
            ]
            if absolute_errors
            else 0.0
        ),
        "median_per_query_kendall_tau": statistics.median(taus) if taus else 0.0,
        "queries_with_rank_comparison": len(taus),
    }


def _target_oracle_ndcg(
    groups: Sequence[QueryCandidates],
    grades: Mapping[str, Mapping[str, float]],
    targets: Mapping[str, Mapping[str, float]],
) -> float | None:
    values: list[float] = []
    for group in groups:
        labels = grades.get(group.query_id, {})
        ideal_grades = targets.get(group.query_id, {})
        ordered = sorted(
            group.candidates,
            key=lambda candidate: (
                -_gain(labels.get(candidate.doc_id, 0.0)),
                candidate.doc_id,
            ),
        )
        value = ndcg(
            [(candidate, 0.0) for candidate in ordered],
            labels,
            ideal_grades=ideal_grades,
        )
        if value is not None:
            values.append(value)
    return sum(values) / len(values) if values else None


def _fit_expanded_ranker(
    training: Sequence[QueryCandidates],
    grades: Mapping[str, Mapping[str, float]],
    targets: Mapping[str, Mapping[str, float]],
    *,
    seed: int,
) -> tuple[LinearRanker, dict[str, Any]]:
    inner_train, inner_validation = split_query_groups(
        training, 0.20, seed=seed + 2026
    )
    selected_l2: float | None = None
    best_score: float | None = None
    for l2 in (0.01, 0.05, 0.1, 0.3, 1.0):
        model = fit_ranker(
            inner_train,
            grades,
            seed=seed,
            epochs=500,
            learning_rate=0.1,
            l2=l2,
            include_computed=True,
            include_typed=True,
            include_typed_unknown=False,
            include_passage_arm=True,
        )
        score = evaluate(
            inner_validation, grades, model, ideal_grades=targets
        )["pairwise_linear_ndcg"]
        if isinstance(score, (int, float)) and (
            best_score is None or score > best_score
        ):
            selected_l2 = l2
            best_score = float(score)
    if selected_l2 is None:
        raise ValueError("expanded fit-only tuning found no eligible inner queries")
    # Ablations are evaluated only on the inner validation slice. They never
    # affect the outer validation comparison or the fitted full-fold nominee.
    inner_model = fit_ranker(
        inner_train,
        grades,
        seed=seed,
        epochs=500,
        learning_rate=0.1,
        l2=selected_l2,
        include_computed=True,
        include_typed=True,
        include_typed_unknown=False,
        include_passage_arm=True,
    )
    feature_index = {
        name: index for index, name in enumerate(inner_model.feature_names)
    }
    inner_ablation: dict[str, Any] = {}
    for label, omitted in (
        ("without_passage_rank", ("passage_arm_rank_rrf",)),
        ("without_passage_score", ("passage_arm_score_z",)),
        (
            "without_passage_rank_or_score",
            ("passage_arm_rank_rrf", "passage_arm_score_z"),
        ),
    ):
        weights = list(inner_model.weights)
        for name in omitted:
            weights[feature_index[name]] = 0.0
        ablated = LinearRanker(tuple(weights), inner_model.feature_names)
        inner_ablation[label] = evaluate(
            inner_validation, grades, ablated, ideal_grades=targets
        )

    nominee = fit_ranker(
        training,
        grades,
        seed=seed,
        epochs=500,
        learning_rate=0.1,
        l2=selected_l2,
        include_computed=True,
        include_typed=True,
        include_typed_unknown=False,
        include_passage_arm=True,
    )
    return nominee, {
        "inner_training_queries": len(inner_train),
        "inner_validation_queries": len(inner_validation),
        "selected_l2": selected_l2,
        "selection_metric": "inner fit-fold NDCG@10",
        "inner_validation_ablation_ndcg": inner_ablation,
    }


def run_experiment(
    candidates_path: Path,
    grades_path: Path,
    targets_path: Path,
    passage_path: Path,
    task_path: Path,
    novel_judgments_path: Path,
    debug_manifest_path: Path,
    frozen_model_path: Path,
    output_path: Path,
    *,
    metadata_path: Path | None = None,
    seed: int = 17,
) -> dict[str, Any]:
    original = read_candidates(candidates_path)
    _validate_task_provenance(original, task_path)
    grades = read_grades(grades_path)
    targets = read_targets(targets_path)
    passage = read_passage_arm(passage_path)
    expanded, novel_by_query = attach_passage_arm(original, passage)
    extra_judgments = _read_novel_positive_judgments(
        novel_judgments_path, novel_by_query
    )
    augmented_grades = {query_id: dict(labels) for query_id, labels in grades.items()}
    for query_id, labels in extra_judgments.items():
        if set(labels) & set(grades.get(query_id, {})):
            raise ValueError("novel judgment unexpectedly overlaps original labels")
        augmented_grades.setdefault(query_id, {}).update(labels)
    positive_novel_judgments = sum(map(len, extra_judgments.values()))
    debug_manifest = _validate_debug_manifest(
        debug_manifest_path,
        eligible_queries=sum(bool(ids) for ids in novel_by_query.values()),
        novel_pairs=sum(map(len, novel_by_query.values())),
        positive_judgments=positive_novel_judgments,
    )

    frozen_payload = json.loads(frozen_model_path.read_text(encoding="utf-8"))
    if not isinstance(frozen_payload, Mapping):
        raise ValueError("frozen model artifact must be a JSON object")
    frozen = LinearRanker.from_json(frozen_payload)
    if frozen.feature_names != TYPED_ENHANCED_NO_UNKNOWN_FEATURE_NAMES:
        raise ValueError("frozen model must be the accepted 6508e87 ranker")

    training, validation = split_query_groups(original, 0.25, seed)
    training_ids = {group.query_id for group in training}
    validation_ids = {group.query_id for group in validation}
    if training_ids & validation_ids or len(training) != 750 or len(validation) != 250:
        raise ValueError("expected the deterministic 750/250 query-disjoint split")
    expanded_by_id = {group.query_id: group for group in expanded}
    expanded_training = [expanded_by_id[group.query_id] for group in training]
    expanded_validation = [expanded_by_id[group.query_id] for group in validation]
    original_by_id = {group.query_id: group for group in original}
    training_original = [original_by_id[group.query_id] for group in training]

    baseline_refit, tune = _fit_only_tune(
        training, grades, targets, seed=seed, include_unknown=False
    )
    frozen_values = _per_query_ndcg(validation, grades, targets, frozen)
    refit_values = _per_query_ndcg(validation, grades, targets, baseline_refit)
    baseline_report = {
        "frozen_6508e87": evaluate(
            validation, grades, frozen, ideal_grades=targets
        ),
        "same_recipe_refit_original_union": evaluate(
            validation, grades, baseline_refit, ideal_grades=targets
        ),
        "paired_refit_minus_frozen_95_percent": _paired_bootstrap(
            refit_values, frozen_values
        ),
        "same_recipe_refit_l2_selected_on_fit_only": tune["selected_l2"],
        "local_latency_ms_per_query": {
            "frozen_p50_p95": _latency_summary(validation, frozen),
            "refit_p50_p95": _latency_summary(validation, baseline_refit),
        },
    }

    fit_additions = {
        query_id: docs for query_id, docs in novel_by_query.items() if query_id in training_ids
    }
    fit_passage = sum(len(passage[group.query_id]) for group in training)
    fit_overlap = sum(
        len(passage[group.query_id])
        - len(fit_additions.get(group.query_id, set()))
        for group in training
    )
    fit_novel_count = sum(map(len, fit_additions.values()))
    fit_original_count = sum(len(group.candidates) for group in training)
    fit_expanded = [expanded_by_id[group.query_id] for group in training]
    fit_target_recovery = _paired_target_recovery(
        fit_expanded, targets, novel_by_query
    )
    fit_original_target_ceiling = candidate_oracle_ndcg(
        training_original, grades, ideal_grades=targets
    )
    fit_expanded_target_ceiling = _target_oracle_ndcg(
        fit_expanded, augmented_grades, targets
    )
    all_novel_ids = sum(map(len, novel_by_query.values()))
    fit_jaccards = []
    for group in training:
        original_ids = {candidate.doc_id for candidate in group.candidates}
        passage_ids = {doc_id for doc_id, _rank, _score in passage[group.query_id]}
        union_size = len(original_ids | passage_ids)
        fit_jaccards.append(
            len(original_ids & passage_ids) / union_size if union_size else 0.0
        )

    metadata_coverage: dict[str, int | float] = {
        "metadata_reads_per_eligible_query_required": sum(
            bool(doc_ids) for doc_ids in novel_by_query.values()
        ),
        "metadata_reads_completed": 0,
        "passage_only_ids_requiring_metadata": sum(map(len, novel_by_query.values())),
        "metadata_id_coverage": 0.0,
    }
    expanded_validation_metrics: dict[str, Any] | None = None
    fit_tuning: dict[str, Any] | None = None
    nominee: LinearRanker | None = None
    nominee_promising = False
    if metadata_path is not None:
        metadata_raw = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata_raw, Mapping):
            raise ValueError("metadata checkpoint must be an object")
        expanded = enrich_passage_only_candidates(
            expanded, novel_by_query, metadata_raw
        )
        expanded_by_id = {group.query_id: group for group in expanded}
        expanded_training = [expanded_by_id[group.query_id] for group in training]
        expanded_validation = [expanded_by_id[group.query_id] for group in validation]
        nominee, fit_tuning = _fit_expanded_ranker(
            expanded_training,
            augmented_grades,
            targets,
            seed=seed,
        )
        expanded_values = _per_query_ndcg(
            expanded_validation, augmented_grades, targets, nominee
        )
        expanded_validation_metrics = evaluate(
            expanded_validation,
            augmented_grades,
            nominee,
            ideal_grades=targets,
        )
        expanded_validation_metrics["paired_vs_frozen_95_percent"] = (
            _paired_bootstrap(expanded_values, frozen_values)
        )
        expanded_validation_metrics["local_latency_ms_per_query"] = (
            _latency_summary(expanded_validation, nominee)
        )
        expanded_validation_metrics["paired_vs_same_recipe_refit_95_percent"] = (
            _paired_bootstrap(expanded_values, refit_values)
        )
        frozen_ci = expanded_validation_metrics["paired_vs_frozen_95_percent"]
        refit_ci = expanded_validation_metrics[
            "paired_vs_same_recipe_refit_95_percent"
        ]
        nominee_promising = bool(
            expanded_validation_metrics["pairwise_linear_ndcg"]
            > max(
                baseline_report["frozen_6508e87"]["pairwise_linear_ndcg"],
                baseline_report["same_recipe_refit_original_union"][
                    "pairwise_linear_ndcg"
                ],
            )
            and frozen_ci["ci95_low"] > 0
            and refit_ci["ci95_low"] > 0
        )
        metadata_coverage = {
            "metadata_reads_per_eligible_query_required": sum(
                bool(doc_ids) for doc_ids in novel_by_query.values()
            ),
            "metadata_reads_completed": len(metadata_raw),
            "passage_only_ids_requiring_metadata": sum(map(len, novel_by_query.values())),
            "metadata_id_coverage": (
                sum(len(docs) for docs in metadata_raw.values()) / all_novel_ids
                if all_novel_ids
                else 1.0
            ),
        }

    source_digest = hashlib.sha256(passage_path.read_bytes()).hexdigest()
    max_ids_per_read = max(map(len, novel_by_query.values()), default=0)
    report: dict[str, Any] = {
        "provenance": {
            "passage_bm25_formulation_commit": PASSAGE_FORMULATION_COMMIT,
            "passage_csv_sha256": source_digest,
            "query_text_locale_verified_against_referee_training_task": True,
            "training_query_count": len(original),
            "passage_query_id_coverage": len(passage),
            "passage_queries_with_no_results": sum(
                not hits for hits in passage.values()
            ),
            "query_split": "seed 17, 750 fit / 250 validation; query-disjoint",
            "passage_ranks_scores_joined_by_exact_query_id": True,
            "original_dense_lexical_ranks_scores_preserved": True,
            "turbopuffer_calls_made": 0,
            "referee_calls_made_by_this_module": 0,
            "novel_candidate_judgments": {
                "source": debug_manifest["source"],
                "token_scope": debug_manifest["token_scope"],
                "debug_batches": debug_manifest["debug_batches"],
                "batch_query_limit": debug_manifest["query_batch_size"],
                "candidate_pair_coverage": (
                    debug_manifest["novel_candidate_pairs_returned"]
                    / debug_manifest["novel_candidate_pairs_requested"]
                    if debug_manifest["novel_candidate_pairs_requested"]
                    else 1.0
                ),
                "positive_judgments": positive_novel_judgments,
                "zero_or_unjudged_ambiguous_ignored": (
                    debug_manifest["zero_or_unjudged_ambiguous_ignored"]
                ),
                "unjudged_used_as_training_labels": False,
            },
        },
        "candidate_coverage": {
            "total_passage_only_ids": all_novel_ids,
            "queries_with_passage_only_ids": sum(
                bool(doc_ids) for doc_ids in novel_by_query.values()
            ),
            "queries_without_passage_only_ids": sum(
                not doc_ids for doc_ids in novel_by_query.values()
            ),
            "passage_only_ids_max_per_query": max_ids_per_read,
            "id_filtered_metadata_request_bound": PASSAGE_CANDIDATE_DEPTH,
            "fit_original_union_pairs": fit_original_count,
            "fit_passage_pairs": fit_passage,
            "fit_overlap_pairs": fit_overlap,
            "fit_passage_only_pairs": fit_novel_count,
            "fit_passage_only_unique_doc_ids": len(
                set().union(*fit_additions.values()) if fit_additions else set()
            ),
            "fit_query_mean_jaccard": statistics.mean(fit_jaccards),
            "fit_passage_only_fraction_of_expanded_pool": (
                fit_novel_count / (fit_original_count + fit_novel_count)
                if fit_original_count + fit_novel_count
                else 0.0
            ),
        },
        "fit_only_diagnostics": {
            "judged_top_target_recovery": fit_target_recovery,
            "judged_gain_ceiling_ndcg_original_union": fit_original_target_ceiling,
            "judged_gain_ceiling_ndcg_expanded": fit_expanded_target_ceiling,
            "judged_gain_ceiling_method": (
                "fit-only candidate oracle over known positive grades; "
                "targets define IDCG; missing labels add no gain and are not "
                "training negatives"
            ),
            "positive_novel_judgments": sum(
                len(labels)
                for query_id, labels in extra_judgments.items()
                if query_id in training_ids
            ),
            "novel_debug_zero_or_unjudged_used_as_training_labels": 0,
        },
        "score_source_consistency": _rank_score_provenance(
            training_original, passage
        ),
        "original_union_validation": baseline_report,
        "expanded_validation_ndcg_at_10": expanded_validation_metrics,
        "expanded_fit_only_tuning": fit_tuning,
        "id_filtered_metadata_coverage": metadata_coverage,
        "expanded_ranking_evaluated": metadata_path is not None,
        "expanded_nominee_promising_on_training_validation": nominee_promising,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if nominee is not None and nominee_promising:
        output_path.with_name("passage_expanded_ranker_weights.json").write_text(
            json.dumps(nominee.to_json(), indent=2) + "\n", encoding="utf-8"
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--grades", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--passage-results", type=Path, required=True)
    parser.add_argument("--referee-task", type=Path, required=True)
    parser.add_argument("--novel-judgments", type=Path, required=True)
    parser.add_argument("--debug-manifest", type=Path, required=True)
    parser.add_argument(
        "--frozen-model",
        type=Path,
        default=Path("models/esci_pairwise_ranker_typed_computed_no_unknown_training.json"),
    )
    parser.add_argument("--metadata-checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    try:
        report = run_experiment(
            args.candidates,
            args.grades,
            args.targets,
            args.passage_results,
            args.referee_task,
            args.novel_judgments,
            args.debug_manifest,
            args.frozen_model,
            args.output,
            metadata_path=args.metadata_checkpoint,
            seed=args.seed,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "training_queries": report["provenance"]["training_query_count"],
                "fit_candidate_coverage": report["candidate_coverage"],
                "fit_only_diagnostics": report["fit_only_diagnostics"],
                "original_union_validation": report["original_union_validation"],
                "expanded_ranking_evaluated": report["expanded_ranking_evaluated"],
                "id_filtered_metadata_coverage": report["id_filtered_metadata_coverage"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
