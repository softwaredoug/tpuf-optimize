"""Training-only typed/computed ranker comparison on the pinned cached fold."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tpuf_optimize.computed_feature_rank import (
    ENHANCED_FEATURE_NAMES,
    LinearRanker,
    QueryCandidates,
    evaluate,
    fit_ranker,
    model_ranking,
    ndcg,
    read_candidates,
    read_grades,
    read_targets,
    split_query_groups,
)


def _per_query_ndcg(
    groups: Sequence[QueryCandidates],
    grades: Mapping[str, Mapping[str, float]],
    targets: Mapping[str, Mapping[str, float]],
    model: LinearRanker,
) -> dict[str, float]:
    values: dict[str, float] = {}
    for group in groups:
        score = ndcg(
            model_ranking(group, model),
            grades.get(group.query_id, {}),
            ideal_grades=targets.get(group.query_id, {}),
        )
        if score is not None:
            values[group.query_id] = score
    return values


def _paired_bootstrap(
    left: Mapping[str, float],
    right: Mapping[str, float],
    *,
    seed: int = 1701,
    samples: int = 10_000,
) -> dict[str, float | int]:
    shared = sorted(left.keys() & right.keys())
    if not shared:
        raise ValueError("paired comparison has no eligible validation queries")
    differences = [left[query_id] - right[query_id] for query_id in shared]
    rng = random.Random(seed)
    means = [
        sum(differences[rng.randrange(len(differences))] for _ in differences)
        / len(differences)
        for _ in range(samples)
    ]
    means.sort()

    def percentile(fraction: float) -> float:
        position = (len(means) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        return means[lower] + (means[upper] - means[lower]) * (position - lower)

    return {
        "paired_queries": len(shared),
        "mean_delta": sum(differences) / len(differences),
        "ci95_low": percentile(0.025),
        "ci95_high": percentile(0.975),
    }


def _latency_summary(
    groups: Sequence[QueryCandidates],
    model: LinearRanker,
    *,
    repetitions: int = 3,
) -> dict[str, float | int | str]:
    """Measure local feature extraction + ranking only, on cached candidates."""
    per_query_ms: list[float] = []
    for group in groups:
        model_ranking(group, model)  # warm-up; excluded
        timings = []
        for _ in range(repetitions):
            start = time.perf_counter_ns()
            model_ranking(group, model)
            timings.append((time.perf_counter_ns() - start) / 1_000_000)
        per_query_ms.append(statistics.median(timings))
    ordered = sorted(per_query_ms)

    def percentile(fraction: float) -> float:
        return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]

    return {
        "measurement": "local feature extraction + ranking; cached candidates; not end-to-end",
        "queries": len(ordered),
        "median_ms_per_query": statistics.median(ordered),
        "p50_ms_per_query": percentile(0.50),
        "p95_ms_per_query": percentile(0.95),
        "repetitions_per_query": repetitions,
    }


def _fit_only_tune(
    outer_training: Sequence[QueryCandidates],
    grades: Mapping[str, Mapping[str, float]],
    targets: Mapping[str, Mapping[str, float]],
    *,
    seed: int,
    include_unknown: bool,
) -> tuple[LinearRanker, dict[str, Any]]:
    """Select L2 on an inner query split entirely inside the outer fit fold."""
    inner_training, inner_validation = split_query_groups(
        outer_training, 0.20, seed=seed + 2026
    )
    selected: tuple[float, LinearRanker, float, float] | None = None
    for l2 in (0.001, 0.01, 0.05, 0.1, 0.3):
        model = fit_ranker(
            inner_training,
            grades,
            seed=seed,
            epochs=500,
            learning_rate=0.1,
            l2=l2,
            include_computed=True,
            include_typed=True,
            include_typed_unknown=include_unknown,
        )
        metrics = evaluate(
            inner_validation, grades, model, ideal_grades=targets
        )
        score = metrics["pairwise_linear_ndcg"]
        if not isinstance(score, (int, float)):
            continue
        if selected is None or score > selected[0]:
            selected = (float(score), model, l2, 0.1)
    if selected is None:
        raise ValueError("fit-only tuning found no eligible inner validation queries")
    _, _inner_model, best_l2, best_learning_rate = selected
    final_model = fit_ranker(
        outer_training,
        grades,
        seed=seed,
        epochs=500,
        learning_rate=best_learning_rate,
        l2=best_l2,
        include_computed=True,
        include_typed=True,
        include_typed_unknown=include_unknown,
    )
    return final_model, {
        "fit_only_tuning": True,
        "inner_training_queries": len(inner_training),
        "inner_validation_queries": len(inner_validation),
        "selected_l2": best_l2,
        "selected_learning_rate": best_learning_rate,
        "epochs": 500,
        "selection_metric": "inner fit-fold NDCG@10",
        "unknown_feature_included": include_unknown,
    }


def run_experiment(
    candidates_path: Path,
    grades_path: Path,
    targets_path: Path,
    frozen_model_path: Path,
    output_dir: Path,
    *,
    seed: int = 17,
) -> dict[str, Any]:
    groups = read_candidates(candidates_path)
    grades = read_grades(grades_path)
    targets = read_targets(targets_path)
    payload = json.loads(frozen_model_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("frozen model artifact must be a JSON object")
    frozen = LinearRanker.from_json(payload)
    if frozen.feature_names != ENHANCED_FEATURE_NAMES:
        raise ValueError("frozen model must be the pinned computed-feature model")

    training, validation = split_query_groups(groups, 0.25, seed)
    computed_refit = fit_ranker(
        training, grades, seed=seed, include_computed=True
    )
    typed_computed_full, tuning_full = _fit_only_tune(
        training, grades, targets, seed=seed, include_unknown=True
    )
    typed_computed_no_unknown, tuning_no_unknown = _fit_only_tune(
        training, grades, targets, seed=seed, include_unknown=False
    )
    # This fit-only diagnostic isolates whether the typed family is useful
    # without the computed BM25/coverage family.
    typed_only = fit_ranker(training, grades, seed=seed, include_typed=True)

    models = {
        "frozen_e64100d": frozen,
        "same_fold_computed_refit": computed_refit,
        "same_fold_typed_computed_refit": typed_computed_full,
        "same_fold_typed_computed_no_unknown_refit": typed_computed_no_unknown,
        "same_fold_typed_only_refit": typed_only,
    }
    validation_grades = {
        "frozen_e64100d": _per_query_ndcg(validation, grades, targets, frozen),
        "same_fold_computed_refit": _per_query_ndcg(
            validation, grades, targets, computed_refit
        ),
        "same_fold_typed_computed_refit": _per_query_ndcg(
            validation, grades, targets, typed_computed_full
        ),
        "same_fold_typed_computed_no_unknown_refit": _per_query_ndcg(
            validation, grades, targets, typed_computed_no_unknown
        ),
        "same_fold_typed_only_refit": _per_query_ndcg(
            validation, grades, targets, typed_only
        ),
    }
    # Feature ablation at inference keeps the fitted typed+computed weights
    # fixed and removes only the unknown/missing-state contribution.
    weights_without_unknown = list(typed_computed_full.weights)
    unknown_index = typed_computed_full.feature_names.index("typed_unknown")
    weights_without_unknown[unknown_index] = 0.0
    no_unknown_contribution = LinearRanker(
        tuple(weights_without_unknown), typed_computed_full.feature_names
    )
    validation_grades["typed_computed_without_unknown_contribution"] = _per_query_ndcg(
        validation, grades, targets, no_unknown_contribution
    )

    summaries = {
        name: evaluate(validation, grades, model, ideal_grades=targets)
        for name, model in models.items()
    }
    summaries["same_fold_typed_computed_no_unknown_refit"] = evaluate(
        validation, grades, typed_computed_no_unknown, ideal_grades=targets
    )
    summaries["typed_computed_without_unknown_contribution"] = evaluate(
        validation, grades, no_unknown_contribution, ideal_grades=targets
    )
    paired = {
        "typed_computed_vs_frozen": _paired_bootstrap(
            validation_grades["same_fold_typed_computed_refit"],
            validation_grades["frozen_e64100d"],
        ),
        "typed_computed_vs_computed_refit": _paired_bootstrap(
            validation_grades["same_fold_typed_computed_refit"],
            validation_grades["same_fold_computed_refit"],
        ),
        "typed_computed_no_unknown_vs_frozen": _paired_bootstrap(
            validation_grades["same_fold_typed_computed_no_unknown_refit"],
            validation_grades["frozen_e64100d"],
        ),
        "typed_computed_no_unknown_vs_computed_refit": _paired_bootstrap(
            validation_grades["same_fold_typed_computed_no_unknown_refit"],
            validation_grades["same_fold_computed_refit"],
        ),
        "typed_computed_vs_without_unknown_contribution": _paired_bootstrap(
            validation_grades["same_fold_typed_computed_refit"],
            validation_grades["typed_computed_without_unknown_contribution"],
        ),
    }
    report: dict[str, Any] = {
        "fold": {
            "seed": seed,
            "training_queries": len(training),
            "validation_queries": len(validation),
            "query_disjoint": not (
                {group.query_id for group in training}
                & {group.query_id for group in validation}
            ),
            "identical_validation_candidate_groups": True,
            "all_expected_validation_queries": len(validation) == 250,
        },
        "typed_model_tuning": {
            "with_unknown_state": tuning_full,
            "without_unknown_state": tuning_no_unknown,
        },
        "validation_ndcg_at_10": summaries,
        "paired_query_bootstrap_95_percent": paired,
        "local_latency": {
            "frozen_e64100d": _latency_summary(validation, frozen),
            "typed_computed_with_unknown": _latency_summary(
                validation, typed_computed_full
            ),
            "typed_computed_without_unknown": _latency_summary(
                validation, typed_computed_no_unknown
            ),
        },
        "feature_isolation": {
            "typed_family_only": summaries["same_fold_typed_only_refit"],
            "typed_missing_unknown_contribution_removed": summaries[
                "typed_computed_without_unknown_contribution"
            ],
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "aggregate_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    # Generalizable weights only; no IDs, queries, candidate rows or labels.
    (output_dir / "typed_computed_with_unknown_ranker_weights.json").write_text(
        json.dumps(typed_computed_full.to_json(), indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "typed_computed_no_unknown_ranker_weights.json").write_text(
        json.dumps(typed_computed_no_unknown.to_json(), indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--grades", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument(
        "--frozen-model",
        type=Path,
        default=Path("models/esci_pairwise_ranker_computed_features_training.json"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    try:
        report = run_experiment(
            args.candidates,
            args.grades,
            args.targets,
            args.frozen_model,
            args.output_dir,
            seed=args.seed,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
