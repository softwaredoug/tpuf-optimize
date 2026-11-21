"""Local lexical/dense score fusion experiments on cached ESCI candidates.

No Turbopuffer client or index is used here. Dense scores are distances (lower
is better); lexical scores are BM25-style scores (higher is better).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


DEFAULT_RRF_CONSTANT = 60
RRF_DENSE_WEIGHT = 2
RRF_LEXICAL_WEIGHT = 1


def _score_map(
    entries: Sequence[Mapping[str, Any]],
    *,
    lower_is_better: bool,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for entry in entries:
        if "doc_id" not in entry or "score" not in entry:
            raise ValueError("Each candidate must contain doc_id and score")
        document_id = str(entry["doc_id"])
        if document_id in scores:
            raise ValueError(f"Duplicate candidate document {document_id!r}")
        try:
            score = float(entry["score"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid score for document {document_id!r}") from error
        if not math.isfinite(score):
            raise ValueError(f"Non-finite score for document {document_id!r}")
        scores[document_id] = -score if lower_is_better else score
    return scores


def _quantile(sorted_values: Sequence[float], fraction: float) -> float:
    """Linear-interpolated quantile (matching common percentile definitions)."""
    if not sorted_values:
        raise ValueError("Cannot calculate a quantile of no scores")
    position = (len(sorted_values) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    remainder = position - lower
    return sorted_values[lower] + remainder * (
        sorted_values[upper] - sorted_values[lower]
    )


def normalize_scores(
    scores: Mapping[str, float],
    method: str,
    *,
    lower_percentile: float = 0.05,
    upper_percentile: float = 0.95,
) -> dict[str, float]:
    """Normalize higher-is-better scores to [0, 1].

    A candidate missing from a retriever has no score in that retriever and
    receives 0 after normalization. Robust normalization clips the observed
    score distribution at the requested percentiles before min-max scaling.
    """
    if not scores:
        return {}
    values = list(scores.values())
    if method == "minmax":
        low, high = min(values), max(values)
    elif method == "percentile":
        if not 0 <= lower_percentile < upper_percentile <= 1:
            raise ValueError("Percentiles must satisfy 0 <= low < high <= 1")
        ordered = sorted(values)
        low = _quantile(ordered, lower_percentile)
        high = _quantile(ordered, upper_percentile)
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    width = high - low
    if width <= 0:
        return {document_id: 0.0 for document_id in scores}
    return {
        document_id: min(1.0, max(0.0, (value - low) / width))
        for document_id, value in scores.items()
    }


def fuse_convex(
    dense: Sequence[Mapping[str, Any]],
    lexical: Sequence[Mapping[str, Any]],
    *,
    dense_weight: float,
    normalization: str,
    missing_score: str = "zero",
) -> list[tuple[str, float]]:
    """Fuse normalized scores with zero-floor or mean missing-score treatment."""
    if not 0 <= dense_weight <= 1:
        raise ValueError("dense_weight must be in [0, 1]")
    if missing_score not in {"zero", "mean"}:
        raise ValueError("missing_score must be 'zero' or 'mean'")
    dense_scores = _score_map(dense, lower_is_better=True)
    lexical_scores = _score_map(lexical, lower_is_better=False)
    dense_normalized = normalize_scores(dense_scores, normalization)
    lexical_normalized = normalize_scores(lexical_scores, normalization)
    dense_missing = (
        statistics.fmean(dense_normalized.values())
        if missing_score == "mean" and dense_normalized
        else 0.0
    )
    lexical_missing = (
        statistics.fmean(lexical_normalized.values())
        if missing_score == "mean" and lexical_normalized
        else 0.0
    )
    document_ids = set(dense_scores) | set(lexical_scores)
    combined = {
        document_id: (
            dense_weight * dense_normalized.get(document_id, dense_missing)
            + (1.0 - dense_weight)
            * lexical_normalized.get(document_id, lexical_missing)
        )
        for document_id in document_ids
    }
    return sorted(combined.items(), key=lambda item: (-item[1], item[0]))


def fuse_rrf(
    dense: Sequence[Mapping[str, Any]],
    lexical: Sequence[Mapping[str, Any]],
    *,
    constant: int = DEFAULT_RRF_CONSTANT,
) -> list[tuple[str, float]]:
    """Weighted RRF with exact 2:1 dense-to-lexical contributions."""
    if constant < 0:
        raise ValueError("RRF constant must be non-negative")
    dense_scores = _score_map(dense, lower_is_better=True)
    lexical_scores = _score_map(lexical, lower_is_better=False)
    dense_ranked = sorted(dense_scores, key=lambda doc_id: (-dense_scores[doc_id], doc_id))
    lexical_ranked = sorted(
        lexical_scores, key=lambda doc_id: (-lexical_scores[doc_id], doc_id)
    )
    combined: dict[str, float] = {}
    for rank, document_id in enumerate(dense_ranked, start=1):
        combined[document_id] = RRF_DENSE_WEIGHT / (constant + rank)
    for rank, document_id in enumerate(lexical_ranked, start=1):
        combined[document_id] = combined.get(document_id, 0.0) + (
            RRF_LEXICAL_WEIGHT / (constant + rank)
        )
    return sorted(combined.items(), key=lambda item: (-item[1], item[0]))


def load_candidates(path: Path) -> dict[str, dict[str, Any]]:
    """Load collector JSONL while preserving opaque query IDs as strings."""
    queries: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as candidates_file:
        for line_number, line in enumerate(candidates_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            required = {"query_id", "query_text", "locale", "dense", "lexical"}
            if not required.issubset(record):
                raise ValueError(
                    f"{path}:{line_number}: missing "
                    f"{', '.join(sorted(required - set(record)))}"
                )
            query_id = str(record["query_id"])
            if query_id in queries:
                raise ValueError(f"Duplicate query_id {query_id!r}")
            queries[query_id] = record
    return queries


def rankings_for(
    records: Mapping[str, Mapping[str, Any]],
    method: str,
    *,
    dense_weight: float = 0.5,
    normalization: str = "percentile",
    missing_score: str = "zero",
) -> dict[str, list[tuple[str, float]]]:
    """Rank every cached query using one fusion configuration."""
    rankings = {}
    for query_id, record in records.items():
        if method == "rrf":
            ranking = fuse_rrf(record["dense"], record["lexical"])
        elif method == "convex":
            ranking = fuse_convex(
                record["dense"],
                record["lexical"],
                dense_weight=dense_weight,
                normalization=normalization,
                missing_score=missing_score,
            )
        else:
            raise ValueError(f"Unknown fusion method: {method}")
        rankings[query_id] = ranking
    return rankings


def _request_json(
    url: str,
    token: str,
    payload: Mapping[str, Any] | None = None,
    *,
    method: str = "POST",
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Referee returned HTTP {error.code}: {detail}") from error


def evaluate(
    records: Mapping[str, Mapping[str, Any]],
    rankings: Mapping[str, Sequence[tuple[str, float]]],
    *,
    token: str,
    referee_url: str,
    output_path: Path,
) -> dict[str, Any]:
    """Create a training task, verify cache alignment, and submit its rankings."""
    base_url = referee_url.rstrip("/")
    task_response = _request_json(
        f"{base_url}/v1/tasks", token, method="POST"
    )
    task_id = str(task_response["task"]["id"])
    queries = task_response["queries"]
    task_ids = {str(query["query_id"]) for query in queries}
    cache_ids = set(records)
    if task_ids != cache_ids:
        raise ValueError(
            "Candidate cache and referee task query IDs differ "
            f"(task={len(task_ids)}, cache={len(cache_ids)})"
        )

    for query in queries:
        query_id = str(query["query_id"])
        cached = records[query_id]
        if str(query["query_text"]) != str(cached["query_text"]):
            raise ValueError(f"Cached query text mismatch for {query_id}")

    results: list[dict[str, Any]] = []
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["query_id", "doc_id", "rank", "score"])
        for query in queries:
            query_id = str(query["query_id"])
            rows = rankings[query_id]
            if not rows:
                writer.writerow([query_id, "-1", -1, -1])
                results.append(
                    {"query_id": query_id, "doc_id": "-1", "rank": -1, "score": -1}
                )
                continue
            for rank, (document_id, score) in enumerate(rows, start=1):
                writer.writerow([query_id, document_id, rank, score])
                results.append(
                    {
                        "query_id": query_id,
                        "doc_id": document_id,
                        "rank": rank,
                        "score": score,
                    }
                )

    return _request_json(
        f"{base_url}/v1/tasks/{task_id}/complete", token, {"results": results}
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--referee-url", default="http://localhost:8080")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "rrf",
            "minmax-0.25",
            "minmax-0.5",
            "minmax-0.75",
            "percentile-0.25",
            "percentile-0.5",
            "percentile-0.75",
            "minmax-0.5-mean",
            "percentile-0.5-mean",
        ],
        help=(
            "RRF and convex weights; suffix -mean tests mean normalized "
            "scores for a missing retriever."
        ),
    )
    return parser


def _method_config(method: str) -> tuple[str, float, str, str]:
    if method == "rrf":
        return "rrf", 0.5, "percentile", "zero"
    missing_score = "mean" if method.endswith("-mean") else "zero"
    base_method = method.removesuffix("-mean")
    try:
        normalization, weight = base_method.rsplit("-", 1)
        dense_weight = float(weight)
    except ValueError as error:
        raise ValueError(f"Invalid method name {method!r}") from error
    if normalization not in {"minmax", "percentile"}:
        raise ValueError(f"Unknown normalization in {method!r}")
    if not 0 <= dense_weight <= 1:
        raise ValueError(f"Dense weight in {method!r} must be in [0, 1]")
    return "convex", dense_weight, normalization, missing_score


def main(argv: Iterable[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not args.token_file.is_file():
        parser.error(f"Token file does not exist: {args.token_file}")
    token = args.token_file.read_text(encoding="utf-8").strip()
    if not token:
        parser.error("Referee token file is empty")
    try:
        records = load_candidates(args.candidates)
        if not records:
            raise ValueError("Candidate cache is empty")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for method in args.methods:
            kind, dense_weight, normalization, missing_score = _method_config(method)
            method_rankings = rankings_for(
                records,
                kind,
                dense_weight=dense_weight,
                normalization=normalization,
                missing_score=missing_score,
            )
            response = evaluate(
                records,
                method_rankings,
                token=token,
                referee_url=args.referee_url,
                output_path=args.output_dir / f"{method}.csv",
            )
            metric = response["metrics"]
            print(
                json.dumps(
                    {
                        "method": method,
                        "metrics": metric,
                        "queries": response["queries"],
                    }
                )
            )
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
