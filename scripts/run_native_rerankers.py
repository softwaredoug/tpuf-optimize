"""Training-only benchmark for Turbopuffer native rerank_by models.

Task queries, rankings, credentials, and per-query metrics stay in a mode-0700
directory outside the repository. This runner deliberately has no result or
candidate cache: every invocation executes the requested API queries anew.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, cast

from turbopuffer import APIError, APIStatusError, Turbopuffer
from turbopuffer._exceptions import RateLimitError
from turbopuffer.types import Row

from tpuf_optimize.native_rerank import (
    NATIVE_RERANK_MODELS,
    RERANK_TOP_K,
    build_submission_rows,
    candidate_queries,
    local_identifier_gated_rrf,
    native_rerank_by,
    paired_delta_summary,
    passage_chars,
    response_performance,
    result_rows,
)
from tpuf_optimize.searcher import SearchRequest
from tpuf_optimize.strategy import TPufStrategy

REFEREE_URL = "http://localhost:8080"
DEFAULT_TOKEN = Path(
    "/private/var/folders/ww/t2bpzntd1990wczd0b7px2640000gn/T/"
    "opencode/esci-search-agent-training-eval-1000/referee.token"
)
DEFAULT_REFERENCE = Path(
    "/private/var/folders/ww/t2bpzntd1990wczd0b7px2640000gn/T/"
    "opencode/esci-search-agent-training-eval-1000/reference-task.json"
)
RETRY_BUDGET_SECONDS = 300.0
RETRY_BASE_SECONDS = 1.0


class RefereeError(RuntimeError):
    """A sanitized error from the local Referee service."""


class RequestFailure(RuntimeError):
    """A failed Turbopuffer request with its retry count."""

    def __init__(self, retries: int, status_code: int | None) -> None:
        super().__init__("Turbopuffer request failed.")
        self.retries = retries
        self.status_code = status_code


def _private_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    temporary.replace(path)


def _read_reference(path: Path) -> tuple[dict[str, object], list[SearchRequest]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("dataset_name") != "esci" or payload.get("task", {}).get(
        "eval_mode"
    ) != "ndcg":
        raise ValueError("The provided reference task is not ESCI NDCG.")
    queries = payload.get("queries")
    if not isinstance(queries, list) or len(queries) != 1000:
        raise ValueError("Expected exactly 1,000 queries in the assigned task.")
    requests = [_request_from_query(query) for query in queries]
    if len({request.query_id for request in requests}) != 1000:
        raise ValueError("The assigned task has duplicate query IDs.")
    return cast(dict[str, object], payload), requests


def _request_from_query(query: Mapping[str, object]) -> SearchRequest:
    query_id = query.get("query_id")
    query_text = query.get("query_text")
    metadata = query.get("metadata")
    locale = metadata.get("locale") if isinstance(metadata, dict) else None
    if not all(isinstance(value, str) and value for value in (
        query_id,
        query_text,
        locale,
    )):
        raise ValueError("A Referee query is missing its ID, text, or locale.")
    return SearchRequest(str(query_id), str(query_text), str(locale))


def _triples(requests: Sequence[SearchRequest]) -> list[tuple[str, str, str]]:
    return [(request.query_id, request.text, request.locale) for request in requests]


def _verify_same_training_queries(
    reference: Sequence[SearchRequest], task_payload: Mapping[str, object]
) -> list[SearchRequest]:
    raw_queries = task_payload.get("queries")
    if not isinstance(raw_queries, list):
        raise RefereeError("Referee task response did not contain a query list.")
    actual = [_request_from_query(query) for query in raw_queries]
    if _triples(actual) != _triples(reference):
        raise RefereeError(
            "New Referee task did not match the assigned training query IDs, "
            "texts, and locales exactly."
        )
    if len(actual) != 1000:
        raise RefereeError("New Referee task did not contain 1,000 queries.")
    return actual


def _referee_request(
    method: str, path: str, token: str, payload: object | None = None
) -> dict[str, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{REFEREE_URL}{path}",
        data=body,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            **({"Content-Type": "application/json"} if body is not None else {}),
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return cast(dict[str, Any], json.loads(response.read()))
    except urllib.error.HTTPError as error:
        # Never include response bodies: task IDs / query data must remain private.
        raise RefereeError(f"Referee returned HTTP {error.code}.") from None
    except urllib.error.URLError:
        raise RefereeError("Referee could not be reached.") from None


def _create_tasks(
    token: str,
    run_dir: Path,
    reference_requests: Sequence[SearchRequest],
) -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    for name in ("baseline", *NATIVE_RERANK_MODELS):
        payload = _referee_request("POST", "/v1/tasks", token)
        _verify_same_training_queries(reference_requests, payload)
        task = payload.get("task")
        if not isinstance(task, dict) or not isinstance(task.get("id"), str):
            raise RefereeError("Referee returned an invalid task response.")
        tasks[name] = payload
        _private_json(run_dir / f"task-{name}.json", payload)
    return tasks


def _retry_after(error: RateLimitError) -> float:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {})
    value = headers.get("Retry-After")
    if value is None:
        return 0.0
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            at = parsedate_to_datetime(str(value))
            if at.tzinfo is None:
                at = at.replace(tzinfo=timezone.utc)
            return max(0.0, (at - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return 0.0


def _request_with_backoff(
    operation: Callable[[], Any],
) -> tuple[Any, float, int]:
    started = time.monotonic()
    attempt = 0
    while True:
        request_started = time.perf_counter()
        try:
            response = operation()
            elapsed_ms = (time.perf_counter() - request_started) * 1000.0
            return response, elapsed_ms, attempt
        except RateLimitError as error:
            elapsed = time.monotonic() - started
            remaining = RETRY_BUDGET_SECONDS - elapsed
            if remaining <= 0:
                raise RequestFailure(
                    attempt, getattr(error, "status_code", 429)
                ) from None
            exponential = min(RETRY_BASE_SECONDS * 2**attempt, 60.0)
            delay = max(random.uniform(exponential / 2, exponential), _retry_after(error))
            if delay >= remaining:
                time.sleep(remaining)
                raise RequestFailure(
                    attempt, getattr(error, "status_code", 429)
                ) from None
            time.sleep(delay)
            attempt += 1


def _latencies(values: Sequence[float | int | None]) -> dict[str, float | None]:
    sorted_values = sorted(float(value) for value in values if value is not None)
    if not sorted_values:
        return {"p50_ms": None, "p95_ms": None}

    def percentile(fraction: float) -> float:
        position = (len(sorted_values) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return sorted_values[lower]
        return sorted_values[lower] + (
            sorted_values[upper] - sorted_values[lower]
        ) * (position - lower)

    return {"p50_ms": percentile(0.5), "p95_ms": percentile(0.95)}


def _task_queries(task_payload: Mapping[str, object]) -> list[SearchRequest]:
    raw = task_payload.get("queries")
    if not isinstance(raw, list):
        raise RefereeError("Saved task response is missing query data.")
    return [_request_from_query(query) for query in raw]


def _run_variant(
    name: str,
    task_payload: Mapping[str, object],
    requests: Sequence[SearchRequest],
    namespace: Any,
    strategy: TPufStrategy,
    token: str,
    reference_requests: Sequence[SearchRequest],
    candidate_counts: Mapping[str, int] | None = None,
    candidate_chars: Mapping[str, int] | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    # Verify the actual saved task once more immediately before the work and
    # once more before submission, rather than assuming task creation was enough.
    if _triples(_task_queries(task_payload)) != _triples(reference_requests):
        raise RefereeError("Task query IDs, texts, or locales changed before execution.")
    task = task_payload.get("task")
    if not isinstance(task, dict) or not isinstance(task.get("id"), str):
        raise RefereeError("Saved task response has no task identifier.")

    task_rows: list[dict[str, str | int | float]] = []
    per_query_stats: list[dict[str, float | int | None]] = []
    query_scores: dict[str, float] = {}
    failures = 0
    request_attempts = 0
    retry_count = 0
    passage_char_total = 0
    token_proxy_chars_total = 0
    candidate_total = 0

    for request in requests:
        dense_query, lexical_query = candidate_queries(
            strategy, request.text, request.locale
        )
        request_was_recorded = False
        try:
            if name == "baseline":
                response, elapsed_ms, retries = _request_with_backoff(
                    lambda: namespace.multi_query(
                        queries=[dense_query, lexical_query]
                    )
                )
                request_attempts += retries + 1
                retry_count += retries
                request_was_recorded = True
                if len(response.results) != 2:
                    raise ValueError("Baseline multi-query response was incomplete.")
                dense_rows = response.results[0].rows or []
                lexical_rows = response.results[1].rows or []
                ranked = local_identifier_gated_rrf(
                    request.text, dense_rows, lexical_rows
                )
                ranked_rows = [(doc_id, score) for doc_id, score in ranked]
                union: dict[str, Row] = {}
                for row in (*dense_rows, *lexical_rows):
                    union.setdefault(str(row.id), row)
                count = len(union)
                chars = passage_chars(list(union.values()))
            else:
                assert name in NATIVE_RERANK_MODELS
                model = NATIVE_RERANK_MODELS[name]
                response, elapsed_ms, retries = _request_with_backoff(
                    lambda: namespace.multi_query(
                        queries=[dense_query, lexical_query],
                        rerank_by=native_rerank_by(model, request.text),
                        limit=RERANK_TOP_K,
                    )
                )
                request_attempts += retries + 1
                retry_count += retries
                request_was_recorded = True
                if len(response.results) != 1:
                    raise ValueError("Reranked multi-query response was incomplete.")
                ranked_rows = result_rows(response.results[0].rows or [])
                if candidate_counts is None or candidate_chars is None:
                    raise ValueError("Native rerank candidate statistics are unavailable.")
                count = candidate_counts[request.query_id]
                chars = candidate_chars[request.query_id]
            task_rows.extend(build_submission_rows(request.query_id, ranked_rows))
            performance = response_performance(response)
            per_query_stats.append(
                {
                    "client_ms": elapsed_ms,
                    "server_total_ms": performance["server_total_ms"],
                    "query_execution_ms": performance["query_execution_ms"],
                    "retries": retries,
                    "candidate_count": count,
                }
            )
            if name == "baseline":
                assert candidate_counts is not None
                assert candidate_chars is not None
                # This same frozen first stage supplies the model candidate
                # count and a conservative text-size proxy without extra calls.
                cast(dict[str, int], candidate_counts)[request.query_id] = count
                cast(dict[str, int], candidate_chars)[request.query_id] = chars
            if name != "baseline":
                token_proxy_chars_total += (
                    chars + len(request.text) * count
                )
            passage_char_total += chars
            candidate_total += count
        except (APIError, TimeoutError, OSError, RequestFailure) as error:
            # No sentinel is emitted for failures. They are reported and the
            # incomplete task is not submitted to Referee.
            failures += 1
            retries = error.retries if isinstance(error, RequestFailure) else 0
            if not request_was_recorded:
                request_attempts += retries + 1
                retry_count += retries
            per_query_stats.append(
                {
                    "client_ms": None,
                    "server_total_ms": None,
                    "query_execution_ms": None,
                    "retries": 0,
                    "candidate_count": 0,
                }
            )
            continue

    if failures:
        return (
            _variant_summary(
                name,
                task_rows,
                per_query_stats,
                request_attempts,
                retry_count,
                failures,
                candidate_total,
                len(requests),
                passage_char_total,
                token_proxy_chars_total,
            ),
            query_scores,
        )
    if len(task_rows) < len(requests):
        # Each query contributes at least one row, including a true no-hit
        # sentinel. This check catches accidental omission without fabricating.
        raise ValueError("Generated Referee rows omitted at least one query.")
    if _triples(_task_queries(task_payload)) != _triples(reference_requests):
        raise RefereeError("Task no longer matches the assigned training split.")
    if {
        str(row["query_id"]) for row in task_rows
    } != {request.query_id for request in requests}:
        raise ValueError("Submission rows do not exactly cover task query IDs.")

    _private_csv(Path(os.environ["TPUF_NATIVE_RUN_DIR"]) / f"{name}.csv", task_rows)
    response = _referee_request(
        "POST",
        f"/v1/tasks/{task['id']}/complete",
        token,
        {"results": task_rows},
    )
    metric = response.get("metrics", {}).get("ndcg")
    raw_scores = response.get("queries")
    if not isinstance(metric, (int, float)) or not isinstance(raw_scores, list):
        raise RefereeError("Referee completion response did not contain NDCG scores.")
    for item in raw_scores:
        if not isinstance(item, dict):
            raise RefereeError("Referee returned an invalid per-query metric.")
        query_id = item.get("query_id")
        score = item.get("ndcg")
        if not isinstance(query_id, str) or not isinstance(score, (int, float)):
            raise RefereeError("Referee returned an incomplete per-query metric.")
        query_scores[query_id] = float(score)
    if query_scores.keys() != {request.query_id for request in requests}:
        raise RefereeError("Referee per-query scores did not cover all task queries.")

    summary = _variant_summary(
        name,
        task_rows,
        per_query_stats,
        request_attempts,
        retry_count,
        failures,
        candidate_total,
        len(requests),
        passage_char_total,
        token_proxy_chars_total,
    )
    summary["ndcg_at_10"] = float(metric)
    summary["submitted_query_count"] = len(requests)
    return summary, query_scores


def _variant_summary(
    name: str,
    task_rows: Sequence[Mapping[str, object]],
    stats: Sequence[Mapping[str, float | int | None]],
    request_attempts: int,
    retries: int,
    failures: int,
    candidate_total: int,
    query_count: int,
    passage_chars_sum: int,
    token_proxy_chars: int,
) -> dict[str, Any]:
    successful = [
        item for item in stats if item.get("client_ms") is not None
    ]
    result: dict[str, Any] = {
        "variant": name,
        "query_count": query_count,
        "completed_query_count": len(successful),
        "submitted_query_count": 0,
        "ndcg_at_10": None,
        "latency": _latencies(
            [cast(float | int | None, item.get("client_ms")) for item in successful]
        ),
        "server_total_latency": _latencies(
            [
                cast(float | int | None, item.get("server_total_ms"))
                for item in successful
            ]
        ),
        "query_execution_latency": _latencies(
            [
                cast(float | int | None, item.get("query_execution_ms"))
                for item in successful
            ]
        ),
        "logical_requests": len(stats),
        "request_attempts": request_attempts,
        "retry_count": retries,
        "failure_count": failures,
        "candidate_count": {
            "total": candidate_total,
            "mean": candidate_total / query_count if query_count else 0.0,
        },
        "server_rerank_timing": (
            "Not separately exposed; server_total_ms/query_execution_ms include "
            "the full native rerank request."
            if name != "baseline"
            else "Not applicable to local RRF baseline."
        ),
        "model_usage": {
            "billing_usage_reported": False,
            "estimated_input_tokens": (
                round(token_proxy_chars / 4)
                if name != "baseline"
                else None
            ),
            "token_estimate_method": (
                "Approximate chars/4 proxy from unique baseline candidate passage "
                "characters plus query characters once per candidate; not actual "
                "model tokenizer or usage."
                if name != "baseline"
                else None
            ),
            "cost_estimate": "Unavailable: no model token usage/billing metric returned.",
        },
    }
    # Aggregate row count only. No result rows, IDs, or per-query scores are
    # included in the committed report.
    result["submitted_result_row_count"] = len(task_rows)
    result["passage_character_count"] = passage_chars_sum or None
    return result


def _private_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        os.fchmod(output.fileno(), 0o600)
        writer = csv.DictWriter(
            output, fieldnames=("query_id", "doc_id", "rank", "score")
        )
        writer.writeheader()
        writer.writerows(rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-file", type=Path, default=DEFAULT_TOKEN)
    parser.add_argument("--reference-task", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(
            "/private/var/folders/ww/t2bpzntd1990wczd0b7px2640000gn/T/"
            "opencode/native-reranker-training"
        ),
    )
    parser.add_argument(
        "--aggregate-output",
        type=Path,
        help="Optional aggregate-only JSON report path, outside private task files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    run_dir = args.run_dir.expanduser().resolve()
    if run_dir == repo or repo in run_dir.parents:
        raise SystemExit("Private run directory must be outside the repository.")
    if not args.token_file.is_file() or not args.reference_task.is_file():
        raise SystemExit("The assigned training token or reference task is missing.")
    if not os.environ.get("TURBOPUFFER_API_KEY"):
        raise SystemExit("Set TURBOPUFFER_API_KEY before running this experiment.")
    run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    run_dir.chmod(0o700)
    os.environ["TPUF_NATIVE_RUN_DIR"] = str(run_dir)

    token = args.token_file.read_text(encoding="utf-8").strip()
    if not token:
        raise SystemExit("The assigned training token file is empty.")
    reference_payload, requests = _read_reference(args.reference_task)
    del reference_payload
    tasks = _create_tasks(token, run_dir, requests)
    candidate_counts: dict[str, int] = {}
    candidate_chars: dict[str, int] = {}
    variants: dict[str, dict[str, Any]] = {}
    scores: dict[str, dict[str, float]] = {}
    strategy = TPufStrategy()

    client = Turbopuffer(
        region=os.environ.get("TURBOPUFFER_REGION", "gcp-us-central1"),
        max_retries=0,
    )
    try:
        with client:
            namespace = client.namespace("esci-products")
            order = ("baseline", "cohere", "voyage")
            for name in order:
                print(f"Running {name} variant on the assigned training split.")
                summary, per_query = _run_variant(
                    name,
                    tasks[name],
                    requests,
                    namespace,
                    strategy,
                    token,
                    requests,
                    candidate_counts if name != "baseline" else candidate_counts,
                    candidate_chars if name != "baseline" else candidate_chars,
                )
                variants[name] = summary
                if summary["submitted_query_count"] == summary["query_count"]:
                    scores[name] = per_query
                print(
                    f"{name}: {summary['completed_query_count']}/"
                    f"{summary['query_count']} completed; "
                    f"{summary['failure_count']} failures."
                )
                if name == "baseline" and summary["failure_count"]:
                    print("Stopping: baseline candidate retrieval was incomplete.")
                    break
    except APIStatusError as error:
        status = getattr(error, "status_code", "unknown")
        print(f"Turbopuffer request failed with HTTP status {status}.")
        return 2
    finally:
        # Run output CSVs and task JSON are already mode 0600 in a private dir.
        pass

    if "baseline" in scores:
        baseline_scores = scores["baseline"]
        for name in ("cohere", "voyage"):
            if name in scores:
                variants[name]["paired_delta_vs_baseline"] = paired_delta_summary(
                    baseline_scores, scores[name]
                )
    report: dict[str, Any] = {
        "protocol": {
            "split": "assigned 1,000-query ESCI NDCG training split only",
            "tasks": (
                "Separate Referee task per baseline and model; exact "
                "ID/text/locale match verified."
            ),
            "retrieval": (
                "Frozen as 7e22661 identifier-gated RRF first stage: locale-filtered "
                "Voyage-4 passage ANN and weighted title/description BM25, top 100 "
                "per arm; passage included in both arms."
            ),
            "baseline": "Local identifier-gated RRF, rank constant 10, base weights 2:1.",
            "model_variants": dict(NATIVE_RERANK_MODELS),
            "model_reranking": (
                "Outer rerank_by on the two-arm multi-query; no local RRF before "
                "model scoring; deduplicated candidate union; final top 10."
            ),
            "cache": (
                "No application candidate/result cache; requests executed "
                "serially, one variant at a time."
            ),
            "latency": (
                "Client end-to-end latency for successful request attempts; 429 "
                "retries backed off and are counted separately."
            ),
            "cost": (
                "No model usage or model billing metric was returned. A chars/4 "
                "token proxy is shown for native models only; monetary cost is unavailable."
            ),
            "paired_ci": (
                "95% normal-approximation interval over per-query paired "
                "NDCG@10 differences."
            ),
            "execution_notes": (
                "An initial setup attempt stopped at locale normalization before "
                "any namespace query. A later baseline attempt was stopped by the "
                "120-second shell limit during retrieval, and another completed "
                "all 1,000 baseline queries but stopped before Referee submission "
                "due to a local output-file bug. Their partial request/retry/"
                "latency details were not retained and are excluded from the "
                "reported complete run. The successful run submitted 1,000 queries "
                "for each of its three tasks."
            ),
        },
        "variants": variants,
    }
    _private_json(run_dir / "aggregate-training-report.json", report)
    if args.aggregate_output is not None:
        destination = args.aggregate_output.resolve()
        if destination == repo or repo not in destination.parents:
            raise SystemExit("Aggregate report path must be inside this repository.")
        _private_json(destination, report)
    complete = all(
        variants.get(name, {}).get("submitted_query_count") == 1000
        for name in ("baseline", "cohere", "voyage")
    )
    if complete:
        print(
            "Finished training-only evaluation. All three 1,000-query tasks "
            "were submitted; task contents, credentials, rankings, and "
            "per-query scores remain private."
        )
        return 0
    print(
        "Training-only evaluation incomplete. Failed tasks were not submitted; "
        "no failed query was represented as a no-hit."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
