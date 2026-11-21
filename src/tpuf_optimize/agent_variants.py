"""Paired, training-only evaluation of agents with pinned retrieval backends."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import math
import os
import random
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any, cast
from urllib.error import URLError

from agents import RunConfig, Runner
from openai import OpenAIError
from turbopuffer import Turbopuffer

from tpuf_optimize.agent_eval import (
    REFEREE_URL,
    _atomic_json,
    _task_completion,
    result_rows,
)
from tpuf_optimize.search_agent import (
    DEFAULT_MODEL,
    MAX_SEARCHES,
    ProductSearch,
    build_agent,
    extract_ranked_product_ids,
)
from tpuf_optimize.searcher import SearchRequest

PRIVATE_DIR = Path(
    "/private/var/folders/ww/t2bpzntd1990wczd0b7px2640000gn/T/opencode/"
    "esci-search-agent-training-eval-1000"
)
TOKEN_FILE = PRIVATE_DIR / "referee.token"
REFERENCE_TASK = PRIVATE_DIR / "reference-task.json"
VARIANTS = ("naive-bm25", "voyage4-ann", "best-typed-computed")
TASK_QUERY_COUNT = 1000
DEFAULT_CONCURRENCY = 8
MAX_CONCURRENCY = 32
STATE_VERSION = 1
OUTPUT_SCHEMA = "ranked-search-results-v3"


def _task_signature(task: object) -> list[tuple[str, str, str]]:
    if (
        not isinstance(task, dict)
        or task.get("dataset_name") != "esci"
        or not isinstance(task.get("task"), dict)
        or not isinstance(task["task"].get("id"), str)
        or not isinstance(task.get("queries"), list)
    ):
        raise ValueError("Referee task is malformed.")
    signature = []
    for row in task["queries"]:
        if not isinstance(row, dict):
            raise ValueError("Referee task contains a malformed query.")
        metadata = row.get("metadata")
        query_id = row.get("query_id")
        text = row.get("query_text")
        locale = metadata.get("locale") if isinstance(metadata, dict) else None
        if not all(isinstance(value, str) and value for value in (query_id, text, locale)):
            raise ValueError("Referee query is missing ID, text, or locale.")
        signature.append((query_id, text, locale))
    if len(signature) != TASK_QUERY_COUNT or len({row[0] for row in signature}) != TASK_QUERY_COUNT:
        raise ValueError("Expected exactly 1,000 unique queries.")
    return signature


def _requests(task: object) -> list[SearchRequest]:
    return [
        SearchRequest(query_id, text, locale)
        for query_id, text, locale in _task_signature(task)
    ]


def _post_task(token: str) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{REFEREE_URL}/v1/tasks",
        method="POST",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return cast(dict[str, Any], json.load(response))


def prepare_tasks() -> dict[str, int]:
    """Create matched tasks from the assigned training split only."""
    if not TOKEN_FILE.is_file() or not REFERENCE_TASK.is_file():
        raise ValueError("The authorized training token/task is missing.")
    token = TOKEN_FILE.read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError("The assigned training token is empty.")
    expected = _task_signature(json.loads(REFERENCE_TASK.read_text(encoding="utf-8")))
    for variant in VARIANTS:
        path = PRIVATE_DIR / f"{OUTPUT_SCHEMA}-{variant}.task.json"
        task = json.loads(path.read_text(encoding="utf-8")) if path.exists() else _post_task(token)
        if _task_signature(task) != expected:
            raise ValueError(f"{variant} task does not exactly match the assigned training task.")
        if not path.exists():
            _atomic_json(path, task)
    return {"matched_queries": len(expected), "variant_tasks": len(VARIANTS)}


class _ObservedSearch:
    """Collect backend timing and valid product IDs for one agent query."""

    def __init__(self, backend: ProductSearch) -> None:
        self.backend = backend
        self.tool_calls = 0
        self.backend_seconds = 0.0
        self.failures = 0
        self.returned_ids: set[str] = set()

    def search(self, query: str, locale: str) -> list[dict[str, object]]:
        self.tool_calls += 1
        started = time.perf_counter()
        try:
            products = self.backend.search(query, locale)
            for product in products:
                doc_id = product.get("doc_id")
                if doc_id is None:
                    raise ValueError("Search result has no document ID.")
                self.returned_ids.add(str(doc_id))
            return products
        except Exception:
            self.failures += 1
            raise
        finally:
            self.backend_seconds += time.perf_counter() - started


async def _query_agent(
    backend: ProductSearch, request: SearchRequest
) -> dict[str, Any]:
    observed = _ObservedSearch(backend)
    agent = build_agent(observed, request.locale, DEFAULT_MODEL)  # type: ignore[arg-type]
    started = time.perf_counter()
    result = await Runner.run(
        agent,
        f"Requested locale: {request.locale}\nProduct request: {request.text}",
        max_turns=MAX_SEARCHES + 1,
        run_config=RunConfig(tracing_disabled=True),
    )
    elapsed = time.perf_counter() - started
    if observed.failures:
        raise RuntimeError("A backend search failed inside the agent tool.")
    if not 1 <= observed.tool_calls <= MAX_SEARCHES:
        raise RuntimeError("Agent violated the tool-call budget.")
    ids, invalid, missing, malformed = extract_ranked_product_ids(
        result.final_output, observed.returned_ids
    )
    return {
        "doc_ids": ids[:10],
        "end_to_end_seconds": elapsed,
        "backend_seconds": observed.backend_seconds,
        "tool_calls": observed.tool_calls,
        "invalid_id_count": invalid,
        "missing_id_queries": int(missing),
        "format_failure_queries": int(malformed),
    }


async def _run_concurrently(
    requests: list[SearchRequest],
    concurrency: int,
    operation,
    on_complete,
) -> None:
    """Run independent query agents concurrently; checkpoint as each finishes."""
    if concurrency < 1:
        raise ValueError("Concurrency must be at least one.")
    semaphore = asyncio.Semaphore(concurrency)

    async def run_one(request: SearchRequest):
        async with semaphore:
            try:
                result = await operation(request)
                return request, result, None
            except Exception as error:
                return request, None, error

    tasks = [asyncio.create_task(run_one(request)) for request in requests]
    for completed in asyncio.as_completed(tasks):
        request, result, error = await completed
        on_complete(request, result, error)


def _read_or_initialize_state(
    path: Path,
    variant: str,
    task_id: str,
    requests: list[SearchRequest],
    concurrency: int,
) -> dict[str, Any]:
    query_ids = [request.query_id for request in requests]
    if not path.exists():
        state: dict[str, Any] = {
            "version": STATE_VERSION,
            "output_schema": OUTPUT_SCHEMA,
            "variant": variant,
            "task_id": task_id,
            "model": DEFAULT_MODEL,
            "concurrency": concurrency,
            "query_ids": query_ids,
            "records": {},
            "failure_counts": {},
        }
        _atomic_json(path, state)
        return state
    state = json.loads(path.read_text(encoding="utf-8"))
    if (
        state.get("version") != STATE_VERSION
        or state.get("output_schema") != OUTPUT_SCHEMA
        or state.get("variant") != variant
        or state.get("task_id") != task_id
        or state.get("model") != DEFAULT_MODEL
        or state.get("concurrency") != concurrency
        or state.get("query_ids") != query_ids
        or not isinstance(state.get("records"), dict)
        or not isinstance(state.get("failure_counts"), dict)
    ):
        raise ValueError("Checkpoint does not match this model, variant, task, or concurrency.")
    return state


def run_variant(variant: str, concurrency: int, *, allow_live: bool) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError("Unknown retrieval variant.")
    if not 1 <= concurrency <= MAX_CONCURRENCY:
        raise ValueError(f"Concurrency must be between 1 and {MAX_CONCURRENCY}.")
    if not allow_live or os.environ.get("ALLOW_LIVE_SEARCH_AGENT_EVAL") != "1":
        raise RuntimeError("Live calls require --allow-live and explicit environment approval.")
    if not os.environ.get("OPENAI_API_KEY") or not os.environ.get("TURBOPUFFER_API_KEY"):
        raise ValueError("OPENAI_API_KEY and TURBOPUFFER_API_KEY are required.")
    lock_file = (PRIVATE_DIR / f"{variant}.lock").open("a")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_variant_locked(variant, concurrency)
    finally:
        lock_file.close()


def _run_variant_locked(variant: str, concurrency: int) -> dict[str, Any]:
    task_path = PRIVATE_DIR / f"{OUTPUT_SCHEMA}-{variant}.task.json"
    reference = json.loads(REFERENCE_TASK.read_text(encoding="utf-8"))
    task = json.loads(task_path.read_text(encoding="utf-8"))
    if _task_signature(task) != _task_signature(reference):
        raise ValueError("Variant task differs from the assigned training queries.")
    requests = _requests(task)
    task_id = task["task"]["id"]
    token = TOKEN_FILE.read_text(encoding="utf-8").strip()
    checkpoint = PRIVATE_DIR / (
        f"{OUTPUT_SCHEMA}-gpt-6-luna-{variant}-c{concurrency}.checkpoint.json"
    )
    result_path = PRIVATE_DIR / (
        f"{OUTPUT_SCHEMA}-gpt-6-luna-{variant}-c{concurrency}.result.json"
    )
    pending = result_path.with_suffix(".completion-pending")
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    if pending.exists():
        raise RuntimeError("Refusing to replay an ambiguous Referee completion.")
    state = _read_or_initialize_state(checkpoint, variant, task_id, requests, concurrency)
    failures_this_run: dict[str, int] = {}

    def completed(
        request: SearchRequest, record: dict[str, Any] | None, error: Exception | None
    ) -> None:
        if error is not None:
            name = type(error).__name__
            failures_this_run[name] = failures_this_run.get(name, 0) + 1
            state["failure_counts"][name] = int(state["failure_counts"].get(name, 0)) + 1
        else:
            state["records"][request.query_id] = record
        _atomic_json(checkpoint, state)

    with Turbopuffer(
        region=os.environ.get("TURBOPUFFER_REGION", "gcp-us-central1"),
        max_retries=0,
    ) as client:
        backend = ProductSearch(client.namespace("esci-products"), variant)
        remaining = [
            request for request in requests
            if request.query_id not in state["records"]
        ]
        asyncio.run(
            _run_concurrently(
                remaining,
                concurrency,
                lambda request: _query_agent(backend, request),
                completed,
            )
        )

    if failures_this_run:
        counts = ", ".join(f"{name}={count}" for name, count in sorted(failures_this_run.items()))
        raise RuntimeError(
            f"{sum(failures_this_run.values())} query failures ({counts}); "
            "no Referee results were submitted."
        )
    if set(state["records"]) != {request.query_id for request in requests}:
        raise RuntimeError("The private checkpoint is incomplete.")

    rows: list[dict[str, object]] = []
    for request in requests:
        rows.extend(result_rows(request.query_id, state["records"][request.query_id]["doc_ids"]))
    _atomic_json(pending, {
        "variant": variant, "task_id": task_id,
        "query_count": len(requests), "state": "completion may be sent once only",
    })
    response = _task_completion(task_id, token, rows)
    raw_metrics = response.get("metrics")
    raw_queries = response.get("queries")
    if not isinstance(raw_metrics, dict) or not isinstance(raw_queries, list):
        raise ValueError("Referee returned malformed aggregate metrics.")
    scores = {}
    for item in raw_queries:
        if not isinstance(item, dict) or not isinstance(item.get("query_id"), str):
            raise ValueError("Referee returned malformed per-query metrics.")
        value = item.get("ndcg")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Referee returned an invalid NDCG value.")
        score = float(value)
        if not math.isfinite(score) or item["query_id"] in scores:
            raise ValueError("Referee returned non-finite or duplicate metric rows.")
        scores[item["query_id"]] = score
    if set(scores) != {request.query_id for request in requests}:
        raise ValueError("Referee metric IDs differ from the training task.")
    sanitized = {
        "variant": variant,
        "model": DEFAULT_MODEL,
        "concurrency": concurrency,
        "task_id": task_id,
        "metrics": raw_metrics,
        "queries": [
            {"query_id": request.query_id, "ndcg": scores[request.query_id]}
            for request in requests
        ],
    }
    _atomic_json(result_path, sanitized)
    pending.unlink(missing_ok=True)
    return sanitized


def _percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(p * len(ordered)) - 1))
    return ordered[index]


def summarize(concurrency: int) -> dict[str, Any]:
    results = {}
    score_maps = {}
    for variant in VARIANTS:
        result_path = PRIVATE_DIR / (
            f"{OUTPUT_SCHEMA}-gpt-6-luna-{variant}-c{concurrency}.result.json"
        )
        checkpoint_path = PRIVATE_DIR / (
            f"{OUTPUT_SCHEMA}-gpt-6-luna-{variant}-c{concurrency}.checkpoint.json"
        )
        result = json.loads(result_path.read_text(encoding="utf-8"))
        state = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        score_maps[variant] = {row["query_id"]: float(row["ndcg"]) for row in result["queries"]}
        if len(score_maps[variant]) != TASK_QUERY_COUNT:
            raise ValueError("Completed variant does not cover 1,000 queries.")
        records = state["records"]
        if set(records) != score_maps[variant].keys():
            raise ValueError("Latency records do not match scored query IDs.")
        end_to_end = [float(row["end_to_end_seconds"]) * 1000 for row in records.values()]
        backend = [float(row["backend_seconds"]) * 1000 for row in records.values()]
        results[variant] = {
            "ndcg_at_10": float(result["metrics"]["ndcg"]),
            "p50_end_to_end_ms": _percentile(end_to_end, 0.50),
            "p95_end_to_end_ms": _percentile(end_to_end, 0.95),
            "p50_backend_ms": _percentile(backend, 0.50),
            "p95_backend_ms": _percentile(backend, 0.95),
            "tool_calls": sum(int(row["tool_calls"]) for row in records.values()),
            "invalid_ids": sum(int(row["invalid_id_count"]) for row in records.values()),
            "missing_id_queries": sum(int(row["missing_id_queries"]) for row in records.values()),
            "format_failures": sum(int(row["format_failure_queries"]) for row in records.values()),
            "failure_counts": state["failure_counts"],
        }
    baseline = score_maps["naive-bm25"]
    comparisons = {}
    rng = random.Random(2606)
    for variant in VARIANTS[1:]:
        shared = sorted(baseline.keys() & score_maps[variant].keys())
        deltas = [score_maps[variant][query_id] - baseline[query_id] for query_id in shared]
        draws = sorted(
            statistics.mean(deltas[rng.randrange(len(deltas))] for _ in deltas)
            for _ in range(5000)
        )
        comparisons[variant] = {
            "paired_queries": len(shared),
            "delta_vs_naive_bm25": statistics.mean(deltas),
            "ci95_low": draws[int(.025 * len(draws))],
            "ci95_high": draws[int(.975 * len(draws))],
        }
    return {
        "split": "ESCI training only",
        "queries": TASK_QUERY_COUNT,
        "model": DEFAULT_MODEL,
        "concurrency": concurrency,
        "variants": results,
        "paired_vs_naive_bm25": comparisons,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="Create three matched training-only Referee tasks.")
    run = commands.add_parser("run", help="Run one retrieval-backed agent variant.")
    run.add_argument("--variant", choices=VARIANTS, required=True)
    run.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    run.add_argument("--allow-live", action="store_true")
    compare = commands.add_parser(
        "compare", help="Summarize completed variants and paired deltas."
    )
    compare.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            print(json.dumps(prepare_tasks(), sort_keys=True))
        elif args.command == "run":
            result = run_variant(args.variant, args.concurrency, allow_live=args.allow_live)
            print(json.dumps({
                "variant": args.variant,
                "model": DEFAULT_MODEL,
                "concurrency": args.concurrency,
                "ndcg_at_10": result["metrics"]["ndcg"],
            }, sort_keys=True))
        else:
            summary = summarize(args.concurrency)
            output = PRIVATE_DIR / (
                "gpt-6-luna-ranked-search-results-v3-agent-variant-summary.json"
            )
            _atomic_json(output, summary)
            print(json.dumps(summary, sort_keys=True))
        return 0
    except (OSError, ValueError, OpenAIError, RuntimeError, URLError) as error:
        print(f"Agent variant evaluation stopped ({type(error).__name__}); private state retained.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
