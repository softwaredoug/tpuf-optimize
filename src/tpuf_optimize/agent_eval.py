"""Gated, paired Referee evaluation of direct and agent ESCI product search.

This module deliberately stores query/document identifiers and aggregate
measurements only in private checkpoints. Query text, agent output, catalog
documents, SDK traces, tokens, and API keys are never written to repository
files or printed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any, cast
from urllib.error import URLError

from openai import OpenAIError
from turbopuffer import Turbopuffer

from tpuf_optimize.search_agent import (
    BEST_RETRIEVER,
    DEFAULT_MODEL,
    MAX_SEARCHES,
    TOP_K,
    ProductSearch,
    build_agent,
    extract_ranked_product_ids,
    load_ranker,
)
from tpuf_optimize.searcher import SearchRequest

REFEREE_URL = "http://localhost:8080"
PRIVATE_DIR = Path(
    "/private/var/folders/ww/t2bpzntd1990wczd0b7px2640000gn/T/opencode/"
    "esci-search-agent-training-eval-1000"
)
TOKEN_FILE = PRIVATE_DIR / "referee.token"
ARM_TASK_FILES = {
    "direct": PRIVATE_DIR / "reference-task.json",
    "agent": PRIVATE_DIR / "agent.task.json",
}
TASK_QUERY_COUNT = 1000
MODEL = DEFAULT_MODEL
CHECKPOINT_VERSION = 2
BOOTSTRAP_SAMPLES = 10_000
ID_LABEL_RE = re.compile(
    r"\b(?:document\s+id|doc(?:ument)?_id|product\s+id)"
    r"\s*[:#]\s*`?([A-Za-z0-9][A-Za-z0-9_-]*)`?",
    re.IGNORECASE,
)
NO_RECOMMENDATION_RE = re.compile(
    r"\b(?:no suitable products?|no products? found|no results|"
    r"could not find|couldn't find|nothing suitable)\b",
    re.IGNORECASE,
)
RECOMMENDATION_RE = re.compile(
    r"\b(?:recommend(?:ation)?s?|products?|matches|options?)\b",
    re.IGNORECASE,
)


def _atomic_json(path: Path, value: object) -> None:
    """Write private run state atomically with restrictive file permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            os.chmod(temporary, 0o600)
            json.dump(value, output, ensure_ascii=False, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_task(path: Path, arm: str) -> tuple[dict[str, Any], list[SearchRequest]]:
    """Load a pre-created task; never create a task or select another split."""
    task = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(task, dict) or task.get("dataset_name") != "esci":
        raise ValueError("The assigned task file is not an ESCI task.")
    task_info = task.get("task")
    queries = task.get("queries")
    if not isinstance(task_info, dict) or not isinstance(task_info.get("id"), str):
        raise ValueError("The assigned task file has no task ID.")
    if not isinstance(queries, list) or len(queries) != TASK_QUERY_COUNT:
        raise ValueError(
            f"Each assigned training task must contain exactly {TASK_QUERY_COUNT} queries."
        )
    requests: list[SearchRequest] = []
    seen: set[str] = set()
    for item in queries:
        metadata = item.get("metadata") if isinstance(item, dict) else None
        query_id = item.get("query_id") if isinstance(item, dict) else None
        text = item.get("query_text") if isinstance(item, dict) else None
        locale = metadata.get("locale") if isinstance(metadata, dict) else None
        if (
            not isinstance(query_id, str)
            or not query_id
            or not isinstance(text, str)
            or not text
            or not isinstance(locale, str)
            or not locale
        ):
            raise ValueError("The assigned task contains a malformed query.")
        if query_id in seen:
            raise ValueError("The assigned task contains duplicate query IDs.")
        seen.add(query_id)
        requests.append(SearchRequest(query_id, text, locale))
    if arm not in ARM_TASK_FILES:
        raise ValueError("Unsupported evaluation arm.")
    return cast(dict[str, Any], task), requests


def _assert_matched_tasks() -> None:
    """Fail closed unless the two private task files contain identical queries."""
    _, direct = _read_task(ARM_TASK_FILES["direct"], "direct")
    _, agent = _read_task(ARM_TASK_FILES["agent"], "agent")
    if direct != agent:
        raise ValueError("The direct and agent tasks are not an exact query match.")


def extract_recommendation_ids(
    final_output: object, returned_ids: set[str]
) -> tuple[list[str], int, bool, bool]:
    """Strictly parse labeled IDs; accept only IDs returned by a tool call.

    Returns valid ordered IDs, invalid citation count, missing-ID flag, and
    format-failure flag. No LLM or backend candidate fill is used.
    """
    if not isinstance(final_output, str):
        return [], 0, bool(returned_ids), bool(returned_ids)
    mentions = ID_LABEL_RE.findall(final_output)
    valid: list[str] = []
    seen: set[str] = set()
    invalid = 0
    for document_id in mentions:
        if document_id not in returned_ids:
            invalid += 1
        elif document_id not in seen:
            valid.append(document_id)
            seen.add(document_id)

    has_no_recommendation = NO_RECOMMENDATION_RE.search(final_output) is not None
    needs_id = bool(
        returned_ids
        and not valid
        and not has_no_recommendation
        and RECOMMENDATION_RE.search(final_output)
    )
    malformed_recommendation = bool(
        returned_ids
        and not mentions
        and not has_no_recommendation
        and final_output.strip()
    )
    return valid, invalid, needs_id, malformed_recommendation


def result_rows(query_id: str, document_ids: list[str]) -> list[dict[str, object]]:
    """Build Referee-compatible rows, with rank-derived finite scores."""
    if not document_ids:
        return [{"query_id": query_id, "doc_id": "-1", "rank": -1, "score": -1}]
    if len(set(document_ids)) != len(document_ids):
        raise ValueError("Duplicate document IDs cannot be ranked.")
    return [
        {
            "query_id": query_id,
            "doc_id": document_id,
            "rank": rank,
            "score": 1.0 / rank,
        }
        for rank, document_id in enumerate(document_ids[:TOP_K], start=1)
    ]


class _ObservedProductSearch:
    """Measure backend search time without changing the frozen search policy."""

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
                document_id = product.get("doc_id")
                if document_id is None:
                    raise ValueError("ProductSearch returned a product without an ID.")
                self.returned_ids.add(str(document_id))
            return products
        except Exception:
            self.failures += 1
            raise
        finally:
            self.backend_seconds += time.perf_counter() - started


def _query_direct(
    backend: ProductSearch, request: SearchRequest
) -> tuple[list[str], float, float]:
    started = time.perf_counter()
    products = backend.search(request.text, request.locale)
    elapsed = time.perf_counter() - started
    identifiers: list[str] = []
    seen: set[str] = set()
    for product in products:
        raw_id = product.get("doc_id")
        if raw_id is None:
            raise ValueError("ProductSearch returned a product without an ID.")
        document_id = str(raw_id)
        if document_id in seen:
            raise ValueError("ProductSearch returned duplicate product IDs.")
        seen.add(document_id)
        identifiers.append(document_id)
    return identifiers[:TOP_K], elapsed, elapsed


def _query_agent(
    backend: ProductSearch, request: SearchRequest
) -> tuple[list[str], float, float, int, int, bool, bool]:
    from agents import RunConfig, Runner

    observed = _ObservedProductSearch(backend)
    agent = build_agent(observed, request.locale, MODEL)  # type: ignore[arg-type]
    started = time.perf_counter()
    result = Runner.run_sync(
        agent,
        f"Requested locale: {request.locale}\nProduct request: {request.text}",
        max_turns=MAX_SEARCHES + 1,
        run_config=RunConfig(tracing_disabled=True),
    )
    elapsed = time.perf_counter() - started
    if observed.failures:
        raise RuntimeError("A backend search call failed inside the agent tool.")
    if observed.tool_calls < 1 or observed.tool_calls > MAX_SEARCHES:
        raise RuntimeError("The agent violated the frozen search tool-call budget.")
    ids, invalid, missing, format_failure = extract_ranked_product_ids(
        result.final_output, observed.returned_ids
    )
    return (
        ids[:TOP_K],
        elapsed,
        observed.backend_seconds,
        observed.tool_calls,
        invalid,
        missing,
        format_failure,
    )


def _read_state(
    path: Path, arm: str, task_id: str, requests: list[SearchRequest]
) -> dict[str, Any]:
    query_ids = [request.query_id for request in requests]
    if not path.exists():
        state: dict[str, Any] = {
            "version": CHECKPOINT_VERSION,
            "arm": arm,
            "task_id": task_id,
            "model": MODEL if arm == "agent" else None,
            "query_ids": query_ids,
            "records": {},
            "failure_counts": {},
        }
        _atomic_json(path, state)
        return state
    state = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(state, dict)
        or state.get("version") != CHECKPOINT_VERSION
        or state.get("arm") != arm
        or state.get("task_id") != task_id
        or state.get("model") != (MODEL if arm == "agent" else None)
        or state.get("query_ids") != query_ids
        or not isinstance(state.get("records"), dict)
        or not isinstance(state.get("failure_counts"), dict)
    ):
        raise ValueError("Private checkpoint does not match this arm and task.")
    return state


def _task_completion(
    task_id: str, token: str, rows: list[dict[str, object]]
) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{REFEREE_URL}/v1/tasks/{task_id}/complete",
        data=json.dumps({"results": rows}, allow_nan=False).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return cast(dict[str, Any], json.load(response))


def run_arm(
    arm: str,
    task_path: Path,
    checkpoint_path: Path,
    result_path: Path,
    namespace_name: str,
    region: str,
    *,
    allow_live: bool,
) -> dict[str, Any]:
    """Run or resume one assigned arm, then complete its existing Referee task."""
    if not allow_live or os.environ.get("ALLOW_LIVE_SEARCH_AGENT_EVAL") != "1":
        raise RuntimeError(
            "Live calls are gated; pass --allow-live and set "
            "ALLOW_LIVE_SEARCH_AGENT_EVAL=1 only after orchestrator approval."
        )
    expected_task_path = ARM_TASK_FILES.get(arm)
    if expected_task_path is None or task_path.resolve() != expected_task_path.resolve():
        raise ValueError("Live runs may use only the pre-created assigned task files.")
    if not task_path.exists() or not TOKEN_FILE.exists():
        raise ValueError("Assigned private task/token file is missing.")
    _assert_matched_tasks()
    task, requests = _read_task(task_path, arm)
    task_id = task["task"]["id"]
    token = TOKEN_FILE.read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError("Assigned Referee token file is empty.")

    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    pending_path = result_path.with_suffix(result_path.suffix + ".completion-pending")
    if pending_path.exists():
        raise RuntimeError(
            "A previous completion may have reached Referee; refusing to replay it."
        )

    state = _read_state(checkpoint_path, arm, task_id, requests)
    if not os.environ.get("TURBOPUFFER_API_KEY"):
        raise ValueError("TURBOPUFFER_API_KEY is not set.")
    if arm == "agent" and not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is not set.")

    failures_this_run: dict[str, int] = {}
    with Turbopuffer(region=region, max_retries=0) as client:
        backend = ProductSearch(
            client.namespace(namespace_name),
            BEST_RETRIEVER,
            load_ranker(),
        )
        for request in requests:
            if request.query_id in state["records"]:
                continue
            try:
                if arm == "direct":
                    ids, end_to_end, backend_seconds = _query_direct(backend, request)
                    tool_calls = 0
                    invalid = 0
                    missing = False
                    format_failure = False
                else:
                    (
                        ids,
                        end_to_end,
                        backend_seconds,
                        tool_calls,
                        invalid,
                        missing,
                        format_failure,
                    ) = _query_agent(backend, request)
                record = {
                    "doc_ids": ids,
                    "end_to_end_seconds": end_to_end,
                    "backend_seconds": backend_seconds,
                    "tool_calls": tool_calls,
                    "invalid_id_count": invalid,
                    "missing_id_queries": int(missing),
                    "format_failure_queries": int(format_failure),
                }
                state["records"][request.query_id] = record
                _atomic_json(checkpoint_path, state)
            except Exception as error:
                failure_name = type(error).__name__
                failures_this_run[failure_name] = (
                    failures_this_run.get(failure_name, 0) + 1
                )
                previous = int(state["failure_counts"].get(failure_name, 0))
                state["failure_counts"][failure_name] = previous + 1
                _atomic_json(checkpoint_path, state)

    if failures_this_run:
        counts = ", ".join(
            f"{name}={count}" for name, count in sorted(failures_this_run.items())
        )
        raise RuntimeError(
            f"{sum(failures_this_run.values())} query attempt(s) failed ({counts}); "
            "no Referee submission was made. Resume after resolving the failure."
        )
    if set(state["records"]) != {request.query_id for request in requests}:
        raise RuntimeError("The private checkpoint is missing task query results.")

    result_rows_all: list[dict[str, object]] = []
    for request in requests:
        result_rows_all.extend(
            result_rows(request.query_id, state["records"][request.query_id]["doc_ids"])
        )
    _atomic_json(
        pending_path,
        {
            "task_id": task_id,
            "arm": arm,
            "result_rows": len(result_rows_all),
            "query_count": len(requests),
            "state": "completion request may be sent once only",
        },
    )
    response = _task_completion(task_id, token, result_rows_all)
    response_queries = response.get("queries")
    if not isinstance(response_queries, list):
        raise ValueError("Referee completion response is malformed.")
    scores: dict[str, float] = {}
    for item in response_queries:
        if not isinstance(item, dict):
            raise ValueError("Referee returned a malformed per-query metric.")
        query_id = item.get("query_id")
        ndcg = item.get("ndcg")
        if not isinstance(query_id, str) or isinstance(ndcg, bool) or not isinstance(
            ndcg, (int, float)
        ):
            raise ValueError("Referee returned an invalid per-query NDCG.")
        value = float(ndcg)
        if not math.isfinite(value) or query_id in scores:
            raise ValueError("Referee returned invalid or duplicate per-query NDCG.")
        scores[query_id] = value
    if set(scores) != {request.query_id for request in requests}:
        raise ValueError("Referee metric query IDs do not match the assigned task.")

    # Referee includes query text in its response; retain scores and IDs only.
    sanitized = {
        "task_id": task_id,
        "metrics": response.get("metrics", {}),
        "queries": [
            {"query_id": request.query_id, "ndcg": scores[request.query_id]}
            for request in requests
        ],
    }
    _atomic_json(result_path, sanitized)
    pending_path.unlink(missing_ok=True)
    return sanitized


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _paired_bootstrap_ci(deltas: list[float], seed: int = 20261121) -> list[float]:
    if not deltas:
        raise ValueError("Cannot calculate a paired confidence interval without queries.")
    rng = random.Random(seed)
    sample_count = len(deltas)
    means = [
        sum(deltas[rng.randrange(sample_count)] for _ in range(sample_count))
        / sample_count
        for _ in range(BOOTSTRAP_SAMPLES)
    ]
    return [_percentile(means, 0.025), _percentile(means, 0.975)]


def compare_results(
    direct_result_path: Path,
    agent_result_path: Path,
    direct_checkpoint_path: Path,
    agent_checkpoint_path: Path,
) -> dict[str, Any]:
    """Calculate paired per-query NDCG delta/CI and latency summaries."""
    direct = json.loads(direct_result_path.read_text(encoding="utf-8"))
    agent = json.loads(agent_result_path.read_text(encoding="utf-8"))
    direct_scores = {item["query_id"]: item["ndcg"] for item in direct["queries"]}
    agent_scores = {item["query_id"]: item["ndcg"] for item in agent["queries"]}
    if (
        set(direct_scores) != set(agent_scores)
        or len(direct_scores) != TASK_QUERY_COUNT
    ):
        raise ValueError(
            f"Completed Referee tasks are not a matched {TASK_QUERY_COUNT}-query pair."
        )
    query_ids = list(direct_scores)
    deltas = [agent_scores[key] - direct_scores[key] for key in query_ids]

    direct_state = json.loads(direct_checkpoint_path.read_text(encoding="utf-8"))
    agent_state = json.loads(agent_checkpoint_path.read_text(encoding="utf-8"))
    direct_records = direct_state["records"]
    agent_records = agent_state["records"]
    if set(direct_records) != set(query_ids) or set(agent_records) != set(query_ids):
        raise ValueError("Latency checkpoints do not cover the paired tasks.")

    def latency_summary(
        records: dict[str, Any], failure_counts: dict[str, int]
    ) -> dict[str, Any]:
        end_to_end = [float(row["end_to_end_seconds"]) * 1000 for row in records.values()]
        backend = [float(row["backend_seconds"]) * 1000 for row in records.values()]
        sdk_overhead = [max(0.0, e2e - back) for e2e, back in zip(end_to_end, backend)]
        tools = [int(row["tool_calls"]) for row in records.values()]
        return {
            "end_to_end_ms": {
                "p50": _percentile(end_to_end, 0.50),
                "p95": _percentile(end_to_end, 0.95),
            },
            "backend_only_ms": {
                "p50": _percentile(backend, 0.50),
                "p95": _percentile(backend, 0.95),
            },
            "sdk_or_wrapper_overhead_ms": {
                "p50": _percentile(sdk_overhead, 0.50),
                "p95": _percentile(sdk_overhead, 0.95),
            },
            "mean_tool_calls": statistics.mean(tools),
            "invalid_id_count": sum(
                int(row["invalid_id_count"]) for row in records.values()
            ),
            "missing_id_queries": sum(
                int(row["missing_id_queries"]) for row in records.values()
            ),
            "format_failure_queries": sum(
                int(row["format_failure_queries"]) for row in records.values()
            ),
            "failure_counts": failure_counts,
        }

    direct_mean = statistics.mean(direct_scores.values())
    agent_mean = statistics.mean(agent_scores.values())
    return {
        "dataset": "esci",
        "split": "assigned training only",
        "queries": len(query_ids),
        "direct_policy": "ProductSearch/e64100d, original query, frozen ranker",
        "agent_policy": (
            f"77fc0d3 OpenAI Agents SDK free-text policy, {MODEL}, "
            "strict labeled-ID extraction; no prompt/model changes"
        ),
        "direct_ndcg10": direct_mean,
        "agent_ndcg10": agent_mean,
        "paired_agent_minus_direct_ndcg10": statistics.mean(deltas),
        "paired_bootstrap_95pct_ci": _paired_bootstrap_ci(deltas),
        "direct_latency": latency_summary(
            direct_records, direct_state["failure_counts"]
        ),
        "agent_latency": latency_summary(
            agent_records, agent_state["failure_counts"]
        ),
        "agent_tool_calls_total": sum(
            int(row["tool_calls"]) for row in agent_records.values()
        ),
    }


def _write_csv(path: Path, requests: list[SearchRequest], records: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["query_id", "doc_id", "rank", "score"])
        for request in requests:
            for row in result_rows(request.query_id, records[request.query_id]["doc_ids"]):
                writer.writerow(
                    [row["query_id"], row["doc_id"], row["rank"], row["score"]]
                )
    os.chmod(path, 0o600)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run/resume one frozen evaluation arm.")
    run.add_argument("--arm", choices=("direct", "agent"), required=True)
    run.add_argument("--task-file", type=Path)
    run.add_argument("--checkpoint", type=Path)
    run.add_argument("--result", type=Path)
    run.add_argument("--csv", type=Path)
    run.add_argument("--namespace", default="esci-products")
    run.add_argument(
        "--region", default=os.environ.get("TURBOPUFFER_REGION", "gcp-us-central1")
    )
    run.add_argument("--allow-live", action="store_true")
    compare = commands.add_parser("compare", help="Summarize the paired private results.")
    compare.add_argument(
        "--output", type=Path, default=PRIVATE_DIR / "paired-summary.json"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "compare":
        summary = compare_results(
            PRIVATE_DIR / "direct.result.json",
            PRIVATE_DIR / "agent.result.json",
            PRIVATE_DIR / "direct.checkpoint.json",
            PRIVATE_DIR / "agent.checkpoint.json",
        )
        _atomic_json(args.output, summary)
        print(json.dumps(summary, sort_keys=True))
        return 0

    defaults = {
        "task_file": ARM_TASK_FILES[args.arm],
        "checkpoint": PRIVATE_DIR / f"{args.arm}.checkpoint.json",
        "result": PRIVATE_DIR / f"{args.arm}.result.json",
        "csv": PRIVATE_DIR / f"{args.arm}.results.csv",
    }
    paths = {key: getattr(args, key) or value for key, value in defaults.items()}
    try:
        task, requests = _read_task(paths["task_file"], args.arm)
        result = run_arm(
            args.arm,
            paths["task_file"],
            paths["checkpoint"],
            paths["result"],
            args.namespace,
            args.region,
            allow_live=args.allow_live,
        )
        state = json.loads(paths["checkpoint"].read_text(encoding="utf-8"))
        _write_csv(paths["csv"], requests, state["records"])
        print(
            f"Completed {args.arm} arm for {len(requests)} assigned queries "
            f"(task {task['task']['id']}); private output saved."
        )
        print(
            "Referee NDCG@10: "
            f"{result.get('metrics', {}).get('ndcg', 'unavailable')}"
        )
    except (OSError, ValueError, URLError, OpenAIError, RuntimeError) as error:
        # Do not print exception strings: upstream errors can contain payloads.
        print(f"Evaluation stopped ({type(error).__name__}); private checkpoint retained.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
