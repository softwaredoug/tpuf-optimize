"""Checkpointed collection of ESCI training retrieval candidates."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, cast
from urllib.error import URLError

from turbopuffer import Turbopuffer
from turbopuffer._exceptions import RateLimitError
from turbopuffer.types import Row
from turbopuffer.types.namespace_multi_query_params import Query

from tpuf_optimize.searcher import SearchRequest, Searcher
from tpuf_optimize.strategy import TPufStrategy

REFEREE_URL = "http://localhost:8080"
DEFAULT_TEMP_DIR = Path(
    "/private/var/folders/ww/t2bpzntd1990wczd0b7px2640000gn/T/opencode"
)
DEFAULT_TASK_FILE = DEFAULT_TEMP_DIR / "esci_research_training_task.json"
DEFAULT_TOKEN_FILE = DEFAULT_TEMP_DIR / "esci_breed_training_only.token"
DEFAULT_OUTPUT = DEFAULT_TEMP_DIR / "esci_research_training_candidates.jsonl"
RESULT_ATTRIBUTES = ("title", "description", "brand", "color", "category")
HYBRID_DEPTH = 100
RRF_TOP_K = 10
RRF_RANK_CONSTANT = 10
RRF_WEIGHTS = (2.0, 1.0)


def _atomic_json(path: Path, value: object) -> None:
    """Write JSON using a same-directory temporary file and atomic replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            json.dump(value, output, ensure_ascii=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _create_training_task(token: str) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{REFEREE_URL}/v1/tasks",
        method="POST",
        headers={"Authorization": f"Bearer {token}"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        task = json.load(response)
    if not isinstance(task, dict) or not isinstance(task.get("queries"), list):
        raise ValueError("Referee returned an invalid training task.")
    return cast(dict[str, Any], task)


def _get_task(task_path: Path, token_path: Path) -> dict[str, Any]:
    if task_path.exists():
        with task_path.open(encoding="utf-8") as task_file:
            task = json.load(task_file)
    else:
        token = token_path.read_text(encoding="utf-8").strip()
        if not token:
            raise ValueError(f"{token_path} is empty.")
        task = _create_training_task(token)
        _atomic_json(task_path, task)
    if not isinstance(task, dict) or not isinstance(task.get("queries"), list):
        raise ValueError(f"{task_path} is not a valid Referee task.")
    return cast(dict[str, Any], task)


def _task_requests(task: dict[str, Any]) -> list[SearchRequest]:
    requests: list[SearchRequest] = []
    seen: set[str] = set()
    for query in task["queries"]:
        if not isinstance(query, dict):
            raise ValueError("Referee task contains a malformed query.")
        query_id = query.get("query_id")
        query_text = query.get("query_text")
        metadata = query.get("metadata")
        locale = metadata.get("locale") if isinstance(metadata, dict) else None
        if (
            not isinstance(query_id, str)
            or not query_id
            or not isinstance(query_text, str)
            or not query_text
            or not isinstance(locale, str)
            or not locale
        ):
            raise ValueError("Referee query must have an ID, text, and locale.")
        if query_id in seen:
            raise ValueError(f"Duplicate query ID in Referee task: {query_id!r}")
        seen.add(query_id)
        requests.append(SearchRequest(query_id, query_text, locale))
    return requests


class CandidateQueryStrategy(TPufStrategy):
    """Use the parent hybrid rank expressions while returning candidate fields."""

    LOCALE_ALIASES = {"en-US": "us", "es-ES": "es", "ja-JP": "jp"}
    mode: str

    def __init__(self, mode: str) -> None:
        super().__init__()
        if mode not in {"dense", "lexical"}:
            raise ValueError(f"Unsupported candidate query mode: {mode}")
        self.mode = mode

    @classmethod
    def normalize_locale(cls, locale: str) -> str:
        return cls.LOCALE_ALIASES.get(locale, locale)

    def query(self, query_text: str, locale: str, top_k: int) -> Query:
        """Construct one parent-compatible dense or BM25 query."""
        if not query_text.strip():
            raise ValueError("query_text must not be empty")
        if not locale.strip():
            raise ValueError("locale must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be greater than zero")
        normalized_locale = self.normalize_locale(locale)
        if self.mode == "dense":
            rank_by: Any = (
                "passage__embed__voyage_4",
                "ANN",
                ("Embed", query_text),
            )
        else:
            # Same Sum(Product(2, title BM25), Product(1, description BM25))
            # expression as the hybrid parent at cc5fc8a.
            rank_by = self.rank_by(query_text, normalized_locale)
        return cast(
            Query,
            {
                "rank_by": rank_by,
                "filters": ("locale", "Eq", normalized_locale),
                "top_k": top_k,
                "include_attributes": list(RESULT_ATTRIBUTES),
            },
        )


def _row_candidate(row: Row) -> dict[str, object]:
    values = row.model_dump()
    doc_id = values.get("id")
    score = values.get("$dist")
    if doc_id is None or not isinstance(score, (float, int)) or not math.isfinite(score):
        raise ValueError("Turbopuffer returned a candidate without id or numeric $dist.")
    return {
        "doc_id": str(doc_id),
        "score": score,
        "title": values.get("title"),
        "brand": values.get("brand"),
        "color": values.get("color"),
        "category": values.get("category"),
    }


def _candidate_list(rows: list[Row]) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    seen: set[str] = set()
    for rank, row in enumerate(rows, start=1):
        candidate = _row_candidate(row)
        doc_id = str(candidate["doc_id"])
        if doc_id in seen:
            raise ValueError(f"Turbopuffer returned duplicate candidate {doc_id!r}.")
        seen.add(doc_id)
        candidates.append(candidate)
    return candidates


def _query_rows(
    searcher: Searcher, request: SearchRequest, top_k: int
) -> list[Row]:
    # A single input is sent at a time; Searcher retains bounded retry and
    # serial fallback behavior while sharing one namespace client.
    found = list(searcher.search([request], top_k))
    if len(found) != 1 or found[0][0] != request:
        raise ValueError("Searcher did not return exactly one matching query.")
    return found[0][1]


def _checkpoint_read(path: Path, requests: list[SearchRequest]) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    request_by_id = {request.query_id: request for request in requests}
    completed: dict[str, dict[str, Any]] = {}
    valid_end = 0
    with path.open("rb+") as checkpoint:
        while line := checkpoint.readline():
            line_end = checkpoint.tell()
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if line_end == path.stat().st_size and not line.endswith(b"\n"):
                    checkpoint.seek(valid_end)
                    checkpoint.truncate()
                    break
                raise ValueError("Checkpoint contains malformed JSONL.")
            if not isinstance(record, dict) or not isinstance(record.get("query_id"), str):
                raise ValueError("Checkpoint contains an invalid record.")
            query_id = record["query_id"]
            request = request_by_id.get(query_id)
            if request is None or query_id in completed:
                raise ValueError("Checkpoint query IDs do not match the task.")
            if record.get("query_text") != request.text or record.get("locale") != request.locale:
                raise ValueError("Checkpoint query inputs do not match the task.")
            if (
                not isinstance(record.get("dense"), list)
                or not isinstance(record.get("lexical"), list)
            ):
                raise ValueError("Checkpoint record is missing candidate lists.")
            completed[query_id] = record
            valid_end = line_end
    return completed


def _append_checkpoint(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n").encode()
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def collect_candidates(
    namespace: object,
    requests: list[SearchRequest],
    checkpoint_path: Path,
    output_path: Path,
) -> int:
    """Collect both top-100 lists, resume partial progress, finalize atomically."""
    if checkpoint_path.resolve() == output_path.resolve():
        raise ValueError("The checkpoint and complete output must be distinct.")
    completed = _checkpoint_read(checkpoint_path, requests)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.touch(exist_ok=True)
    strategy = CandidateQueryStrategy("dense")
    searcher = Searcher(namespace, strategy, batch_size=1)  # type: ignore[arg-type]
    for request in requests:
        if request.query_id in completed:
            continue
        strategy.mode = "dense"
        dense = _candidate_list(_query_rows(searcher, request, HYBRID_DEPTH))
        strategy.mode = "lexical"
        lexical = _candidate_list(_query_rows(searcher, request, HYBRID_DEPTH))
        record: dict[str, Any] = {
            "query_id": request.query_id,
            "query_text": request.text,
            "locale": request.locale,
            "dense": dense,
            "lexical": lexical,
        }
        _append_checkpoint(checkpoint_path, record)
        completed[request.query_id] = record

    if len(completed) != len(requests):
        raise ValueError("Candidate collection is missing task queries.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            for request in requests:
                output.write(
                    json.dumps(
                        completed[request.query_id],
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return len(completed)


def _rrf_candidates(record: dict[str, Any], top_k: int = RRF_TOP_K) -> list[str]:
    """Reproduce the parent hybrid RRF ordering from the cached list ranks."""
    scores: dict[str, float] = {}
    for field, weight in zip(("dense", "lexical"), RRF_WEIGHTS):
        for rank, candidate in enumerate(record[field], start=1):
            doc_id = str(candidate["doc_id"])
            scores[doc_id] = scores.get(doc_id, 0.0) + (
                weight / (RRF_RANK_CONSTANT + rank)
            )
    return [
        doc_id
        for doc_id, _score in sorted(
            scores.items(), key=lambda item: item[1], reverse=True
        )[:top_k]
    ]


def _evaluate_parent_rrf(
    task: dict[str, Any], cache_path: Path, token_path: Path
) -> dict[str, Any]:
    token = token_path.read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError(f"{token_path} is empty.")
    cache: dict[str, dict[str, Any]] = {}
    with cache_path.open(encoding="utf-8") as cache_file:
        for line in cache_file:
            record = json.loads(line)
            cache[record["query_id"]] = record
    task_requests = _task_requests(task)
    if len(cache) != len(task_requests) or set(cache) != {
        request.query_id for request in task_requests
    }:
        raise ValueError("Training cache coverage does not match the task.")

    eval_task = _create_training_task(token)
    eval_requests = _task_requests(eval_task)
    if [request.query_id for request in eval_requests] != [
        request.query_id for request in task_requests
    ]:
        raise ValueError("Separate Referee task does not match cached query IDs.")
    results: list[dict[str, object]] = []
    for request in eval_requests:
        doc_ids = _rrf_candidates(cache[request.query_id])
        if not doc_ids:
            results.append(
                {"query_id": request.query_id, "doc_id": "-1", "rank": -1, "score": -1}
            )
            continue
        results.extend(
            {
                "query_id": request.query_id,
                "doc_id": doc_id,
                "rank": rank,
                "score": 1.0 / (RRF_RANK_CONSTANT + rank),
            }
            for rank, doc_id in enumerate(doc_ids, start=1)
        )

    complete_request = urllib.request.Request(
        f"{REFEREE_URL}/v1/tasks/{eval_task['task']['id']}/complete",
        data=json.dumps({"results": results}).encode(),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(complete_request, timeout=120) as response:
        return cast(dict[str, Any], json.load(response))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-file", type=Path, default=DEFAULT_TASK_FILE)
    parser.add_argument("--token-file", type=Path, default=DEFAULT_TOKEN_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--namespace", default="esci-products")
    parser.add_argument("--region", default=os.environ.get("TURBOPUFFER_REGION", "gcp-us-central1"))
    parser.add_argument("--evaluate-parent-rrf", action="store_true")
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=DEFAULT_TEMP_DIR / "esci_research_training_rrf_metrics.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not os.environ.get("TURBOPUFFER_API_KEY"):
        parser.error("Set TURBOPUFFER_API_KEY before querying.")
    checkpoint_path = args.checkpoint or args.output.with_suffix(
        args.output.suffix + ".partial"
    )
    try:
        task = _get_task(args.task_file, args.token_file)
        requests = _task_requests(task)
        with Turbopuffer(region=args.region) as client:
            namespace = client.namespace(args.namespace)
            count = collect_candidates(
                namespace, requests, checkpoint_path, args.output
            )
        print(f"Collected candidate lists for {count} training queries.")
        if args.evaluate_parent_rrf:
            metrics = _evaluate_parent_rrf(task, args.output, args.token_file)
            _atomic_json(args.metrics_output, metrics)
            metric_values = metrics.get("metrics", {})
            print(f"Parent RRF metrics: {json.dumps(metric_values, sort_keys=True)}")
    except RateLimitError:
        print(
            f"Rate limit persisted; checkpoint preserved at {checkpoint_path}.",
            flush=True,
        )
        return 1
    except (OSError, ValueError, KeyError, URLError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
