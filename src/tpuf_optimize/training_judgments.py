"""Read training-split judgments for cached candidates via Referee debug only."""

from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from tpuf_optimize.computed_feature_collector import (
    FEATURES,
    ID_FILTER_METHOD,
    ComputedFeatureQueryStrategy,
    _load_candidate_cache,
)
from tpuf_optimize.searcher import SearchRequest
from tpuf_optimize.training_collector import (
    DEFAULT_TASK_FILE,
    DEFAULT_TEMP_DIR,
    DEFAULT_TOKEN_FILE,
    _task_requests,
)

REFEREE_URL = "http://localhost:8080"
DEFAULT_CANDIDATE_CACHE = (
    DEFAULT_TEMP_DIR / "esci_research_training_candidates.jsonl"
)
DEFAULT_ENRICHED_CACHE = (
    DEFAULT_TEMP_DIR / "esci_research_training_idfiltered_candidates.jsonl"
)
DEFAULT_OUTPUT = DEFAULT_TEMP_DIR / "esci_research_training_idfiltered"
BATCH_QUERIES = 10
MAX_429_RETRIES = 3
MAX_BACKOFF_SECONDS = 30.0


def _atomic_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _read_checkpoint(
    path: Path,
    requests: list[SearchRequest],
    source_sha256: str,
    task_id: str,
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    expected = {request.query_id: request for request in requests}
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as checkpoint:
        for line_number, line in enumerate(checkpoint, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: malformed judgment checkpoint"
                ) from error
            query_id = record.get("query_id") if isinstance(record, dict) else None
            if (
                not isinstance(query_id, str)
                or query_id not in expected
                or query_id in records
                or record.get("source_sha256") != source_sha256
                or record.get("task_id") != task_id
                or not isinstance(record.get("grades"), dict)
                or not isinstance(record.get("targets"), list)
            ):
                raise ValueError("Judgment checkpoint does not match training task/cache.")
            records[query_id] = record
    return records


def _load_enriched_cache(
    path: Path, requests: list[SearchRequest]
) -> tuple[list[dict[str, Any]], str]:
    """Validate that feature records are this training task's enriched cache."""
    records: list[dict[str, Any]] = []
    source_hash: str | None = None
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: malformed enriched cache.") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: malformed enriched record.")
            current_hash = record.get("source_sha256")
            if (
                record.get("feature_collection_method") != ID_FILTER_METHOD
                or record.get("feature_schema") != list(FEATURES)
                or not isinstance(current_hash, str)
                or (source_hash is not None and current_hash != source_hash)
            ):
                raise ValueError("Enriched cache provenance/schema is invalid.")
            source_hash = current_hash
            records.append(record)
    if source_hash is None or len(records) != len(requests):
        raise ValueError("Enriched cache does not cover the training task.")
    for record, request in zip(records, requests):
        if (
            record.get("query_id") != request.query_id
            or record.get("query_text") != request.text
            or record.get("locale")
            != ComputedFeatureQueryStrategy.normalize_locale(request.locale)
        ):
            raise ValueError("Enriched cache query order does not match training task.")
        for arm in ("dense", "lexical"):
            candidates = record.get(arm)
            if not isinstance(candidates, list):
                raise ValueError("Enriched cache is missing a candidate arm.")
            seen: set[str] = set()
            for candidate in candidates:
                if not isinstance(candidate, dict) or not isinstance(
                    candidate.get("features"), dict
                ):
                    raise ValueError("Enriched cache has incomplete features.")
                doc_id = str(candidate.get("doc_id"))
                if doc_id in seen or set(candidate["features"]) != set(FEATURES):
                    raise ValueError("Enriched cache contains duplicate/stale candidates.")
                seen.add(doc_id)
    return records, source_hash


def _validate_enriched_against_source(
    enriched: list[dict[str, Any]], source: list[dict[str, Any]]
) -> None:
    """Prove feature fields did not alter raw candidate attributes/ranks/scores."""
    if len(enriched) != len(source):
        raise ValueError("Enriched/source candidate query counts differ.")
    for enriched_record, source_record in zip(enriched, source):
        if enriched_record.get("query_id") != source_record.get("query_id"):
            raise ValueError("Enriched/source candidate query order differs.")
        for arm in ("dense", "lexical"):
            enriched_candidates = enriched_record[arm]
            source_candidates = source_record[arm]
            if len(enriched_candidates) != len(source_candidates):
                raise ValueError("Enriched/source candidate arm counts differ.")
            for with_features, original in zip(enriched_candidates, source_candidates):
                without_features = {
                    key: value for key, value in with_features.items() if key != "features"
                }
                if without_features != original:
                    raise ValueError(
                        "Enriched cache changed raw candidate fields/order/ranks/scores."
                    )


def _post_debug(
    task_id: str,
    token: str,
    payload: dict[str, object],
) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{REFEREE_URL}/v1/tasks/{task_id}/debug",
        data=json.dumps(payload).encode(),
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    started = time.monotonic()
    attempts = 0
    while True:
        attempts += 1
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.load(response)
                if not isinstance(body, dict):
                    raise ValueError("Referee debug response must be an object.")
                return {
                    "body": body,
                    "attempts": attempts,
                    "elapsed_ms": (time.monotonic() - started) * 1000,
                }
        except urllib.error.HTTPError as error:
            if error.code != 429 or attempts > MAX_429_RETRIES:
                raise
            remaining = MAX_BACKOFF_SECONDS - (time.monotonic() - started)
            if remaining <= 0:
                raise
            retry_after = 0.0
            header = error.headers.get("Retry-After")
            if header:
                try:
                    retry_after = max(0.0, float(header))
                except ValueError:
                    retry_after = 0.0
            nominal = min(0.5 * (2 ** (attempts - 1)), 8.0)
            delay = max(random.uniform(nominal / 2, nominal), retry_after)
            if delay > remaining:
                raise
            time.sleep(delay)


def _debug_payload(
    requests: list[SearchRequest],
    candidate_records: dict[str, dict[str, Any]],
) -> dict[str, object]:
    query_ids: list[str] = []
    result_rows: list[dict[str, object]] = []
    for request in requests:
        record = candidate_records[request.query_id]
        query_ids.append(request.query_id)
        candidates: dict[str, dict[str, Any]] = {}
        for arm in ("dense", "lexical"):
            for candidate in record[arm]:
                candidates.setdefault(str(candidate["doc_id"]), candidate)
        for rank, doc_id in enumerate(candidates, 1):
            result_rows.append(
                {
                    "query_id": request.query_id,
                    "doc_id": doc_id,
                    "rank": rank,
                    "score": 0.0,
                }
            )
    return {
        "query_ids": query_ids,
        "results": result_rows,
        "include_target": True,
    }


def _parse_debug_batch(
    body: dict[str, Any],
    requests: list[SearchRequest],
    candidate_records: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    raw_queries = body.get("queries")
    if not isinstance(raw_queries, dict):
        raise ValueError("Referee debug response is missing queries.")
    output: list[dict[str, Any]] = []
    for request in requests:
        raw_query = raw_queries.get(request.query_id)
        if not isinstance(raw_query, dict) or not isinstance(
            raw_query.get("results"), list
        ):
            raise ValueError("Referee debug response omitted a training query.")
        expected_ids = {
            str(candidate["doc_id"])
            for arm in ("dense", "lexical")
            for candidate in candidate_records[request.query_id][arm]
        }
        grades: dict[str, int] = {}
        for result in raw_query["results"]:
            if not isinstance(result, dict):
                raise ValueError("Referee debug result is malformed.")
            doc_id = result.get("doc_id")
            grade = result.get("relevance_grade")
            if (
                not isinstance(doc_id, str)
                or doc_id not in expected_ids
                or isinstance(grade, bool)
                or not isinstance(grade, int)
                or grade < 0
                or doc_id in grades
            ):
                raise ValueError("Referee returned invalid candidate judgment data.")
            grades[doc_id] = grade
        if set(grades) != expected_ids:
            raise ValueError("Referee judgments do not cover the candidate union.")
        raw_targets = raw_query.get("target")
        targets: list[dict[str, Any]] = []
        if raw_targets is not None:
            if not isinstance(raw_targets, list):
                raise ValueError("Referee target judgments are malformed.")
            for item in raw_targets:
                if (
                    not isinstance(item, dict)
                    or not isinstance(item.get("doc_id"), str)
                    or isinstance(item.get("relevance_grade"), bool)
                    or not isinstance(item.get("relevance_grade"), int)
                    or item["relevance_grade"] <= 0
                ):
                    raise ValueError("Referee target judgment is malformed.")
                targets.append(
                    {
                        "doc_id": item["doc_id"],
                        "grade": item["relevance_grade"],
                    }
                )
        output.append(
            {"query_id": request.query_id, "grades": grades, "targets": targets}
        )
    return output


def collect_training_judgments(
    task: dict[str, Any],
    requests: list[SearchRequest],
    candidate_records: list[dict[str, Any]],
    source_sha256: str,
    token: str,
    checkpoint_path: Path,
    grades_path: Path,
    targets_path: Path,
) -> dict[str, object]:
    """Checkpoint batched debug judgments; never complete or inspect another split."""
    task_info = task.get("task")
    task_id = task_info.get("id") if isinstance(task_info, dict) else None
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("Training task is missing its task ID.")
    candidate_by_id = {record["query_id"]: record for record in candidate_records}
    if len(candidate_by_id) != len(requests) or set(candidate_by_id) != {
        request.query_id for request in requests
    }:
        raise ValueError("Candidate cache does not exactly match training task.")
    completed = _read_checkpoint(
        checkpoint_path, requests, source_sha256, task_id
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.touch(exist_ok=True)
    pending = [request for request in requests if request.query_id not in completed]
    batch_count = 0
    attempt_count = 0
    elapsed_sum_ms = 0.0
    for start in range(0, len(pending), BATCH_QUERIES):
        batch = pending[start:start + BATCH_QUERIES]
        posted = _post_debug(
            task_id,
            token,
            _debug_payload(batch, candidate_by_id),
        )
        parsed = _parse_debug_batch(posted["body"], batch, candidate_by_id)
        with checkpoint_path.open("a", encoding="utf-8") as checkpoint:
            for record in parsed:
                full_record = {
                    **record,
                    "task_id": task_id,
                    "source_sha256": source_sha256,
                }
                checkpoint.write(json.dumps(full_record, ensure_ascii=False) + "\n")
                completed[record["query_id"]] = full_record
            checkpoint.flush()
            os.fsync(checkpoint.fileno())
        batch_count += 1
        attempt_count += int(posted["attempts"])
        elapsed_sum_ms += float(posted["elapsed_ms"])

    if len(completed) != len(requests):
        raise ValueError("Referee debug judgment collection is incomplete.")
    ordered = [completed[request.query_id] for request in requests]
    grade_counts: dict[str, int] = {}
    candidate_judgments = 0
    target_judgments = 0
    for record in ordered:
        candidate_judgments += len(record["grades"])
        target_judgments += len(record["targets"])
        for grade in record["grades"].values():
            label = str(grade)
            grade_counts[label] = grade_counts.get(label, 0) + 1
    _atomic_jsonl(
        grades_path,
        [
            {"query_id": record["query_id"], "grades": record["grades"]}
            for record in ordered
        ],
    )
    _atomic_jsonl(
        targets_path,
        [
            {"query_id": record["query_id"], "targets": record["targets"]}
            for record in ordered
        ],
    )
    return {
        "queries": len(ordered),
        "debug_batches": batch_count,
        "debug_request_attempts": attempt_count,
        "debug_elapsed_ms_sum": elapsed_sum_ms,
        "candidate_judgments": candidate_judgments,
        "target_judgments": target_judgments,
        "grade_counts": grade_counts,  # aggregate counts only; no labels/IDs
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-file", type=Path, default=DEFAULT_TASK_FILE)
    parser.add_argument("--token-file", type=Path, default=DEFAULT_TOKEN_FILE)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=DEFAULT_ENRICHED_CACHE,
    )
    parser.add_argument(
        "--source-candidates", type=Path, default=DEFAULT_CANDIDATE_CACHE
    )
    parser.add_argument("--output-prefix", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        with args.task_file.open(encoding="utf-8") as source:
            task = json.load(source)
        requests = _task_requests(task)
        candidate_records, source_sha256 = _load_enriched_cache(args.candidates, requests)
        source_records, expected_sha256 = _load_candidate_cache(
            args.source_candidates, requests
        )
        if source_sha256 != expected_sha256:
            raise ValueError("Enriched features do not cite the current training cache.")
        _validate_enriched_against_source(candidate_records, source_records)
        # The input is enriched, but validation uses the same cached dense /
        # lexical IDs and does not alter the raw scores or source ranks.
        token = args.token_file.read_text(encoding="utf-8").strip()
        if not token:
            raise ValueError("The assigned training token file is empty.")
        grades_path = args.output_prefix.with_name(
            args.output_prefix.name + "_grades.jsonl"
        )
        targets_path = args.output_prefix.with_name(
            args.output_prefix.name + "_targets.jsonl"
        )
        report = collect_training_judgments(
            task,
            requests,
            candidate_records,
            source_sha256,
            token,
            args.output_prefix.with_name(args.output_prefix.name + "_judgments.partial"),
            grades_path,
            targets_path,
        )
    except (OSError, ValueError, urllib.error.URLError) as error:
        parser.error(str(error))
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
