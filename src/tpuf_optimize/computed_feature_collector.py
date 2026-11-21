"""Collect fixed ANN/BM25 candidates with non-ranking computed signals.

This experiment makes two top-100 requests per query, matching the pinned
training collector's Voyage passage ANN and weighted locale BM25 arms. The
computed fields are metadata only and never participate in rank_by.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any, cast

from turbopuffer import Turbopuffer
from turbopuffer._exceptions import RateLimitError
from turbopuffer.types import Row
from turbopuffer.types.namespace_multi_query_params import Query

from tpuf_optimize.searcher import SearchRequest, Searcher
from tpuf_optimize.strategy import TPufStrategy
from tpuf_optimize.training_collector import (
    DEFAULT_TASK_FILE,
    DEFAULT_TEMP_DIR,
    DEFAULT_TOKEN_FILE,
    HYBRID_DEPTH,
    _atomic_json,
    _append_checkpoint,
    _checkpoint_read,
    _get_task,
    _task_requests,
)

FEATURES = ("title_bm25", "passage_bm25", "brand_bm25", "title_full_coverage")
_MISSING = object()


class ComputedFeatureQueryStrategy(TPufStrategy):
    """Build baseline retrieval requests with result-only BM25 features."""

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

    def computed_attributes(self, query_text: str, locale: str) -> dict[str, Any]:
        normalized_locale = self.normalize_locale(locale)
        analyzer = self.LOCALE_ANALYZERS.get(normalized_locale)
        suffix = analyzer[0] if analyzer is not None else self.DEFAULT_ANALYZER
        return {
            "title_bm25": (f"title__{suffix}", "BM25", query_text),
            "passage_bm25": (f"passage__{suffix}", "BM25", query_text),
            "brand_bm25": (f"brand__{suffix}", "BM25", query_text),
        }

    def query(self, query_text: str, locale: str, top_k: int) -> Query:
        if not query_text.strip():
            raise ValueError("query_text must not be empty")
        if not locale.strip():
            raise ValueError("locale must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be greater than zero")
        normalized_locale = self.normalize_locale(locale)
        rank_by: Any
        if self.mode == "dense":
            # Keep the pinned Voyage-4 native-embedding ANN request verbatim.
            rank_by = (
                "passage__embed__voyage_4",
                "ANN",
                ("Embed", query_text),
            )
        else:
            # Keep the pinned parent 2x title + 1x description BM25 expression.
            rank_by = self.rank_by(query_text, normalized_locale)
        return cast(
            Query,
            {
                "rank_by": rank_by,
                "filters": ("locale", "Eq", normalized_locale),
                "top_k": top_k,
                "include_attributes": ["title", "brand"],
                "compute_attributes": {
                    **self.computed_attributes(query_text, normalized_locale),
                    # BM25 operators expose scores, not a boolean coverage op.
                    # This marker is extracted from the returned raw title below.
                },
            },
        )


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _title_full_coverage(query: str, title: object) -> bool | None:
    if not isinstance(title, str):
        return None
    query_tokens = set(TPufStrategy.pretokenize(query))
    if not query_tokens:
        return None
    title_tokens = set(TPufStrategy.pretokenize(title))
    return query_tokens.issubset(title_tokens)


def _candidate_list(rows: list[Row], query_text: str) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in rows:
        values = row.model_dump()
        doc_id = values.get("id")
        score = _number(values.get("$dist"))
        if doc_id is None or score is None:
            raise ValueError("Turbopuffer returned a candidate without id or numeric $dist.")
        key = str(doc_id)
        if key in seen:
            raise ValueError(f"Turbopuffer returned duplicate candidate {key!r}.")
        seen.add(key)
        title = values.get("title")
        features: dict[str, object] = {}
        # Keep null for absent/unsupported signals; do not turn a real zero into
        # null or treat a missing brand as a zero BM25 score.
        for name in ("title_bm25", "passage_bm25", "brand_bm25"):
            raw = values.get(name, _MISSING)
            features[name] = None if raw is _MISSING else _number(raw)
        features["title_full_coverage"] = _title_full_coverage(query_text, title)
        candidate: dict[str, object] = {
            "doc_id": key,
            "score": score,
            "title": title,
            "brand": values.get("brand"),
            "features": features,
        }
        candidates.append(candidate)
    return candidates


def _read_feature_checkpoint(
    path: Path, requests: list[SearchRequest]
) -> dict[str, dict[str, Any]]:
    """Reject legacy candidate caches lacking this experiment's feature payload."""
    records = _checkpoint_read(path, requests)
    for record in records.values():
        if record.get("feature_schema") != list(FEATURES):
            raise ValueError(
                "Checkpoint is not computed-feature data; choose a fresh "
                "checkpoint instead of reusing a legacy cache."
            )
        for arm in ("dense", "lexical"):
            for candidate in record[arm]:
                if not isinstance(candidate, dict) or not isinstance(
                    candidate.get("features"), dict
                ):
                    raise ValueError("Computed-feature checkpoint is incomplete.")
                if set(candidate["features"]) != set(FEATURES):
                    raise ValueError("Computed-feature checkpoint has a stale schema.")
    return records


def collect_feature_candidates(
    namespace: object,
    requests: list[SearchRequest],
    checkpoint_path: Path,
    output_path: Path,
) -> int:
    """Collect two ordered top-100 arms per query with per-result feature data."""
    if checkpoint_path.resolve() == output_path.resolve():
        raise ValueError("The checkpoint and complete output must be distinct.")
    completed = _read_feature_checkpoint(checkpoint_path, requests)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.touch(exist_ok=True)
    strategy = ComputedFeatureQueryStrategy("dense")
    searcher = Searcher(namespace, strategy, batch_size=1)  # type: ignore[arg-type]
    for request in requests:
        if request.query_id in completed:
            continue
        arms: dict[str, list[dict[str, object]]] = {}
        for mode in ("dense", "lexical"):
            strategy.mode = mode
            found = list(searcher.search([request], HYBRID_DEPTH))
            if len(found) != 1 or found[0][0] != request:
                raise ValueError("Searcher did not return exactly one matching query.")
            arms[mode] = _candidate_list(found[0][1], request.text)
        record: dict[str, Any] = {
            "query_id": request.query_id,
            "query_text": request.text,
            "locale": request.locale,
            "dense": arms["dense"],
            "lexical": arms["lexical"],
            "feature_schema": list(FEATURES),
        }
        _append_checkpoint(checkpoint_path, record)
        completed[request.query_id] = record

    if len(completed) != len(requests):
        raise ValueError("Feature candidate collection is missing task queries.")
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


ID_FILTER_METHOD = "cached_union_id_filter_v1"
ID_FILTER_MAX_RETRIES = 3
ID_FILTER_MAX_BACKOFF_SECONDS = 30.0


def _load_candidate_cache(
    path: Path, requests: list[SearchRequest]
) -> tuple[list[dict[str, Any]], str]:
    """Read and validate the pinned training collector's cache, without edits."""
    digest = hashlib.sha256()
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("rb") as source:
        for line_number, raw_line in enumerate(source, 1):
            digest.update(raw_line)
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid JSON in source candidates"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            query_id = record.get("query_id")
            if not isinstance(query_id, str) or query_id in seen:
                raise ValueError(f"{path}:{line_number}: invalid/duplicate query_id")
            if (
                not isinstance(record.get("query_text"), str)
                or not isinstance(record.get("locale"), str)
                or not isinstance(record.get("dense"), list)
                or not isinstance(record.get("lexical"), list)
            ):
                raise ValueError(f"{path}:{line_number}: incomplete source candidates")
            for arm in ("dense", "lexical"):
                ids: set[str] = set()
                for candidate in record[arm]:
                    if not isinstance(candidate, dict) or candidate.get("doc_id") is None:
                        raise ValueError(f"{path}:{line_number}: malformed {arm} candidate")
                    doc_id = str(candidate["doc_id"])
                    if doc_id in ids:
                        raise ValueError(f"{path}:{line_number}: duplicate {arm} doc_id")
                    ids.add(doc_id)
                    if "features" in candidate:
                        raise ValueError(
                            "Source cache already contains feature payloads; "
                            "use an unmodified pinned candidate cache."
                        )
            seen.add(query_id)
            records.append(record)

    if len(records) != len(requests) or [r["query_id"] for r in records] != [
        request.query_id for request in requests
    ]:
        raise ValueError("Source candidate IDs/order do not match the training task.")
    for record, request in zip(records, requests):
        if (
            record["query_text"] != request.text
            or record["locale"] != ComputedFeatureQueryStrategy.normalize_locale(
                request.locale
            )
        ):
            raise ValueError("Source candidate query text/locale does not match task.")
    return records, digest.hexdigest()


def _query_id_filtered_union(
    namespace: object,
    query_record: dict[str, Any],
) -> tuple[dict[str, dict[str, object]], int, float, dict[str, object]]:
    """Request only cached IDs once, retaining their retrieval scores/ranks locally."""
    query_text = str(query_record["query_text"])
    locale = str(query_record["locale"])
    strategy = ComputedFeatureQueryStrategy("dense")
    candidate_ids = list(
        dict.fromkeys(
            str(candidate["doc_id"])
            for arm in ("dense", "lexical")
            for candidate in query_record[arm]
        )
    )
    if not candidate_ids:
        raise ValueError("Cannot compute features for an empty candidate union.")
    query: Query = cast(
        Query,
        {
            "filters": (
                "And",
                (
                    ("locale", "Eq", locale),
                    ("id", "In", candidate_ids),
                ),
            ),
            "rank_by": ("id", "asc"),
            "top_k": len(candidate_ids),
            "include_attributes": ["title", "brand"],
            "compute_attributes": strategy.computed_attributes(query_text, locale),
        },
    )
    started = time.monotonic()
    attempts = 0
    while True:
        attempts += 1
        try:
            response = cast(Any, namespace).query(**query)
            break
        except RateLimitError as error:
            if attempts > ID_FILTER_MAX_RETRIES:
                raise
            remaining = ID_FILTER_MAX_BACKOFF_SECONDS - (
                time.monotonic() - started
            )
            if remaining <= 0:
                raise
            nominal = min(0.5 * (2 ** (attempts - 1)), 8.0)
            delay = max(
                random.uniform(nominal / 2, nominal),
                Searcher._retry_after_seconds(error),
            )
            if delay > remaining:
                raise
            time.sleep(delay)

    elapsed_ms = (time.monotonic() - started) * 1000
    by_id: dict[str, dict[str, object]] = {}
    for row in response.rows or []:
        values = row.model_dump()
        raw_id = values.get("id")
        if raw_id is None:
            raise ValueError("ID-filtered feature query returned a row without id.")
        doc_id = str(raw_id)
        if doc_id in by_id:
            raise ValueError(f"ID-filtered feature query duplicated {doc_id!r}.")
        computed: dict[str, object] = {
            name: _number(values.get(name))
            for name in ("title_bm25", "passage_bm25", "brand_bm25")
        }
        computed["title_full_coverage"] = _title_full_coverage(
            query_text, values.get("title")
        )
        by_id[doc_id] = computed
    if set(by_id) != set(candidate_ids):
        raise ValueError(
            "ID-filtered feature result did not exactly cover the cached candidate union."
        )
    raw_performance = getattr(response, "performance", None)
    performance = (
        raw_performance.model_dump()
        if raw_performance is not None and hasattr(raw_performance, "model_dump")
        else {}
    )
    return by_id, attempts, elapsed_ms, performance


def _feature_summary(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, dict[str, int]]]:
    """Count availability and true zeros by locale, source arm and feature."""
    summary: dict[str, dict[str, dict[str, int]]] = {}
    for record in records:
        locale = str(record["locale"])
        for arm in ("dense", "lexical"):
            for feature in FEATURES:
                counter = summary.setdefault(locale, {}).setdefault(
                    f"{arm}.{feature}",
                    {
                        "available": 0,
                        "unavailable": 0,
                        "zero": 0,
                        "nonzero": 0,
                        "true": 0,
                        "false": 0,
                    },
                )
                for candidate in record[arm]:
                    value = candidate.get("features", {}).get(feature)
                    if feature == "title_full_coverage":
                        if value is None:
                            counter["unavailable"] += 1
                        else:
                            counter["available"] += 1
                            counter["true" if value else "false"] += 1
                    elif value is None:
                        counter["unavailable"] += 1
                    else:
                        counter["available"] += 1
                        counter["zero" if float(value) == 0.0 else "nonzero"] += 1
    return summary


def collect_cached_union_features(
    namespace: object,
    requests: list[SearchRequest],
    source_records: list[dict[str, Any]],
    source_sha256: str,
    checkpoint_path: Path,
    output_path: Path,
    task_id: str,
) -> dict[str, Any]:
    """Checkpoint one ID-filtered compute_attributes query per cached query."""
    if checkpoint_path.resolve() == output_path.resolve():
        raise ValueError("The checkpoint and complete output must be distinct.")
    completed = _checkpoint_read(checkpoint_path, requests)
    for record in completed.values():
        if (
            record.get("feature_collection_method") != ID_FILTER_METHOD
            or record.get("feature_schema") != list(FEATURES)
        ):
            raise ValueError("Checkpoint is not from this ID-filtered collection mode.")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.touch(exist_ok=True)
    source_by_id = {record["query_id"]: record for record in source_records}
    for request in requests:
        if request.query_id in completed:
            cached_record = completed[request.query_id]
            if cached_record.get("source_sha256") != source_sha256:
                raise ValueError("Checkpoint was created from a different source cache.")
            continue
        source = source_by_id[request.query_id]
        features_by_id, attempts, elapsed_ms, performance = (
            _query_id_filtered_union(namespace, source)
        )
        record: dict[str, Any] = {
            "query_id": request.query_id,
            "query_text": request.text,
            "locale": source["locale"],
            "dense": [
                {**candidate, "features": features_by_id[str(candidate["doc_id"])]}
                for candidate in source["dense"]
            ],
            "lexical": [
                {**candidate, "features": features_by_id[str(candidate["doc_id"])]}
                for candidate in source["lexical"]
            ],
            "feature_schema": list(FEATURES),
            "feature_collection_method": ID_FILTER_METHOD,
            "source_sha256": source_sha256,
            "request_attempts": attempts,
            "client_elapsed_ms": elapsed_ms,
            "server_performance": performance,
        }
        for arm in ("dense", "lexical"):
            before = [
                (str(candidate["doc_id"]), candidate["score"])
                for candidate in source[arm]
            ]
            after = [
                (str(candidate["doc_id"]), candidate["score"])
                for candidate in record[arm]
            ]
            if before != after:
                raise ValueError("Feature enrichment altered cached rank/order/score.")
        _append_checkpoint(checkpoint_path, record)
        completed[request.query_id] = record

    if len(completed) != len(requests):
        raise ValueError("ID-filtered feature collection did not cover the task.")
    ordered_records = [completed[request.query_id] for request in requests]
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
            for record in ordered_records:
                output.write(
                    json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    successful_requests = sum(
        int(record.get("request_attempts", 0)) > 0 for record in ordered_records
    )
    total_attempts = sum(int(record.get("request_attempts", 0)) for record in ordered_records)
    elapsed_values = [
        float(record.get("client_elapsed_ms", 0.0)) for record in ordered_records
    ]
    server_values = [
        float(record["server_performance"]["server_total_ms"])
        for record in ordered_records
        if isinstance(record.get("server_performance"), dict)
        and isinstance(
            record["server_performance"].get("server_total_ms"), (int, float)
        )
    ]
    return {
        "method": ID_FILTER_METHOD,
        "task_id": task_id,
        "source_sha256": source_sha256,
        "query_count": len(ordered_records),
        "successful_feature_requests": successful_requests,
        "total_request_attempts": total_attempts,
        "client_elapsed_ms_sum": sum(elapsed_values),
        "client_elapsed_ms_mean": (
            sum(elapsed_values) / len(elapsed_values) if elapsed_values else None
        ),
        "server_total_ms_sum": sum(server_values),
        "server_total_ms_mean": (
            sum(server_values) / len(server_values) if server_values else None
        ),
        "feature_counts_by_locale_and_arm": _feature_summary(ordered_records),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-file", type=Path, default=DEFAULT_TASK_FILE)
    parser.add_argument("--token-file", type=Path, default=DEFAULT_TOKEN_FILE)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_TEMP_DIR / "esci_research_training_computed_candidates.jsonl",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--source-candidates",
        type=Path,
        help="Use one ID-filtered request/query on this existing dense+lexical cache.",
    )
    parser.add_argument("--namespace", default="esci-products")
    parser.add_argument(
        "--region",
        default=os.environ.get("TURBOPUFFER_REGION", "gcp-us-central1"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if not os.environ.get("TURBOPUFFER_API_KEY"):
        parser.error("Set TURBOPUFFER_API_KEY before querying.")
    if args.source_candidates is not None and args.output == (
        DEFAULT_TEMP_DIR / "esci_research_training_computed_candidates.jsonl"
    ):
        args.output = DEFAULT_TEMP_DIR / "esci_research_training_idfiltered_candidates.jsonl"
    checkpoint = args.checkpoint or args.output.with_suffix(args.output.suffix + ".partial")
    try:
        task = _get_task(args.task_file, args.token_file)
        requests = _task_requests(task)
        if args.source_candidates is not None:
            source_records, source_sha256 = _load_candidate_cache(
                args.source_candidates, requests
            )
        with Turbopuffer(
            region=args.region, max_retries=0, telemetry=False
        ) as client:
            namespace = client.namespace(args.namespace)
            if args.source_candidates is None:
                count = collect_feature_candidates(
                    namespace, requests, checkpoint, args.output
                )
                report = {
                    "method": "paired_ann_bm25",
                    "queries": count,
                    "successful_arm_requests": 2 * count,
                }
            else:
                report = collect_cached_union_features(
                    namespace,
                    requests,
                    source_records,
                    source_sha256,
                    checkpoint,
                    args.output,
                    str(task.get("task", {}).get("id", "")),
                )
        print(json.dumps(report, sort_keys=True))
        if args.source_candidates is not None:
            _atomic_json(args.output.with_suffix(".manifest.json"), report)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    except RateLimitError:
        print(
            f"Rate limit persisted; checkpoint preserved at {checkpoint}.",
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
