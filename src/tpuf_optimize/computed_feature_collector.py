"""Collect fixed ANN/BM25 candidates with non-ranking computed signals.

This experiment makes two top-100 requests per query, matching the pinned
training collector's Voyage passage ANN and weighted locale BM25 arms. The
computed fields are metadata only and never participate in rank_by.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, cast

from turbopuffer import Turbopuffer
from turbopuffer.types import Row
from turbopuffer.types.namespace_multi_query_params import Query

from tpuf_optimize.searcher import SearchRequest, Searcher
from tpuf_optimize.strategy import TPufStrategy
from tpuf_optimize.training_collector import (
    DEFAULT_TASK_FILE,
    DEFAULT_TEMP_DIR,
    DEFAULT_TOKEN_FILE,
    HYBRID_DEPTH,
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
    checkpoint = args.checkpoint or args.output.with_suffix(args.output.suffix + ".partial")
    try:
        task = _get_task(args.task_file, args.token_file)
        requests = _task_requests(task)
        with Turbopuffer(region=args.region) as client:
            namespace = client.namespace(args.namespace)
            count = collect_feature_candidates(
                namespace, requests, checkpoint, args.output
            )
        print(
            f"Collected feature-enriched candidates for {count} training queries; "
            f"{2 * count} top-{HYBRID_DEPTH} arm queries."
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
