"""Run queries from a CSV file and write ranked results to another CSV."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from turbopuffer import Turbopuffer

from tpuf_optimize.searcher import SearchRequest, Searcher
from tpuf_optimize.strategy import (
    DEFAULT_CONFIDENCE_GATE,
    DEFAULT_HYBRID_WEIGHTS,
    DEFAULT_RRF_RANK_CONSTANT,
    TPufStrategy,
)


def _read_queries(path: Path) -> list[SearchRequest]:
    with path.open(newline="", encoding="utf-8") as queries_file:
        reader = csv.DictReader(queries_file)
        if reader.fieldnames is None:
            raise ValueError(f"{path} is empty or missing a CSV header")

        missing_columns = {"query", "locale"} - set(reader.fieldnames)
        if missing_columns:
            names = ", ".join(sorted(missing_columns))
            raise ValueError(f"{path} is missing required column(s): {names}")

        requests = []
        for query_number, row in enumerate(reader, start=1):
            text = (row.get("query") or "").strip()
            locale = (row.get("locale") or "").strip()
            if not text or not locale:
                raise ValueError(
                    f"{path}, row {query_number + 1}: query and locale must not be empty"
                )
            requests.append(SearchRequest(f"q{query_number}", text, locale))

    return requests


def _write_results(
    output_path: Path,
    requests: list[SearchRequest],
    searcher: Searcher,
    top_k: int,
) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["query_id", "doc_id", "rank", "score"])

        for request, rows in searcher.search(requests, top_k):
            if not rows:
                writer.writerow([request.query_id, -1, -1, -1])
                continue

            seen_document_ids: set[str] = set()
            for rank, result in enumerate(rows, start=1):
                document_id = str(result.id)
                if document_id in seen_document_ids:
                    raise ValueError(
                        f"Duplicate document {document_id!r} in results for "
                        f"{request.query_id}"
                    )
                seen_document_ids.add(document_id)
                score = result["$dist"]
                if not isinstance(score, (float, int)):
                    raise ValueError(
                        f"Missing numeric BM25 score for document {document_id!r}"
                    )
                writer.writerow([request.query_id, document_id, rank, score])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--namespace", default="esci-products")
    parser.add_argument(
        "--region",
        default=os.environ.get("TURBOPUFFER_REGION", "gcp-us-central1"),
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--lexical-only",
        action="store_true",
        help="Disable Voyage dense retrieval and hybrid rank fusion.",
    )
    parser.add_argument(
        "--confidence-gate",
        choices=("identifier", "margin", "none"),
        default=DEFAULT_CONFIDENCE_GATE,
        help=(
            "Per-query hybrid weight gate; 'none' uses fixed 2:1 dense:lexical "
            "RRF."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=Searcher.MAX_QUERIES_PER_BATCH,
        help=(
            "Maximum input queries per batch; hybrid mode submits two retrieval "
            "queries per input and is capped at 8 input queries."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be greater than zero")
    if args.batch_size < 1:
        parser.error("--batch-size must be greater than zero")
    if not os.environ.get("TURBOPUFFER_API_KEY"):
        parser.error("Set TURBOPUFFER_API_KEY before querying.")

    try:
        requests = _read_queries(args.queries)
        with Turbopuffer(region=args.region) as client:
            namespace = client.namespace(args.namespace)
            strategy = TPufStrategy(
                hybrid_weights=(
                    None if args.lexical_only else DEFAULT_HYBRID_WEIGHTS
                ),
                rank_constant=DEFAULT_RRF_RANK_CONSTANT,
                confidence_gate=(
                    None
                    if args.lexical_only or args.confidence_gate == "none"
                    else args.confidence_gate
                ),
            )
            searcher = Searcher(namespace, strategy, args.batch_size)
            _write_results(args.output, requests, searcher, args.top_k)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(f"Wrote results for {len(requests)} queries to {args.output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
