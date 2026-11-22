"""Run queries from a CSV file and write ranked results to another CSV."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sqlite3
from pathlib import Path

from turbopuffer import APIStatusError, Turbopuffer
from turbopuffer.types import Row

from tpuf_optimize.benchmark import run_benchmark
from tpuf_optimize.searcher import SearchRequest, Searcher
from tpuf_optimize.strategy import TPufStrategy


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
        seen = set()
        for query_number, row in enumerate(reader, start=1):
            text = (row.get("query") or "").strip()
            locale = (row.get("locale") or "").strip()
            if not text or not locale:
                raise ValueError(
                    f"{path}, row {query_number + 1}: query and locale must not be empty"
                )
            query_id = (row.get("query_id") or "").strip()
            if "query_id" in reader.fieldnames and not query_id:
                raise ValueError(f"{path}, row {query_number + 1}: missing query_id")
            query_id = query_id or f"q{query_number}"
            if query_id in seen:
                raise ValueError(f"{path}, row {query_number + 1}: duplicate query_id")
            seen.add(query_id)
            requests.append(SearchRequest(query_id, text, locale))

    return requests


def _ranked_rows(
    request: SearchRequest, rows: list[Row]
) -> list[tuple[str, str, int, float]]:
    if not rows:
        return [(request.query_id, "-1", -1, -1)]
    ranked = []
    seen: set[str] = set()
    for rank, result in enumerate(rows, start=1):
        document_id = str(result.id)
        if not document_id or document_id == "-1" or document_id in seen:
            raise ValueError(f"Invalid or duplicate document in results for {request.query_id}")
        seen.add(document_id)
        score = result["$dist"]
        if not isinstance(score, (float, int)) or not math.isfinite(score):
            raise ValueError(f"Missing or invalid retrieval score for {request.query_id}")
        ranked.append((request.query_id, document_id, rank, float(score)))
    return ranked


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
            writer.writerows(_ranked_rows(request, rows))


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
        "--batch-size",
        type=int,
        help=(
            "Defaults to 1 in benchmark mode and 16 otherwise "
            f"(capped at {Searcher.MAX_QUERIES_PER_BATCH})."
        ),
    )
    parser.add_argument(
        "--benchmark-db", type=Path,
        help="Record rankings and latency together in a private SQLite file.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Continue the same benchmark run in an existing checkpoint.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be greater than zero")
    if args.resume and args.benchmark_db is None:
        parser.error("--resume requires --benchmark-db")
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = 1 if args.benchmark_db else Searcher.MAX_QUERIES_PER_BATCH
    if batch_size < 1:
        parser.error("--batch-size must be greater than zero")
    if not os.environ.get("TURBOPUFFER_API_KEY"):
        parser.error("Set TURBOPUFFER_API_KEY before querying.")

    try:
        requests = _read_queries(args.queries)
        if args.benchmark_db and len({
            args.queries.resolve(), args.output.resolve(), args.benchmark_db.resolve()
        }) != 3:
            raise ValueError("Benchmark input, output, and checkpoint must be separate files")
        client = (
            Turbopuffer(region=args.region, max_retries=0)
            if args.benchmark_db
            else Turbopuffer(region=args.region)
        )
        with client:
            namespace = client.namespace(args.namespace)
            searcher = Searcher(
                namespace,
                TPufStrategy(),
                batch_size,
                serial_requests=True,
            )
            if args.benchmark_db:
                run_benchmark(
                    args.benchmark_db, args.output, requests, searcher, args.top_k,
                    (args.region, args.namespace), args.resume, _ranked_rows,
                )
            else:
                _write_results(args.output, requests, searcher, args.top_k)
    except (OSError, ValueError, sqlite3.Error) as error:
        parser.error(str(error))
    except APIStatusError as error:
        if not args.benchmark_db:
            raise
        parser.error(f"Benchmark stopped after Turbopuffer HTTP {error.status_code}")

    print(f"Wrote results for {len(requests)} queries to {args.output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
