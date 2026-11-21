"""Run queries from a CSV file and write ranked results to another CSV."""

from __future__ import annotations

import argparse
import csv
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from turbopuffer import Turbopuffer
from turbopuffer.types import Row

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
            _write_query_rows(writer, request, rows)


def _write_query_rows(
    writer: Any, request: SearchRequest, rows: Iterable[Row]
) -> None:
    if not rows:
        writer.writerow([request.query_id, -1, -1, -1])
        return

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


def _write_counterfactual_arms(
    treatment_path: Path,
    control_path: Path,
    requests: list[SearchRequest],
    searcher: Searcher,
    top_k: int,
) -> None:
    if treatment_path.resolve() == control_path.resolve():
        raise ValueError("treatment and control output paths must differ")
    with (
        treatment_path.open("w", newline="", encoding="utf-8") as treatment_file,
        control_path.open("w", newline="", encoding="utf-8") as control_file,
    ):
        treatment_writer = csv.writer(treatment_file)
        control_writer = csv.writer(control_file)
        header = ["query_id", "doc_id", "rank", "score"]
        treatment_writer.writerow(header)
        control_writer.writerow(header)
        for arms in searcher.search_counterfactual_arms(requests, top_k):
            _write_query_rows(
                treatment_writer, arms.request, arms.treatment_rows
            )
            _write_query_rows(control_writer, arms.request, arms.control_rows)


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
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument(
        "--counterfactual-control",
        type=Path,
        help=(
            "Also write the matched original-ranking control here and write "
            "the counterfactual treatment to --output."
        ),
    )
    parser.add_argument(
        "--lexical-only",
        action="store_true",
        help="Disable hybrid fusion and use lexical BM25 only.",
    )
    parser.add_argument("--dense-weight", type=float, default=2.0)
    parser.add_argument("--lexical-weight", type=float, default=1.0)
    parser.add_argument("--rrf-constant", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be greater than zero")
    if args.batch_size < 1:
        parser.error("--batch-size must be greater than zero")
    if args.rrf_constant < 0:
        parser.error("--rrf-constant must not be negative")
    if args.dense_weight < 0 or args.lexical_weight < 0:
        parser.error("hybrid weights must not be negative")
    if not args.lexical_only and args.dense_weight + args.lexical_weight == 0:
        parser.error("at least one hybrid weight must be greater than zero")
    if not os.environ.get("TURBOPUFFER_API_KEY"):
        parser.error("Set TURBOPUFFER_API_KEY before querying.")

    try:
        requests = _read_queries(args.queries)
        with Turbopuffer(region=args.region) as client:
            namespace = client.namespace(args.namespace)
            hybrid_weights = (
                None
                if args.lexical_only
                else (args.dense_weight, args.lexical_weight)
            )
            strategy = TPufStrategy(
                hybrid_weights=hybrid_weights,
                rank_constant=args.rrf_constant,
            )
            searcher = Searcher(namespace, strategy, args.batch_size)
            if args.counterfactual_control:
                _write_counterfactual_arms(
                    args.output,
                    args.counterfactual_control,
                    requests,
                    searcher,
                    args.top_k,
                )
            else:
                _write_results(args.output, requests, searcher, args.top_k)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(f"Wrote results for {len(requests)} queries to {args.output}.")
    if args.counterfactual_control:
        print(f"Wrote matched control results to {args.counterfactual_control}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
