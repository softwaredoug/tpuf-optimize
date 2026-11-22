"""Run queries from a CSV file and write ranked results to another CSV."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from turbopuffer import Turbopuffer

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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--lexical-candidate-fusion",
        action="store_true",
        help="Union all-token and exact-phrase candidates with the baseline channel.",
    )
    parser.add_argument(
        "--lexical-rerank",
        action="store_true",
        help="apply a bounded lexical position adjustment to BM25 candidates",
    )
    parser.add_argument(
        "--lexical-rerank-boost",
        type=float,
        default=TPufStrategy.LEXICAL_RERANK_MAX_BOOST,
        help="maximum normalized score adjustment (0 through 1)",
    )
    parser.add_argument(
        "--field-fusion",
        choices=("none", *TPufStrategy.FIELD_FUSION_CHANNELS),
        default="none",
        help="Add an existing lexical field/analyzer channel to the baseline rank.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be greater than zero")
    if args.batch_size < 1:
        parser.error("--batch-size must be greater than zero")
    if not 0.0 <= args.lexical_rerank_boost <= 1.0:
        parser.error("--lexical-rerank-boost must be between 0 and 1")
    if not os.environ.get("TURBOPUFFER_API_KEY"):
        parser.error("Set TURBOPUFFER_API_KEY before querying.")

    try:
        requests = _read_queries(args.queries)
        with Turbopuffer(region=args.region) as client:
            namespace = client.namespace(args.namespace)
            searcher = Searcher(
                namespace,
                TPufStrategy(field_fusion=args.field_fusion),
                args.batch_size,
                lexical_candidate_fusion=args.lexical_candidate_fusion,
                lexical_rerank=args.lexical_rerank,
                lexical_rerank_boost=args.lexical_rerank_boost,
            )
            _write_results(
                args.output,
                requests,
                searcher,
                args.top_k,
            )
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(f"Wrote results for {len(requests)} queries to {args.output}.")
    if args.lexical_candidate_fusion:
        print(
            "Lexical candidate coverage: "
            f"zero_result_arms={searcher.lexical_stats['zero_result_arms']}/"
            f"{searcher.lexical_stats['arm_queries']}, "
            f"mean_union_candidates="
            f"{searcher.lexical_stats['union_candidates'] / len(requests):.2f}."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
