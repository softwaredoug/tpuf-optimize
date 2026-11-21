"""Index the ESCI product corpus into a Turbopuffer namespace."""

from __future__ import annotations

import argparse
import os

from turbopuffer import Turbopuffer
from tqdm import tqdm

from tpuf_optimize.indexer import DEFAULT_BATCH_SIZE, Indexer
from tpuf_optimize.strategy import TPufStrategy


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="esci-products")
    parser.add_argument(
        "--region",
        default=os.environ.get("TURBOPUFFER_REGION", "gcp-us-central1"),
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--combined-text",
        action="store_true",
        help="Index title-emphasized combined text for the experimental strategy.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and count corpus documents without writing them.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be greater than zero")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be greater than zero")
    if not args.dry_run and not os.environ.get("TURBOPUFFER_API_KEY"):
        parser.error("Set TURBOPUFFER_API_KEY before indexing.")

    # cheat-at-search loads the product dataset lazily on this import.
    from cheat_at_search.esci_data import corpus  # type: ignore[attr-defined]

    strategy = TPufStrategy(combined_text=args.combined_text)
    documents = strategy.documents(corpus, args.limit)
    total_rows = len(corpus)
    if args.limit is not None:
        total_rows = min(args.limit, total_rows)

    description = "Counting ESCI" if args.dry_run else "Indexing ESCI"
    with tqdm(total=total_rows, desc=description, unit="doc") as progress:
        def tracked_documents():
            for document in documents:
                progress.update(1)
                yield document

        if args.dry_run:
            count = sum(1 for _ in tracked_documents())
            print(f"Would index {count:,} documents into {args.namespace!r}.")
            return 0

        with Turbopuffer(region=args.region) as client:
            namespace = client.namespace(args.namespace)
            total = Indexer(namespace, strategy, args.batch_size).index(
                tracked_documents()
            )

    print(f"Indexed {total:,} documents into {args.namespace!r}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
