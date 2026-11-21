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
    namespace_action = parser.add_mutually_exclusive_group()
    namespace_action.add_argument(
        "--drop-existing",
        action="store_true",
        help="Delete the target namespace before indexing it.",
    )
    namespace_action.add_argument(
        "--skip-existing-docs",
        action="store_true",
        help="Check each batch for existing document IDs and skip those rows. "
        "Use only when resuming with the same schema.",
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
    if args.batch_size > TPufStrategy.MAX_BATCH_SIZE:
        parser.error(
            f"--batch-size cannot exceed {TPufStrategy.MAX_BATCH_SIZE} "
            "when using native embeddings"
        )
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be greater than zero")
    if not args.dry_run and not os.environ.get("TURBOPUFFER_API_KEY"):
        parser.error("Set TURBOPUFFER_API_KEY before indexing.")

    # cheat-at-search loads the product dataset lazily on this import.
    from cheat_at_search.esci_data import corpus  # type: ignore[attr-defined]

    strategy = TPufStrategy()
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
            if args.drop_existing:
                exists = any(
                    item.id == args.namespace
                    for item in client.namespaces(prefix=args.namespace)
                )
                if exists:
                    namespace.delete_all()
            skip_existing = args.skip_existing_docs and namespace.exists()
            result = Indexer(namespace, strategy, args.batch_size).index(
                tracked_documents(),
                skip_existing_docs=skip_existing,
            )

    print(
        f"Indexed {result.indexed:,} documents into {args.namespace!r}; "
        f"skipped {result.skipped:,} existing documents."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
