"""Index the ESCI product corpus into a Turbopuffer namespace."""

from __future__ import annotations

import argparse
import itertools
import math
import os
from collections.abc import Iterable, Iterator
from typing import Any, cast

import pandas as pd
from turbopuffer import Turbopuffer
from turbopuffer.types import AttributeSchemaParam, RowParam
from tqdm import tqdm


TEXT_COLUMNS = {
    "title": "title",
    "description": "description",
    "brand": "product_brand",
    "bullet_point": "product_bullet_point",
    "color": "product_color",
    "category": "product_category",
}
DOCUMENT_COLUMNS = {**TEXT_COLUMNS, "locale": "product_locale"}
FULL_TEXT_SCHEMA: dict[str, AttributeSchemaParam] = {
    name: {"type": "string", "full_text_search": True}
    for name in TEXT_COLUMNS
}


def _as_text(value: object) -> str | None:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    return text or None


def _document_rows(
    corpus: pd.DataFrame, limit: int | None = None
) -> Iterator[dict[str, Any]]:
    id_column = "doc_id" if "doc_id" in corpus.columns else "product_id"
    selected_columns = [id_column]
    selected_columns.extend(
        source for source in DOCUMENT_COLUMNS.values() if source in corpus.columns
    )
    output_names = {
        source: name
        for name, source in DOCUMENT_COLUMNS.items()
        if source in corpus.columns
    }

    emitted = 0
    for values in corpus[selected_columns].itertuples(index=False, name=None):
        if limit is not None and emitted >= limit:
            break
        document_id = _as_text(values[0])
        if document_id is None:
            continue

        document: dict[str, Any] = {"id": document_id}
        for source, value in zip(selected_columns[1:], values[1:]):
            text = _as_text(value)
            if text is not None:
                document[output_names[source]] = text
        yield document
        emitted += 1


def _batches(
    rows: Iterator[dict[str, Any]], batch_size: int
) -> Iterator[list[dict[str, Any]]]:
    while batch := list(itertools.islice(rows, batch_size)):
        yield batch


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="esci-products")
    parser.add_argument(
        "--region",
        default=os.environ.get("TURBOPUFFER_REGION", "gcp-us-central1"),
    )
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--limit", type=int)
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

    rows = _document_rows(corpus, args.limit)
    total_rows = len(corpus)
    if args.limit is not None:
        total_rows = min(args.limit, total_rows)

    description = "Counting ESCI" if args.dry_run else "Indexing ESCI"
    with tqdm(total=total_rows, desc=description, unit="doc") as progress:
        if args.dry_run:
            count = 0
            for _ in rows:
                count += 1
                progress.update(1)
            print(f"Would index {count:,} documents into {args.namespace!r}.")
            return 0

        total = 0
        with Turbopuffer(region=args.region) as client:
            namespace = client.namespace(args.namespace)
            for batch in _batches(rows, args.batch_size):
                namespace.write(
                    upsert_rows=cast(Iterable[RowParam], batch),
                    schema=FULL_TEXT_SCHEMA,
                )
                total += len(batch)
                progress.update(len(batch))

    print(f"Indexed {total:,} documents into {args.namespace!r}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
