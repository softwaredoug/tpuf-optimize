"""Generic batched writes to a Turbopuffer namespace."""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from turbopuffer.types import (
    AttributeSchemaParam,
    DistanceMetric,
    Filter,
    NamespaceQueryResponse,
    RowParam,
)

from tpuf_optimize.strategy import TPufStrategy


DEFAULT_BATCH_SIZE = 256


@dataclass(frozen=True)
class IndexResult:
    """Counts from an indexing run."""

    indexed: int
    skipped: int


class NamespaceWriter(Protocol):
    """The namespace query/write surface required by Indexer."""

    def write(
        self,
        *,
        upsert_rows: Iterable[RowParam],
        schema: dict[str, AttributeSchemaParam],
        distance_metric: DistanceMetric,
    ) -> object: ...

    def query(
        self,
        *,
        filters: Filter,
        top_k: int,
        include_attributes: list[str],
    ) -> NamespaceQueryResponse: ...


class Indexer:
    """Batch strategy-produced documents into namespace writes."""

    def __init__(
        self,
        namespace: NamespaceWriter,
        strategy: TPufStrategy,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be greater than zero")
        if batch_size > strategy.MAX_BATCH_SIZE:
            raise ValueError(
                f"batch_size cannot exceed {strategy.MAX_BATCH_SIZE} when "
                "using native embeddings"
            )
        self.namespace = namespace
        self.strategy = strategy
        self.batch_size = batch_size

    def index(
        self,
        documents: Iterable[RowParam],
        skip_existing_docs: bool = False,
    ) -> IndexResult:
        """Write documents in batches, optionally skipping IDs already in the namespace."""
        iterator = iter(documents)
        indexed = 0
        skipped = 0
        while batch := list(itertools.islice(iterator, self.batch_size)):
            if skip_existing_docs:
                existing_rows = self.namespace.query(
                    filters=("id", "In", [row["id"] for row in batch]),
                    top_k=len(batch),
                    include_attributes=["id"],
                ).rows or []
                existing_ids = {str(row.id) for row in existing_rows}
                skipped += sum(
                    str(row["id"]) in existing_ids for row in batch
                )
                batch = [row for row in batch if str(row["id"]) not in existing_ids]

            if not batch:
                continue
            self.namespace.write(
                upsert_rows=batch,
                schema=self.strategy.schema,
                distance_metric=self.strategy.distance_metric,
            )
            indexed += len(batch)
        return IndexResult(indexed=indexed, skipped=skipped)
