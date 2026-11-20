"""Generic batched writes to a Turbopuffer namespace."""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from typing import Protocol

from turbopuffer.types import AttributeSchemaParam, RowParam

from tpuf_optimize.strategy import TPufStrategy


DEFAULT_BATCH_SIZE = 5000


class NamespaceWriter(Protocol):
    """The namespace write surface required by Indexer."""

    def write(
        self,
        *,
        upsert_rows: Iterable[RowParam],
        schema: dict[str, AttributeSchemaParam],
    ) -> object: ...


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
        self.namespace = namespace
        self.strategy = strategy
        self.batch_size = batch_size

    def index(self, documents: Iterable[RowParam]) -> int:
        """Write all documents in batches and return the indexed document count."""
        iterator = iter(documents)
        total = 0
        while batch := list(itertools.islice(iterator, self.batch_size)):
            self.namespace.write(
                upsert_rows=batch,
                schema=self.strategy.schema,
            )
            total += len(batch)
        return total
