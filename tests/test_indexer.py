"""Tests for batched skip-existing indexing."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from turbopuffer.types import Row

from tpuf_optimize.indexer import Indexer
from tpuf_optimize.strategy import TPufStrategy


class MemoryNamespace:
    def __init__(self, existing_ids: set[str]) -> None:
        self.existing_ids = existing_ids
        self.written_rows: list[dict[str, str]] = []
        self.lookup_count = 0

    def query(self, *, filters, top_k, include_attributes):
        del top_k, include_attributes
        self.lookup_count += 1
        ids = filters[2]
        return SimpleNamespace(
            rows=[
                Row.from_dict({"id": document_id})
                for document_id in ids
                if document_id in self.existing_ids
            ]
        )

    def write(self, *, upsert_rows, schema, distance_metric):
        del schema, distance_metric
        rows = list(upsert_rows)
        self.written_rows.extend(rows)
        self.existing_ids.update(str(row["id"]) for row in rows)


class IndexerTests(unittest.TestCase):
    def test_skip_existing_checks_once_per_batch_and_only_writes_missing_ids(self):
        namespace = MemoryNamespace({"doc-1"})
        documents = [
            {"id": "doc-1"},
            {"id": "doc-2"},
            {"id": "doc-3"},
            {"id": "doc-4"},
        ]

        result = Indexer(namespace, TPufStrategy(), batch_size=2).index(
            documents,
            skip_existing_docs=True,  # type: ignore[arg-type]
        )

        self.assertEqual(result.indexed, 3)
        self.assertEqual(result.skipped, 1)
        self.assertEqual(namespace.lookup_count, 2)
        self.assertEqual(
            [row["id"] for row in namespace.written_rows],
            ["doc-2", "doc-3", "doc-4"],
        )


if __name__ == "__main__":
    unittest.main()
