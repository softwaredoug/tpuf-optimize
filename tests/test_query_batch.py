"""Tests for batch query CSV parsing."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tpuf_optimize.query_batch import _read_queries


class QueryBatchTests(unittest.TestCase):
    def test_preserves_explicit_query_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queries.csv"
            path.write_text(
                "query_id,query,locale\noriginal-id,blue shoe,en-US\n",
                encoding="utf-8",
            )

            requests = _read_queries(path)

        self.assertEqual(requests[0].query_id, "original-id")
        self.assertEqual(requests[0].text, "blue shoe")
        self.assertEqual(requests[0].locale, "en-US")


if __name__ == "__main__":
    unittest.main()
