"""Tests for batched focused-query retrieval and RRF."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import cast

from turbopuffer.types import Row

from tpuf_optimize.searcher import NamespaceSearcher, SearchRequest, Searcher
from tpuf_optimize.strategy import TPufStrategy


class FakeNamespace:
    def __init__(self) -> None:
        self.calls: list[list[object]] = []
        self.result_rows = [
            [
                Row.model_validate({"id": "a", "$dist": 1.0}),
                Row.model_validate({"id": "b", "$dist": 0.5}),
            ],
            [
                Row.model_validate({"id": "b", "$dist": 1.0}),
                Row.model_validate({"id": "a", "$dist": 0.5}),
            ],
            [Row.model_validate({"id": "a", "$dist": 1.0})],
            [Row.model_validate({"id": "c", "$dist": 1.0})],
        ]

    def multi_query(
        self, *, queries: object
    ) -> SimpleNamespace:
        query_list = list(queries)  # type: ignore[arg-type]
        self.calls.append(query_list)
        return SimpleNamespace(
            results=[
                SimpleNamespace(rows=self.result_rows[index % 4])
                for index in range(len(query_list))
            ]
        )


class SearcherTests(unittest.TestCase):
    def test_batches_focused_queries_and_preserves_request_order(self) -> None:
        namespace = FakeNamespace()
        searcher = Searcher(cast(NamespaceSearcher, namespace), TPufStrategy())
        requests = [
            SearchRequest("q1", "blue shoe", "us"),
            SearchRequest("q2", "zapato azul", "es"),
        ]

        results = list(searcher.search(requests, top_k=10))

        self.assertEqual([request.query_id for request, _ in results], ["q1", "q2"])
        self.assertEqual(len(namespace.calls), 1)
        self.assertEqual(len(namespace.calls[0]), 8)
        fused_rows = results[0][1]
        self.assertEqual([str(row.id) for row in fused_rows], ["c", "a", "b"])
        self.assertAlmostEqual(fused_rows[0]["$dist"], 2 / 11)
        self.assertAlmostEqual(
            fused_rows[1]["$dist"], 1 / 11 + 0.5 / 12 + 0.25 / 11
        )
        self.assertAlmostEqual(fused_rows[2]["$dist"], 1 / 12 + 0.5 / 11)


if __name__ == "__main__":
    unittest.main()
