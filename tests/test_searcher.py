"""Tests for single-query and hybrid batched retrieval."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from turbopuffer.types import Row

from tpuf_optimize.searcher import SearchRequest, Searcher
from tpuf_optimize.strategy import TPufStrategy


class MemoryNamespace:
    def __init__(self) -> None:
        self.queries: list[object] = []
        self.call_sizes: list[int] = []

    def multi_query(self, *, queries):
        batch = list(queries)
        self.call_sizes.append(len(batch))
        self.queries.extend(batch)
        return SimpleNamespace(
            results=[
                SimpleNamespace(rows=[Row.from_dict({"id": f"doc-{i}", "$dist": 1})])
                for i, _ in enumerate(batch)
            ]
        )


class SearcherTests(unittest.TestCase):
    def test_hybrid_batches_fuse_each_vector_and_lexical_pair(self) -> None:
        namespace = MemoryNamespace()
        strategy = TPufStrategy(hybrid_weights=(1.0, 2.0))
        requests = [
            SearchRequest("q1", "blue shoe", "us"),
            SearchRequest("q2", "red shoe", "us"),
        ]

        results = list(Searcher(namespace, strategy, batch_size=100).search(
            requests, top_k=5
        ))

        self.assertEqual(len(namespace.queries), 4)
        self.assertEqual([request.query_id for request, _ in results], ["q1", "q2"])
        self.assertEqual(
            [[str(row.id) for row in rows] for _, rows in results],
            [["doc-1", "doc-0"], ["doc-3", "doc-2"]],
        )

    def test_hybrid_batches_do_not_exceed_multi_query_limit(self) -> None:
        namespace = MemoryNamespace()
        requests = [
            SearchRequest(f"q{i}", "product query", "us") for i in range(9)
        ]

        results = list(Searcher(namespace, TPufStrategy(), batch_size=100).search(
            requests, top_k=5
        ))

        self.assertEqual(len(results), 9)
        self.assertEqual(namespace.call_sizes, [16, 2])

    def test_all_tokens_signal_fuses_three_candidate_lists(self) -> None:
        namespace = MemoryNamespace()
        strategy = TPufStrategy(
            title_signal="all_tokens",
            title_signal_weight=0.5,
        )

        results = list(
            Searcher(namespace, strategy, batch_size=100).search(
                [SearchRequest("q1", "blue shoe", "en-US")],
                top_k=5,
            )
        )

        self.assertEqual(len(namespace.queries), 3)
        self.assertEqual(len(results), 1)
        self.assertEqual([str(row.id) for row in results[0][1]], ["doc-0", "doc-1", "doc-2"])

    def test_all_tokens_batches_respect_multi_query_limit(self) -> None:
        namespace = MemoryNamespace()
        strategy = TPufStrategy(title_signal="all_tokens")
        requests = [
            SearchRequest(f"q{i}", "product query", "us") for i in range(6)
        ]

        results = list(Searcher(namespace, strategy).search(requests, top_k=5))

        self.assertEqual(len(results), 6)
        self.assertEqual(namespace.call_sizes, [15, 3])


if __name__ == "__main__":
    unittest.main()
