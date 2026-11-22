"""Tests for optional lexical pseudo-relevance feedback."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Iterable

from turbopuffer.types import Row

from tpuf_optimize.searcher import SearchRequest, Searcher
from tpuf_optimize.strategy import TPufStrategy


class FeedbackNamespace:
    def __init__(self) -> None:
        self.call_count = 0

    def multi_query(self, *, queries: Iterable[object]) -> SimpleNamespace:
        self.call_count += 1
        query_count = len(list(queries))
        if self.call_count == 1:
            rows = [
                Row.from_dict(
                    {
                        "id": "original-1",
                        "$dist": 2.0,
                        "title": "Blue hiking boots",
                        "description": "waterproof trail shoe",
                        "brand__word_v2__default": "North Star",
                    }
                ),
                Row.from_dict(
                    {
                        "id": "original-2",
                        "$dist": 1.0,
                        "title": "Blue hiking boots",
                        "description": "waterproof trail shoe",
                        "brand__word_v2__default": "North Star",
                    }
                ),
            ]
        else:
            rows = [
                Row.from_dict({"id": "original-2", "$dist": 3.0}),
                Row.from_dict({"id": "original-1", "$dist": 2.0}),
                Row.from_dict({"id": "feedback-only", "$dist": 1.0}),
            ]
        return SimpleNamespace(
            results=[SimpleNamespace(rows=rows) for _ in range(query_count)]
        )


class SearcherTests(unittest.TestCase):
    def test_feedback_fuses_expansion_without_dropping_original_candidates(self):
        namespace = FeedbackNamespace()
        searcher = Searcher(
            namespace,
            TPufStrategy(),
            batch_size=1,
            pseudo_relevance_feedback=True,
        )

        results = list(
            searcher.search(
                [SearchRequest("q1", "blue shoe", "us")],
                top_k=3,
            )
        )

        self.assertEqual(namespace.call_count, 2)
        self.assertEqual(
            [str(row.id) for row in results[0][1]],
            ["original-1", "original-2", "feedback-only"],
        )


if __name__ == "__main__":
    unittest.main()
