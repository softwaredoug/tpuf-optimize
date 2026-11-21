"""Offline tests for deterministic query-shape fusion routing."""

from __future__ import annotations

import unittest

from tpuf_optimize.routing import QueryRouter, rerank_query_candidates


def _rows(*document_ids: str) -> list[dict[str, object]]:
    return [{"doc_id": document_id} for document_id in document_ids]


class QueryRouterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.router = QueryRouter()

    def test_identifier_like_queries_keep_the_baseline_profile(self) -> None:
        self.assertEqual(
            self.router.route("wireless headphones X200").name,
            "balanced",
        )
        self.assertEqual(
            self.router.route("12V adapter").name,
            "balanced",
        )

    def test_routes_long_descriptive_queries_to_semantic_profile(self) -> None:
        self.assertEqual(
            self.router.route("comfortable shoes for long distance running").name,
            "descriptive",
        )

    def test_short_queries_keep_baseline_profile(self) -> None:
        self.assertEqual(self.router.route("red running shoe").name, "balanced")

    def test_profiles_choose_distinct_results_on_shared_union(self) -> None:
        dense = _rows(
            "dense-first",
            "shared",
            *[f"dense-{index}" for index in range(3, 11)],
        )
        lexical = _rows(
            "lexical-first",
            *[f"lexical-{index}" for index in range(2, 55)],
            "shared",
        )

        identifier = rerank_query_candidates(
            "model X200", dense, lexical, top_k=200
        )
        balanced = rerank_query_candidates(
            "red running shoe", dense, lexical, top_k=200
        )
        descriptive = rerank_query_candidates(
            "comfortable shoes for long distance running",
            dense,
            lexical,
            top_k=200,
        )

        identifier_ids = [row["doc_id"] for row in identifier]
        balanced_ids = [row["doc_id"] for row in balanced]
        descriptive_ids = [row["doc_id"] for row in descriptive]
        self.assertEqual(identifier_ids, balanced_ids)
        self.assertEqual(balanced_ids[0], "shared")
        self.assertEqual(descriptive_ids[0], "dense-first")
        self.assertEqual(set(balanced_ids), set(descriptive_ids))

    def test_ties_are_dense_first_and_duplicate_candidates_are_collapsed(self) -> None:
        rows = rerank_query_candidates(
            "shoe",
            _rows("shared", "dense-tie"),
            _rows("lexical-tie", "shared"),
            top_k=10,
            rrf_k=0,
        )
        ids = [row["doc_id"] for row in rows]
        self.assertEqual(ids, ["shared", "dense-tie", "lexical-tie"])
        self.assertEqual(len(ids), len(set(ids)))

    def test_invalid_limits_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            rerank_query_candidates("shoe", [], [], top_k=0)
        with self.assertRaises(ValueError):
            rerank_query_candidates("shoe", [], [], rrf_k=-1)


if __name__ == "__main__":
    unittest.main()
