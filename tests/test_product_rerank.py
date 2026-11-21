"""Tests for bounded, deterministic ESCI product reranking."""

from __future__ import annotations

import unittest

from tpuf_optimize.product_rerank import rerank_candidates


class ProductRerankTests(unittest.TestCase):
    def test_uses_lexical_two_to_one_dense_rrf_as_base(self) -> None:
        results = rerank_candidates(
            "ordinary query",
            "us",
            [{"doc_id": "a", "score": 0.1}, {"doc_id": "b", "score": 0.2}],
            [{"doc_id": "b", "score": 5.0}, {"doc_id": "a", "score": 4.0}],
        )

        by_id = {row["doc_id"]: row for row in results}
        self.assertAlmostEqual(by_id["a"]["rrf_score"], 2 / 62 + 1 / 61)
        self.assertAlmostEqual(by_id["b"]["rrf_score"], 2 / 61 + 1 / 62)
        # Backend score scales are intentionally ignored; rank is the fusion input.
        self.assertEqual(len(results), 2)

    def test_title_order_coverage_and_proximity_help_exact_title(self) -> None:
        results = rerank_candidates(
            "red running shoes",
            "us",
            [{"doc_id": "near"}, {"doc_id": "exact"}],
            [
                {"doc_id": "near", "title": "Running shoes"},
                {"doc_id": "exact", "title": "Red running shoes"},
            ],
        )

        self.assertEqual(results[0]["doc_id"], "exact")
        self.assertGreater(results[0]["rerank_adjustment"], results[1]["rerank_adjustment"])

    def test_matching_units_and_model_code_beat_conflicting_specs(self) -> None:
        results = rerank_candidates(
            "Acme X120 12 oz bottle",
            "us",
            [{"doc_id": "bad"}, {"doc_id": "good"}],
            [
                {"doc_id": "bad", "title": "Acme X130 16 oz bottle"},
                {"doc_id": "good", "title": "Acme X120 12 ounces bottle"},
            ],
        )

        self.assertEqual(results[0]["doc_id"], "good")
        self.assertGreater(results[0]["rerank_adjustment"], results[1]["rerank_adjustment"])

    def test_missing_specifications_and_attributes_are_not_demoted(self) -> None:
        results = rerank_candidates(
            "Acme 12 oz blue bottle",
            "us",
            [{"doc_id": "sparse"}, {"doc_id": "conflict"}],
            [
                {"doc_id": "sparse", "title": "Bottle"},
                {
                    "doc_id": "conflict",
                    "title": "Bottle 16 oz",
                    "brand": "OtherBrand",
                    "color": "red",
                },
            ],
        )

        # No numeric or attribute fields is neutral; an explicit contradiction
        # is the only reason the complete title is lowered here.
        self.assertGreater(
            next(row for row in results if row["doc_id"] == "sparse")["score"],
            next(row for row in results if row["doc_id"] == "conflict")["score"],
        )

    def test_brand_color_are_soft_and_only_used_when_query_names_them(self) -> None:
        candidates = [
            {"doc_id": "match", "title": "wireless headphones", "brand": "Acme", "color": "blue"},
            {"doc_id": "other", "title": "wireless headphones", "brand": "Nova", "color": "red"},
            {"doc_id": "unknown", "title": "wireless headphones"},
        ]
        neutral = rerank_candidates(
            "wireless headphones", "us", [], candidates
        )
        explicit = rerank_candidates(
            "Acme blue wireless headphones", "us", [], candidates
        )

        self.assertEqual(neutral[0]["doc_id"], "match")
        self.assertEqual(explicit[0]["doc_id"], "match")
        self.assertGreater(
            next(row for row in explicit if row["doc_id"] == "match")["score"],
            next(row for row in explicit if row["doc_id"] == "unknown")["score"],
        )

    def test_unicode_normalization_and_japanese_phrase_signal(self) -> None:
        unicode_results = rerank_candidates(
            "Ｃａｆé mug",
            "us",
            [],
            [
                {"doc_id": "wrong", "title": "Tea mug"},
                {"doc_id": "right", "title": "Café mug"},
            ],
        )
        japanese_results = rerank_candidates(
            "青い靴",
            "jp",
            [],
            [
                {"doc_id": "partial", "title": "黒い靴"},
                {"doc_id": "exact", "title": "青い靴"},
            ],
        )

        self.assertEqual(unicode_results[0]["doc_id"], "right")
        self.assertEqual(japanese_results[0]["doc_id"], "exact")

    def test_is_bounded_stable_deduplicated_and_does_not_mutate_inputs(self) -> None:
        lexical = [
            {"doc_id": "a", "title": "blue shoes"},
            {"doc_id": "a", "title": "duplicate"},
            {"doc_id": "b", "title": "red shoes"},
        ]
        before = [dict(row) for row in lexical]
        first = rerank_candidates("blue shoe 12 oz", "us", [], lexical)
        second = rerank_candidates("blue shoe 12 oz", "us", [], lexical)

        self.assertEqual([row["doc_id"] for row in first], ["a", "b"])
        self.assertEqual(first, second)
        self.assertEqual(lexical, before)
        self.assertTrue(all(abs(row["rerank_adjustment"]) <= 0.012 for row in first))

    def test_top_k_and_validation(self) -> None:
        self.assertEqual(
            len(
                rerank_candidates(
                    "test",
                    "us",
                    [],
                    [{"doc_id": "a"}, {"doc_id": "b"}],
                    top_k=1,
                )
            ),
            1,
        )
        with self.assertRaises(ValueError):
            rerank_candidates("", "us", [], [])


if __name__ == "__main__":
    unittest.main()
