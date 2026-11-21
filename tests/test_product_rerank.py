"""Tests for bounded, deterministic ESCI product reranking."""

from __future__ import annotations

import unittest

from tpuf_optimize.product_rerank import (
    DEFAULT_DENSE_WEIGHT,
    DEFAULT_LEXICAL_WEIGHT,
    DEFAULT_RANK_CONSTANT,
    PINNED_MAX_ADJUSTMENT,
    PINNED_TOP_K,
    rerank_candidates,
)


class ProductRerankTests(unittest.TestCase):
    def test_pinned_configuration(self) -> None:
        self.assertEqual(DEFAULT_DENSE_WEIGHT, 2.0)
        self.assertEqual(DEFAULT_LEXICAL_WEIGHT, 1.0)
        self.assertEqual(DEFAULT_RANK_CONSTANT, 10)
        self.assertEqual(PINNED_MAX_ADJUSTMENT, 0.05)
        self.assertEqual(PINNED_TOP_K, 10)

    def test_uses_dense_two_to_one_lexical_rrf_as_base(self) -> None:
        results = rerank_candidates(
            "ordinary query",
            "us",
            [{"doc_id": "a", "score": 0.1}, {"doc_id": "b", "score": 0.2}],
            [{"doc_id": "b", "score": 5.0}, {"doc_id": "a", "score": 4.0}],
        )

        by_id = {row["doc_id"]: row for row in results}
        self.assertAlmostEqual(by_id["a"]["rrf_score"], 2 / 11 + 1 / 12)
        self.assertAlmostEqual(by_id["b"]["rrf_score"], 2 / 12 + 1 / 11)
        # Backend score scales are intentionally ignored; rank is the fusion input.
        self.assertEqual(len(results), 2)

    def test_zero_adjustment_matches_dense_weighted_parent_not_inverted_weights(
        self,
    ) -> None:
        dense = [{"doc_id": "dense-first"}, {"doc_id": "lexical-first"}]
        lexical = [{"doc_id": "lexical-first"}, {"doc_id": "dense-first"}]

        parent_order = rerank_candidates(
            "plain query", "us", dense, lexical, max_adjustment=0
        )
        inverted_order = rerank_candidates(
            "plain query",
            "us",
            dense,
            lexical,
            dense_weight=1,
            lexical_weight=2,
            max_adjustment=0,
        )

        self.assertEqual(
            [row["doc_id"] for row in parent_order],
            ["dense-first", "lexical-first"],
        )
        self.assertEqual(
            [row["doc_id"] for row in inverted_order],
            ["lexical-first", "dense-first"],
        )
        self.assertTrue(all(row["rerank_adjustment"] == 0 for row in parent_order))

    def test_zero_adjustment_matches_exact_parent_rrf_and_dense_first_ties(
        self,
    ) -> None:
        dense = [
            {"doc_id": "shared"},
            {"doc_id": "dense-only"},
            {"doc_id": "tie-a"},
            {"doc_id": "tie-b"},
        ]
        lexical = [
            {"doc_id": "shared"},
            {"doc_id": "lexical-only"},
            {"doc_id": "tie-b"},
            {"doc_id": "tie-a"},
        ]
        # Direct offline transcription of TPufStrategy.reciprocal_rank_fusion
        # at the accepted parent's weights and rank constant.
        parent_scores: dict[str, float] = {}
        for ranked, weight in ((dense, 2.0), (lexical, 1.0)):
            for rank, row in enumerate(ranked, start=1):
                doc_id = str(row["doc_id"])
                parent_scores[doc_id] = parent_scores.get(doc_id, 0.0) + (
                    weight / (10 + rank)
                )
        parent_rows = sorted(
            parent_scores.items(), key=lambda item: item[1], reverse=True
        )

        results = rerank_candidates(
            "plain query", "us", dense, lexical, max_adjustment=0
        )

        self.assertEqual(
            [(row["doc_id"], row["score"]) for row in results],
            parent_rows,
        )
        self.assertEqual(
            [row["doc_id"] for row in results if row["doc_id"].startswith("tie-")],
            ["tie-a", "tie-b"],
        )
        self.assertTrue(all(row["rrf_score"] == row["score"] for row in results))

    def test_title_order_coverage_and_proximity_reward_exact_title(self) -> None:
        results = rerank_candidates(
            "red running shoes",
            "us",
            [{"doc_id": "near"}, {"doc_id": "exact"}],
            [
                {"doc_id": "near", "title": "Running shoes"},
                {"doc_id": "exact", "title": "Red running shoes"},
            ],
        )

        by_id = {row["doc_id"]: row for row in results}
        self.assertGreater(
            by_id["exact"]["rerank_adjustment"],
            by_id["near"]["rerank_adjustment"],
        )

    def test_matching_units_and_model_code_reward_exact_specs(self) -> None:
        results = rerank_candidates(
            "Acme X120 12 oz bottle",
            "us",
            [{"doc_id": "bad"}, {"doc_id": "good"}],
            [
                {"doc_id": "bad", "title": "Acme X130 16 oz bottle"},
                {"doc_id": "good", "title": "Acme X120 12 ounces bottle"},
            ],
        )

        by_id = {row["doc_id"]: row for row in results}
        self.assertGreater(
            by_id["good"]["rerank_adjustment"],
            by_id["bad"]["rerank_adjustment"],
        )

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

        unicode_by_id = {row["doc_id"]: row for row in unicode_results}
        japanese_by_id = {row["doc_id"]: row for row in japanese_results}
        self.assertGreater(
            unicode_by_id["right"]["rerank_adjustment"],
            unicode_by_id["wrong"]["rerank_adjustment"],
        )
        self.assertGreater(
            japanese_by_id["exact"]["rerank_adjustment"],
            japanese_by_id["partial"]["rerank_adjustment"],
        )

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
        self.assertTrue(
            all(
                abs(row["rerank_adjustment"]) <= PINNED_MAX_ADJUSTMENT
                for row in first
            )
        )

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
