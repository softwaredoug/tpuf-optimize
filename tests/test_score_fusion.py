"""Tests for cached lexical/dense score-fusion experiments."""

from __future__ import annotations

import unittest

from tpuf_optimize.score_fusion import (
    fuse_convex,
    fuse_rrf,
    normalize_scores,
)


def candidate(document_id: str, score: float) -> dict[str, str | float]:
    return {"doc_id": document_id, "score": score}


class ScoreFusionTests(unittest.TestCase):
    def test_dense_distances_are_reversed_and_ties_are_stable(self) -> None:
        results = fuse_convex(
            [candidate("z", 0.1), candidate("a", 0.1), candidate("m", 0.9)],
            [],
            dense_weight=1,
            normalization="minmax",
        )
        self.assertEqual([document_id for document_id, _ in results], ["a", "z", "m"])

    def test_missing_retriever_score_gets_zero_contribution(self) -> None:
        results = fuse_convex(
            [candidate("dense-only", 0.1), candidate("both", 0.2)],
            [candidate("lex-only", 8), candidate("both", 2)],
            dense_weight=0.5,
            normalization="minmax",
        )
        by_id = dict(results)
        self.assertEqual(by_id["dense-only"], 0.5)
        self.assertEqual(by_id["lex-only"], 0.5)
        self.assertEqual(by_id["both"], 0.0)
        self.assertEqual([doc_id for doc_id, _ in results], ["dense-only", "lex-only", "both"])

    def test_mean_missing_scores_are_compared_separately(self) -> None:
        results = fuse_convex(
            [candidate("dense-only", 0.1), candidate("both", 0.2)],
            [candidate("lex-only", 8), candidate("both", 2)],
            dense_weight=0.5,
            normalization="minmax",
            missing_score="mean",
        )
        by_id = dict(results)
        self.assertEqual(by_id["dense-only"], 0.75)
        self.assertEqual(by_id["lex-only"], 0.75)
        self.assertEqual(by_id["both"], 0.0)

    def test_percentile_normalization_clips_extremes(self) -> None:
        scores = {"low": 0.0, "a": 10.0, "b": 20.0, "c": 30.0, "high": 1000.0}
        robust = normalize_scores(scores, "percentile")
        minmax = normalize_scores(scores, "minmax")
        self.assertEqual(robust["low"], 0.0)
        self.assertEqual(robust["high"], 1.0)
        self.assertGreater(robust["b"], minmax["b"])
        self.assertLess(robust["c"], 1.0)

    def test_equal_range_normalizes_to_zero(self) -> None:
        self.assertEqual(
            normalize_scores({"a": 4.0, "b": 4.0}, "minmax"),
            {"a": 0.0, "b": 0.0},
        )

    def test_rrf_is_exact_dense_to_lexical_two_to_one(self) -> None:
        results = dict(
            fuse_rrf(
                [candidate("dense-first", 0.01)],
                [candidate("lex-first", 9.0)],
            )
        )
        self.assertEqual(results["dense-first"], 2 / 11)
        self.assertEqual(results["lex-first"], 1 / 11)

    def test_rrf_rank_order_uses_score_direction_and_union(self) -> None:
        results = fuse_rrf(
            [candidate("shared", 0.1), candidate("d2", 0.2)],
            [candidate("shared", 2.0), candidate("l2", 1.0)],
        )
        self.assertEqual({document_id for document_id, _ in results},
                         {"d2", "l2", "shared"})
        self.assertEqual(results[0][0], "shared")

    def test_rrf_uses_first_ten_from_each_top_hundred_arm(self) -> None:
        dense = [
            candidate(f"d{rank}", float(101 - rank))
            for rank in range(1, 101)
        ]
        lexical = [candidate(f"l{rank}", float(101 - rank)) for rank in range(1, 101)]

        results = dict(fuse_rrf(dense, lexical))

        self.assertEqual(len(results), 20)
        self.assertEqual(results["d1"], 2 / 11)
        self.assertEqual(results["d10"], 2 / 20)
        self.assertEqual(results["l1"], 1 / 11)
        self.assertEqual(results["l10"], 1 / 20)
        self.assertNotIn("d11", results)
        self.assertNotIn("l11", results)

    def test_rrf_equal_scores_keep_dense_first_insertion_order(self) -> None:
        dense = [
            candidate(f"d{rank}", float(11 - rank))
            for rank in range(1, 11)
        ]
        dense[7] = candidate("shared-at-dense-eight", 3.0)
        lexical = [
            candidate(f"l{rank}", float(11 - rank))
            for rank in range(1, 11)
        ]
        lexical[7] = candidate("shared-at-dense-eight", 3.0)

        results = fuse_rrf(dense, lexical)
        scores = dict(results)

        self.assertEqual(scores["d2"], 2 / 12)
        self.assertEqual(scores["shared-at-dense-eight"], 2 / 18 + 1 / 18)
        self.assertEqual(scores["d2"], scores["shared-at-dense-eight"])
        ordered_ids = [document_id for document_id, _ in results]
        self.assertLess(
            ordered_ids.index("d2"),
            ordered_ids.index("shared-at-dense-eight"),
        )

    def test_duplicate_or_nonfinite_candidates_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            fuse_rrf(
                [candidate("x", 1.0), candidate("x", 2.0)],
                [],
            )
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            fuse_convex(
                [candidate("x", float("nan"))],
                [],
                dense_weight=0.5,
                normalization="minmax",
            )


if __name__ == "__main__":
    unittest.main()
