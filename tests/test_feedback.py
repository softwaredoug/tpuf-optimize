"""Tests for anchored, consensus-gated pseudo-relevance feedback."""

from __future__ import annotations

import unittest

from tpuf_optimize.feedback import (
    bounded_followup_fusion,
    offline_feedback_rerank,
    plan_feedback,
    weighted_rrf,
)


def _candidate(doc_id: str, title: str, color: str = "blue") -> dict[str, object]:
    return {
        "doc_id": doc_id,
        "title": title,
        "brand": "Northstar",
        "color": color,
        "category": "outdoor gear",
    }


class FeedbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dense = [
            _candidate("d1", "blue shoe waterproof trail"),
            _candidate("d2", "blue shoe waterproof hiking"),
            _candidate("d3", "blue shoe waterproof outdoor"),
            _candidate("d4", "blue shoe durable hiking"),
            _candidate("d5", "blue shoe durable trail"),
            _candidate("d6", "blue shoe lightweight trail"),
        ]
        self.lexical = [
            self.dense[2],
            self.dense[0],
            self.dense[1],
            self.dense[4],
            self.dense[3],
            self.dense[5],
        ]

    def test_terms_need_joint_distinct_support_and_followup_keeps_anchor(self) -> None:
        plan = plan_feedback(
            "blue shoe",
            "us",
            self.dense,
            self.lexical,
            depth=6,
            min_support=3,
        )

        self.assertIn("waterproof", plan.terms)
        self.assertLessEqual(len(plan.terms), 2)
        self.assertTrue(plan.followup_query.startswith("blue shoe "))
        self.assertEqual(plan.locale, "us")

    def test_generic_terms_and_query_terms_are_not_expanded(self) -> None:
        plan = plan_feedback(
            "blue shoe",
            "us",
            self.dense,
            self.lexical,
            depth=6,
        )

        self.assertNotIn("blue", plan.terms)
        self.assertNotIn("shoe", plan.terms)
        self.assertNotIn("outdoor", plan.terms)

    def test_identifier_query_and_low_consensus_are_suppressed(self) -> None:
        identifier = plan_feedback(
            "B0A12345",
            "us",
            self.dense,
            self.lexical,
            depth=6,
        )
        low_consensus = plan_feedback(
            "blue shoe",
            "us",
            self.dense[:2],
            self.lexical[:2],
            depth=6,
        )

        self.assertEqual(identifier.terms, ())
        self.assertIsNone(identifier.followup_query)
        self.assertEqual(low_consensus.terms, ())

    def test_attribute_conflicts_do_not_support_expansion(self) -> None:
        conflicting = [
            _candidate(f"red-{index}", "red shoe waterproof trail", "red")
            for index in range(3)
        ]
        consistent = [
            _candidate("blue-1", "blue shoe durable trail"),
            _candidate("blue-2", "blue shoe durable hiking"),
        ]

        plan = plan_feedback(
            "blue shoe",
            "us",
            conflicting + consistent,
            consistent + conflicting,
            depth=5,
        )

        # Without rejecting the three red products, "waterproof" would meet
        # the three-document support threshold; the two blue products cannot.
        self.assertEqual(plan.terms, ())

    def test_weighted_rrf_is_2_to_1_and_stable_dense_first(self) -> None:
        dense = [_candidate("first", "item"), _candidate("tied", "item")]
        lexical = [_candidate("tied", "item"), _candidate("first", "item")]

        ranked = weighted_rrf(dense, lexical, top_k=2)

        self.assertEqual([row[0] for row in ranked], ["first", "tied"])
        self.assertGreater(ranked[0][1], 0)

    def test_feedback_probe_is_capped_and_unexpanded_control_is_unchanged(self) -> None:
        plan, baseline = offline_feedback_rerank(
            "blue shoe",
            "us",
            self.dense,
            self.lexical,
            top_k=6,
            depth=6,
        )
        _, no_feedback = offline_feedback_rerank(
            "B0A12345",
            "us",
            self.dense,
            self.lexical,
            top_k=6,
            depth=6,
        )
        original = weighted_rrf(self.dense, self.lexical, top_k=6)

        self.assertTrue(plan.terms)
        self.assertEqual([row[0] for row in no_feedback], [row[0] for row in original])
        self.assertLessEqual(
            max(row[1] for row in baseline),
            original[0][1] * 1.10 + 1e-12,
        )

    def test_one_followup_list_has_a_hard_score_cap(self) -> None:
        baseline = weighted_rrf(self.dense, self.lexical, top_k=6)
        followup = [
            _candidate("new", "waterproof outdoor hiking"),
            _candidate("d1", "blue shoe waterproof trail"),
            _candidate("new", "duplicate result"),
        ]

        fused = bounded_followup_fusion(baseline, followup, top_k=8)
        baseline_scores = {row[0]: row[1] for row in baseline}
        max_contribution = baseline[0][1] * 0.10

        self.assertLessEqual(
            fused[[row[0] for row in fused].index("new")][1],
            max_contribution + 1e-12,
        )
        self.assertLessEqual(
            fused[[row[0] for row in fused].index("d1")][1]
            - baseline_scores["d1"],
            max_contribution + 1e-12,
        )
        self.assertEqual(sum(row[0] == "new" for row in fused), 1)


if __name__ == "__main__":
    unittest.main()
