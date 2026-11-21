"""Offline tests for locale-weighted dense/lexical fusion."""

from __future__ import annotations

import unittest

from tpuf_optimize.locale_hybrid import (
    grouped_validation_split,
    paired_metric_summaries,
    shrink_locale_weights,
    weighted_rrf,
)


def arm(*doc_ids: str) -> list[dict[str, str | float]]:
    return [
        {"doc_id": doc_id, "score": 1.0 / rank, "title": doc_id}
        for rank, doc_id in enumerate(doc_ids, start=1)
    ]


class LocaleHybridTests(unittest.TestCase):
    def test_parent_uses_one_based_weighted_rrf_and_dense_first_ties(self) -> None:
        dense = arm("d1", "shared", *(f"d{i}" for i in range(3, 13)))
        lexical = arm("l1", "shared")

        results = weighted_rrf(dense, lexical, result_limit=20)

        result_ids = [result.doc_id for result in results]
        self.assertEqual(result_ids[:2], ["shared", "d1"])
        expected_shared = 2 / 12 + 1 / 12
        self.assertAlmostEqual(results[0].score, expected_shared)
        # dense rank 12 and lexical rank 1 both score 1/11.
        self.assertAlmostEqual(results[result_ids.index("d12")].score, 1 / 11)
        self.assertAlmostEqual(results[result_ids.index("l1")].score, 1 / 11)
        self.assertLess(result_ids.index("d12"), result_ids.index("l1"))

    def test_fusion_limits_candidates_and_output(self) -> None:
        dense = arm(*(f"d{i}" for i in range(1, 105)))
        lexical = arm(*(f"l{i}" for i in range(1, 105)))

        results = weighted_rrf(dense, lexical, result_limit=10)

        self.assertEqual(len(results), 10)
        self.assertNotIn("d101", {result.doc_id for result in results})
        self.assertNotIn("l101", {result.doc_id for result in results})

    def test_locale_shrinkage_has_minimum_support_fallback(self) -> None:
        self.assertEqual(shrink_locale_weights((3.0, 1.0), 19), (2.0, 1.0))

        dense, lexical = shrink_locale_weights((3.0, 1.0), 20)

        self.assertAlmostEqual(dense, 2.0 + 20 / 22)
        self.assertAlmostEqual(lexical, 1.0)

    def test_validation_keeps_identical_queries_together(self) -> None:
        query_texts = {
            "q1": "red  shoe",
            "q2": " RED shoe ",
            **{f"q{i}": f"query {i}" for i in range(3, 40)},
        }

        fit, validation = grouped_validation_split(query_texts)

        self.assertFalse(fit & validation)
        self.assertEqual(fit | validation, set(query_texts))
        self.assertEqual("q1" in validation, "q2" in validation)
        self.assertGreater(len(validation), 0)
        self.assertGreater(len(fit), 0)

    def test_paired_metrics_report_overall_and_locale_uncertainty(self) -> None:
        parent = {"q1": 0.5, "q2": 0.6, "q3": 0.3}
        candidate = {"q1": 0.6, "q2": 0.5, "q3": 0.5}
        locales = {"q1": "us", "q2": "us", "q3": "jp"}

        summaries = paired_metric_summaries(parent, candidate, locales)

        self.assertEqual(summaries["overall"].count, 3)
        self.assertAlmostEqual(summaries["overall"].mean or 0, 0.0666666667)
        self.assertIsNotNone(summaries["overall"].standard_error)
        self.assertAlmostEqual(summaries["us"].mean or 0, 0.0)
        self.assertEqual(summaries["jp"].count, 1)
        self.assertIsNone(summaries["jp"].standard_error)


if __name__ == "__main__":
    unittest.main()
