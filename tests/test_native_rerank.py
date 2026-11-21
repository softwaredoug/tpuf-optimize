"""Tests for the frozen candidate arms and native reranking request."""

from __future__ import annotations

import unittest

from turbopuffer.types import Row

from tpuf_optimize.native_rerank import (
    NATIVE_RERANK_MODELS,
    candidate_queries,
    identifier_gated_weights,
    local_identifier_gated_rrf,
    native_rerank_by,
    paired_delta_summary,
)
from tpuf_optimize.strategy import TPufStrategy


class NativeRerankTests(unittest.TestCase):
    def test_candidate_arms_include_passage_and_keep_locale_filter(self) -> None:
        dense, lexical = candidate_queries(
            TPufStrategy(), "blue shoe", "en-US"
        )

        self.assertEqual(dense["top_k"], 100)
        self.assertEqual(lexical["top_k"], 100)
        self.assertEqual(dense["filters"], ("locale", "Eq", "us"))
        self.assertEqual(lexical["filters"], ("locale", "Eq", "us"))
        self.assertEqual(
            dense["rank_by"],
            ("passage__embed__voyage_4", "ANN", ("Embed", "blue shoe")),
        )
        self.assertIn("passage", dense["include_attributes"])
        self.assertIn("passage", lexical["include_attributes"])
        self.assertIn("title__word_v4__en__stem_stop", str(lexical["rank_by"]))
        self.assertIn("description__word_v4__en__stem_stop", str(lexical["rank_by"]))
        self.assertEqual(
            candidate_queries(TPufStrategy(), "青い靴", "jp")[0]["filters"],
            ("locale", "Eq", "jp"),
        )

    def test_native_rerank_is_outer_model_on_passage(self) -> None:
        self.assertEqual(
            native_rerank_by(NATIVE_RERANK_MODELS["cohere"], "blue shoe"),
            {
                "model": "cohere/rerank-v3.5",
                "attribute": "passage",
                "query": "blue shoe",
            },
        )
        self.assertEqual(
            native_rerank_by(NATIVE_RERANK_MODELS["voyage"], "red shoe")["model"],
            "voyage/rerank-3",
        )

    def test_identifier_gate_matches_7e22661_and_local_rrf(self) -> None:
        dense = [
            {"id": "other-1", "title": "Different item"},
            {"id": "other-2", "title": "Something else"},
            {"id": "other-3", "title": "Another item"},
        ]
        lexical = [
            {"id": "SKU123_us", "title": "SKU123 Premium Shoe"},
            {"id": "second", "title": "Shoe"},
        ]
        self.assertEqual(
            identifier_gated_weights("SKU123 shoe", dense, lexical),
            (1.0, 2.0),
        )

        dense_rows = [
            Row.from_dict({"id": item["id"], "$dist": 0.1, **item})
            for item in dense
        ]
        lexical_rows = [
            Row.from_dict({"id": item["id"], "$dist": 0.1, **item})
            for item in lexical
        ]
        ranked = local_identifier_gated_rrf(
            "SKU123 shoe", dense_rows, lexical_rows
        )
        self.assertEqual(ranked[0][0], "SKU123_us")
        self.assertEqual(len(ranked), 5)

    def test_paired_delta_summary(self) -> None:
        summary = paired_delta_summary(
            {"a": 0.2, "b": 0.4, "c": 0.8},
            {"a": 0.3, "b": 0.5, "c": 0.9},
        )

        self.assertEqual(summary["n"], 3)
        self.assertAlmostEqual(float(summary["mean_delta"]), 0.1)
        self.assertAlmostEqual(float(summary["ci95_low"]), 0.1)
        self.assertAlmostEqual(float(summary["ci95_high"]), 0.1)


if __name__ == "__main__":
    unittest.main()
