"""Synthetic query-disjoint tests for the computed-feature ablation."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from tpuf_optimize.computed_feature_rank import (
    COMPUTED_FEATURE_NAMES,
    ENHANCED_FEATURE_NAMES,
    FEATURE_NAMES,
    LinearRanker,
    TYPED_ENHANCED_FEATURE_NAMES,
    TYPED_ENHANCED_NO_UNKNOWN_FEATURE_NAMES,
    compare_ablations,
    feature_vectors,
    model_ranking,
    parse_query_record,
)


def _record(index: int):
    query = f"red shoe {index}"
    good = f"good-{index}"
    bad = f"bad-{index}"
    return {
        "query_id": f"q{index}",
        "query_text": query,
        "locale": "en-US",
        "dense": [
            {
                "doc_id": bad,
                "score": 0.9,
                "title": f"blue boot {index}",
                "features": {
                    "title_bm25": 0.0,
                    "passage_bm25": 0.0,
                    "brand_bm25": None,
                    "title_full_coverage": False,
                },
            },
            {
                "doc_id": good,
                "score": 0.5,
                "title": f"red shoe {index}",
                "features": {
                    "title_bm25": 2.0,
                    "passage_bm25": 1.0,
                    "brand_bm25": 0.0,
                    "title_full_coverage": True,
                },
            },
        ],
        "lexical": [
            {
                "doc_id": good,
                "score": 0.8,
                "title": f"red shoe {index}",
                "features": {
                    "title_bm25": 2.0,
                    "passage_bm25": 1.0,
                    "brand_bm25": 0.0,
                    "title_full_coverage": True,
                },
            },
            {
                "doc_id": bad,
                "score": 0.2,
                "title": f"blue boot {index}",
                "features": {
                    "title_bm25": 0.0,
                    "passage_bm25": 0.0,
                    "brand_bm25": None,
                    "title_full_coverage": False,
                },
            },
        ],
    }


class ComputedFeatureRankTests(unittest.TestCase):
    def test_zero_and_absent_are_distinguished_by_value_and_availability(self) -> None:
        group = parse_query_record(_record(0))
        vectors = feature_vectors(group, include_computed=True)
        self.assertEqual(len(vectors["bad-0"]), len(ENHANCED_FEATURE_NAMES))
        self.assertEqual(len(COMPUTED_FEATURE_NAMES), 8)
        zero_index = ENHANCED_FEATURE_NAMES.index("brand_bm25_z")
        available_index = ENHANCED_FEATURE_NAMES.index("brand_bm25_available")
        # For the positive candidate, brand BM25 is exactly zero but available.
        self.assertEqual(vectors["good-0"][zero_index], 0.0)
        self.assertEqual(vectors["good-0"][available_index], 1.0)
        # Brand is unavailable for the other candidate, not represented as
        # an observed score of zero.
        self.assertEqual(vectors["bad-0"][available_index], 0.0)
        self.assertEqual(len(next(iter(feature_vectors(group).values()))), len(FEATURE_NAMES))

    def test_ablation_uses_one_query_disjoint_fold_and_same_candidates(self) -> None:
        groups = [parse_query_record(_record(index)) for index in range(12)]
        grades = {
            f"q{index}": {f"good-{index}": 3.0, f"bad-{index}": 1.0}
            for index in range(12)
        }
        frozen_path = (
            Path(__file__).resolve().parents[1]
            / "models"
            / "esci_pairwise_ranker_adf9768.json"
        )
        frozen = LinearRanker.from_json(
            json.loads(frozen_path.read_text(encoding="utf-8"))
        )
        reports, detail = compare_ablations(
            groups, grades, frozen, validation_fraction=0.25, seed=31, epochs=60
        )
        self.assertEqual(
            set(reports),
            {
                "frozen_adf9768",
                "same_fold_base_refit",
                "same_fold_computed_refit",
            },
        )
        self.assertEqual(detail["fold"]["train_queries"], 9)
        self.assertEqual(detail["fold"]["validation_queries"], 3)
        self.assertTrue(detail["fold"]["query_disjoint"])
        self.assertTrue(detail["fold"]["identical_validation_candidates"])
        self.assertEqual(
            detail["models"]["same_fold_base_refit"].feature_names, FEATURE_NAMES
        )
        self.assertEqual(
            detail["models"]["same_fold_computed_refit"].feature_names,
            ENHANCED_FEATURE_NAMES,
        )
        for metrics in reports.values():
            self.assertEqual(metrics["evaluated_queries"], 3)

    def test_training_nominee_artifact_contains_weights_only(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "models"
            / "esci_pairwise_ranker_computed_features_training.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(set(payload), {"feature_names", "weights"})
        model = LinearRanker.from_json(payload)
        self.assertEqual(model.feature_names, ENHANCED_FEATURE_NAMES)
        self.assertEqual(len(model.weights), len(ENHANCED_FEATURE_NAMES))

    def test_typed_computed_nominee_artifact_is_weights_only(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "models"
            / "esci_pairwise_ranker_typed_computed_no_unknown_training.json"
        )
        payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(set(payload), {"feature_names", "weights"})
        model = LinearRanker.from_json(payload)
        self.assertIn("typed_product_match", model.feature_names)
        self.assertIn("typed_contradiction", model.feature_names)
        self.assertNotIn("typed_unknown", model.feature_names)
        self.assertEqual(len(model.weights), len(model.feature_names))

    def test_typed_product_evidence_separates_support_conflict_and_unknown(self) -> None:
        record = {
            "query_id": "typed",
            "query_text": "Nike red running shoe size 10",
            "locale": "en-US",
            "dense": [
                {
                    "doc_id": "supported",
                    "score": 1.0,
                    "title": "Nike red running shoe size 10",
                    "brand": "Nike",
                    "color": "Red",
                },
                {
                    "doc_id": "contradiction",
                    "score": 0.9,
                    "title": "Adidas blue running shoe size 8",
                    "brand": "Adidas",
                    "color": "Blue",
                },
                {
                    "doc_id": "unknown",
                    "score": 0.8,
                    "title": "Running shoe",
                    "brand": "",
                    "color": "",
                },
            ],
            "lexical": [],
        }
        group = parse_query_record(record)
        vectors = feature_vectors(group, include_computed=True, include_typed=True)
        typed_indices = [
            TYPED_ENHANCED_FEATURE_NAMES.index(name)
            for name in (
                "typed_product_match",
                "typed_support",
                "typed_contradiction",
                "typed_unknown",
            )
        ]
        support = [vectors["supported"][index] for index in typed_indices]
        conflict = [vectors["contradiction"][index] for index in typed_indices]
        unknown = [vectors["unknown"][index] for index in typed_indices]
        self.assertEqual(support, [1.0, 1.0, 0.0, 0.0])
        self.assertGreater(conflict[2], 0.0)
        self.assertEqual(conflict[3], 0.0)
        self.assertEqual(unknown[2], 0.0)
        self.assertGreater(unknown[3], 0.0)
        self.assertEqual(
            tuple(
                candidate.doc_id
                for candidate in group.candidates
            ),
            ("supported", "contradiction", "unknown"),
        )
        supported = group.candidates[0]
        self.assertEqual((supported.dense_rank, supported.dense_score), (1, 1.0))
        self.assertEqual((supported.lexical_rank, supported.lexical_score), (None, None))
        typed_model = LinearRanker(
            tuple(0.1 for _ in TYPED_ENHANCED_FEATURE_NAMES),
            TYPED_ENHANCED_FEATURE_NAMES,
        )
        ranked_ids = {
            candidate.doc_id
            for candidate, _score in model_ranking(group, typed_model)
        }
        self.assertEqual(ranked_ids, {candidate.doc_id for candidate in group.candidates})
        no_unknown_vectors = feature_vectors(
            group,
            include_computed=True,
            include_typed=True,
            include_typed_unknown=False,
        )
        no_unknown_model = LinearRanker(
            tuple(0.1 for _ in TYPED_ENHANCED_NO_UNKNOWN_FEATURE_NAMES),
            TYPED_ENHANCED_NO_UNKNOWN_FEATURE_NAMES,
        )
        self.assertEqual(
            len(no_unknown_vectors["supported"]),
            len(TYPED_ENHANCED_NO_UNKNOWN_FEATURE_NAMES),
        )
        self.assertEqual(
            {
                candidate.doc_id
                for candidate, _score in model_ranking(group, no_unknown_model)
            },
            ranked_ids,
        )


if __name__ == "__main__":
    unittest.main()
