"""Offline tests for passage-only candidate expansion."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from tpuf_optimize.computed_feature_rank import (
    PASSAGE_ARM_FEATURE_NAMES,
    PASSAGE_TYPED_COMPUTED_FEATURE_NAMES,
    LinearRanker,
    feature_vectors,
    model_ranking,
    parse_query_record,
)
from tpuf_optimize.passage_candidate_expansion import (
    attach_passage_arm,
    collect_passage_metadata_checkpoint,
    enrich_passage_only_candidates,
    passage_metadata_query,
    read_passage_arm,
    _read_novel_positive_judgments,
)


def _record(index: int) -> dict[str, object]:
    return {
        "query_id": f"q{index}",
        "query_text": f"red shoe {index}",
        "locale": "us",
        "dense": [
            {
                "doc_id": f"dense-{index}",
                "score": 0.8,
                "title": "red running shoe",
                "features": {
                    "title_bm25": 1.0,
                    "passage_bm25": 2.0,
                    "brand_bm25": 0.0,
                    "title_full_coverage": False,
                },
            },
            {
                "doc_id": f"shared-{index}",
                "score": 0.6,
                "title": "red shoe",
                "brand": "Acme",
                "color": "Red",
                "features": {
                    "title_bm25": 2.0,
                    "passage_bm25": 3.0,
                    "brand_bm25": 1.0,
                    "title_full_coverage": True,
                },
            },
        ],
        "lexical": [
            {
                "doc_id": f"shared-{index}",
                "score": 0.9,
                "title": "red shoe",
                "brand": "Acme",
                "color": "Red",
                "features": {
                    "title_bm25": 2.0,
                    "passage_bm25": 3.0,
                    "brand_bm25": 1.0,
                    "title_full_coverage": True,
                },
            }
        ],
    }


class PassageCandidateExpansionTests(unittest.TestCase):
    def test_read_passage_arm_accepts_ranked_lists_and_sentinel(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "passage.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("query_id", "doc_id", "rank", "score"))
                writer.writerow(("q1", "novel-1", 1, 2.5))
                writer.writerow(("q1", "shared-1", 2, 1.5))
                writer.writerow(("q2", -1, -1, -1))

            arm = read_passage_arm(path)

        self.assertEqual(arm["q1"], (("novel-1", 1, 2.5), ("shared-1", 2, 1.5)))
        self.assertEqual(arm["q2"], ())

    def test_join_preserves_original_arms_and_adds_passage_only_features(self) -> None:
        group = parse_query_record(_record(1))
        original = {
            candidate.doc_id: (
                candidate.dense_rank,
                candidate.dense_score,
                candidate.lexical_rank,
                candidate.lexical_score,
            )
            for candidate in group.candidates
        }
        expanded, novel = attach_passage_arm(
            [group],
            {
                "q1": (
                    ("novel-1", 1, 4.0),
                    ("shared-1", 2, 2.0),
                )
            },
        )

        self.assertEqual(novel, {"q1": {"novel-1"}})
        self.assertEqual(
            tuple(candidate.doc_id for candidate in expanded[0].candidates),
            ("dense-1", "shared-1", "novel-1"),
        )
        self.assertEqual(
            {
                candidate.doc_id: (
                    candidate.dense_rank,
                    candidate.dense_score,
                    candidate.lexical_rank,
                    candidate.lexical_score,
                )
                for candidate in expanded[0].candidates[:2]
            },
            original,
        )
        vectors = feature_vectors(
            expanded[0],
            include_computed=True,
            include_typed=True,
            include_typed_unknown=False,
            include_passage_arm=True,
        )
        self.assertEqual(len(vectors["novel-1"]), len(PASSAGE_TYPED_COMPUTED_FEATURE_NAMES))
        self.assertNotIn("passage_bm25_z", PASSAGE_TYPED_COMPUTED_FEATURE_NAMES)
        self.assertEqual(
            PASSAGE_ARM_FEATURE_NAMES,
            (
                "passage_arm_present",
                "passage_arm_rank_rrf",
                "passage_arm_score_z",
            ),
        )
        # A ranker over the expanded feature schema can score the joined pool.
        model = LinearRanker(
            tuple(0.0 for _ in PASSAGE_TYPED_COMPUTED_FEATURE_NAMES),
            PASSAGE_TYPED_COMPUTED_FEATURE_NAMES,
        )
        self.assertEqual(len(model_ranking(expanded[0], model)), 3)

    def test_id_filtered_metadata_query_is_bounded_and_has_no_embedding(self) -> None:
        query = passage_metadata_query("red shoe", "en-US", ["new-1", "new-2"])
        self.assertEqual(query["rank_by"], ("id", "asc"))
        self.assertEqual(query["top_k"], 2)
        self.assertEqual(
            query["include_attributes"], ["title", "brand", "color", "locale"]
        )
        self.assertNotIn("ANN", str(query))
        self.assertNotIn("Embed", str(query))
        self.assertIn(("id", "In", ["new-1", "new-2"]), query["filters"][1])

    def test_metadata_enrichment_preserves_zero_and_missing(self) -> None:
        group = parse_query_record(_record(3))
        expanded, novel = attach_passage_arm(
            [group], {"q3": (("novel-3", 1, 2.0),)}
        )
        enriched = enrich_passage_only_candidates(
            expanded,
            novel,
            {
                "q3": {
                    "novel-3": {
                        "title": "red shoe",
                        "brand": "Acme",
                        "color": "Red",
                        "locale": "us",
                        "features": {
                            "title_bm25": 0.0,
                            "passage_bm25": None,
                            "brand_bm25": 1.0,
                            "title_full_coverage": False,
                        },
                    }
                }
            },
        )
        candidate = enriched[0].candidates[-1]
        self.assertEqual(candidate.title_bm25, 0.0)
        self.assertIsNone(candidate.passage_bm25)
        self.assertEqual(candidate.brand_bm25, 1.0)
        self.assertFalse(candidate.title_full_coverage)
        self.assertEqual((candidate.brand, candidate.color), ("Acme", "Red"))

    def test_metadata_enrichment_rejects_partial_id_coverage(self) -> None:
        group = parse_query_record(_record(4))
        expanded, novel = attach_passage_arm(
            [group], {"q4": (("novel-4", 1, 2.0),)}
        )
        with self.assertRaisesRegex(ValueError, "exactly cover"):
            enrich_passage_only_candidates(expanded, novel, {"q4": {}})

    def test_ambiguous_grade_zero_is_not_a_novel_training_label(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "positive-only.json"
            path.write_text('{"q6":{"new-6":0}}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "positive Referee"):
                _read_novel_positive_judgments(path, {"q6": {"new-6"}})

    def test_metadata_collection_checkpoints_one_bounded_read_per_query(self) -> None:
        class FakeRow:
            def model_dump(self) -> dict[str, object]:
                return {
                    "id": "novel-5",
                    "title": "red shoe",
                    "brand": "Acme",
                    "color": "Red",
                    "locale": "us",
                    "title_bm25": 0.0,
                    "passage_bm25": None,
                    "brand_bm25": 1.0,
                }

        class FakeNamespace:
            def __init__(self) -> None:
                self.queries: list[dict[str, object]] = []

            def query(self, **kwargs: object) -> SimpleNamespace:
                self.queries.append(kwargs)
                return SimpleNamespace(rows=[FakeRow()])

        group = parse_query_record(_record(5))
        expanded, novel = attach_passage_arm(
            [group], {"q5": (("novel-5", 1, 2.0),)}
        )
        namespace = FakeNamespace()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "checkpoint.jsonl"
            output = root / "metadata.json"
            first = collect_passage_metadata_checkpoint(
                namespace, expanded, novel, checkpoint, output
            )
            second = collect_passage_metadata_checkpoint(
                namespace, expanded, novel, checkpoint, output
            )

            saved = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(first, second)
        self.assertEqual(len(namespace.queries), 1)
        self.assertEqual(namespace.queries[0]["rank_by"], ("id", "asc"))
        self.assertEqual(saved["q5"]["novel-5"]["features"]["title_bm25"], 0.0)


if __name__ == "__main__":
    unittest.main()
