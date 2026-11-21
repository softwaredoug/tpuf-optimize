"""Offline tests for deterministic query-grouped learned reranking."""

from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from tpuf_optimize.learned_rank import (
    FEATURE_NAMES,
    candidate_oracle_ndcg,
    evaluate,
    feature_vectors,
    fit_ranker,
    main,
    model_ranking,
    parse_query_record,
    read_candidates,
    read_grades,
    rrf_ranking,
    split_query_groups,
)


def _record(index: int) -> dict[str, object]:
    query = f"red shoe {index}"
    return {
        "query_id": f"q{index}",
        "query_text": query,
        "locale": "en-US",
        "dense": [
            {
                "doc_id": f"bad-{index}",
                "score": 0.9,
                "title": f"blue boot {index}",
            },
            {
                "doc_id": f"good-{index}",
                "score": 0.5,
                "title": f"red shoe {index} walking sneaker",
            },
        ],
        "lexical": [
            {
                "doc_id": f"good-{index}",
                "score": 0.8,
                "title": f"red shoe {index} walking sneaker",
            },
            {
                "doc_id": f"bad-{index}",
                "score": 0.2,
                "title": f"blue boot {index}",
            },
        ],
    }


def _groups(count: int = 8):
    return [parse_query_record(_record(index)) for index in range(count)]


def _grades(count: int = 8) -> dict[str, dict[str, float]]:
    return {
        f"q{index}": {f"good-{index}": 3.0, f"bad-{index}": 1.0}
        for index in range(count)
    }


class LearnedRankTests(unittest.TestCase):
    def test_feature_extraction_uses_scores_ranks_and_text_only(self) -> None:
        group = parse_query_record(_record(1))
        vectors = feature_vectors(group)
        self.assertEqual(set(vectors), {"bad-1", "good-1"})
        self.assertEqual(len(vectors["good-1"]), len(FEATURE_NAMES))
        self.assertTrue(
            all("id" not in name for name in FEATURE_NAMES)
        )
        self.assertGreater(
            vectors["good-1"][FEATURE_NAMES.index("title_query_coverage")],
            vectors["bad-1"][FEATURE_NAMES.index("title_query_coverage")],
        )
        self.assertEqual(group.candidates[0].dense_rank, 1)
        self.assertEqual(group.candidates[0].lexical_rank, 2)

    def test_rrf_baseline_is_weighted_and_deterministic(self) -> None:
        group = parse_query_record(_record(2))
        ranking = rrf_ranking(group)
        self.assertEqual([item.doc_id for item, _score in ranking], ["bad-2", "good-2"])
        self.assertAlmostEqual(
            ranking[0][1], 2 / 61 + 1 / 62
        )
        self.assertEqual(ranking, rrf_ranking(group))

    def test_pairwise_fit_ignores_grade_zero_as_negative(self) -> None:
        groups = _groups(2)
        with self.assertRaisesRegex(ValueError, "no unequal positive-grade pairs"):
            fit_ranker(
                groups[:1],
                {"q0": {"good-0": 3.0, "bad-0": 0.0}},
                epochs=5,
            )

    def test_pairwise_fit_and_eval_compare_same_candidates_to_rrf(self) -> None:
        groups = _groups()
        grades = _grades()
        training, validation = split_query_groups(groups, 0.25, 31)
        train_ids = {group.query_id for group in training}
        validation_ids = {group.query_id for group in validation}
        self.assertFalse(train_ids.intersection(validation_ids))
        model = fit_ranker(training, grades, epochs=600, learning_rate=0.15)
        report = evaluate(validation, grades, model)
        self.assertEqual(report["evaluated_queries"], len(validation))
        self.assertGreater(
            report["pairwise_linear_ndcg"], report["rrf_2_1_ndcg"]
        )
        self.assertEqual(candidate_oracle_ndcg(training, grades), 1.0)
        for group in validation:
            self.assertEqual(
                {candidate.doc_id for candidate, _ in model_ranking(group, model)},
                {candidate.doc_id for candidate, _score in rrf_ranking(group)},
            )

    def test_readers_and_cli_run_query_disjoint_synthetic_data(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate_file = root / "candidates.jsonl"
            grade_file = root / "grades.jsonl"
            output_file = root / "validation.csv"
            model_file = root / "model.json"
            candidate_file.write_text(
                "".join(json.dumps(_record(index)) + "\n" for index in range(8)),
                encoding="utf-8",
            )
            grade_file.write_text(
                "".join(
                    json.dumps(
                        {
                            "query_id": f"q{index}",
                            "grades": {
                                f"good-{index}": 3,
                                f"bad-{index}": 0 if index == 0 else 1,
                            },
                        }
                    )
                    + "\n"
                    for index in range(8)
                ),
                encoding="utf-8",
            )
            self.assertEqual(len(read_candidates(candidate_file)), 8)
            self.assertEqual(read_grades(grade_file)["q0"]["bad-0"], 0.0)
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                status = main(
                    [
                        "--candidates",
                        str(candidate_file),
                        "--grades",
                        str(grade_file),
                        "--output",
                        str(output_file),
                        "--model-output",
                        str(model_file),
                        "--seed",
                        "31",
                        "--epochs",
                        "600",
                        "--learning-rate",
                        "0.15",
                    ]
                )
            self.assertEqual(status, 0)
            report = json.loads(stdout.getvalue())
            self.assertEqual(report["train_queries"], 6)
            self.assertEqual(report["validation_queries"], 2)
            self.assertTrue(model_file.exists())
            self.assertIn("weights", json.loads(model_file.read_text(encoding="utf-8")))
            output_rows = output_file.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(output_rows), 1 + 2 * 2)
            self.assertEqual(output_rows[0], "query_id,doc_id,rank,score")


if __name__ == "__main__":
    unittest.main()
