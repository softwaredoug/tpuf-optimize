"""Offline tests for the gated, paired agent evaluator."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tpuf_optimize.agent_eval import (
    _paired_bootstrap_ci,
    _read_state,
    compare_results,
    extract_recommendation_ids,
    result_rows,
    run_arm,
)
from tpuf_optimize.searcher import SearchRequest


class RecommendationExtractionTests(unittest.TestCase):
    def test_accepts_only_explicit_ids_returned_by_a_tool_call(self) -> None:
        ids, invalid, missing, format_failure = extract_recommendation_ids(
            "Top pick — Document ID: `p-1`. Alternative doc_id: p-2. "
            "Product ID: invented.",
            {"p-1", "p-2", "p-3"},
        )
        self.assertEqual(ids, ["p-1", "p-2"])
        self.assertEqual(invalid, 1)
        self.assertFalse(missing)
        self.assertFalse(format_failure)

    def test_unlabeled_or_non_string_output_is_not_filled_from_candidates(self) -> None:
        ids, invalid, missing, format_failure = extract_recommendation_ids(
            "I recommend the blue shoe, which is a great match.",
            {"candidate-only"},
        )
        self.assertEqual(ids, [])
        self.assertEqual(invalid, 0)
        self.assertTrue(missing)
        self.assertTrue(format_failure)
        self.assertEqual(
            extract_recommendation_ids(None, {"candidate-only"}),
            ([], 0, True, True),
        )

    def test_explicit_no_match_is_not_counted_as_a_format_failure(self) -> None:
        self.assertEqual(
            extract_recommendation_ids(
                "No suitable products were found.", {"candidate-only"}
            ),
            ([], 0, False, False),
        )

    def test_empty_candidates_produce_no_hits_without_format_error(self) -> None:
        self.assertEqual(
            extract_recommendation_ids("No products found.", set()),
            ([], 0, False, False),
        )

    def test_result_rows_have_contiguous_ranks_and_exact_sentinel(self) -> None:
        self.assertEqual(
            result_rows("q1", ["p1", "p2"]),
            [
                {"query_id": "q1", "doc_id": "p1", "rank": 1, "score": 1.0},
                {"query_id": "q1", "doc_id": "p2", "rank": 2, "score": 0.5},
            ],
        )
        self.assertEqual(
            result_rows("q2", []),
            [{"query_id": "q2", "doc_id": "-1", "rank": -1, "score": -1}],
        )
        with self.assertRaises(ValueError):
            result_rows("q3", ["p1", "p1"])


class CheckpointTests(unittest.TestCase):
    def test_live_runner_is_disabled_without_explicit_approval(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Live calls are gated"):
            run_arm(
                "direct",
                Path("unused-task.json"),
                Path("unused-checkpoint.json"),
                Path("unused-result.json"),
                "unused-namespace",
                "unused-region",
                allow_live=False,
            )

    def test_checkpoint_binds_arm_task_and_identical_ordered_query_ids(self) -> None:
        requests = [SearchRequest("q1", "secret query", "en-US")]
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "state.json"
            state = _read_state(checkpoint, "direct", "task-1", requests)
            self.assertEqual(state["query_ids"], ["q1"])
            self.assertNotIn("secret query", checkpoint.read_text(encoding="utf-8"))
            state["records"]["q1"] = {
                "doc_ids": ["p1"],
                "end_to_end_seconds": 0.1,
                "backend_seconds": 0.1,
                "tool_calls": 0,
                "invalid_id_count": 0,
                "missing_id_queries": 0,
                "format_failure_queries": 0,
            }
            checkpoint.write_text(json.dumps(state), encoding="utf-8")
            self.assertEqual(
                _read_state(checkpoint, "direct", "task-1", requests)["records"],
                state["records"],
            )
            with self.assertRaises(ValueError):
                _read_state(checkpoint, "agent", "task-1", requests)
            with self.assertRaises(ValueError):
                _read_state(
                    checkpoint,
                    "direct",
                    "task-1",
                    [SearchRequest("q2", "other", "en-US")],
                )

    def test_paired_bootstrap_is_deterministic_and_respects_constant_deltas(self) -> None:
        first = _paired_bootstrap_ci([0.1, 0.1, 0.1])
        second = _paired_bootstrap_ci([0.1, 0.1, 0.1])
        self.assertEqual(first, second)
        self.assertAlmostEqual(first[0], 0.1)
        self.assertAlmostEqual(first[1], 0.1)

    def test_compare_reports_paired_ndcg_latency_and_failure_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            query_ids = [f"q{index}" for index in range(500)]
            direct_result = {
                "metrics": {"ndcg": 0.5},
                "queries": [{"query_id": key, "ndcg": 0.5} for key in query_ids],
            }
            agent_result = {
                "metrics": {"ndcg": 0.6},
                "queries": [{"query_id": key, "ndcg": 0.6} for key in query_ids],
            }
            direct_records = {
                key: {
                    "end_to_end_seconds": 0.1,
                    "backend_seconds": 0.1,
                    "tool_calls": 0,
                    "invalid_id_count": 0,
                    "missing_id_queries": 0,
                    "format_failure_queries": 0,
                }
                for key in query_ids
            }
            agent_records = {
                key: {
                    "end_to_end_seconds": 1.1,
                    "backend_seconds": 0.1,
                    "tool_calls": 2,
                    "invalid_id_count": 1,
                    "missing_id_queries": 2,
                    "format_failure_queries": 3,
                }
                for key in query_ids
            }
            for name, value in (
                ("direct.result.json", direct_result),
                ("agent.result.json", agent_result),
                (
                    "direct.checkpoint.json",
                    {"records": direct_records, "failure_counts": {}},
                ),
                (
                    "agent.checkpoint.json",
                    {
                        "records": agent_records,
                        "failure_counts": {"RuntimeError": 4},
                    },
                ),
            ):
                (root / name).write_text(json.dumps(value), encoding="utf-8")
            summary = compare_results(
                root / "direct.result.json",
                root / "agent.result.json",
                root / "direct.checkpoint.json",
                root / "agent.checkpoint.json",
            )
            self.assertAlmostEqual(
                summary["paired_agent_minus_direct_ndcg10"], 0.1
            )
            self.assertAlmostEqual(
                summary["paired_bootstrap_95pct_ci"][0], 0.1
            )
            self.assertAlmostEqual(
                summary["paired_bootstrap_95pct_ci"][1], 0.1
            )
            self.assertEqual(summary["agent_tool_calls_total"], 1000)
            self.assertEqual(
                summary["agent_latency"]["failure_counts"], {"RuntimeError": 4}
            )
            self.assertEqual(summary["agent_latency"]["invalid_id_count"], 500)


if __name__ == "__main__":
    unittest.main()
