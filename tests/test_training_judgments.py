"""Synthetic tests for training-only Referee debug judgment caching."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tpuf_optimize.searcher import SearchRequest
from tpuf_optimize.training_judgments import (
    _debug_payload,
    _parse_debug_batch,
    collect_training_judgments,
)


def _candidate_record(index: int):
    return {
        "query_id": f"q{index}",
        "query_text": f"red shoe {index}",
        "locale": "us",
        "dense": [{"doc_id": f"dense-{index}", "score": 0.2}],
        "lexical": [{"doc_id": f"lexical-{index}", "score": 1.0}],
    }


def _debug_body(requests, candidates):
    queries = {}
    for request in requests:
        ids = {
            str(candidate["doc_id"])
            for arm in ("dense", "lexical")
            for candidate in candidates[request.query_id][arm]
        }
        queries[request.query_id] = {
            "query_id": request.query_id,
            "query_text": request.text,
            "results": [
                {
                    "doc_id": doc_id,
                    "rank": rank,
                    "score": 0.0,
                    "relevance_grade": 3 if "dense" in doc_id else 0,
                }
                for rank, doc_id in enumerate(sorted(ids), 1)
            ],
            "target": [{"doc_id": f"dense-{request.query_id[1:]}", "relevance_grade": 3}],
        }
    return {"queries": queries}


class TrainingJudgmentTests(unittest.TestCase):
    def test_debug_payload_and_response_validation_keep_query_groups(self) -> None:
        requests = [SearchRequest("q1", "red shoe 1", "us")]
        records = {"q1": _candidate_record(1)}
        payload = _debug_payload(requests, records)
        self.assertEqual(payload["query_ids"], ["q1"])
        self.assertEqual(len(payload["results"]), 2)
        body = _debug_body(requests, records)
        parsed = _parse_debug_batch(body, requests, records)
        self.assertEqual(parsed[0]["grades"], {"dense-1": 3, "lexical-1": 0})
        self.assertEqual(
            parsed[0]["targets"], [{"doc_id": "dense-1", "grade": 3}]
        )

    def test_training_judgments_checkpoint_and_output_are_aggregate_safe(self) -> None:
        requests = [
            SearchRequest(f"q{index}", f"red shoe {index}", "us")
            for index in range(3)
        ]
        records = {f"q{index}": _candidate_record(index) for index in range(3)}
        task = {"task": {"id": "training-task"}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            posted = []

            def fake_post(task_id, token, payload):
                posted.append((task_id, token, len(payload["query_ids"])))
                batch = [
                    request
                    for request in requests
                    if request.query_id in payload["query_ids"]
                ]
                return {
                    "body": _debug_body(batch, records),
                    "attempts": 1,
                    "elapsed_ms": 4.0,
                }

            with patch(
                "tpuf_optimize.training_judgments._post_debug", side_effect=fake_post
            ):
                report = collect_training_judgments(
                    task,
                    requests,
                    list(records.values()),
                    "source-hash",
                    "fake-training-token",
                    root / "judgments.partial",
                    root / "grades.jsonl",
                    root / "targets.jsonl",
                )
            self.assertEqual(posted, [("training-task", "fake-training-token", 3)])
            self.assertEqual(report["queries"], 3)
            self.assertEqual(report["candidate_judgments"], 6)
            grade_lines = [
                json.loads(line)
                for line in (root / "grades.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(grade_lines), 3)
            self.assertEqual(grade_lines[0]["grades"]["dense-0"], 3)
            self.assertEqual(grade_lines[0]["grades"]["lexical-0"], 0)
            target_lines = [
                json.loads(line)
                for line in (root / "targets.jsonl").read_text().splitlines()
            ]
            self.assertEqual(target_lines[0]["targets"][0]["grade"], 3)


if __name__ == "__main__":
    unittest.main()
