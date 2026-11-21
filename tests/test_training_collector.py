"""Tests for the checkpointed training candidate collector."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from turbopuffer.types import Row

from tpuf_optimize.searcher import SearchRequest
from tpuf_optimize.training_collector import (
    CandidateQueryStrategy,
    _candidate_list,
    _checkpoint_read,
    _rrf_candidates,
    collect_candidates,
)


class FakeNamespace:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def multi_query(self, *, queries):
        query_list = list(queries)
        self.calls.extend(query_list)
        results = []
        for query in query_list:
            mode = query["rank_by"][1]
            docs = (
                [{"id": "dense-1"}, {"id": "shared"}, {"id": "dense-3"}]
                if mode == "ANN"
                else [{"id": "lexical-1"}, {"id": "shared"}, {"id": "lexical-3"}]
            )
            rows = [
                Row.from_dict(
                    {
                        **doc,
                        "$dist": float(index),
                        "title": "Product",
                        "brand": "Brand",
                        "color": "Blue",
                        "category": "Shoes",
                    }
                )
                for index, doc in enumerate(docs, start=1)
            ]
            results.append(SimpleNamespace(rows=rows))
        return SimpleNamespace(results=results)

    def query(self, **_query):
        raise AssertionError("Serial fallback was not expected in this test.")


class TrainingCollectorTests(unittest.TestCase):
    def test_parent_query_expressions_and_locale_mapping(self) -> None:
        lexical = CandidateQueryStrategy("lexical").query("red shoes", "en-US", 100)
        dense = CandidateQueryStrategy("dense").query("red shoes", "en-US", 100)

        self.assertEqual(lexical["filters"], ("locale", "Eq", "us"))
        self.assertEqual(
            lexical["rank_by"],
            (
                "Sum",
                (
                    (
                        "Product",
                        2.0,
                        ("title__word_v4__en__stem_stop", "BM25", "red shoes"),
                    ),
                    (
                        "Product",
                        1.0,
                        (
                            "description__word_v4__en__stem_stop",
                            "BM25",
                            "red shoes",
                        ),
                    ),
                ),
            ),
        )
        self.assertEqual(
            dense["rank_by"],
            ("passage__embed__voyage_4", "ANN", ("Embed", "red shoes")),
        )
        self.assertEqual(lexical["top_k"], 100)

    def test_collect_writes_atomic_complete_jsonl_and_resumes(self) -> None:
        namespace = FakeNamespace()
        requests = [SearchRequest("esci-1", "red shoes", "en-US")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            partial = root / "candidates.jsonl.partial"
            output = root / "candidates.jsonl"
            count = collect_candidates(namespace, requests, partial, output)

            self.assertEqual(count, 1)
            self.assertTrue(output.is_file())
            self.assertEqual(len(namespace.calls), 2)
            self.assertEqual(namespace.calls[0]["top_k"], 100)
            records = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["query_id"], "esci-1")
            self.assertEqual(records[0]["query_text"], "red shoes")
            self.assertEqual(records[0]["dense"][0]["doc_id"], "dense-1")
            self.assertEqual(records[0]["dense"][0]["score"], 1.0)
            self.assertEqual(records[0]["dense"][0]["brand"], "Brand")
            self.assertEqual(records[0]["lexical"][0]["doc_id"], "lexical-1")

            before = len(namespace.calls)
            collect_candidates(namespace, requests, partial, output)
            self.assertEqual(len(namespace.calls), before)

    def test_partial_reader_truncates_only_incomplete_final_record(self) -> None:
        requests = [SearchRequest("esci-1", "red shoes", "en-US")]
        record = {
            "query_id": "esci-1",
            "query_text": "red shoes",
            "locale": "en-US",
            "dense": [],
            "lexical": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "candidates.partial"
            checkpoint.write_text(json.dumps(record) + "\n" + '{"query_id":')
            loaded = _checkpoint_read(checkpoint, requests)
            self.assertEqual(set(loaded), {"esci-1"})
            self.assertEqual(checkpoint.read_text(), json.dumps(record) + "\n")

    def test_rrf_matches_two_to_one_parent_fusion(self) -> None:
        record = {
            "dense": [{"doc_id": "shared"}, {"doc_id": "dense"}],
            "lexical": [{"doc_id": "lexical"}, {"doc_id": "shared"}],
        }
        self.assertEqual(_rrf_candidates(record), ["shared", "dense", "lexical"])

    def test_duplicate_candidate_ids_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate candidate"):
            _candidate_list(
                [
                    Row.from_dict({"id": "same", "$dist": 0.1}),
                    Row.from_dict({"id": "same", "$dist": 0.2}),
                ]
            )


if __name__ == "__main__":
    unittest.main()
