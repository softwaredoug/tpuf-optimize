"""Synthetic smoke tests for computed feature collection requests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from turbopuffer.types import Row

from tpuf_optimize.computed_feature_collector import (
    ComputedFeatureQueryStrategy,
    collect_feature_candidates,
)
from tpuf_optimize.searcher import SearchRequest
from tpuf_optimize.training_collector import (
    CandidateQueryStrategy,
    collect_candidates,
)


class FakeNamespace:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def multi_query(self, *, queries):
        query_list = list(queries)
        self.calls.extend(query_list)
        results = []
        for query in query_list:
            mode = query["rank_by"][1]
            ids = ["d1", "d2"] if mode == "ANN" else ["d2", "d1"]
            rows = []
            for index, doc_id in enumerate(ids, 1):
                row = {
                    "id": doc_id,
                    "$dist": float(index),
                    "title": "red shoe" if doc_id == "d1" else "blue boot",
                    "title_bm25": 0.0 if doc_id == "d1" else 1.25,
                    "passage_bm25": 0.0 if doc_id == "d1" else 2.0,
                }
                if doc_id == "d1":
                    # Missing brand feature must not be coerced to a real zero.
                    row["brand"] = None
                else:
                    row["brand"] = "Acme"
                    row["brand_bm25"] = 0.0
                rows.append(Row.from_dict(row))
            results.append(SimpleNamespace(rows=rows))
        return SimpleNamespace(results=results)

    def query(self, **_query):
        raise AssertionError("serial fallback was not expected")


class ComputedFeatureCollectorTests(unittest.TestCase):
    def test_feature_requests_keep_retrieval_intent_and_are_locale_specific(self) -> None:
        request = SearchRequest("q1", "red shoes", "en-US")
        feature_strategy = ComputedFeatureQueryStrategy("dense")
        base_strategy = CandidateQueryStrategy("dense")
        feature_dense = feature_strategy.query(request.text, request.locale, 100)
        base_dense = base_strategy.query(request.text, "us", 100)
        self.assertEqual(feature_dense["rank_by"], base_dense["rank_by"])
        self.assertEqual(feature_dense["filters"], base_dense["filters"])
        self.assertEqual(feature_dense["top_k"], base_dense["top_k"])
        self.assertEqual(
            feature_dense["compute_attributes"],
            {
                "title_bm25": ("title__word_v4__en__stem_stop", "BM25", "red shoes"),
                "passage_bm25": (
                    "passage__word_v4__en__stem_stop",
                    "BM25",
                    "red shoes",
                ),
                "brand_bm25": ("brand__word_v4__en__stem_stop", "BM25", "red shoes"),
            },
        )

        feature_strategy.mode = "lexical"
        feature_lexical = feature_strategy.query(request.text, request.locale, 100)
        base_lexical = CandidateQueryStrategy("lexical").query(
            request.text, "us", 100
        )
        self.assertEqual(feature_lexical["rank_by"], base_lexical["rank_by"])
        self.assertEqual(feature_lexical["filters"], base_lexical["filters"])
        self.assertEqual(feature_lexical["top_k"], base_lexical["top_k"])
        self.assertTrue(
            all(
                expression[1] == "BM25"
                for expression in feature_lexical["compute_attributes"].values()
            )
        )

    def test_paired_collection_preserves_order_and_zero_vs_missing(self) -> None:
        request = SearchRequest("q1", "red shoe", "en-US")
        namespace = FakeNamespace()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "candidates.partial"
            output = root / "candidates.jsonl"
            self.assertEqual(
                collect_feature_candidates(namespace, [request], checkpoint, output),
                1,
            )
            record = json.loads(output.read_text(encoding="utf-8").strip())
            base_namespace = FakeNamespace()
            base_output = root / "base.jsonl"
            collect_candidates(
                base_namespace, [request], root / "base.partial", base_output
            )
            base_record = json.loads(base_output.read_text(encoding="utf-8").strip())

        # Rank expression and ordering are unchanged in both arms; compute
        # attributes are response-only and don't alter retrieval intent.
        self.assertEqual(
            [item["doc_id"] for item in record["dense"]], ["d1", "d2"]
        )
        self.assertEqual(
            [item["doc_id"] for item in record["dense"]],
            [item["doc_id"] for item in base_record["dense"]],
        )
        self.assertEqual(
            [item["doc_id"] for item in record["lexical"]], ["d2", "d1"]
        )
        self.assertEqual(
            [item["doc_id"] for item in record["lexical"]],
            [item["doc_id"] for item in base_record["lexical"]],
        )
        self.assertEqual([call["top_k"] for call in namespace.calls], [100, 100])
        self.assertEqual(namespace.calls[0]["rank_by"][1], "ANN")
        self.assertEqual(
            namespace.calls[1]["rank_by"],
            CandidateQueryStrategy("lexical").query(
                request.text, "us", 100
            )["rank_by"],
        )
        # There is just one native Embed use: the dense rank_by. Computed
        # features use BM25 and never invoke a second embedding/model.
        self.assertEqual(
            sum(
                len(call["rank_by"]) == 3
                and call["rank_by"][2][0] == "Embed"
                for call in namespace.calls
            ),
            1,
        )
        for call in namespace.calls:
            self.assertEqual(call["filters"], ("locale", "Eq", "us"))
            self.assertEqual(
                set(call["compute_attributes"]),
                {"title_bm25", "passage_bm25", "brand_bm25"},
            )
        d1 = record["dense"][0]["features"]
        self.assertEqual(d1["title_bm25"], 0.0)
        self.assertEqual(d1["passage_bm25"], 0.0)
        self.assertIsNone(d1["brand_bm25"])
        self.assertTrue(d1["title_full_coverage"])
        d2 = record["dense"][1]["features"]
        self.assertEqual(d2["brand_bm25"], 0.0)
        self.assertFalse(d2["title_full_coverage"])


if __name__ == "__main__":
    unittest.main()
