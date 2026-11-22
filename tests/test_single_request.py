"""Tests for the single-request lexical candidate and local reranker."""

from __future__ import annotations

import unittest

from tpuf_optimize.single_request import (
    SingleRequestLexicalStrategy,
    local_lexical_evidence,
    rerank_candidates,
    search_one,
)


class SingleRequestLexicalStrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy = SingleRequestLexicalStrategy()

    def test_query_matches_cdb2d64_title_passage_bm25(self) -> None:
        query = self.strategy.query("red shoe", "us", 50)
        self.assertEqual(query["top_k"], 50)
        self.assertEqual(query["filters"], ("locale", "Eq", "us"))
        self.assertEqual(
            query["rank_by"],
            (
                "Sum",
                (
                    (
                        "Product",
                        1.0,
                        ("title__word_v4__en__stem_stop", "BM25", "red shoe"),
                    ),
                    (
                        "Product",
                        0.5,
                        ("passage__word_v4__en__stem_stop", "BM25", "red shoe"),
                    ),
                ),
            ),
        )
        self.assertEqual(
            query["include_attributes"],
            ["title", "description", "brand", "locale"],
        )

    def test_locale_analyzers_are_used_without_schema_changes(self) -> None:
        spanish = self.strategy.query("zapatos rojos", "es", 100)
        japanese = self.strategy.query("青い靴", "jp", 100)
        self.assertIn("title__word_v4__es__stem_stop_fold", str(spanish["rank_by"]))
        self.assertIn("passage__word_v4__es__stem_stop_fold", str(spanish["rank_by"]))
        self.assertIn("title__kuromoji__ja", str(japanese["rank_by"]))
        self.assertIn("passage__kuromoji__ja", str(japanese["rank_by"]))

    def test_top_k_is_bounded(self) -> None:
        for invalid_top_k in (0, 101):
            with self.subTest(top_k=invalid_top_k):
                with self.assertRaises(ValueError):
                    self.strategy.query("red shoe", "us", invalid_top_k)


class LocalLexicalRerankTests(unittest.TestCase):
    def test_search_one_uses_one_serial_namespace_request(self) -> None:
        class FakeNamespace:
            def __init__(self) -> None:
                self.calls = []

            def query(self, **kwargs: object) -> object:
                self.calls.append(kwargs)
                return type(
                    "Response",
                    (),
                    {
                        "rows": [
                            {"id": "a", "title": "rack"},
                            {"id": "b", "title": "blue shoe"},
                            {"id": "c", "title": "shoe holder"},
                        ]
                    },
                )()

        namespace = FakeNamespace()
        ranked = search_one(namespace, "blue shoe", "us", top_k=50, boost=0.35)
        self.assertEqual(len(namespace.calls), 1)
        self.assertEqual(namespace.calls[0]["top_k"], 50)
        self.assertEqual([row["id"] for row in ranked], ["b", "a", "c"])

    def test_title_phrase_and_ordered_evidence(self) -> None:
        rows = [
            {"id": "a", "title": "Shoe rack", "brand": "", "description": ""},
            {"id": "b", "title": "Blue shoe", "brand": "", "description": ""},
            {"id": "c", "title": "Shoe blue", "brand": "", "description": ""},
        ]
        ranked = rerank_candidates("blue shoe", rows, boost=0.5)
        self.assertEqual([row["id"] for row in ranked], ["b", "a", "c"])
        self.assertGreater(
            local_lexical_evidence("blue shoe", rows[1]),
            local_lexical_evidence("blue shoe", rows[2]),
        )

    def test_brand_and_description_are_supporting_field_evidence(self) -> None:
        title_only = {
            "title": "rack",
            "brand": "",
            "description": "",
        }
        brand_match = {
            "title": "rack",
            "brand": "Blue",
            "description": "shoe organizer",
        }
        self.assertGreater(
            local_lexical_evidence("blue shoe", brand_match),
            local_lexical_evidence("blue shoe", title_only),
        )

    def test_empty_and_zero_boost_preserve_original_order(self) -> None:
        rows = [{"title": "shoe"}, {"title": "blue shoe"}]
        self.assertEqual(rerank_candidates("blue shoe", rows, 0.0), rows)
        self.assertEqual(rerank_candidates("blue shoe", [], 0.25), [])

    def test_boost_must_be_finite_range(self) -> None:
        rows = [{"title": "shoe"}]
        for invalid_boost in (-0.01, 1.01):
            with self.subTest(boost=invalid_boost):
                with self.assertRaises(ValueError):
                    rerank_candidates("shoe", rows, invalid_boost)


if __name__ == "__main__":
    unittest.main()
