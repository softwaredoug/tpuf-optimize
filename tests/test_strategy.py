"""Tests for analyzer-specific ESCI indexing fields."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import pandas as pd

from tpuf_optimize.strategy import TPufStrategy


class TPufStrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.strategy = TPufStrategy()
        self.corpus = pd.DataFrame(
            [
                {
                    "doc_id": "B001_us",
                    "product_locale": "us",
                    "title": "Blue Shoe",
                    "description": "Light running shoe",
                    "product_brand": "North Star",
                    "product_bullet_point": "Flexible sole",
                    "product_color": "Blue",
                    "product_category": "Shoes",
                },
                {
                    "doc_id": "B001_es",
                    "product_locale": "es",
                    "title": "Zapato Azul",
                    "description": "Zapato ligero para correr",
                    "product_brand": "Estrella",
                    "product_bullet_point": "Suela flexible",
                    "product_color": "Azul",
                    "product_category": "Zapatos",
                },
                {
                    "doc_id": "B001_jp",
                    "product_locale": "jp",
                    "title": "青い靴",
                    "description": "軽いランニングシューズ",
                    "product_brand": "星",
                    "product_bullet_point": "柔軟なソール",
                    "product_color": "青",
                    "product_category": "靴",
                },
            ]
        )

    def test_emits_locale_specific_analyzer_fields(self) -> None:
        documents = list(self.strategy.documents(self.corpus))

        self.assertEqual(len(documents), 3)
        self.assertEqual(
            [document["id"] for document in documents],
            ["B001_us", "B001_es", "B001_jp"],
        )
        self.assertIn("title__word_v4__en__stem_stop", documents[0])
        self.assertIn("title__word_v4__es__stem_stop_fold", documents[1])
        self.assertIn("title__kuromoji__ja", documents[2])
        self.assertNotIn("title__kuromoji__ja", documents[0])

    def test_emits_alternative_tokenizers_and_passage_embedding(self) -> None:
        document = next(self.strategy.documents(self.corpus.head(1)))

        self.assertEqual(
            document["passage"], "## Blue Shoe\n\nLight running shoe"
        )
        self.assertIn("passage__word_v4__default", document)
        self.assertIn("passage__word_v4__en__stem_stop", document)
        self.assertIn("passage__pretokenized_array__simple", document)
        self.assertIn("title__word_v2__default", document)
        self.assertIn("title__whitespace__default", document)
        self.assertIn("title__pretokenized_array__simple", document)
        self.assertEqual(
            document["passage__embed__voyage_4"],
            "## Blue Shoe\n\nLight running shoe",
        )

    def test_schema_has_voyage_vectors_and_analyzer_variants(self) -> None:
        schema = self.strategy.schema

        self.assertEqual(
            schema["title__word_v4__es__stem_stop_fold"]["full_text_search"][
                "language"
            ],
            "spanish",
        )
        self.assertIn("passage__word_v4__en__stem_stop", schema)
        self.assertEqual(
            schema["passage__embed__voyage_4"]["embed"]["model"],
            "voyage/voyage-4",
        )
        self.assertEqual(
            schema["embedding__voyage_4__passage"]["type"], "[1024]f16"
        )

    def test_passage_is_description_when_title_is_blank(self) -> None:
        corpus = pd.DataFrame(
            [
                {
                    "doc_id": "B002_us",
                    "product_locale": "us",
                    "title": " ",
                    "description": "Only the description",
                }
            ]
        )

        document = next(self.strategy.documents(corpus))

        self.assertEqual(document["passage"], "Only the description")
        self.assertEqual(
            document["passage__embed__voyage_4"], "Only the description"
        )

    def test_default_query_selects_locale_analyzer(self) -> None:
        us_query = self.strategy.query("blue shoe", "us", top_k=10)
        es_query = self.strategy.query("zapato azul", "es", top_k=10)
        jp_query = self.strategy.query("青い靴", "jp", top_k=10)

        self.assertIn("title__word_v4__en__stem_stop", str(us_query["rank_by"]))
        self.assertIn("title__word_v4__es__stem_stop_fold", str(es_query["rank_by"]))
        self.assertIn("title__kuromoji__ja", str(jp_query["rank_by"]))

    def test_pretokenizer_handles_latin_and_japanese(self) -> None:
        self.assertEqual(
            self.strategy.pretokenize("A-B Café"),
            ["a", "b", "café"],
        )
        self.assertEqual(self.strategy.pretokenize("青い靴"), ["青", "い", "靴"])

    def test_binary_presence_query_sums_token_predicates_not_bm25(self) -> None:
        query = self.strategy.binary_presence_query(
            "Blue shoe BLUE the", "us", top_k=25
        )

        self.assertIsNotNone(query)
        assert query is not None
        self.assertEqual(query["filters"], ("locale", "Eq", "us"))
        self.assertEqual(query["top_k"], 25)
        self.assertEqual(
            query["include_attributes"], ["title", "description", "locale"]
        )
        rank_by = query["rank_by"]
        self.assertEqual(rank_by[0], "Sum")
        clauses = rank_by[1]
        self.assertEqual(
            clauses,
            [
                (
                    "Max",
                    [
                        (
                            "Product",
                            2.0,
                            (
                                "title__word_v4__en__stem_stop",
                                "ContainsAllTokens",
                                "blue",
                            ),
                        ),
                        (
                            "passage__word_v4__en__stem_stop",
                            "ContainsAllTokens",
                            "blue",
                        ),
                    ],
                ),
                (
                    "Max",
                    [
                        (
                            "Product",
                            2.0,
                            (
                                "title__word_v4__en__stem_stop",
                                "ContainsAllTokens",
                                "shoe",
                            ),
                        ),
                        (
                            "passage__word_v4__en__stem_stop",
                            "ContainsAllTokens",
                            "shoe",
                        ),
                    ],
                ),
            ],
        )
        self.assertNotIn("BM25", str(rank_by))

    def test_binary_presence_query_guards_empty_stopwords_and_japanese(self) -> None:
        self.assertIsNone(
            self.strategy.binary_presence_query("the and", "us", top_k=10)
        )
        self.assertIsNone(
            self.strategy.binary_presence_query("青い靴", "jp", top_k=10)
        )
        self.assertIsNone(
            self.strategy.binary_presence_query("...", "us", top_k=10)
        )
        with self.assertRaises(ValueError):
            self.strategy.binary_presence_query("   ", "us", top_k=10)

    def test_binary_presence_query_uses_locale_specific_passage_field(self) -> None:
        query = self.strategy.binary_presence_query("zapato blanco", "es", 10)

        self.assertIsNotNone(query)
        assert query is not None
        rank_by = query["rank_by"]
        self.assertIn("title__word_v4__es__stem_stop_fold", str(rank_by))
        self.assertIn("passage__word_v4__es__stem_stop_fold", str(rank_by))

    def test_binary_presence_terms_are_unique_and_clause_bounded(self) -> None:
        duplicate_terms = self.strategy.binary_presence_terms(
            "Café cafe\u0301 shoe shoe", "us"
        )
        self.assertEqual(duplicate_terms, ["café", "shoe"])

        long_query = " ".join(f"word{index}" for index in range(25))
        terms = self.strategy.binary_presence_terms(long_query, "us")
        self.assertEqual(len(terms), self.strategy.MAX_BINARY_PRESENCE_TERMS)

    def test_binary_presence_rows_sort_ties_by_document_id(self) -> None:
        rows = [
            SimpleNamespace(id="z", **{"$dist": 3.0}),
            SimpleNamespace(id="b", **{"$dist": 2.0}),
            SimpleNamespace(id="a", **{"$dist": 2.0}),
            SimpleNamespace(id="x", **{"$dist": 1.0}),
        ]

        sorted_rows = self.strategy.sort_binary_presence_rows(rows)

        self.assertEqual([row.id for row in sorted_rows], ["z", "a", "b", "x"])


if __name__ == "__main__":
    unittest.main()
