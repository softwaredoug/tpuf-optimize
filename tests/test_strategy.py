"""Tests for analyzer-specific ESCI indexing fields."""

from __future__ import annotations

import unittest

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

    def test_emits_alternative_tokenizers_and_embedding_views(self) -> None:
        document = next(self.strategy.documents(self.corpus.head(1)))

        self.assertIn("title__word_v2__default", document)
        self.assertIn("title__whitespace__default", document)
        self.assertIn("title__pretokenized_array__simple", document)
        self.assertEqual(document["title__embed__voyage_4"], "Blue Shoe")
        self.assertIn("Blue Shoe", document["product_text__embed__voyage_4"])
        self.assertIn("Flexible sole", document["product_text__embed__voyage_4"])

    def test_schema_has_voyage_vectors_and_analyzer_variants(self) -> None:
        schema = self.strategy.schema

        self.assertEqual(
            schema["title__word_v4__es__stem_stop_fold"]["full_text_search"][
                "language"
            ],
            "spanish",
        )
        self.assertEqual(
            schema["title__embed__voyage_4"]["embed"]["model"],
            "voyage/voyage-4",
        )
        self.assertEqual(
            schema["embedding__voyage_4__title"]["type"], "[1024]f16"
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


if __name__ == "__main__":
    unittest.main()
