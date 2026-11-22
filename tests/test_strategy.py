"""Tests for analyzer-specific ESCI indexing fields."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import cast

import pandas as pd
from turbopuffer.types import Row

from tpuf_optimize.searcher import SearchRequest, Searcher
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

        self.assertIn("passage__word_v4__en__stem_stop", str(us_query["rank_by"]))
        self.assertIn("title__word_v4__en__stem_stop", str(us_query["rank_by"]))
        self.assertIn("passage__word_v4__es__stem_stop_fold", str(es_query["rank_by"]))
        self.assertIn("title__word_v4__es__stem_stop_fold", str(es_query["rank_by"]))
        self.assertIn("passage__kuromoji__ja", str(jp_query["rank_by"]))
        self.assertIn("title__kuromoji__ja", str(jp_query["rank_by"]))

    def test_candidate_queries_are_filter_only_side_channels(self) -> None:
        queries = self.strategy.lexical_candidate_queries(
            "blue shoe", "us", top_k=30
        )

        self.assertEqual(len(queries), 3)
        self.assertEqual(
            queries[0]["filters"],
            (
                "And",
                (
                    ("locale", "Eq", "us"),
                    (
                        "title__word_v4__en__stem_stop",
                        "ContainsAllTokens",
                        "blue shoe",
                    ),
                ),
            ),
        )
        self.assertEqual(queries[1]["rank_by"][0], "passage__word_v4__en__stem_stop")
        self.assertEqual(queries[1]["filters"][1][1][1], "ContainsTokenSequence")
        self.assertEqual(queries[2]["rank_by"][0], "brand__word_v4__default")

    def test_candidate_query_supports_japanese_locale(self) -> None:
        queries = self.strategy.lexical_candidate_queries("青い靴", "jp", top_k=10)

        self.assertEqual(queries[0]["rank_by"][0], "title__kuromoji__ja")
        self.assertEqual(queries[1]["rank_by"][0], "passage__kuromoji__ja")

    def test_union_fusion_deduplicates_and_rewards_cross_arm_matches(self) -> None:
        title_match = cast(Row, SimpleNamespace(id="title-match"))
        consensus_match = cast(Row, SimpleNamespace(id="consensus-match"))
        passage_match = cast(Row, SimpleNamespace(id="passage-match"))

        fused = Searcher._reciprocal_rank_fusion(
            [
                [title_match, consensus_match],
                [consensus_match, passage_match],
            ]
        )

        self.assertEqual(
            [row.id for row in fused],
            ["consensus-match", "title-match", "passage-match"],
        )

    def test_rank_by_adds_fuzzy_boosts_for_long_latin_terms(self) -> None:
        rank_by = str(self.strategy.rank_by("blue running shoe", "us"))

        self.assertIn("title__word_v4__fuzzy", rank_by)
        self.assertIn("brand__word_v4__fuzzy", rank_by)
        self.assertIn("min_query_chars", rank_by)

    def test_rank_by_skips_fuzzy_boosts_for_japanese(self) -> None:
        rank_by = str(self.strategy.rank_by("軽いランニング", "jp"))

        self.assertIn("passage__kuromoji__ja", rank_by)
        self.assertNotIn("__fuzzy", rank_by)

    def test_folded_passage_channel_is_opt_in(self) -> None:
        baseline = str(self.strategy.rank_by("blue shoe", "us"))
        passage_fold = str(
            TPufStrategy(field_fusion="passage_fold").rank_by("blue shoe", "us")
        )

        self.assertNotIn("passage__word_v4__fold", baseline)
        self.assertIn("passage__word_v4__fold", passage_fold)
        with self.assertRaises(ValueError):
            TPufStrategy(field_fusion="unknown")

    def test_pretokenizer_handles_latin_and_japanese(self) -> None:
        self.assertEqual(
            self.strategy.pretokenize("A-B Café"),
            ["a", "b", "café"],
        )
        self.assertEqual(self.strategy.pretokenize("青い靴"), ["青", "い", "靴"])

    def test_position_score_prefers_title_phrase_and_order(self) -> None:
        exact = self.strategy.lexical_position_score(
            "blue running shoe",
            "Blue Running Shoe",
            "A comfortable everyday product",
            None,
        )
        scattered = self.strategy.lexical_position_score(
            "blue running shoe",
            "Shoe for running, color blue",
            "A comfortable everyday product",
            None,
        )

        self.assertGreater(exact, scattered)

    def test_position_score_handles_japanese_text(self) -> None:
        exact = self.strategy.lexical_position_score("青い靴", "青い靴", None, None)
        partial = self.strategy.lexical_position_score("青い靴", "青い", None, None)

        self.assertGreater(exact, partial)

    def test_lexical_rerank_is_bounded_and_stable_for_missing_fields(self) -> None:
        rows = [
            cast(Row, {"id": "first"}),
            cast(Row, {"id": "second"}),
            cast(Row, {"id": "third"}),
        ]

        self.assertEqual(
            [row["id"] for row in self.strategy.rerank_lexically("blue shoe", rows)],
            ["first", "second", "third"],
        )

    def test_lexical_rerank_uses_only_returned_lexical_attributes(self) -> None:
        rows = [
            cast(Row, {"id": "no-match", "title": "Red jacket"}),
            cast(Row, {"id": "phrase", "title": "Blue running shoe"}),
            *[cast(Row, {"id": f"other-{index}"}) for index in range(8)],
        ]

        ranked = self.strategy.rerank_lexically("blue running shoe", rows)

        self.assertEqual(ranked[0]["id"], "phrase")

    def test_lexical_rerank_rejects_unbounded_boost(self) -> None:
        with self.assertRaises(ValueError):
            self.strategy.rerank_lexically("blue shoe", [], max_boost=1.01)

    def test_candidate_union_is_capped_then_position_reranked(self) -> None:
        class FakeRow(dict[str, object]):
            @property
            def id(self) -> str:
                return str(self["id"])

            def model_dump(self) -> dict[str, object]:
                return dict(self)

        baseline_rows = [
            cast(
                Row,
                FakeRow(
                    id="weak",
                    **{"$dist": 1.0},
                    title="Red jacket",
                    description="Outdoor apparel",
                    brand="North Star",
                ),
            ),
            cast(
                Row,
                FakeRow(
                    id="exact",
                    **{"$dist": 0.8},
                    title="Blue running shoe",
                    description="Everyday footwear",
                    brand="North Star",
                ),
            ),
            *[
                cast(
                    Row,
                    FakeRow(id=f"other-{index}", **{"$dist": 0.5}),
                )
                for index in range(48)
            ],
        ]

        class FakeNamespace:
            def multi_query(self, *, queries):
                self.queries = list(queries)
                return SimpleNamespace(
                    results=[
                        SimpleNamespace(rows=baseline_rows),
                        SimpleNamespace(rows=[]),
                        SimpleNamespace(rows=[]),
                        SimpleNamespace(rows=[]),
                    ]
                )

        namespace = FakeNamespace()
        searcher = Searcher(
            namespace,
            self.strategy,
            batch_size=16,
            lexical_candidate_fusion=True,
            lexical_rerank=True,
        )
        results = list(
            searcher.search(
                [SearchRequest("q1", "blue running shoe", "us")],
                top_k=1,
            )
        )

        self.assertEqual(len(namespace.queries), 4)
        self.assertEqual({query["top_k"] for query in namespace.queries}, {50})
        self.assertEqual([row.id for row in results[0][1]], ["exact"])


if __name__ == "__main__":
    unittest.main()
