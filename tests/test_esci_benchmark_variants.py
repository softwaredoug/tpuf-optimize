"""Verify all pinned benchmark adapters without network calls."""

import unittest
from types import SimpleNamespace

from turbopuffer.types import Row

from scripts.esci_benchmark_variants import VARIANTS, FrozenSearcher
from tpuf_optimize.query_batch import _ranked_rows
from tpuf_optimize.searcher import SearchRequest


class Namespace:
    def __init__(self):
        self.calls = []
        self.options = []

    def multi_query(self, *, queries, **options):
        self.calls.append(list(queries))
        self.options.append(options)
        result = SimpleNamespace(rows=[Row.from_dict({
                "id": "doc-a", "$dist": .3, "title": "blue shoe", "brand": "example",
                "title_bm25": 2.0, "passage_bm25": 1.5, "brand_bm25": 0.0,
            })])
        count = 1 if "rerank_by" in options else len(queries)
        return SimpleNamespace(results=[result for _ in range(count)])


class FrozenVariantTests(unittest.TestCase):
    def test_each_leader_has_an_explicit_strategy_class(self):
        categories = {variant.slug: variant.category for variant in VARIANTS}
        self.assertEqual(categories["bm25"], "Pure lexical")
        self.assertEqual(categories["hybrid"], "Hybrid (lexical + embedding)")
        self.assertEqual(categories["typed"], "Learned reranker")
        self.assertEqual(categories["voyage-rerank"], "Third-party reranker")
        self.assertEqual(categories["voyage"], "Pure embedding")
        self.assertEqual(categories["product"], "Handcrafted reranker")

    def test_all_variants_rank_independently_on_every_execution(self):
        request = SearchRequest("qid", "blue shoe", "us")
        for variant in VARIANTS:
            with self.subTest(variant=variant.slug):
                namespace = Namespace()
                searcher = FrozenSearcher(namespace, variant)
                for _ in range(2):
                    [rows], latency, retries = searcher.measure_batch([request], 10)
                    self.assertEqual(_ranked_rows(request, rows)[0][:3], ("qid", "doc-a", 1))
                    self.assertGreaterEqual(latency, 0)
                    self.assertEqual(retries, 0)
                expected_calls = 4 if variant.slug in {"computed", "typed"} else 2
                self.assertEqual(len(namespace.calls), expected_calls)
                if variant.slug == "voyage":
                    rank_by = namespace.calls[0][0]["rank_by"]
                    self.assertEqual(rank_by[0], "passage__embed__voyage_4")
                    self.assertEqual(rank_by[1], "ANN")
                if variant.slug == "voyage-rerank":
                    self.assertEqual(
                        namespace.options[0]["rerank_by"]["model"],
                        "voyage/rerank-3",
                    )
                    self.assertEqual(
                        namespace.options[0]["rerank_by"]["attribute"], "passage"
                    )
                    self.assertEqual(namespace.options[0]["limit"], 10)
                    self.assertTrue(all(
                        "passage" in query["include_attributes"]
                        for query in namespace.calls[0]
                    ))
