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

    def multi_query(self, *, queries):
        self.calls.append(list(queries))
        return SimpleNamespace(results=[
            SimpleNamespace(rows=[Row.from_dict({
                "id": "doc-a", "$dist": .3, "title": "blue shoe", "brand": "example",
                "title_bm25": 2.0, "passage_bm25": 1.5, "brand_bm25": 0.0,
            })])
            for _ in queries
        ])


class FrozenVariantTests(unittest.TestCase):
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
