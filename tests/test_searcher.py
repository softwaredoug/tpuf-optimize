"""Tests for bounded multi-query batching and serial fallback."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx2
from turbopuffer._exceptions import RateLimitError
from turbopuffer.types import Row

from tpuf_optimize.searcher import SearchRequest, Searcher
from tpuf_optimize.strategy import TPufStrategy


class FakeNamespace:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.serial_calls = 0
        self.always_rate_limit = False

    def multi_query(self, *, queries):
        self.batch_sizes.append(len(queries))
        if self.always_rate_limit:
            raise self.rate_limit_error()
        return SimpleNamespace(
            results=[SimpleNamespace(rows=[]) for _ in queries]
        )

    def query(self, **_query):
        self.serial_calls += 1
        return SimpleNamespace(rows=[SimpleNamespace(id="serial-result")])

    @staticmethod
    def rate_limit_error() -> RateLimitError:
        request = httpx2.Request("POST", "https://example.test/query")
        response = httpx2.Response(429, request=request)
        return RateLimitError(
            "rate limited",
            response=response,
            body=None,
        )


class FakeHybridNamespace:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def multi_query(self, *, queries):
        self.batch_sizes.append(len(queries))
        results = []
        for query in queries:
            if query["rank_by"][0] == "passage__embed__voyage_4":
                rows = [
                    Row.from_dict(
                        {
                            "id": "dense-result",
                            "$dist": 0.9,
                            "title": "generic shoe",
                        }
                    )
                ]
            else:
                rows = [
                    Row.from_dict(
                        {
                            "id": "B08ABC1234",
                            "$dist": 12.0,
                            "title": "B08ABC1234 running shoe",
                        }
                    )
                ]
            results.append(SimpleNamespace(rows=rows))
        return SimpleNamespace(results=results)


class SearcherTests(unittest.TestCase):
    def test_caps_batches_at_sixteen_queries(self) -> None:
        namespace = FakeNamespace()
        requests = [
            SearchRequest(f"q{index}", "blue shoe", "us")
            for index in range(17)
        ]

        results = list(
            Searcher(namespace, TPufStrategy(), batch_size=100).search(
                requests,
                top_k=10,
            )
        )

        self.assertEqual(len(results), 17)
        self.assertEqual(namespace.batch_sizes, [16, 1])
        self.assertEqual(namespace.serial_calls, 0)

    def test_confidence_hybrid_search_fuses_the_two_channels(self) -> None:
        namespace = FakeHybridNamespace()
        strategy = TPufStrategy(
            hybrid_weights=(2.0, 1.0),
            confidence_gate="identifier",
        )

        results = list(
            Searcher(namespace, strategy).search(
                [SearchRequest("q1", "B08ABC1234", "en-US")],
                top_k=10,
            )
        )

        self.assertEqual(namespace.batch_sizes, [2])
        self.assertEqual(len(results), 1)
        self.assertEqual(
            str(results[0][1][0].id),
            "B08ABC1234",
        )
        self.assertEqual(len(results[0][1]), 2)

    def test_retries_rate_limited_batch_then_uses_serial_api(self) -> None:
        namespace = FakeNamespace()
        namespace.always_rate_limit = True
        clock = [0.0]
        sleep_durations: list[float] = []

        def fake_sleep(duration: float) -> None:
            sleep_durations.append(duration)
            clock[0] += duration

        requests = [
            SearchRequest("q1", "blue shoe", "us"),
            SearchRequest("q2", "red shoe", "us"),
        ]
        with (
            patch("tpuf_optimize.searcher.time.monotonic", lambda: clock[0]),
            patch("tpuf_optimize.searcher.time.sleep", fake_sleep),
            patch("tpuf_optimize.searcher.random.uniform", lambda low, _high: low),
        ):
            results = list(
                Searcher(namespace, TPufStrategy()).search(requests, top_k=10)
            )

        self.assertEqual(len(namespace.batch_sizes), 10)
        self.assertEqual(sum(sleep_durations), 300.0)
        self.assertEqual(namespace.serial_calls, 2)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(result_rows for _, result_rows in results))


if __name__ == "__main__":
    unittest.main()
