"""Tests for bounded multi-query batching and serial fallback."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx2
from turbopuffer._exceptions import RateLimitError

from tpuf_optimize.searcher import SearchRequest, Searcher
from tpuf_optimize.strategy import TPufStrategy


class FakeNamespace:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.serial_calls = 0
        self.rate_limits_remaining = 0

    def multi_query(self, *, queries):
        self.batch_sizes.append(len(queries))
        return SimpleNamespace(
            results=[SimpleNamespace(rows=[]) for _ in queries]
        )

    def query(self, **_query):
        self.serial_calls += 1
        if self.rate_limits_remaining:
            self.rate_limits_remaining -= 1
            raise self.rate_limit_error()
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


class SearcherTests(unittest.TestCase):
    def test_issues_one_serial_request_per_query(self) -> None:
        namespace = FakeNamespace()
        requests = [
            SearchRequest(f"q{index}", "blue shoe", "us")
            for index in range(17)
        ]

        results = list(
            Searcher(
                namespace,
                TPufStrategy(),
                batch_size=100,
                serial_requests=True,
            ).search(
                requests,
                top_k=10,
            )
        )

        self.assertEqual(len(results), 17)
        self.assertEqual(namespace.batch_sizes, [])
        self.assertEqual(namespace.serial_calls, 17)

    def test_retries_rate_limited_serial_requests(self) -> None:
        namespace = FakeNamespace()
        namespace.rate_limits_remaining = 2
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
                Searcher(
                    namespace,
                    TPufStrategy(),
                    serial_requests=True,
                ).search(requests, top_k=10)
            )

        self.assertEqual(namespace.batch_sizes, [])
        self.assertEqual(len(sleep_durations), 2)
        self.assertEqual(namespace.serial_calls, 4)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(result_rows for _, result_rows in results))

    def test_measured_serial_mode_rejects_multiple_queries(self) -> None:
        searcher = Searcher(
            FakeNamespace(),
            TPufStrategy(),
            batch_size=2,
            serial_requests=True,
        )
        requests = [
            SearchRequest("q1", "blue shoe", "us"),
            SearchRequest("q2", "red shoe", "us"),
        ]

        with self.assertRaisesRegex(ValueError, "one query"):
            searcher.measure_batch(requests, top_k=10)


if __name__ == "__main__":
    unittest.main()
