"""Offline tests for matched agent retrieval variants and concurrency."""

import asyncio
import unittest

from tpuf_optimize.agent_variants import _run_concurrently
from tpuf_optimize.searcher import SearchRequest


class AgentVariantTests(unittest.TestCase):
    def test_query_workers_obey_concurrency_and_keep_independent_results(self):
        requests = [
            SearchRequest(f"q{index}", f"query {index}", "us")
            for index in range(20)
        ]
        active = 0
        maximum = 0
        completed = []

        async def operation(request):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0.002)
            active -= 1
            return {"query_id": request.query_id}

        def on_complete(request, result, error):
            self.assertIsNone(error)
            completed.append((request.query_id, result["query_id"]))

        asyncio.run(_run_concurrently(requests, 4, operation, on_complete))

        self.assertEqual(maximum, 4)
        self.assertEqual(len(completed), len(requests))
        self.assertEqual(
            {query_id for query_id, result_id in completed},
            {result_id for query_id, result_id in completed},
        )

    def test_rejects_zero_concurrency(self):
        with self.assertRaises(ValueError):
            asyncio.run(_run_concurrently([], 0, None, None))


if __name__ == "__main__":
    unittest.main()
