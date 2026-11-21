"""Offline tests for matched agent retrieval variants and concurrency."""

import asyncio
import unittest
from types import SimpleNamespace

from tpuf_optimize.agent_variants import _model_usage, _run_concurrently
from tpuf_optimize.search_agent import RetrievalUsage
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

    def test_cost_uses_measured_input_reasoning_output_and_cache_tokens(self):
        response = SimpleNamespace(
            raw_responses=[
                SimpleNamespace(
                    usage=SimpleNamespace(
                        input_tokens=100,
                        output_tokens=50,
                        input_tokens_details=SimpleNamespace(
                            cached_tokens=10, cache_write_tokens=5
                        ),
                        output_tokens_details=SimpleNamespace(reasoning_tokens=20),
                    )
                ),
                SimpleNamespace(
                    usage=SimpleNamespace(
                        input_tokens=200,
                        output_tokens=30,
                        input_tokens_details=SimpleNamespace(
                            cached_tokens=0, cache_write_tokens=0
                        ),
                        output_tokens_details=SimpleNamespace(reasoning_tokens=10),
                    )
                ),
            ]
        )
        usage = _model_usage(response)
        expected = (
            (285 * 0.10) + (10 * 0.01) + (5 * 0.125) + (80 * 0.50)
        ) / 1_000_000
        self.assertEqual(usage["model_requests"], 2)
        self.assertEqual(usage["input_tokens"], 300)
        self.assertEqual(usage["reasoning_tokens"], 30)
        self.assertEqual(usage["output_tokens"], 80)
        self.assertEqual(usage["visible_output_tokens"], 50)
        self.assertAlmostEqual(usage["openai_cost_usd"], expected)

    def test_retrieval_usage_reads_native_embedding_and_billing_metrics(self):
        usage = RetrievalUsage()
        usage.record(SimpleNamespace(
            results=[SimpleNamespace(
                performance=SimpleNamespace(embedding_tokens=13, embedding_ms=4),
                billing=SimpleNamespace(
                    billable_logical_bytes_queried=100,
                    billable_logical_bytes_returned=200,
                ),
            )]
        ))
        self.assertEqual(usage.to_dict(), {
            "turbopuffer_requests": 1,
            "embedding_usage_responses": 1,
            "embedding_tokens": 13,
            "embedding_ms": 4,
            "billable_bytes_queried": 100,
            "billable_bytes_returned": 200,
        })


if __name__ == "__main__":
    unittest.main()
