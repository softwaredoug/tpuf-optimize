"""Offline tests for the interactive product search tool."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agents.tool_context import ToolContext
from turbopuffer.types import Row

from tpuf_optimize.search_agent import (
    ProductRecommendations,
    ProductSearch,
    build_agent,
    extract_ranked_product_ids,
    load_ranker,
)


class ProductSearchTests(unittest.TestCase):
    @patch("tpuf_optimize.search_agent._query_id_filtered_union")
    def test_search_merges_and_reranks_both_arms(self, collect) -> None:
        row = Row.from_dict({
            "id": "shoe", "$dist": 0.25, "title": "Blue shoe",
            "description": "Blue running shoe with a wide toe box",
            "brand": "Runner", "color": "blue", "category": "shoes",
        })
        ns = Mock()
        ns.multi_query.return_value = SimpleNamespace(
            results=[
                SimpleNamespace(
                    rows=[row],
                    performance=SimpleNamespace(embedding_tokens=5, embedding_ms=2),
                ),
                SimpleNamespace(
                    rows=[row],
                    performance=SimpleNamespace(embedding_tokens=0, embedding_ms=0),
                ),
            ]
        )
        collect.return_value = (
            {"shoe": {
                "title_bm25": 1.0, "title_full_coverage": True,
                "typed_product_match": 1.0,
            }},
            1, 0.0, {},
        )
        results = ProductSearch(
            ns, "best-typed-computed", load_ranker()
        ).search("blue shoe", "en-US")
        self.assertEqual(results[0]["doc_id"], "shoe")
        self.assertEqual(results[0]["title"], "Blue shoe")
        self.assertEqual(
            results[0]["description"], "Blue running shoe with a wide toe box"
        )
        self.assertEqual(results[0]["color"], "blue")
        [dense, lexical] = ns.multi_query.call_args.kwargs["queries"]
        for arm in (dense, lexical):
            self.assertEqual(arm["filters"], ("locale", "Eq", "us"))
            self.assertEqual(arm["top_k"], 100)
        self.assertEqual(dense["rank_by"][1], "ANN")
        self.assertEqual(lexical["rank_by"][0], "Sum")
        collect.assert_called_once()

    @patch("tpuf_optimize.search_agent._query_id_filtered_union")
    def test_empty_results_skip_feature_request(self, collect) -> None:
        ns = Mock()
        ns.multi_query.return_value = SimpleNamespace(
            results=[
                SimpleNamespace(
                    rows=[],
                    performance=SimpleNamespace(embedding_tokens=1, embedding_ms=1),
                ),
                SimpleNamespace(
                    rows=None,
                    performance=SimpleNamespace(embedding_tokens=0, embedding_ms=0),
                ),
            ]
        )
        self.assertEqual(
            ProductSearch(ns, "best-typed-computed", load_ranker()).search(
                "shoe", "us"
            ),
            [],
        )
        collect.assert_not_called()

    def test_baselines_use_single_retrieval_arm_and_return_description(self) -> None:
        row = Row.from_dict({
            "id": "shoe", "$dist": 0.25, "title": "Blue shoe",
            "description": "Lightweight trail shoe",
        })
        for variant, ranker in (
            ("naive-bm25", "Sum"),
            ("voyage4-ann", "ANN"),
        ):
            with self.subTest(variant=variant):
                ns = Mock()
                ns.multi_query.return_value = SimpleNamespace(
                    results=[SimpleNamespace(
                        rows=[row],
                        performance=SimpleNamespace(
                            embedding_tokens=3 if variant == "voyage4-ann" else 0,
                            embedding_ms=2 if variant == "voyage4-ann" else 0,
                        ),
                        billing=SimpleNamespace(
                            billable_logical_bytes_queried=100,
                            billable_logical_bytes_returned=200,
                        ),
                    )]
                )
                result, usage = ProductSearch(ns, variant).search_with_usage(
                    "blue shoe", "en-US"
                )
                self.assertEqual(result[0]["description"], "Lightweight trail shoe")
                self.assertEqual(usage["turbopuffer_requests"], 1)
                self.assertEqual(
                    usage["embedding_tokens"],
                    3 if variant == "voyage4-ann" else 0,
                )
                [query] = ns.multi_query.call_args.kwargs["queries"]
                if ranker == "ANN":
                    self.assertEqual(query["rank_by"][1], "ANN")
                else:
                    self.assertEqual(query["rank_by"][0], "Sum")


class AgentToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_requires_tool_and_enforces_locale_and_call_budget(self) -> None:
        search = Mock()
        search.search.return_value = [{
            "doc_id": "shoe", "title": "Blue shoe",
            "description": "A lightweight shoe",
        }]
        agent = build_agent(search, "en-US", "gpt-6-luna")
        self.assertEqual(agent.model_settings.tool_choice, "required")
        self.assertFalse(agent.model_settings.parallel_tool_calls)
        self.assertIs(agent.output_type, ProductRecommendations)
        tool = agent.tools[0]
        self.assertEqual(tool.name, "search_products")
        self.assertIn("descriptions", tool.description.lower())

        async def invoke(locale: str):
            arguments = json.dumps({"query": "blue shoe", "locale": locale})
            context = ToolContext(
                context=None, tool_name=tool.name, tool_call_id="test",
                tool_arguments=arguments,
            )
            return await tool.on_invoke_tool(context, arguments)

        self.assertIn("error", await invoke("jp"))
        for _ in range(3):
            self.assertEqual(
                (await invoke("us"))["products"], search.search.return_value
            )
        self.assertIn("error", await invoke("us"))
        self.assertEqual(search.search.call_count, 3)
        search.search.assert_called_with("blue shoe", "us")

    def test_structured_output_keeps_only_unique_ids_seen_in_tool_results(self):
        output = ProductRecommendations(
            ranked_results=[
                {"doc_id": "p2"},
                {"doc_id": "invented"},
                {"doc_id": "p1"},
                {"doc_id": "p2"},
            ]
        )
        self.assertEqual(
            extract_ranked_product_ids(output, {"p1", "p2"}),
            (["p2", "p1"], 2, False, 0),
        )
        self.assertEqual(
            extract_ranked_product_ids(
                ProductRecommendations(ranked_results=[]), {"p1"}
            ),
            ([], 0, True, 0),
        )
        self.assertEqual(
            extract_ranked_product_ids("free text", {"p1"}),
            ([], 0, True, 1),
        )
