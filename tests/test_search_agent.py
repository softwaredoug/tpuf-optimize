"""Offline tests for the interactive product search tool."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from agents.tool_context import ToolContext
from turbopuffer.types import Row

from tpuf_optimize.search_agent import ProductSearch, build_agent, load_ranker


class ProductSearchTests(unittest.TestCase):
    @patch("tpuf_optimize.search_agent._query_id_filtered_union")
    def test_search_merges_and_reranks_both_arms(self, collect) -> None:
        row = Row.from_dict({"id": "shoe", "$dist": 0.25, "title": "Blue shoe"})
        ns = Mock()
        ns.multi_query.return_value = SimpleNamespace(
            results=[SimpleNamespace(rows=[row]), SimpleNamespace(rows=[row])]
        )
        collect.return_value = (
            {"shoe": {"title_bm25": 1.0, "title_full_coverage": True}},
            1, 0.0, {},
        )
        results = ProductSearch(ns, load_ranker()).search("blue shoe", "en-US")
        self.assertEqual(results, [{"doc_id": "shoe", "title": "Blue shoe"}])
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
            results=[SimpleNamespace(rows=[]), SimpleNamespace(rows=None)]
        )
        self.assertEqual(ProductSearch(ns, load_ranker()).search("shoe", "us"), [])
        collect.assert_not_called()


class AgentToolTests(unittest.IsolatedAsyncioTestCase):
    async def test_requires_tool_and_enforces_locale_and_call_budget(self) -> None:
        search = Mock()
        search.search.return_value = [{"doc_id": "shoe", "title": "Blue shoe"}]
        agent = build_agent(search, "en-US", "gpt-5-mini")
        self.assertEqual(agent.model_settings.tool_choice, "required")
        tool = agent.tools[0]
        self.assertEqual(tool.name, "search_products")

        async def invoke(locale: str):
            arguments = json.dumps({"query": "blue shoe", "locale": locale})
            context = ToolContext(
                context=None, tool_name=tool.name, tool_call_id="test",
                tool_arguments=arguments,
            )
            return await tool.on_invoke_tool(context, arguments)

        self.assertIn("error", await invoke("jp"))
        for _ in range(3):
            self.assertEqual((await invoke("us"))["products"], search.search.return_value)
        self.assertIn("error", await invoke("us"))
        self.assertEqual(search.search.call_count, 3)
        search.search.assert_called_with("blue shoe", "us")
