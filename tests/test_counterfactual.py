"""Tests for conservative, candidate-matched token ablation."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from turbopuffer.types import Row

from tpuf_optimize.counterfactual import rerank_with_probe, select_probe_token
from tpuf_optimize.searcher import SearchRequest, Searcher
from tpuf_optimize.strategy import TPufStrategy


def row(
    doc_id: str,
    title: str,
    *,
    score: float = 1.0,
    brand: str = "",
    color: str = "",
    category: str = "",
) -> dict[str, object]:
    return {
        "id": doc_id,
        "title": title,
        "description": "",
        "brand": brand,
        "color": color,
        "category": category,
        "$dist": score,
    }


class CounterfactualTests(unittest.TestCase):
    def test_selects_one_numeric_identifier_and_ablates_only_that_token(self) -> None:
        candidates = [
            row("with", "Blue Shoe 13"),
            row("without", "Blue Shoe"),
        ]

        choice = select_probe_token("blue shoe 13", candidates)

        self.assertIsNotNone(choice)
        assert choice is not None
        self.assertEqual(choice.token, "13")
        self.assertEqual(choice.ablated_query, "blue shoe")
        self.assertEqual(choice.reason, "identifier")

    def test_attribute_probe_requires_incomplete_structured_coverage(self) -> None:
        candidates = [
            row("red", "Runner", color="Red"),
            row("blue", "Runner", color="Blue"),
            row("other", "Runner", color="Green"),
        ]

        choice = select_probe_token("red running shoes", candidates)

        self.assertIsNotNone(choice)
        assert choice is not None
        self.assertEqual(choice.token, "red")
        self.assertEqual(choice.reason, "attribute_coverage")

    def test_no_probe_for_single_term_or_uncovered_generic_query(self) -> None:
        candidates = [row("a", "Blue shoe"), row("b", "Red shoe")]

        self.assertIsNone(select_probe_token("shoes", candidates))
        self.assertIsNone(select_probe_token("comfortable everyday", candidates))

    def test_treatment_uses_probe_rank_only_as_a_soft_signal(self) -> None:
        original = [
            row("identifier-match", "Blue Shoe 13", score=9.0),
            row("no-identifier", "Blue Shoe", score=8.0),
            row("other", "Blue Sandal", score=7.0),
        ]
        control = list(original)
        probe = [
            {
                "id": "identifier-match",
                "original_lexical_score": 0.9,
                "counterfactual_score": 0.1,
            },
            {
                "id": "no-identifier",
                "original_lexical_score": 0.8,
                "counterfactual_score": 0.9,
            },
            {
                "id": "other",
                "original_lexical_score": 0.4,
                "counterfactual_score": 0.4,
            },
        ]

        treatment = rerank_with_probe(original, probe, "13")

        self.assertEqual([item["id"] for item in control], [
            "identifier-match",
            "no-identifier",
            "other",
        ])
        self.assertEqual(
            {item["id"] for item in treatment},
            {item["id"] for item in original},
        )
        self.assertEqual(treatment[0]["id"], "no-identifier")
        self.assertEqual(original[0]["id"], "identifier-match")

    def test_strategy_builds_true_same_candidate_loo_score_query(self) -> None:
        strategy = TPufStrategy()

        query = strategy.counterfactual_query(
            "blue shoe 13",
            "blue shoe",
            "us",
            ["doc-a", "doc-b"],
        )

        self.assertEqual(query["rank_by"], ("id", "asc"))
        self.assertEqual(query["top_k"], 2)
        self.assertEqual(
            query["filters"],
            (
                "And",
                (
                    ("locale", "Eq", "us"),
                    ("id", "In", ["doc-a", "doc-b"]),
                ),
            ),
        )
        compute_attributes = query["compute_attributes"]
        score_expr = compute_attributes["counterfactual_score"]
        original_expr = compute_attributes["original_lexical_score"]
        self.assertEqual(score_expr[0], "Sum")
        self.assertEqual(
            score_expr[1][0][2],
            ("title__word_v4__en__stem_stop", "BM25", "blue shoe"),
        )
        self.assertEqual(
            score_expr[1][1][2],
            ("description__word_v4__en__stem_stop", "BM25", "blue shoe"),
        )
        self.assertEqual(
            original_expr[1][0][2],
            ("title__word_v4__en__stem_stop", "BM25", "blue shoe 13"),
        )


class CounterfactualSearcherTests(unittest.TestCase):
    def test_uses_one_batched_loo_query_and_returns_matched_control(self) -> None:
        original_rows = [
            row("with-13", "Blue Shoe 13", score=9.0),
            row("without-13", "Blue Shoe", score=8.0),
        ]

        class Namespace:
            def __init__(self) -> None:
                self.queries: list[list[dict[str, object]]] = []

            def multi_query(self, *, queries):
                query_list = list(queries)
                self.queries.append(query_list)
                if len(self.queries) == 1:
                    return SimpleNamespace(
                        results=[
                            SimpleNamespace(rows=original_rows)
                            for _ in query_list
                        ]
                    )
                self.assert_probe_query(query_list[0])
                return SimpleNamespace(
                    results=[
                        SimpleNamespace(
                            rows=[
                                {
                                    "id": "with-13",
                                    "original_lexical_score": 0.9,
                                    "counterfactual_score": 0.1,
                                },
                                {
                                    "id": "without-13",
                                    "original_lexical_score": 0.8,
                                    "counterfactual_score": 0.9,
                                },
                            ]
                        )
                    ]
                )

            @staticmethod
            def assert_probe_query(query: dict[str, object]) -> None:
                assert query["rank_by"] == ("id", "asc")
                assert query["top_k"] == 2
                assert "counterfactual_score" in query["compute_attributes"]
                assert query["filters"][1][1][1] == "In"
                assert query["filters"][1][1][2] == ["with-13", "without-13"]

        namespace = Namespace()
        request = SearchRequest("q1", "blue shoe 13", "us")

        arms = list(
            Searcher(
                namespace,
                TPufStrategy(hybrid_weights=None),
            ).search_counterfactual_arms([request], top_k=2)
        )

        self.assertEqual(len(namespace.queries), 2)
        self.assertEqual(arms[0].probe_token, "13")
        self.assertEqual(
            [item["id"] for item in arms[0].control_rows],
            ["with-13", "without-13"],
        )
        self.assertEqual(
            [item["id"] for item in arms[0].treatment_rows],
            ["without-13", "with-13"],
        )
        self.assertIs(arms[0].control_rows[0], original_rows[0])

    def test_hybrid_probe_is_confined_to_fused_top_k_and_keeps_attributes(self) -> None:
        dense_rows = [
            Row.from_dict(
                {
                    "id": "with-13",
                    "$dist": 0.1,
                    "title": "Blue Shoe 13",
                    "description": "Everyday shoes",
                    "brand": "North",
                    "color": "Blue",
                    "category": "Shoes",
                }
            ),
            Row.from_dict(
                {
                    "id": "without-13",
                    "$dist": 0.2,
                    "title": "Blue Shoe",
                    "description": "Everyday shoes",
                    "brand": "North",
                    "color": "Blue",
                    "category": "Shoes",
                }
            ),
        ]
        lexical_rows = [
            Row.from_dict({"id": "with-13", "$dist": 4.0}),
            Row.from_dict({"id": "without-13", "$dist": 3.0}),
        ]

        class Namespace:
            def __init__(self) -> None:
                self.calls: list[list[dict[str, object]]] = []

            def multi_query(self, *, queries):
                query_list = list(queries)
                self.calls.append(query_list)
                if len(self.calls) == 1:
                    return SimpleNamespace(
                        results=[
                            SimpleNamespace(rows=dense_rows),
                            SimpleNamespace(rows=lexical_rows),
                        ]
                    )
                probe_filter = query_list[0]["filters"]
                assert probe_filter[1][1][2] == ["with-13", "without-13"]
                return SimpleNamespace(
                    results=[
                        SimpleNamespace(
                            rows=[
                                {
                                    "id": "with-13",
                                    "original_lexical_score": 0.9,
                                    "counterfactual_score": 0.1,
                                },
                                {
                                    "id": "without-13",
                                    "original_lexical_score": 0.8,
                                    "counterfactual_score": 0.9,
                                },
                            ]
                        )
                    ]
                )

        namespace = Namespace()
        arms = list(
            Searcher(namespace, TPufStrategy()).search_counterfactual_arms(
                [SearchRequest("q1", "blue shoe 13", "us")],
                top_k=2,
            )
        )[0]

        self.assertEqual([len(batch) for batch in namespace.calls], [2, 1])
        self.assertEqual(arms.probe_token, "13")
        self.assertEqual(
            [str(result.id) for result in arms.control_rows],
            ["with-13", "without-13"],
        )
        self.assertEqual(
            [str(result.id) for result in arms.treatment_rows],
            ["without-13", "with-13"],
        )
        self.assertEqual(arms.control_rows[0]["brand"], "North")


if __name__ == "__main__":
    unittest.main()
