"""Verify the leaderboard chart without live benchmark data."""

import unittest
from xml.etree import ElementTree

from scripts.plot_pareto import Point, pareto_frontier, parse_leaderboard, render_svg


class ParetoPlotTests(unittest.TestCase):
    def test_frontier_excludes_slower_or_less_relevant_points(self) -> None:
        fast = Point("Fast", 10, 0.35)
        relevant = Point("Relevant", 20, 0.42)
        tie = Point("Same metrics", 20, 0.42)
        points = [fast, relevant, tie, Point("Slower", 21, 0.42), Point("Lower", 20, 0.40)]
        self.assertEqual(pareto_frontier(points), [fast, relevant, tie])

    def test_readme_parsing_and_xml_escaping(self) -> None:
        points, pending = parse_leaderboard(
            "## ESCI 5,000-query leaderboard\n\n"
            "| Variant | Latency p50 / p95 (ms) | NDCG@10 |\n"
            "|---|---:|---:|\n"
            "| Title & brand | 42.5 / 80 | 0.41 |\n"
            "| Waiting | Pending | 0.42 |\n"
        )
        self.assertEqual(points, [Point("Title & brand", 42.5, 0.41)])
        self.assertEqual(pending, 1)
        chart = render_svg(points, pending)
        self.assertIn("Title &amp; brand", chart)
        ElementTree.fromstring(chart)

    def test_pending_chart_has_no_points(self) -> None:
        chart = render_svg([], 10)
        self.assertIn("0 measured; 10 pending", chart)
        self.assertNotIn("<circle", chart)
        ElementTree.fromstring(chart)

    def test_tail_latency_axis_uses_p95(self) -> None:
        chart = render_svg([Point("Variant", 80, 0.41)], 0, "p95")
        self.assertIn("End-to-end p95 latency (ms)", chart)
        self.assertNotIn("End-to-end p50 latency (ms)", chart)
        ElementTree.fromstring(chart)

    def test_cost_chart_accepts_a_custom_axis(self) -> None:
        chart = render_svg(
            [Point("Variant", 6.34, 0.45)],
            0,
            "cost",
            chart_title="Relevance vs. model cost",
            x_axis_label="Estimated cost per 5,000 queries (USD)",
            x_description="Estimated model inference cost",
        )
        self.assertIn("Estimated cost per 5,000 queries (USD)", chart)
        self.assertIn("Relevance vs. model cost", chart)
        ElementTree.fromstring(chart)
