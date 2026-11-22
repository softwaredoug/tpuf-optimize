"""Tests for the query CLI options."""

from __future__ import annotations

import unittest

from tpuf_optimize.query_batch import _parser


class QueryBatchTests(unittest.TestCase):
    def test_fuzzy_position_boost_defaults_to_disabled(self) -> None:
        args = _parser().parse_args(
            ["--queries", "queries.csv", "--output", "results.csv"]
        )

        self.assertEqual(args.lexical_fuzzy_position_boost, 0.0)

    def test_fuzzy_position_boost_accepts_a_value(self) -> None:
        args = _parser().parse_args(
            [
                "--queries",
                "queries.csv",
                "--output",
                "results.csv",
                "--lexical-fuzzy-position-boost",
                "0.15",
            ]
        )

        self.assertEqual(args.lexical_fuzzy_position_boost, 0.15)


if __name__ == "__main__":
    unittest.main()
