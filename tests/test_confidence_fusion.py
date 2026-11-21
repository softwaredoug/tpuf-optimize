"""Offline tests for query-local confidence gates."""

from __future__ import annotations

import unittest

from tpuf_optimize.confidence_fusion import confidence_aware_weights


def _candidates(
    prefix: str, scores: list[float], *, title: str = ""
) -> list[dict[str, object]]:
    return [
        {
            "doc_id": f"{prefix}-{index}",
            "score": score,
            "title": title if index == 0 else f"other product {index}",
            "brand": "",
            "color": "",
            "category": "",
        }
        for index, score in enumerate(scores)
    ]


class ConfidenceFusionTests(unittest.TestCase):
    def test_margin_gate_prefers_channel_with_clear_gap_and_low_overlap(self) -> None:
        dense = _candidates("d", [0.99, 0.98, 0.97, 0.96, 0.95, 0.10])
        lexical = _candidates("l", [12, 11.9, 11.8, 11.7, 11.6, 11.5])

        weights = confidence_aware_weights("blue shoes", dense, lexical)

        self.assertEqual(weights, (3.0, 1.0))

    def test_margin_gate_keeps_baseline_when_lists_agree(self) -> None:
        dense = _candidates("d", [0.99, 0.98, 0.97, 0.96, 0.95, 0.10])
        lexical = [
            {**candidate, "score": score}
            for candidate, score in zip(
                dense,
                [12, 11.9, 11.8, 11.7, 11.6, 11.5],
            )
        ]

        weights = confidence_aware_weights("blue shoes", dense, lexical)

        self.assertEqual(weights, (2.0, 1.0))

    def test_margin_gate_can_favor_lexical_channel(self) -> None:
        dense = _candidates("d", [0.99, 0.98, 0.97, 0.96, 0.95, 0.94])
        lexical = _candidates("l", [12, 11, 10, 9, 8, 1])

        weights = confidence_aware_weights("blue shoes", dense, lexical)

        self.assertEqual(weights, (1.0, 2.0))

    def test_identifier_gate_requires_a_lexical_only_identifier_hit(self) -> None:
        dense = _candidates("DENSE", [0.9, 0.8, 0.7], title="other shoe")
        lexical = _candidates(
            "B08ABC1234", [12, 8, 2], title="B08ABC1234 running shoe"
        )

        weights = confidence_aware_weights(
            "find B08ABC1234", dense, lexical, gate="identifier"
        )

        self.assertEqual(weights, (1.0, 2.0))

    def test_identifier_gate_does_not_override_dense_corroboration(self) -> None:
        dense = _candidates(
            "B08ABC1234", [0.9, 0.8, 0.7], title="B08ABC1234 running shoe"
        )
        lexical = _candidates(
            "B08ABC1234", [12, 8, 2], title="B08ABC1234 running shoe"
        )

        weights = confidence_aware_weights(
            "find B08ABC1234", dense, lexical, gate="identifier"
        )

        self.assertEqual(weights, (2.0, 1.0))

    def test_unusable_score_signals_keep_baseline(self) -> None:
        dense = _candidates("d", [0.9, 0.9, 0.9])
        lexical = _candidates("l", [1.0, 1.0, 1.0])

        weights = confidence_aware_weights("shoe", dense, lexical)

        self.assertEqual(weights, (2.0, 1.0))

    def test_invalid_gate_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            confidence_aware_weights("shoe", [], [], gate="unsupported")  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
