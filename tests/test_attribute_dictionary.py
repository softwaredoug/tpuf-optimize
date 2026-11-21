"""Offline synthetic-catalog tests for attribute intent matching."""

from __future__ import annotations

import unittest

from tpuf_optimize.attribute_dictionary import (
    AttributeDictionary,
    rerank_with_attribute_intents,
)


def _row(
    doc_id: str,
    *,
    locale: str = "en-US",
    title: str = "",
    brand: str = "",
    color: str = "",
    category: str = "",
) -> dict[str, str]:
    return {
        "doc_id": doc_id,
        "locale": locale,
        "title": title,
        "brand": brand,
        "color": color,
        "category": category,
    }


class AttributeDictionaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = [
            _row("1", title="North Star trail runner", brand="North Star", color="Blue"),
            _row("2", title="North Star road runner", brand="North Star", color="Blue"),
            _row("3", title="North Star hiking shoe", brand="North Star", color="Red"),
            _row("4", title="Blue kids sneaker", brand="Acme", color="Blue"),
            _row("5", title="Blue canvas sneaker", brand="Acme", color="Blue"),
            _row("6", title="Azul zapato ligero", locale="es-ES", brand="Marca", color="Azul"),
            _row("7", title="Azul zapato cómodo", locale="es-ES", brand="Marca", color="Azul"),
            _row("8", title="Star headphones", brand="Star", color="Black"),
            _row("9", title="Star speaker", brand="Star", color="Black"),
        ]
        self.dictionary = AttributeDictionary.fit(self.catalog)

    def test_detects_normalized_whole_token_and_longest_phrase(self) -> None:
        intents = self.dictionary.detect("NORTH-STAR blueish BLUE", "en-US")

        self.assertEqual(
            [(intent.attribute, intent.value) for intent in intents],
            [("brand", "North Star"), ("color", "Blue")],
        )
        self.assertEqual(intents[0].support, 3)
        self.assertEqual(intents[1].support, 4)

    def test_dictionary_values_are_locale_scoped(self) -> None:
        self.assertEqual(
            [intent.value for intent in self.dictionary.detect("azul", "es-ES")],
            ["Azul"],
        )
        self.assertEqual(self.dictionary.detect("azul", "en-US"), [])
        self.assertEqual(self.dictionary.detect("blue", "es-ES"), [])

    def test_duplicate_document_does_not_inflate_support(self) -> None:
        rows = [
            _row("same", brand="One Off", color="Green"),
            _row("same", brand="One Off", color="Green"),
        ]
        dictionary = AttributeDictionary.fit(rows, min_support=2)

        self.assertEqual(dictionary.detect("one off", "en-US"), [])
        self.assertEqual(dictionary.detect("green", "en-US"), [])

    def test_common_single_token_brand_is_guarded_as_ambiguous(self) -> None:
        rows = [
            _row("b1", title="Home smart speaker", brand="Home"),
            _row("b2", title="Home security camera", brand="Home"),
            _row("c1", title="Home theater receiver", brand="Other"),
            _row("c2", title="Home storage cabinet", brand="Other"),
            _row("c3", title="Home office desk", brand="Other"),
            _row("c4", title="Home garden light", brand="Other"),
        ]
        dictionary = AttributeDictionary.fit(rows)

        self.assertFalse(
            any(
                intent.attribute == "brand"
                for intent in dictionary.detect("home", "en-US")
            )
        )

    def test_intent_requires_a_direct_attribute_match_for_soft_boost(self) -> None:
        docs = [
            {
                "doc_id": "title-only",
                "title": "Blue shoe",
                "brand": "Acme",
                "color": "Black",
                "score": 0.10,
            },
            {
                "doc_id": "exact-color",
                "title": "Running sneaker",
                "brand": "Acme",
                "color": "Blue",
                "score": 0.099,
            },
            {
                "doc_id": "other",
                "title": "Running shoe",
                "brand": "Other",
                "color": "Red",
                "score": 0.098,
            },
        ]

        ranked = rerank_with_attribute_intents(
            "blue sneaker", "en-US", docs, self.dictionary
        )

        self.assertEqual(
            [row["doc_id"] for row in ranked],
            ["exact-color", "title-only", "other"],
        )
        self.assertGreater(ranked[0]["attribute_boost"], 0)
        self.assertEqual(ranked[1]["attribute_boost"], 0)
        self.assertEqual(len(ranked), len(docs))

    def test_singleton_attributes_are_not_admitted_by_default(self) -> None:
        dictionary = AttributeDictionary.fit(
            [_row("single", brand="Solo", color="Violet")]
        )
        self.assertEqual(dictionary.detect("solo violet", "en-US"), [])


if __name__ == "__main__":
    unittest.main()
