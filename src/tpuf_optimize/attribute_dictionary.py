"""Catalog-derived attribute intent matching and conservative reranking.

The dictionary is deliberately fitted from document metadata already returned
by retrieval, rather than from external product data or query judgments. Fit it
once on a query-disjoint catalog sample, then freeze it before evaluating.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)
_ATTRIBUTE_FIELDS = {"brand": "brand", "color": "color"}


def _text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _tokens(value: object) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", _text(value)).casefold()
    return tuple(_WORD_RE.findall(normalized))


def _contains_phrase(haystack: Sequence[str], phrase: Sequence[str]) -> bool:
    if not phrase or len(phrase) > len(haystack):
        return False
    return any(
        tuple(haystack[index:index + len(phrase)]) == tuple(phrase)
        for index in range(len(haystack) - len(phrase) + 1)
    )


@dataclass(frozen=True)
class AttributeIntent:
    """A catalog-supported attribute phrase found in a query."""

    attribute: str
    value: str
    support: int
    start: int
    end: int


@dataclass(frozen=True)
class _Entry:
    value: str
    tokens: tuple[str, ...]
    support: int


class AttributeDictionary:
    """Locale-scoped, support-filtered brand/color phrase dictionary."""

    def __init__(
        self,
        entries: Mapping[str, Mapping[str, Sequence[_Entry]]],
    ) -> None:
        self._entries = {
            locale: {
                attribute: tuple(sorted(values, key=lambda entry: entry.tokens))
                for attribute, values in attributes.items()
            }
            for locale, attributes in entries.items()
        }
        self._by_first_token: dict[
            str, dict[str, dict[str, tuple[_Entry, ...]]]
        ] = {}
        for locale, attributes in self._entries.items():
            self._by_first_token[locale] = {}
            for attribute, values in attributes.items():
                grouped: dict[str, list[_Entry]] = defaultdict(list)
                for entry in values:
                    grouped[entry.tokens[0]].append(entry)
                self._by_first_token[locale][attribute] = {
                    token: tuple(entries) for token, entries in grouped.items()
                }

    @classmethod
    def fit(
        cls,
        rows: Iterable[Mapping[str, object]],
        *,
        min_support: int = 2,
        ambiguous_brand_ratio: float = 2.0,
        min_ambiguous_competitors: int = 3,
    ) -> AttributeDictionary:
        """Fit from catalog metadata, counting each doc_id only once.

        Single-token brand values that appear repeatedly in unrelated titles or
        categories are discarded as ambiguous. This helps avoid treating
        generic product words (for example, "home") as brand intents. Colors
        are not subjected to that brand-specific ambiguity penalty.
        """
        if min_support < 1:
            raise ValueError("min_support must be at least one")
        if ambiguous_brand_ratio <= 0:
            raise ValueError("ambiguous_brand_ratio must be positive")
        if min_ambiguous_competitors < 1:
            raise ValueError("min_ambiguous_competitors must be at least one")

        unique_rows: dict[str, dict[str, str]] = {}
        for number, row in enumerate(rows):
            doc_id = _text(row.get("doc_id")) or f"__row_{number}"
            normalized = {
                "doc_id": doc_id,
                "locale": _text(row.get("locale")).casefold() or "default",
                "title": _text(row.get("title")),
                "category": _text(row.get("category")),
                "brand": _text(row.get("brand")),
                "color": _text(row.get("color")),
            }
            # Keep a duplicate ID's first non-empty metadata; duplicates must
            # never increase support counts.
            existing = unique_rows.setdefault(doc_id, normalized)
            for field, value in normalized.items():
                if not existing[field] and value:
                    existing[field] = value

        supports: dict[tuple[str, str, str], Counter[str]] = defaultdict(Counter)
        competitor_titles: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        display_values: dict[tuple[str, str, str], str] = {}

        for row in unique_rows.values():
            locale = row["locale"]
            for attribute in _ATTRIBUTE_FIELDS:
                value_tokens = _tokens(row[attribute])
                if value_tokens:
                    key = (locale, attribute, " ".join(value_tokens))
                    supports[key][row["doc_id"]] += 1
                    display_values.setdefault(key, row[attribute])

        brand_phrases = {
            (locale, phrase)
            for locale, attribute, phrase in supports
            if attribute == "brand" and len(phrase.split()) == 1
        }
        for locale, phrase in brand_phrases:
            phrase_tokens = tuple(phrase.split())
            for row in unique_rows.values():
                if row["locale"] != locale:
                    continue
                if " ".join(_tokens(row["brand"])) == phrase:
                    continue
                # Mentioning the value in another brand's title/category is
                # evidence that this single-token brand is generic. Count
                # each competing document only once.
                if any(
                    _contains_phrase(_tokens(row[field]), phrase_tokens)
                    for field in ("title", "category", "color")
                ):
                    competitor_titles[(locale, phrase)][row["doc_id"]] += 1

        entries: dict[str, dict[str, list[_Entry]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for (locale, attribute, phrase), docs in supports.items():
            support = len(docs)
            if support < min_support:
                continue
            if attribute == "brand":
                competitors = len(competitor_titles[(locale, phrase)])
                if (
                    competitors >= min_ambiguous_competitors
                    and competitors >= support * ambiguous_brand_ratio
                ):
                    continue
            entries[locale][attribute].append(
                _Entry(
                    value=display_values[(locale, attribute, phrase)],
                    tokens=tuple(phrase.split()),
                    support=support,
                )
            )
        return cls(entries)

    def detect(self, query: str, locale: str) -> list[AttributeIntent]:
        """Find whole-token attribute phrases, preferring longer overlaps."""
        query_tokens = _tokens(query)
        if not query_tokens:
            return []
        locale_entries = self._by_first_token.get(locale.casefold(), {})
        matches: list[AttributeIntent] = []
        for attribute, entries_by_first_token in locale_entries.items():
            for start, first_token in enumerate(query_tokens):
                for entry in entries_by_first_token.get(first_token, ()):
                    width = len(entry.tokens)
                    if query_tokens[start:start + width] == entry.tokens:
                        matches.append(
                            AttributeIntent(
                                attribute=attribute,
                                value=entry.value,
                                support=entry.support,
                                start=start,
                                end=start + width,
                            )
                        )

        # Resolve nested phrases independently per attribute ("star" inside
        # "north star"), while retaining distinct brand and color signals.
        selected: list[AttributeIntent] = []
        for attribute in _ATTRIBUTE_FIELDS:
            candidates = sorted(
                (match for match in matches if match.attribute == attribute),
                key=lambda match: (-(match.end - match.start), match.start),
            )
            occupied: set[int] = set()
            for match in candidates:
                span = set(range(match.start, match.end))
                if span.isdisjoint(occupied):
                    selected.append(match)
                    occupied.update(span)
        return sorted(selected, key=lambda match: (match.start, match.attribute))


def _field_matches(value: object, intent: AttributeIntent) -> bool:
    return _contains_phrase(_tokens(value), _tokens(intent.value))


def rerank_with_attribute_intents(
    query: str,
    locale: str,
    documents: Sequence[Mapping[str, Any]],
    dictionary: AttributeDictionary,
    *,
    relative_boost: float = 0.05,
    max_relative_boost: float = 0.08,
) -> list[dict[str, Any]]:
    """Apply a small soft boost only for direct query-intent/document matches.

    Each row must have its base fused score in ``score``. The function
    preserves the original tie order and never filters a candidate. A detected
    intent alone is not enough: its matching brand/color must also be present
    on that candidate document.
    """
    if relative_boost < 0 or max_relative_boost < 0:
        raise ValueError("boost values must be non-negative")
    intents = dictionary.detect(query, locale)
    ranked: list[tuple[int, float, dict[str, Any]]] = []
    for original_position, source in enumerate(documents):
        row = dict(source)
        base_score = float(row.get("score", 0.0))
        matched = {
            intent.attribute
            for intent in intents
            if _field_matches(row.get(intent.attribute), intent)
        }
        relative = min(max_relative_boost, relative_boost * len(matched))
        row["score"] = base_score * (1.0 + relative)
        row["attribute_boost"] = row["score"] - base_score
        ranked.append((original_position, row["score"], row))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return [row for _, _, row in ranked]
