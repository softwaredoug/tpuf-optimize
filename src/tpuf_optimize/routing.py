"""Query-shape routing for reciprocal-rank fusion of existing result lists."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

Candidate: TypeAlias = Mapping[str, object]

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True)
class FusionWeights:
    """Dense and lexical weights for one route."""

    name: str
    dense: float
    lexical: float


class QueryRouter:
    """Choose an RRF balance from observable query structure only.

    Longer natural-language requests favor semantic retrieval. Short queries
    and queries containing alphanumeric identifiers retain the established
    2:1 dense-to-lexical balance; training evidence did not support shifting
    identifiers toward lexical retrieval.
    """

    BALANCED = FusionWeights("balanced", dense=2.0, lexical=1.0)
    DESCRIPTIVE = FusionWeights("descriptive", dense=3.0, lexical=1.0)
    DESCRIPTIVE_MIN_TOKENS = 4

    @staticmethod
    def _tokens(query: str) -> list[str]:
        normalized = unicodedata.normalize("NFKC", query).casefold()
        return _TOKEN_RE.findall(normalized)

    def route(self, query: str) -> FusionWeights:
        """Return a deterministic scoring profile without using query labels."""
        tokens = self._tokens(query)
        for token in tokens:
            has_letters = any(character.isalpha() for character in token)
            has_digits = any(character.isdigit() for character in token)
            if has_letters and has_digits:
                return self.BALANCED
        if len(tokens) >= self.DESCRIPTIVE_MIN_TOKENS:
            return self.DESCRIPTIVE
        return self.BALANCED


def rerank_query_candidates(
    query: str,
    dense: Sequence[Candidate],
    lexical: Sequence[Candidate],
    *,
    top_k: int = 10,
    rrf_k: int = 10,
    router: QueryRouter | None = None,
) -> list[dict[str, object]]:
    """Fuse dense and lexical lists with a routed weighted RRF profile.

    Input sequence order is rank order (1-based). The union is built
    dense-first, which also provides deterministic tie-breaking. Candidate
    metadata is copied from its first occurrence, preferring its dense row.
    """
    if top_k < 1:
        raise ValueError("top_k must be greater than zero")
    if rrf_k < 0:
        raise ValueError("rrf_k must not be negative")

    weights = (router or QueryRouter()).route(query)
    rows: dict[str, dict[str, object]] = {}
    dense_ranks: dict[str, int] = {}
    lexical_ranks: dict[str, int] = {}

    for rank, candidate in enumerate(dense, start=1):
        document_id = str(candidate["doc_id"])
        dense_ranks.setdefault(document_id, rank)
        rows.setdefault(document_id, dict(candidate))
    for rank, candidate in enumerate(lexical, start=1):
        document_id = str(candidate["doc_id"])
        lexical_ranks.setdefault(document_id, rank)
        rows.setdefault(document_id, dict(candidate))

    fused: list[tuple[str, float, int]] = []
    for dense_order, document_id in enumerate(rows):
        score = 0.0
        if document_id in dense_ranks:
            score += weights.dense / (rrf_k + dense_ranks[document_id])
        if document_id in lexical_ranks:
            score += weights.lexical / (rrf_k + lexical_ranks[document_id])
        fused.append((document_id, score, dense_order))

    fused.sort(key=lambda item: (-item[1], item[2]))
    results: list[dict[str, object]] = []
    for document_id, score, _dense_order in fused[:top_k]:
        row = rows[document_id]
        result = dict(row)
        result["doc_id"] = document_id
        result["fusion_score"] = score
        results.append(result)
    return results
