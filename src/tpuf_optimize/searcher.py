"""Generic batched queries against a Turbopuffer namespace."""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Protocol

from turbopuffer.types import Row
from turbopuffer.types.namespace_multi_query_params import Query
from turbopuffer.types.namespace_multi_query_response import NamespaceMultiQueryResponse

from tpuf_optimize.strategy import TPufStrategy


@dataclass(frozen=True)
class SearchRequest:
    """A query and its associated output identifier."""

    query_id: str
    text: str
    locale: str


class NamespaceSearcher(Protocol):
    """The namespace multi-query surface required by Searcher."""

    def multi_query(
        self, *, queries: Iterable[Query]
    ) -> NamespaceMultiQueryResponse: ...


class Searcher:
    """Batch search requests and preserve their input order."""

    def __init__(
        self,
        namespace: NamespaceSearcher,
        strategy: TPufStrategy,
        batch_size: int = 16,
        lexical_candidate_fusion: bool = False,
        lexical_rerank: bool = False,
        lexical_rerank_boost: float = TPufStrategy.LEXICAL_RERANK_MAX_BOOST,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be greater than zero")
        self.namespace = namespace
        self.strategy = strategy
        self.batch_size = batch_size
        self.lexical_candidate_fusion = lexical_candidate_fusion
        self.lexical_stats = {
            "arm_queries": 0,
            "zero_result_arms": 0,
            "union_candidates": 0,
        }
        self.lexical_rerank = lexical_rerank
        self.lexical_rerank_boost = lexical_rerank_boost

    def search(
        self, requests: Iterable[SearchRequest], top_k: int
    ) -> Iterator[tuple[SearchRequest, list[Row]]]:
        """Yield each request with its rows, issuing one API call per batch."""
        iterator = iter(requests)
        arm_count = 4 if self.lexical_candidate_fusion else 1
        request_batch_size = min(self.batch_size, 16 // arm_count)
        while request_batch := list(
            itertools.islice(iterator, request_batch_size)
        ):
            candidate_depth = 50 if (
                self.lexical_candidate_fusion or self.lexical_rerank
            ) else top_k
            if self.lexical_candidate_fusion:
                queries = [
                    query
                    for request in request_batch
                    for query in (
                        self.strategy.query(
                            request.text, request.locale, candidate_depth
                        ),
                        *self.strategy.lexical_candidate_queries(
                            request.text, request.locale, candidate_depth
                        ),
                    )
                ]
            else:
                queries = [
                    self.strategy.query(
                        request.text, request.locale, candidate_depth
                    )
                    for request in request_batch
                ]
            response = self.namespace.multi_query(queries=queries)
            if len(response.results) != len(queries):
                raise ValueError(
                    "Turbopuffer returned a different number of results than queries"
                )
            if self.lexical_candidate_fusion:
                for request_number, request in enumerate(request_batch):
                    start = request_number * arm_count
                    arm_results = response.results[start:start + arm_count]
                    arm_rows = [result.rows or [] for result in arm_results]
                    self.lexical_stats["arm_queries"] += len(arm_rows)
                    self.lexical_stats["zero_result_arms"] += sum(
                        not rows for rows in arm_rows
                    )
                    fused_rows = self._reciprocal_rank_fusion(arm_rows)
                    self.lexical_stats["union_candidates"] += len(fused_rows)
                    rows = fused_rows[:candidate_depth]
                    if self.lexical_rerank:
                        rows = self.strategy.rerank_lexically(
                            request.text, rows, self.lexical_rerank_boost
                        )
                    yield request, rows[:top_k]
            else:
                for request, result in zip(request_batch, response.results):
                    rows = result.rows or []
                    if self.lexical_rerank:
                        rows = self.strategy.rerank_lexically(
                            request.text, rows, self.lexical_rerank_boost
                        )
                    yield request, rows[:top_k]

    @staticmethod
    def _reciprocal_rank_fusion(arm_rows: list[list[Row]]) -> list[Row]:
        """Merge independent lexical candidate lists by reciprocal rank."""
        rank_constant = 20
        scores: dict[str, float] = {}
        documents: dict[str, Row] = {}
        for rows in arm_rows:
            for rank, row in enumerate(rows, start=1):
                document_id = str(row.id)
                scores[document_id] = scores.get(document_id, 0.0) + 1 / (
                    rank_constant + rank
                )
                documents.setdefault(document_id, row)
        return [
            documents[document_id]
            for document_id in sorted(
                scores, key=lambda document_id: (-scores[document_id], document_id)
            )
        ]
