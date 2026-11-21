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
    """Batch focused retrievals, fuse them locally, and preserve input order."""

    MAX_SUBQUERIES_PER_REQUEST = 16

    def __init__(
        self,
        namespace: NamespaceSearcher,
        strategy: TPufStrategy,
        batch_size: int = 100,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be greater than zero")
        self.namespace = namespace
        self.strategy = strategy
        self.batch_size = batch_size

    def search(
        self, requests: Iterable[SearchRequest], top_k: int
    ) -> Iterator[tuple[SearchRequest, list[Row]]]:
        """Yield results with reciprocal-rank fusion across focused candidates."""
        iterator = iter(requests)
        max_requests = min(
            self.batch_size,
            self.MAX_SUBQUERIES_PER_REQUEST
            // self.strategy.FUSION_QUERY_COUNT,
        )
        while request_batch := list(itertools.islice(iterator, max_requests)):
            query_groups = [
                self.strategy.fusion_queries(request.text, request.locale, top_k)
                for request in request_batch
            ]
            queries = [query for group in query_groups for query in group]
            response = self.namespace.multi_query(queries=queries)
            if len(response.results) != len(queries):
                raise ValueError(
                    "Turbopuffer returned a different number of results than queries"
                )
            for index, (request, query_group) in enumerate(
                zip(request_batch, query_groups)
            ):
                result_offset = index * len(query_group)
                result_group = response.results[
                    result_offset:result_offset + len(query_group)
                ]
                rows_by_id: dict[str, Row] = {}
                fused_scores: dict[str, float] = {}
                for result, channel_weight in zip(
                    result_group, self.strategy.FUSION_CHANNEL_WEIGHTS
                ):
                    for rank, row in enumerate(result.rows or [], start=1):
                        document_id = str(row.id)
                        rows_by_id[document_id] = row
                        fused_scores[document_id] = fused_scores.get(
                            document_id, 0.0
                        ) + channel_weight / (
                            self.strategy.RRF_RANK_CONSTANT + rank
                        )
                ranked_ids = sorted(
                    fused_scores,
                    key=lambda document_id: (
                        -fused_scores[document_id],
                        document_id,
                    ),
                )[:top_k]
                fused_rows = [
                    rows_by_id[document_id].model_copy(
                        update={"$dist": fused_scores[document_id]}
                    )
                    for document_id in ranked_ids
                ]
                yield request, fused_rows
