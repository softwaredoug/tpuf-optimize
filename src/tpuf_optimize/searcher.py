"""Generic batched queries against a Turbopuffer namespace."""

from __future__ import annotations

import itertools
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Protocol

from turbopuffer import RateLimitError
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
        batch_size: int = 100,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be greater than zero")
        self.namespace = namespace
        self.strategy = strategy
        self.batch_size = batch_size

    def _multi_query(self, queries: list[Query]) -> NamespaceMultiQueryResponse:
        """Retry transient model/query throttling with bounded backoff."""
        for attempt in range(6):
            try:
                return self.namespace.multi_query(queries=queries)
            except RateLimitError:
                if attempt == 5:
                    raise
                time.sleep(15 * (attempt + 1))
        raise AssertionError("unreachable")

    def search(
        self, requests: Iterable[SearchRequest], top_k: int
    ) -> Iterator[tuple[SearchRequest, list[Row]]]:
        """Yield each request with its rows, issuing one API call per batch."""
        iterator = iter(requests)
        hybrid = getattr(self.strategy, "hybrid_weights", None)
        queries_per_request = 1
        if hybrid is not None:
            queries_per_request = (
                3
                if getattr(self.strategy, "title_signal", None) is not None
                else 2
            )
        request_limit = min(self.batch_size, 16 // queries_per_request)
        while request_batch := list(itertools.islice(iterator, request_limit)):
            if hybrid is None:
                query_groups = [
                    (self.strategy.query(request.text, request.locale, top_k),)
                    for request in request_batch
                ]
            else:
                query_groups = [
                    self.strategy.hybrid_queries(
                        request.text, request.locale, top_k
                    )
                    for request in request_batch
                ]
            queries = [query for group in query_groups for query in group]
            response = self._multi_query(queries)
            if len(response.results) != len(queries):
                raise ValueError(
                    "Turbopuffer returned a different number of results than queries"
                )
            results = iter(response.results)
            for request, group in zip(request_batch, query_groups):
                rows = [result.rows or [] for result in itertools.islice(
                    results, len(group)
                )]
                if hybrid is None:
                    yield request, rows[0]
                else:
                    weights = hybrid
                    if getattr(self.strategy, "title_signal", None) is not None:
                        weights = (*hybrid, self.strategy.title_signal_weight)
                    yield request, self.strategy.reciprocal_rank_fusion(
                        rows,
                        weights=weights,
                        rank_constant=self.strategy.rank_constant,
                        top_k=top_k,
                    )
