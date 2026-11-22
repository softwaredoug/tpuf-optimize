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
        batch_size: int = 100,
        lexical_rerank: bool = False,
        lexical_rerank_boost: float = TPufStrategy.LEXICAL_RERANK_MAX_BOOST,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be greater than zero")
        self.namespace = namespace
        self.strategy = strategy
        self.batch_size = batch_size
        self.lexical_rerank = lexical_rerank
        self.lexical_rerank_boost = lexical_rerank_boost

    def search(
        self, requests: Iterable[SearchRequest], top_k: int
    ) -> Iterator[tuple[SearchRequest, list[Row]]]:
        """Yield each request with its rows, issuing one API call per batch."""
        iterator = iter(requests)
        while request_batch := list(
            itertools.islice(iterator, self.batch_size)
        ):
            queries = [
                self.strategy.query(request.text, request.locale, top_k)
                for request in request_batch
            ]
            response = self.namespace.multi_query(queries=queries)
            if len(response.results) != len(request_batch):
                raise ValueError(
                    "Turbopuffer returned a different number of results than queries"
                )
            for request, result in zip(request_batch, response.results):
                rows = result.rows or []
                if self.lexical_rerank:
                    rows = self.strategy.rerank_lexically(
                        request.text, rows, self.lexical_rerank_boost
                    )
                yield request, rows
