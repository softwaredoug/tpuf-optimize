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
        pseudo_relevance_feedback: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be greater than zero")
        self.namespace = namespace
        self.strategy = strategy
        self.batch_size = batch_size
        self.pseudo_relevance_feedback = pseudo_relevance_feedback

    def search(
        self, requests: Iterable[SearchRequest], top_k: int
    ) -> Iterator[tuple[SearchRequest, list[Row]]]:
        """Yield each request with its rows, issuing one API call per batch."""
        iterator = iter(requests)
        while request_batch := list(
            itertools.islice(iterator, self.batch_size)
        ):
            depth = max(top_k, 20) if self.pseudo_relevance_feedback else top_k
            queries = [
                self.strategy.query(request.text, request.locale, depth)
                for request in request_batch
            ]
            response = self.namespace.multi_query(queries=queries)
            if len(response.results) != len(request_batch):
                raise ValueError(
                    "Turbopuffer returned a different number of results than queries"
                )
            first_pass = [
                result.rows or [] for result in response.results
            ]
            if not self.pseudo_relevance_feedback:
                for request, rows in zip(request_batch, first_pass):
                    yield request, rows
                continue

            expanded_requests: list[SearchRequest] = []
            for request, rows in zip(request_batch, first_pass):
                terms = self.strategy.feedback_terms(
                    request.text, request.locale, rows
                )
                if terms:
                    expanded_requests.append(
                        SearchRequest(
                            request.query_id,
                            f"{request.text} {' '.join(terms)}",
                            request.locale,
                        )
                    )

            expanded_by_id: dict[str, list[Row]] = {}
            if expanded_requests:
                expanded_queries = [
                    self.strategy.query(request.text, request.locale, depth)
                    for request in expanded_requests
                ]
                expanded_response = self.namespace.multi_query(
                    queries=expanded_queries
                )
                if len(expanded_response.results) != len(expanded_requests):
                    raise ValueError(
                        "Turbopuffer returned a different number of feedback "
                        "results than queries"
                    )
                expanded_by_id = {
                    request.query_id: result.rows or []
                    for request, result in zip(
                        expanded_requests, expanded_response.results
                    )
                }

            for request, original_rows in zip(request_batch, first_pass):
                if not original_rows:
                    yield request, []
                    continue
                rows_by_id = {str(row.id): row for row in original_rows}
                scores = {
                    document_id: 1.0 / (60 + rank)
                    for rank, document_id in enumerate(rows_by_id, start=1)
                }
                for rank, row in enumerate(
                    expanded_by_id.get(request.query_id, []), start=1
                ):
                    document_id = str(row.id)
                    rows_by_id.setdefault(document_id, row)
                    scores[document_id] = scores.get(document_id, 0.0) + (
                        0.25 / (60 + rank)
                    )
                ordered_ids = sorted(
                    scores, key=lambda document_id: (-scores[document_id], document_id)
                )[:top_k]
                yield request, [rows_by_id[document_id] for document_id in ordered_ids]
