"""Generic batched queries against a Turbopuffer namespace."""

from __future__ import annotations

import itertools
import time
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, cast

from turbopuffer import RateLimitError
from turbopuffer.types import Row
from turbopuffer.types.namespace_multi_query_params import Query
from turbopuffer.types.namespace_multi_query_response import NamespaceMultiQueryResponse

from tpuf_optimize.counterfactual import rerank_with_probe, select_probe_token
from tpuf_optimize.strategy import TPufStrategy


def _row_id(row: object) -> str:
    if isinstance(row, Mapping):
        return str(row["id"])
    return str(row.id)  # type: ignore[attr-defined]


@dataclass(frozen=True)
class SearchRequest:
    """A query and its associated output identifier."""

    query_id: str
    text: str
    locale: str


@dataclass(frozen=True)
class SearchArms:
    """Matched original/control and counterfactual-treatment result lists."""

    request: SearchRequest
    probe_token: str | None
    probe_reason: str | None
    control_rows: list[Row]
    treatment_rows: list[Row]


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

    def _query_batch(self, queries: list[Query]) -> list[list[Row]]:
        response = self._multi_query(queries)
        if len(response.results) != len(queries):
            raise ValueError(
                "Turbopuffer returned a different number of results than queries"
            )
        return [result.rows or [] for result in response.results]

    def _query_groups(
        self,
        request_batch: list[SearchRequest],
        top_k: int,
        *,
        include_counterfactual_attributes: bool = False,
    ) -> list[tuple[Query, ...]]:
        hybrid = getattr(self.strategy, "hybrid_weights", None)
        if hybrid is None:
            return [
                (
                    self.strategy.query(
                        request.text,
                        request.locale,
                        top_k,
                        include_counterfactual_attributes=(
                            include_counterfactual_attributes
                        ),
                    ),
                )
                for request in request_batch
            ]

        groups = [
            self.strategy.hybrid_queries(request.text, request.locale, top_k)
            for request in request_batch
        ]
        if not include_counterfactual_attributes:
            return groups
        requested_attributes = [
            "title",
            "description",
            "locale",
            "brand",
            "color",
            "category",
        ]
        return [
            tuple(
                cast(
                    Query,
                    {
                        **query,
                        "include_attributes": requested_attributes,
                    },
                )
                for query in group
            )
            for group in groups
        ]

    def _fuse_group(
        self,
        ranked_rows: list[list[Row]],
        top_k: int,
    ) -> list[Row]:
        hybrid = getattr(self.strategy, "hybrid_weights", None)
        if hybrid is None:
            return ranked_rows[0][:top_k]
        return self.strategy.reciprocal_rank_fusion(
            ranked_rows,
            weights=hybrid,
            rank_constant=self.strategy.rank_constant,
            top_k=top_k,
        )

    @staticmethod
    def _attach_candidate_attributes(
        fused_rows: list[Row],
        source_rows: list[list[Row]],
    ) -> list[Row]:
        """Retain RRF candidates and their existing attributes for token checks."""
        attributes_by_id: dict[str, dict[str, object]] = {}
        for rows in source_rows:
            for row in rows:
                extras = row.model_extra or {}
                attributes: dict[str, object] = {}
                for attribute in (
                    "title",
                    "description",
                    "brand",
                    "color",
                    "category",
                ):
                    value = extras.get(attribute)
                    if value is not None:
                        attributes[attribute] = value
                attributes_by_id.setdefault(
                    str(row.id),
                    attributes,
                )
        return [
            Row.from_dict(
                cast(
                    Any,
                    {
                        "id": str(row.id),
                        "$dist": row["$dist"],
                        **attributes_by_id.get(str(row.id), {}),
                    },
                )
            )
            for row in fused_rows
        ]

    def search(
        self, requests: Iterable[SearchRequest], top_k: int
    ) -> Iterator[tuple[SearchRequest, list[Row]]]:
        """Yield each request with its rows, issuing one API call per batch."""
        iterator = iter(requests)
        hybrid = getattr(self.strategy, "hybrid_weights", None)
        request_limit = min(self.batch_size, 8 if hybrid is not None else 16)
        while request_batch := list(itertools.islice(iterator, request_limit)):
            query_groups = self._query_groups(request_batch, top_k)
            queries = [query for group in query_groups for query in group]
            result_rows = self._query_batch(queries)
            offset = 0
            for request, group in zip(request_batch, query_groups):
                rows = result_rows[offset:offset + len(group)]
                offset += len(group)
                yield request, self._fuse_group(rows, top_k)

    def search_counterfactual_arms(
        self, requests: Iterable[SearchRequest], top_k: int
    ) -> Iterator[SearchArms]:
        """Yield matched baseline and treatment lists with same-candidate LOO.

        Eligible requests incur one additional logical query. It is restricted
        to the original top-k candidate IDs and computes the ablated BM25 score
        as an attribute, so it cannot add or remove candidate documents.
        """
        iterator = iter(requests)
        hybrid = getattr(self.strategy, "hybrid_weights", None)
        request_limit = min(self.batch_size, 8 if hybrid is not None else 16)
        while request_batch := list(itertools.islice(iterator, request_limit)):
            query_groups = self._query_groups(
                request_batch,
                top_k,
                include_counterfactual_attributes=True,
            )
            queries = [query for group in query_groups for query in group]
            baseline_query_rows = self._query_batch(queries)
            offset = 0
            control_by_request: list[list[Row]] = []
            for group in query_groups:
                source_rows = baseline_query_rows[offset:offset + len(group)]
                offset += len(group)
                fused = self._fuse_group(source_rows, top_k)
                if hybrid is not None:
                    fused = self._attach_candidate_attributes(fused, source_rows)
                control_by_request.append(fused)

            probe_specs: list[tuple[int, str, str]] = []
            probe_queries: list[Query] = []
            for index, (request, rows) in enumerate(
                zip(request_batch, control_by_request)
            ):
                choice = select_probe_token(request.text, rows)
                if choice is None or not rows:
                    continue
                candidate_ids = list(dict.fromkeys(_row_id(row) for row in rows))
                probe_specs.append((index, choice.token, choice.reason))
                probe_queries.append(
                    self.strategy.counterfactual_query(
                        request.text,
                        choice.ablated_query,
                        request.locale,
                        candidate_ids,
                    )
                )

            probes_by_index: dict[int, list[Row]] = {}
            if probe_queries:
                probe_rows = self._query_batch(probe_queries)
                probes_by_index = {
                    spec[0]: rows
                    for spec, rows in zip(probe_specs, probe_rows)
                }

            choices_by_index = {
                index: (token, reason)
                for index, token, reason in probe_specs
            }
            for index, (request, original_rows) in enumerate(
                zip(request_batch, control_by_request)
            ):
                choice = choices_by_index.get(index)
                if choice is None:
                    yield SearchArms(
                        request=request,
                        probe_token=None,
                        probe_reason=None,
                        control_rows=original_rows,
                        treatment_rows=original_rows,
                    )
                    continue
                token, reason = choice
                treatment = cast(
                    list[Row],
                    rerank_with_probe(
                        original_rows,
                        probes_by_index.get(index, []),
                        token,
                    ),
                )
                yield SearchArms(
                    request=request,
                    probe_token=token,
                    probe_reason=reason,
                    control_rows=original_rows,
                    treatment_rows=treatment,
                )
