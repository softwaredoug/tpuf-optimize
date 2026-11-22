"""Generic batched queries against a Turbopuffer namespace."""

from __future__ import annotations

import itertools
import logging
import random
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, TypeVar, cast

from turbopuffer._exceptions import APIError, RateLimitError
from turbopuffer.types import Row
from turbopuffer.types.namespace_multi_query_params import Query
from tpuf_optimize.strategy import TPufStrategy

_Result = TypeVar("_Result")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchRequest:
    """A query and its associated output identifier."""

    query_id: str
    text: str
    locale: str


class Searcher:
    """Batch search requests and preserve their input order."""

    MAX_QUERIES_PER_BATCH = 16
    MAX_RETRY_WAIT_SECONDS = 5 * 60
    INITIAL_BACKOFF_SECONDS = 1.0

    def __init__(
        self,
        namespace: Any,
        strategy: TPufStrategy,
        batch_size: int = MAX_QUERIES_PER_BATCH,
        serial_requests: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be greater than zero")
        self.namespace = namespace
        self.strategy = strategy
        self.batch_size = min(batch_size, self.MAX_QUERIES_PER_BATCH)
        self.serial_requests = serial_requests

    @staticmethod
    def _retry_after_seconds(error: RateLimitError) -> float:
        """Return a numeric Retry-After value when the server supplied one."""
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", {})
        retry_after = headers.get("Retry-After")
        if retry_after is None:
            return 0.0
        try:
            return max(0.0, float(retry_after))
        except (TypeError, ValueError):
            try:
                retry_at = parsedate_to_datetime(str(retry_after))
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                return max(
                    0.0,
                    (retry_at - datetime.now(timezone.utc)).total_seconds(),
                )
            except (TypeError, ValueError, OverflowError):
                return 0.0

    def _with_rate_limit_backoff(
        self, operation: Callable[[], _Result]
    ) -> _Result:
        """Retry a request with jittered exponential backoff for at most 5 minutes."""
        result, _elapsed_ms, _retry_count = (
            self._with_rate_limit_backoff_measured(operation)
        )
        return result

    def _with_rate_limit_backoff_measured(
        self, operation: Callable[[], _Result]
    ) -> tuple[_Result, float, int]:
        """Return the successful attempt's latency and number of 429 retries."""
        started_at = time.monotonic()
        attempt = 0
        while True:
            request_started = time.perf_counter()
            try:
                result = operation()
                elapsed_ms = (time.perf_counter() - request_started) * 1000
                return result, elapsed_ms, attempt
            except RateLimitError as error:
                elapsed = time.monotonic() - started_at
                remaining = self.MAX_RETRY_WAIT_SECONDS - elapsed
                if remaining <= 0:
                    raise

                exponential_delay = min(
                    self.INITIAL_BACKOFF_SECONDS * (2**attempt),
                    self.MAX_RETRY_WAIT_SECONDS,
                )
                jittered_delay = random.uniform(
                    exponential_delay / 2,
                    exponential_delay,
                )
                delay = max(
                    jittered_delay,
                    self._retry_after_seconds(error),
                )
                if delay >= remaining:
                    time.sleep(remaining)
                    raise
                time.sleep(delay)
                attempt += 1

    def _query_serially(self, queries: list[Query]) -> list[list[Row]]:
        """Run one HTTP query request for each input query."""
        serial_query = getattr(self.namespace, "query", None)
        if not callable(serial_query):
            raise RuntimeError(
                "Namespace does not provide the single-query API required "
                "by this lexical strategy"
            )

        results: list[list[Row]] = []
        for query in queries:
            response = self._with_rate_limit_backoff(
                lambda query=query: cast(Any, serial_query)(**query)
            )
            results.append(response.rows or [])
        return results

    def _query_batch(self, queries: list[Query]) -> list[list[Row]]:
        """Run a bounded multi-query request, falling back to serial calls."""
        try:
            response = self._with_rate_limit_backoff(
                lambda: self.namespace.multi_query(queries=queries)
            )
            if len(response.results) == len(queries):
                return [result.rows or [] for result in response.results]
            logger.warning(
                "multi_query returned %d results for %d queries; retrying serially",
                len(response.results),
                len(queries),
            )
        except APIError:
            logger.warning(
                "multi_query batch failed; retrying %d queries serially",
                len(queries),
            )
        return self._query_serially(queries)

    def search(
        self, requests: Iterable[SearchRequest], top_k: int
    ) -> Iterator[tuple[SearchRequest, list[Row]]]:
        """Yield each request in order, issuing one serial API call per query."""
        iterator = iter(requests)
        while request_batch := list(
            itertools.islice(iterator, self.batch_size)
        ):
            queries = [
                self.strategy.query(request.text, request.locale, top_k)
                for request in request_batch
            ]
            rows_by_query = (
                self._query_serially(queries)
                if self.serial_requests
                else self._query_batch(queries)
            )
            for request, rows in zip(request_batch, rows_by_query):
                yield request, rows

    def _retrieve_measured_batch(
        self, requests: list[SearchRequest], top_k: int
    ) -> list[list[Row]]:
        """Retrieve a batch; serial mode insists on exactly one query request."""
        queries = [
            self.strategy.query(request.text, request.locale, top_k)
            for request in requests
        ]
        if self.serial_requests:
            if len(requests) != 1:
                raise ValueError("Measured serial requests must contain one query")
            serial_query = getattr(self.namespace, "query", None)
            if not callable(serial_query):
                raise RuntimeError("Namespace does not provide the single-query API")
            response = cast(Any, serial_query)(**queries[0])
            return [response.rows or []]

        response = self.namespace.multi_query(queries=queries)
        if len(response.results) != len(requests):
            raise ValueError("Incomplete response for measured query batch")
        return [result.rows or [] for result in response.results]

    def measure_batch(
        self, requests: list[SearchRequest], top_k: int
    ) -> tuple[list[list[Row]], float, int]:
        """Return one ranking, successful request latency, and 429 retries."""
        if not 1 <= len(requests) <= self.batch_size:
            raise ValueError("Measured batch must fit the configured batch size")
        if self.serial_requests and len(requests) != 1:
            raise ValueError("Measured serial requests must contain one query")
        rows, elapsed, retries = self._with_rate_limit_backoff_measured(
            lambda: self._retrieve_measured_batch(requests, top_k)
        )
        if len(rows) != len(requests):
            raise ValueError("Incomplete response for measured query batch")
        top_rows = [hits[:top_k] for hits in rows]
        return top_rows, elapsed / len(requests), retries
