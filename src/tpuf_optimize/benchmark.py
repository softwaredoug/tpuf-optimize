"""Checkpoint uncached query executions and export their Referee rankings."""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
import os
import sqlite3
import tempfile
from collections.abc import Callable
from contextlib import closing
from pathlib import Path

from turbopuffer.types import Row

from tpuf_optimize.searcher import SearchRequest, Searcher

RankedRows = list[tuple[str, str, int, float]]


def run_benchmark(
    db_path: Path,
    output: Path,
    requests: list[SearchRequest],
    searcher: Searcher,
    top_k: int,
    target: tuple[str, str],
    resume: bool,
    rank: Callable[[SearchRequest, list[Row]], RankedRows],
) -> None:
    if not requests:
        raise ValueError("Benchmark requires at least one query")
    signature = hashlib.sha256(json.dumps([
        target, top_k, searcher.batch_size,
        [(request.query_id, request.text, request.locale) for request in requests],
        [Path(inspect.getfile(cls)).read_text(encoding="utf-8") for cls in (
            type(searcher), type(searcher.strategy)
        )],
    ]).encode()).hexdigest()
    if not resume:
        fd = os.open(db_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.close(fd)
    elif not db_path.is_file():
        raise ValueError("Cannot resume without an existing benchmark checkpoint")

    with closing(sqlite3.connect(db_path)) as db:
        if not resume:
            with db:
                db.execute("CREATE TABLE run (signature TEXT NOT NULL)")
                db.execute("INSERT INTO run VALUES (?)", (signature,))
                db.execute("""
                    CREATE TABLE queries (
                        query_id TEXT PRIMARY KEY, rankings TEXT NOT NULL,
                        latency_ms REAL NOT NULL, batch_size INTEGER NOT NULL,
                        batch_429_retries INTEGER NOT NULL
                    )
                """)
        if db.execute("SELECT signature FROM run").fetchone() != (signature,):
            raise ValueError("Checkpoint doesn't match the current queries or search configuration")
        done = {row[0] for row in db.execute("SELECT query_id FROM queries")}
        size = searcher.batch_size
        for start in range(0, len(requests), size):
            batch = requests[start:start + size]
            if all(request.query_id in done for request in batch):
                continue
            if any(request.query_id in done for request in batch):
                raise ValueError("Checkpoint contains an incomplete batch")
            rows, latency_ms, retries = searcher.measure_batch(batch, top_k)
            with db:
                db.executemany(
                    "INSERT INTO queries VALUES (?, ?, ?, ?, ?)",
                    [
                        (request.query_id, json.dumps(rank(request, hits), allow_nan=False),
                         latency_ms, len(batch), retries)
                        for request, hits in zip(batch, rows, strict=True)
                    ],
                )
        _export_rankings(db, output, requests)


def _export_rankings(
    db: sqlite3.Connection, output: Path, requests: list[SearchRequest]
) -> None:
    saved = dict(db.execute("SELECT query_id, rankings FROM queries"))
    if saved.keys() != {request.query_id for request in requests}:
        raise ValueError("Benchmark rankings must exactly match the input query IDs")
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=output.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            writer = csv.writer(handle)
            writer.writerow(["query_id", "doc_id", "rank", "score"])
            for request in requests:
                writer.writerows(json.loads(saved[request.query_id]))
            handle.close()
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
