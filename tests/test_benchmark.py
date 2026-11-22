"""Offline tests for benchmark timing, retries, IDs, and checkpoints."""

import csv
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from tpuf_optimize.benchmark import run_benchmark
from tpuf_optimize.query_batch import _ranked_rows, _read_queries
from tpuf_optimize.searcher import Searcher
from tpuf_optimize.strategy import TPufStrategy

from test_searcher import FakeNamespace


class BenchmarkTests(unittest.TestCase):
    def test_retry_times_only_the_successful_attempt_and_amortizes(self):
        namespace = FakeNamespace()
        namespace.rate_limits_remaining = 1
        requests = _read_requests()
        with (
            patch.object(namespace, "multi_query") as multi_query,
            patch("tpuf_optimize.searcher.time.sleep") as sleep,
            patch("tpuf_optimize.searcher.time.perf_counter", side_effect=[1, 10, 10.25]),
        ):
            rows, elapsed, retries = Searcher(
                namespace,
                TPufStrategy(),
                serial_requests=True,
            ).measure_batch(requests[:1], top_k=10)

        self.assertEqual(len(rows), 1)
        self.assertEqual(elapsed, 250)
        self.assertEqual(retries, 1)
        sleep.assert_called_once()
        multi_query.assert_not_called()
        self.assertEqual(namespace.serial_calls, 2)

    def test_resume_reuses_only_completed_progress_of_the_same_run(self):
        requests = _read_requests()
        namespace = FakeNamespace()
        searcher = Searcher(namespace, TPufStrategy(), batch_size=1)
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "benchmark.sqlite"
            output = Path(directory) / "rankings.csv"

            def run(resume, queries=requests):
                run_benchmark(
                    db_path, output, queries, searcher, 10,
                    ("test-region", "test-namespace"), resume, _ranked_rows,
                )

            with patch.object(searcher, "measure_batch", side_effect=[
                ([[]], 1.25, 2), RuntimeError("interrupted"),
            ]):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    run(False)

            self.assertFalse(output.exists())
            with self.assertRaises(FileExistsError):
                run(False)
            with self.assertRaisesRegex(ValueError, "doesn't match"):
                run(True, list(reversed(requests)))
            run(True)
            self.assertEqual(namespace.batch_sizes, [1])
            with closing(sqlite3.connect(db_path)) as db:
                self.assertEqual(db.execute(
                    "SELECT latency_ms, batch_429_retries FROM queries WHERE query_id = 'ref-b'"
                ).fetchone(), (1.25, 2))
            with output.open(newline="") as handle:
                self.assertEqual(list(csv.reader(handle)), [
                    ["query_id", "doc_id", "rank", "score"],
                    ["ref-b", "-1", "-1", "-1"],
                    ["ref-a", "-1", "-1", "-1"],
                ])


def _read_requests():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "queries.csv"
        path.write_text("query_id,query,locale\nref-b,shoe,us\nref-a,shoe,us\n")
        return _read_queries(path)
