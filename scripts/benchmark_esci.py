"""Run the frozen ESCI benchmark serially, keeping Referee data private."""

import argparse
import csv
import hashlib
import json
import math
import os
import secrets
import sqlite3
import statistics
import subprocess
import sys
import urllib.request
from contextlib import closing
from pathlib import Path

from turbopuffer import Turbopuffer

from esci_benchmark_variants import VARIANTS, FrozenSearcher
from tpuf_optimize.benchmark import run_benchmark
from tpuf_optimize.query_batch import _ranked_rows
from tpuf_optimize.searcher import SearchRequest

COUNT = 5000
ALIASES = {"en-us": "us", "es-es": "es", "ja-jp": "jp"}


def save(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def post(path, token=None, payload=None):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    data = None
    if payload is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload, allow_nan=False).encode()
    request = urllib.request.Request(
        "http://localhost:8080" + path, data=data, headers=headers, method="POST",
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)


def key(text, locale):
    normalized = locale.strip().lower()
    return (" ".join(text.casefold().split()), ALIASES.get(normalized, normalized))


def signature(task):
    return {
        q["query_id"]: (q["query_text"], q["metadata"]["locale"]) for q in task["queries"]
    }


def seen_queries(root):
    paths = {*root.glob("*task*.json"), *root.glob("private-esci-eval-*/tasks.json")}
    seen = set()
    matched = 0
    for path in sorted(paths):
        data = json.loads(path.read_text(encoding="utf-8"))
        tasks = data if isinstance(data, list) else [data]
        covered = False
        for task in tasks:
            if not isinstance(task, dict):
                continue
            for row in task.get("queries", []):
                if not isinstance(row, dict):
                    continue
                text = row.get("query_text")
                metadata = row.get("metadata") or {}
                locale = metadata.get("locale") or row.get("locale")
                if isinstance(text, str) and isinstance(locale, str):
                    seen.add(key(text, locale))
                    covered = True
        matched += int(covered)
    return seen, matched


def metrics(db_path, evaluation, expected, outside_ids):
    with closing(sqlite3.connect(db_path)) as db:
        entries = db.execute(
            "SELECT query_id, latency_ms, batch_429_retries FROM queries"
        ).fetchall()
    scores = {q["query_id"]: float(q["ndcg"]) for q in evaluation["queries"]}
    if len(evaluation["queries"]) != COUNT or scores.keys() != expected.keys():
        raise ValueError("Referee completion did not cover the exact benchmark query IDs")
    if len(entries) != COUNT or {q[0] for q in entries} != expected.keys():
        raise ValueError("Timing checkpoint did not cover the exact benchmark query IDs")
    latency = sorted(entry[1] for entry in entries)
    ndcg = float(evaluation["metrics"]["ndcg"])
    if not math.isclose(statistics.mean(scores.values()), ndcg, abs_tol=1e-10):
        raise ValueError("Referee aggregate disagrees with per-query NDCG")
    return {
        "ndcg_at_10": ndcg,
        "disjoint_ndcg_at_10": statistics.mean(scores[q] for q in outside_ids),
        "p50_ms": statistics.median(latency),
        "p95_ms": latency[math.ceil(.95 * COUNT) - 1],
        "retries_429": sum(entry[2] for entry in entries),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True, help="Private run directory")
    parser.add_argument("--publish", type=Path, help="Destination for aggregate-only JSON")
    parser.add_argument(
        "--render", action="store_true", help="Refresh the README and plots on publish",
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.render and args.publish is None:
        parser.error("--render requires --publish")
    os.umask(0o077)
    state = args.state.resolve()
    if state.is_relative_to(Path.cwd()):
        raise ValueError("Private benchmark state must be outside the repository")
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    if state.stat().st_mode & 0o077:
        raise ValueError("Private benchmark directory must not be accessible to other users")
    # A single orchestrator process per run, even if launched twice.
    import fcntl

    lock = (state / "run.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(args, state)
    finally:
        lock.close()


def run(args, state):
    token_path = state / "session.json"
    if not token_path.exists():
        seed = secrets.randbits(63)
        session = post("/v1/sessions", payload={
            "dataset_name": "esci", "eval_mode": "ndcg", "splits": [1.0],
            "num_queries": COUNT, "seed": seed,
        })
        save(token_path, {"seed": seed, "token": session["split_tokens"][0]})
    session = json.loads(token_path.read_text(encoding="utf-8"))
    token = session["token"]
    reference = state / "reference-task.json"
    if not reference.exists():
        save(reference, post("/v1/tasks", token))
    task = json.loads(reference.read_text(encoding="utf-8"))
    expected = signature(task)
    if len(expected) != COUNT or len(task["queries"]) != COUNT:
        raise ValueError("Expected 5,000 distinct Referee query IDs")
    overlap_file = state / "overlap.json"
    if not overlap_file.exists():
        known, artifacts = seen_queries(state.parent)
        outside = sorted(
            q for q, (text, locale) in expected.items() if key(text, locale) not in known
        )
        save(overlap_file, {"outside_ids": outside, "checked_artifacts": artifacts})
    overlap = json.loads(overlap_file.read_text(encoding="utf-8"))
    outside = overlap["outside_ids"]
    if not outside or not set(outside) <= expected.keys():
        raise ValueError("Invalid previously seen query audit")
    print(f"Verified {COUNT} queries; {COUNT - len(outside)} overlap "
          f"across {overlap['checked_artifacts']} historical task artifacts.", flush=True)
    if args.prepare_only:
        return
    requests = [
        SearchRequest(q, text, ALIASES.get(locale.lower(), locale))
        for q, (text, locale) in expected.items()
    ]
    # Identical order for every variant without exposing the random split seed.
    requests.sort(key=lambda request: hashlib.sha256(request.query_id.encode()).digest())
    aggregate = state / "aggregate.json"
    region = os.environ.get("TURBOPUFFER_REGION", "gcp-us-central1")
    namespace = "esci-products"
    cohort = {
        "query_count": COUNT, "disjoint_count": len(outside),
        "checked_artifacts": overlap["checked_artifacts"],
        "namespace": namespace, "region": region, "batch_size": 1,
    }
    summary = (
        json.loads(aggregate.read_text()) if aggregate.exists() else {**cohort, "variants": {}}
    )
    if any(summary[field] != value for field, value in cohort.items()):
        raise ValueError("Benchmark configuration changed")
    revisions = {
        v.slug: subprocess.check_output(
            ["git", "rev-parse", f"{v.revision}^{{commit}}"], text=True,
        ).strip()
        for v in VARIANTS
    }
    for slug, result in summary["variants"].items():
        if result["commit"] != revisions[slug]:
            raise ValueError("A completed variant's frozen revision changed")
    for variant in VARIANTS:
        if variant.slug in summary["variants"]:
            continue
        run_variant(state, token, expected, outside, requests, region, namespace,
                    variant, revisions[variant.slug], summary)
        save(aggregate, summary)
        if args.publish:
            # Public artifact contains no query IDs, query text, tokens, or split seed.
            save(args.publish, summary)
            if args.render:
                subprocess.run([
                    sys.executable, str(Path(__file__).with_name("render_esci_benchmark.py")),
                    "--aggregate", str(args.publish),
                ], check=True)


def run_variant(
    state, token, expected, outside, requests, region, namespace, variant, revision, summary,
):
    slug = variant.slug
    print(f"Starting {slug} at {revision[:7]} (serial).", flush=True)
    db = state / f"{slug}.sqlite"
    output = state / f"{slug}-rankings.csv"
    evaluated = state / f"{slug}-evaluation.json"
    if not evaluated.exists():
        with Turbopuffer(region=region, max_retries=0) as client:
            runner = FrozenSearcher(client.namespace(namespace), variant)

            def progress(n):
                if n % 100 == 0:
                    print(f"{slug}: {n}/{COUNT} queries checkpointed.", flush=True)

            run_benchmark(db, output, requests, runner, 10, (region, namespace, revision),
                          db.exists(), _ranked_rows, progress=progress)
        with output.open(newline="") as handle:
            rankings = [
                {**row, "rank": int(row["rank"]), "score": float(row["score"])}
                for row in csv.DictReader(handle)
            ]
        task = post("/v1/tasks", token)
        if len(task["queries"]) != COUNT or signature(task) != expected:
            raise ValueError("Variant received different benchmark queries")
        evaluation = post(
            f"/v1/tasks/{task['task']['id']}/complete", token, {"results": rankings},
        )
        save(evaluated, evaluation)
    result = metrics(db, json.loads(evaluated.read_text()), expected, outside)
    summary["variants"][slug] = {"commit": revision, **result}
    print(json.dumps({"variant": slug, **result}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        status = getattr(error, "status_code", getattr(error, "code", None))
        print(f"Benchmark stopped: {type(error).__name__}; HTTP status={status}. "
              "Private checkpoint retained.", file=sys.stderr)
        raise SystemExit(1) from None
