# Turbopuffer experiments

Utilities for indexing datasets into Turbopuffer and experimenting with search
parameters.

## Optimization workflow

This repository is a harness for controlled retrieval experiments. The `main`
branch keeps the simple title/description BM25 baseline; experiment branches
contain alternate query strategies and their evaluation notes. Keep training
and holdout data separate, and use a fresh matched Referee session for final
benchmarks. The shared `Indexer`, `Searcher`, and batch runners provide the
common data and I/O layer.

## Setup

This project uses Python 3.15.2, managed by asdf, and `uv` for the environment
and dependencies:

```sh
uv sync
uv run pre-commit install
```

The `tpuf-helpers` and `cheat-at-search` dependencies are pinned to specific Git
revisions in `pyproject.toml`.

## Index the ESCI corpus

Set `TURBOPUFFER_API_KEY`, then run:

```sh
uv run index-esci --namespace esci-products --region gcp-us-central1
```

The command uses `cheat_at_search.esci_data.corpus` and writes raw fields,
analyzer-specific field copies, a `passage` field, and one Voyage 4 embedding
of that passage.
Searchable names use `logical_field__tokenizer__language__options`, such as
`title__word_v4__es__stem_stop_fold` and `title__kuromoji__ja`. It includes
`word_v4`, `word_v2`, whitespace, pre-tokenized, fuzzy, and BM25 length
normalization variants. Native embedding writes are currently limited to 256
documents per batch, which is also the indexing default.

Use `--limit N` to index a subset or `--dry-run` to count documents without
writing. `--drop-existing` explicitly deletes the target namespace before
indexing, so use it with care. `--skip-existing-docs` looks up IDs once per
batch and skips rows already in the namespace; use it to resume with the same
schema without re-embedding completed documents. These options are mutually
exclusive. The region defaults to `TURBOPUFFER_REGION` or `gcp-us-central1`.

## Query the ESCI corpus

Provide a CSV with `query` and `locale` columns, then run:

```sh
uv run query --queries queries.csv --output results.csv
```

The batch command assigns query IDs (`q1`, `q2`, ...), applies the locale filter,
and writes ranked document IDs and scores. Requests are sent in batches through
Turbopuffer's multi-query API. Queries combine BM25 title and description scores,
with title weighted twice as heavily. Queries with no hits are emitted with the
`-1,-1,-1` sentinel row.

## Search architecture

`TPufStrategy` owns the ESCI document mapping, namespace schema, and query
construction. The outer `Indexer` and `Searcher` handle batched namespace I/O,
so relevance experiments can focus on the strategy without changing the runners.

## Experiment: one-request lexical reranking

`experiment/lexical-one-query-rerank` is based on `main` and does not change the
index schema. `SingleRequestLexicalStrategy` issues the cdb2d64-style locale
filtered query: title BM25 at weight 1.0 plus passage BM25 at weight 0.5, with
`top_k=100` and the existing title, description, brand, and locale attributes
returned. `search_one` makes one serial `namespace.query` call; reranking only
uses those results.

The reranker combines the original candidate rank with exact local evidence:

```text
local =
    0.45 * unique-query-term coverage in title
  + 0.20 * adjacent query-pair coverage in title
  + 0.15 * exact title phrase
  + 0.10 * unique-query-term coverage in brand
  + 0.10 * unique-query-term coverage in description
score = (1 - boost) * normalized_original_rank + boost * local
```

Boost `0.35` was selected from `{0.05, 0.10, 0.20, 0.35}` on a deterministic,
query-disjoint 105-query validation fold of the assigned Referee training
split; it was positive on the remaining 395 queries as well. There is no query
fan-out, candidate union, embedding, second attribute fetch, or reindex.

### Training-split result

| Variant | Referee queries | NDCG@10 | p50 latency | p95 latency | Requests/query |
|---|---:|---:|---:|---:|---:|
| cdb2d64 one-request BM25 | 500 | 0.36803 | 68.67 ms | 81.25 ms | 1 |
| BM25 + local lexical rerank (boost 0.35) | 500 | 0.38922 | 78.77 ms | 99.47 ms | 1 |

The paired NDCG@10 change is **+0.02119** (95% paired bootstrap interval
**[+0.01392, +0.02929]**); on the disjoint 395-query selection-train fold the
change was **+0.02035**. The 105-query selection-validation-fold change was
**+0.02433**. Across all queries, 207 improved, 171 were unchanged, and 122
declined. Latencies are unbatched successful network time; candidate latency
also includes local reranking. They were measured in a separate fresh serial
pass from the NDCG submissions. The returned title/description/brand values
were non-empty for 100.0% / 49.3% / 93.8% of top-100 candidates, respectively.
On the orchestrator-evaluated private 500-query holdout, baseline/reranker
NDCG@10 was 0.3693039 / 0.3754472 (**+0.0061433**). Baseline/reranker p50/p95
latency was 61.97/76.62 ms and 82.62/101.67 ms, respectively. Both used one
serial TPuf request per query; the holdout run had zero 429 retries.

## ESCI 5,000-query leaderboard

Every variant in the table and Pareto charts uses one Referee session created
with seed `42`, `num_queries=5000`, and `splits=[1.0]`. The benchmark runner
verifies each variant receives the same query IDs, text, and locale.

To measure a single checkout without a client-side result or feature cache,
use an explicit `query_id,query,locale` CSV and keep the inputs and outputs
outside the repository in a private directory:

```sh
uv run query --queries "$RUN/queries.csv" --output "$RUN/rankings.csv" \
  --benchmark-db "$RUN/timings.sqlite"
```

Benchmark mode defaults to one query per batch. It records rankings and
latency together, counting only the successful attempt after any HTTP 429s.
For larger `--batch-size` values, `latency_ms` in the checkpoint's `queries`
table is successful batch time divided by the number of user queries. This
is amortized time, not individual response latency. `--resume` continues
the same run after interruption; use a fresh database for every new run or
variant. Turbopuffer's internal caching is outside this tool's control.

The benchmark runner measures each pinned variant in
`scripts/esci_benchmark_variants.py` serially against that same seed-42 query
pool. Each run produces both NDCG@10 and per-query latency; rankings and latency
come from the same uncached client execution.
The Pareto charts use full-split NDCG; pending variants are not plotted. The
cost column estimates external embedding/reranker inference per 5,000-query run;
Turbopuffer database charges are excluded.

<!-- esci-benchmark:start -->

All variants used the same 5,000 Referee queries with seed 42. The second NDCG column covers the 3,588 queries not found in 62 available historical task artifacts.

Classes: lexical; hybrid (lexical + embedding, including rule-based reranking on both); LTR (Learning to Rank), a classic ML reranker using query-dependent, query, and document features; third-party reranker. Pure embedding retrieval is listed separately.
| Variant | Class | NDCG@10 (full) | NDCG@10 (disjoint) | p50 (ms) | p95 (ms) | Cost* (USD/5,000 queries) | Commit |
|---|---|---:|---:|---:|---:|---:|---|
| [Voyage rerank-3 over ANN + BM25](https://github.com/softwaredoug/tpuf-optimize/tree/experiment/tpuf-native-rerankers) | Third-party reranker | 0.44762 | 0.44785 | 243.14 | 368.59 | $6.346635* | [`0a5816e`](https://github.com/softwaredoug/tpuf-optimize/commit/0a5816e) |
| [Typed + computed reranker](https://github.com/softwaredoug/tpuf-optimize/tree/innovate-typed-computed-ltr) | LTR | 0.41377 | 0.41299 | 200.09 | 253.67 | $0.001635* | [`6508e87`](https://github.com/softwaredoug/tpuf-optimize/commit/6508e87) |
| [Computed-feature reranker](https://github.com/softwaredoug/tpuf-optimize/tree/innovate-computed-ltr-features) | LTR | 0.40830 | 0.40805 | 194.06 | 266.37 | $0.001635* | [`e64100d`](https://github.com/softwaredoug/tpuf-optimize/commit/e64100d) |
| [Learned pairwise reranker](https://github.com/softwaredoug/tpuf-optimize/tree/research/learned-rank) | LTR | 0.40152 | 0.40097 | 113.79 | 155.29 | $0.001635* | [`adf9768`](https://github.com/softwaredoug/tpuf-optimize/commit/adf9768) |
| [Identifier-gated RRF](https://github.com/softwaredoug/tpuf-optimize/tree/confidence-aware-fusion) | Hybrid | 0.39563 | 0.39469 | 113.99 | 164.62 | $0.001635* | [`7e22661`](https://github.com/softwaredoug/tpuf-optimize/commit/7e22661) |
| [Deterministic product reranker](https://github.com/softwaredoug/tpuf-optimize/tree/research-product-rerank) | Hybrid | 0.38879 | 0.38823 | 154.08 | 230.99 | $0.001635* | [`7ffd772`](https://github.com/softwaredoug/tpuf-optimize/commit/7ffd772) |
| [RMS stack + fuzzy-position boost 0.15](https://github.com/softwaredoug/tpuf-optimize/tree/experiment/lexical-union-position) | Lexical | 0.38784 | 0.38788 | 94.42 | 172.13 | $0.000000* | [`d663a92`](https://github.com/softwaredoug/tpuf-optimize/commit/d663a92) |
| [RMS lexical candidate stack](https://github.com/softwaredoug/tpuf-optimize/tree/experiment/lexical-union-position) | Lexical | 0.38748 | 0.38729 | 87.13 | 133.64 | $0.000000* | [`038e845`](https://github.com/softwaredoug/tpuf-optimize/commit/038e845) |
| [Min-max score fusion](https://github.com/softwaredoug/tpuf-optimize/tree/research-score-fusion) | Hybrid | 0.38567 | 0.38327 | 111.42 | 164.56 | $0.001635* | [`41e64d4`](https://github.com/softwaredoug/tpuf-optimize/commit/41e64d4) |
| [2:1 Voyage-4 + BM25 RRF](https://github.com/softwaredoug/tpuf-optimize/tree/experiment-6-hybrid-rrf) | Hybrid | 0.38125 | 0.38006 | 122.66 | 199.40 | $0.001635* | [`cc5fc8a`](https://github.com/softwaredoug/tpuf-optimize/commit/cc5fc8a) |
| [Passage BM25 + fuzzy title/brand](https://github.com/softwaredoug/tpuf-optimize/tree/exp3-bm25-passage-fuzzy) | Lexical | 0.36754 | 0.36931 | 68.83 | 98.71 | $0.000000* | [`c46c935`](https://github.com/softwaredoug/tpuf-optimize/commit/c46c935) |
| [Title + locale-aware passage BM25](https://github.com/softwaredoug/tpuf-optimize/tree/relevance/field-weights-passage) | Lexical | 0.36627 | 0.36821 | 66.46 | 98.16 | $0.000000* | [`cdb2d64`](https://github.com/softwaredoug/tpuf-optimize/commit/cdb2d64) |
| [Four-channel RRF](https://github.com/softwaredoug/tpuf-optimize/tree/exp7-rank-fusion) | Hybrid | 0.36616 | 0.36442 | 125.48 | 227.12 | $0.001635* | [`3f4c9bc`](https://github.com/softwaredoug/tpuf-optimize/commit/3f4c9bc) |
| [Baseline: Voyage-4 ANN (Turbopuffer)](https://github.com/softwaredoug/tpuf-optimize/tree/experiment-6-hybrid-rrf) | Embedding | 0.36437 | 0.36312 | 103.21 | 160.41 | $0.001635* | [`cc5fc8a`](https://github.com/softwaredoug/tpuf-optimize/commit/cc5fc8a) |
| [Baseline: weighted title/description BM25](https://github.com/softwaredoug/tpuf-optimize/tree/main) | Lexical | 0.30372 | 0.30203 | 63.77 | 77.25 | $0.000000* | [`1768df7`](https://github.com/softwaredoug/tpuf-optimize/commit/1768df7) |

*Estimated external model inference per 5,000-query run. Voyage-4 query embedding uses a UTF-8-bytes/4 token proxy (0.06 USD/million tokens); lexical-only and local CPU rankers add no model inference charge. Excludes Turbopuffer database/query charges and index-time embedding. Voyage rerank-3 uses $0.05/M and a 25,380-token-per-query training proxy.

- **Lexical** — BM25 and other lexical matching only.
- **Hybrid** — combines lexical + embedding retrieval/fusion, including rule-based reranking over those candidates.
- **LTR** — classic ML reranker using query-dependent, query, and document features.
- **Third-party reranker** — hosted reranking model such as Voyage rerank-3.
- **Embedding** — pure embedding retrieval; listed separately from hybrid.

![ESCI NDCG@10 versus p50 latency](assets/esci-pareto-p50.svg)

![ESCI NDCG@10 versus p95 latency](assets/esci-pareto-p95.svg)

![ESCI NDCG@10 versus estimated model cost per 5,000 queries](assets/esci-pareto-cost.svg)

<!-- esci-benchmark:end -->

Refresh the table and all three Pareto charts from the aggregate-only JSON with
`uv run python scripts/render_esci_benchmark.py`.

For context, the separate 1,000-query training-only native reranker experiment
estimated 25.38M rerank-input tokens by a chars/4 proxy. At $0.05/M, linear
scaling to 5,000 queries estimates `$6.35` for Voyage rerank-3; its leaderboard
cost cell also includes the small Voyage-4 query-embedding estimate. Cohere's
training-only estimate was `$12.69` per 5,000 queries. The rerank API does not
return model token usage or per-request cost.
