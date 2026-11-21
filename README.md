# Turbopuffer experiments

Utilities for indexing datasets into Turbopuffer and experimenting with search
parameters.

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

## Computed-feature candidate experiment

`collect-computed-candidates` is a separate, opt-in collection mode based on the
pinned training collector. For each training query it issues exactly two
top-100 requests: the existing `passage__embed__voyage_4` ANN query and the
existing weighted locale-specific title/description BM25 query, both filtered
by locale. Each arm adds `title_bm25`, `passage_bm25`, and `brand_bm25` through
`compute_attributes`; these response-only expressions do not affect matching
or ranking. Binary full-query title coverage is computed from the returned
title, because the compute-attributes expression set does not provide a
conditional/boolean token-coverage operator. No second embedding, vector
distance, or reranking model is requested. Missing/unsupported BM25 values are
recorded as `null`, while numeric zero remains `0.0`; the offline ranker adds
availability indicators so missing values are not silently treated as zero.

The paired smoke test verified that an ID-filtered request on the pinned cached
candidate union returns exactly the same IDs and leaves the cached dense/lexical
ranks and raw scores untouched. Prefer this lower-cost mode when that cache is
available: one serial request per query, `rank_by=("id", "asc")`, filtered by
locale and the cached candidate IDs. Its `compute_attributes` are the same
BM25 features as the paired-arm path; coverage is derived from the returned
title. SDK retries/telemetry are disabled; explicit bounded 429 backoff and a
per-query checkpoint are used. For 1,000 queries this costs 1,000 successful
feature requests (each returning at most 200 candidate rows), with no Voyage
embedding calls. The resumable command is:

```sh
uv run collect-computed-candidates \
  --source-candidates /path/to/esci_research_training_candidates.jsonl
```

Without `--source-candidates`, the command uses the original two-arm collector
and costs two top-100 requests per query.

With an enriched cache and matching training grades, compare the frozen
`adf9768` model, a same-fold base-feature refit, and an enriched-feature refit:

```sh
uv run rank-computed-features \
  --candidates /path/to/computed_candidates.jsonl \
  --grades /path/to/training_grades.jsonl \
  --output-dir /path/to/ablation
```

All three rankers use the same deterministic query-disjoint train/validation
fold and identical candidate lists. The experiment writes validation CSVs,
weight artifacts, and an aggregate report; labels never enter feature
extraction or ranking.

The training-only 750/250 split (seed 17) evaluated all 250 validation queries
on identical cached candidate lists. NDCG@10 was `0.422458` for both frozen
`adf9768` and the same-fold base-feature refit, and `0.428803` for the
computed-feature refit (2:1 RRF: `0.395673`). The computed-feature weights in
`models/esci_pairwise_ranker_computed_features_training.json` are nominated
for independent holdout evaluation only; this is not a holdout acceptance.

To obtain labels only for the assigned training split, the debug collector
submits candidate IDs to Referee's read-only `/debug` endpoint in serial
10-query batches, using the configured training-only task and bearer token.
It checkpoints each batch and writes grade/target JSONL under the approved
temporary output prefix; it never completes a task:

```sh
uv run collect-training-judgments
uv run rank-computed-features \
  --candidates /path/to/esci_research_training_idfiltered_candidates.jsonl \
  --grades /path/to/esci_research_training_idfiltered_grades.jsonl \
  --targets /path/to/esci_research_training_idfiltered_targets.jsonl \
  --output-dir /path/to/ablation
```
