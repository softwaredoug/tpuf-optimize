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

## Offline learned-rank experiment

`learned-rank` trains and evaluates an offline, query-grouped pairwise linear
reranker from cached retrieval candidates. It makes no Turbopuffer or Referee
calls. Candidate JSONL rows contain `query_id`, `query_text`, `locale`, and
rank-ordered `dense` and `lexical` arrays of documents with `doc_id`, `score`,
and `title`. Grade JSONL rows contain `query_id` and `grades`, either a
document-ID-to-grade object or a list of `{ "doc_id": ..., "grade": ... }`
objects. Example:

```sh
uv run learned-rank \
  --candidates esci_research_training_candidates.jsonl \
  --grades training_grades.jsonl \
  --output validation_results.csv \
  --model-output learned_rank_model.json
```

The default seeded split assigns whole query groups to training or validation.
Features use retrieval ranks/scores and query/title token coverage only; query
and document IDs are join/tie-break keys, not model features. Pairwise fitting
uses unequal positive-grade pairs, so grade zero is never treated as a negative.
The command compares validation NDCG for its reranker and the deterministic
dense:lexical 2:1 reciprocal-rank-fusion baseline on identical candidate sets.
It also reports a candidate-set oracle on training queries only as a diagnostic;
the oracle is not used to produce rankings. Unjudged/zero-grade documents do
not contribute gain in the NDCG calculation.

## Search architecture

`TPufStrategy` owns the ESCI document mapping, namespace schema, and query
construction. The outer `Indexer` and `Searcher` handle batched namespace I/O,
so relevance experiments can focus on the strategy without changing the runners.
