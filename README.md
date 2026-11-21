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
Turbopuffer's multi-query API. By default, it fuses Voyage 4 passage ANN and
lexical BM25 with weighted reciprocal-rank fusion; the lexical ranker combines
title and description scores, with title weighted twice as heavily. Queries
with no hits are emitted with the `-1,-1,-1` sentinel row.

Hybrid search retrieves up to 100 candidates from each retriever.
The default fusion weights are 2:1 (dense:lexical) with an RRF constant of 10;
`--dense-weight`, `--lexical-weight`, and `--rrf-constant` can override them.
Use `--lexical-only` to disable hybrid search.

```sh
uv run query --queries queries.csv --output results.csv \
  --dense-weight 2 --lexical-weight 1
```

### Counterfactual token-probe arm

For a paired experiment, pass `--counterfactual-control control.csv`. The usual
`--output` is then the treatment: it retains only the original top-k candidates
and may softly reorder candidates that lack one selected identifier or
partially-covered attribute token. `control.csv` contains the same candidates
in their original hybrid/fused order, without using probe evidence.

Eligible queries issue one additional logical query. It filters to the original
candidate IDs, orders by ID so zero-score candidates remain present, and uses
`compute_attributes` to evaluate both the original and token-ablated weighted
title/description BM25 on those same candidates. Thus the rank-lift signal
compares true lexical leave-one-out scores, rather than a separately retrieved
list or the fused score. For default hybrid search, the baseline uses two
logical query subrequests (dense and lexical); the counterfactual adds at most
one lexical subrequest, for a maximum of three logical query subrequests per
eligible query. Ineligible queries use only the baseline pair. The probe adds
no embedding query.

## Search architecture

`TPufStrategy` owns the ESCI document mapping, namespace schema, and query
construction. The outer `Indexer` and `Searcher` handle batched namespace I/O,
so relevance experiments can focus on the strategy without changing the runners.
