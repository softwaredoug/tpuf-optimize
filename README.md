# Turbopuffer experiments

Utilities for indexing datasets into Turbopuffer and experimenting with search
parameters.

## Branch experiment: lexical candidate union + position reranking

This branch keeps the original lexical BM25 channel and adds selective
all-token/phrase retrieval, then uses a local reranker over returned text. It
uses no embeddings, training labels, or third-party reranker. The opt-in
one-edit fuzzy-position adjustment is set with
`--lexical-fuzzy-position-boost 0.15`.

For query `q` and locale analyzer suffix `L`, the broad query's Turbopuffer DSL
is:

```python
base = {
    "rank_by": ("Sum", (
        (f"passage__{L}", "BM25", q),
        ("Product", 2.0, (f"title__{L}", "BM25", q)),
        ("Product", 0.5, ("passage__word_v4__fold", "BM25", q)),
        ("Product", 0.5, ("brand__whitespace__default", "BM25", q)),
        # For each non-Japanese token >= 6 characters:
        ("Product", 0.35, (
            "title__word_v4__fuzzy", "Fuzzy", token,
            {"max_edit_distance": [{"min_query_chars": 6, "distance": 1}],
             "case_sensitive": False},
        )),
        ("Product", 0.15, (
            "brand__word_v4__fuzzy", "Fuzzy", token,
            {"max_edit_distance": [{"min_query_chars": 6, "distance": 1}],
             "case_sensitive": False},
        )),
    )),
    "filters": ("locale", "Eq", locale),
    "top_k": 50,
    "include_attributes": ["title", "description", "brand", "locale"],
}
```

Three locale-filtered side queries retrieve title `ContainsAllTokens`, passage
`ContainsTokenSequence`, and brand `ContainsTokenSequence` matches (also top 50).
Merge all four lists with `RRF(doc) += 1 / (20 + rank)`, then cap the union at 50.
At query time, rerank using:

```text
adjusted =
    1 - fused_rank / max(candidate_count - 1, 1)
    + 0.18 * RMS(title/description/brand position quality)
    + 0.03 * cross-field term coverage
    + optional 0.15 * one-edit title/brand match coverage
```

Position quality combines term coverage, query order, compactness, early
occurrence, and exact phrase evidence. The optional fuzzy adjustment only
considers tokens of at least six characters and is disabled for Japanese.
Reproduction command, matched training/holdout results, and the fresh timed
5,000-query benchmark are documented in
[`experiments/lexical-union-position.md`](experiments/lexical-union-position.md).

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
