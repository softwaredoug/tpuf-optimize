# Lexical candidate union: assigned training split

## Protocol

- Dataset: ESCI, 1,000 queries from the assigned training split.
- Compared the unchanged `c46c935` lexical leader with candidate fusion on two
  Referee tasks created from the same training token.
- Confirmed the tasks had identical query IDs, texts, and locales. Query text,
  per-query scores, document rows, task credentials, and raw CSVs remain in the
  assigned private temporary directory and are not committed.
- NDCG@10 was calculated by Referee. The confidence interval is a paired
  percentile bootstrap over query-level candidate-minus-baseline NDCG deltas
  (20,000 resamples; seed 91827).
- No holdout or final benchmark data was accessed.

## Candidate strategy and result

The original passage/title/fuzzy BM25 channel remains intact and returns 50
candidates. Three independent, filter-only lexical retrieval arms each return
up to 50 candidates: all query tokens in the locale-analyzed title, an exact
ordered phrase in the locale-analyzed passage, and an exact ordered phrase in
the indexed brand. Their candidate sets are unioned and ranked using
reciprocal-rank fusion (constant 20), then truncated to 10.

This is not an additive `ContainsAllTokens` title score boost, nor an exact
title-phrase boost. It treats high-precision matches as separate candidate
channels and uses rank fusion to preserve the broad baseline channel. In
particular, phrase retrieval is over the indexed passage and brand attributes,
not a rank-by phrase bonus.

| Variant | NDCG@10 |
| --- | ---: |
| Baseline `c46c935` | 0.367401 |
| Candidate union + RRF | 0.381733 |
| Paired delta | +0.014333 |

Paired bootstrap 95% CI for the delta: **[+0.009892, +0.018968]**. Candidate
fusion improved 269 queries, tied 565, and regressed 166 of the 1,000 training
queries.

Across the 4,000 channel executions (baseline plus three candidate arms),
1,730 returned no candidates. The mean distinct candidate union was 54.88
documents/query before final truncation. This records channel-level empty
results and union coverage without retaining query-level data.

## Turbopuffer considerations

`ContainsAllTokens` is a filter-only operator and requires a full-text indexed
field. `ContainsTokenSequence` is also a filter-only operator; Turbopuffer
documents a partial-postfilter implementation that can increase FTS/filter-only
latency. The baseline ranking channel is never filtered by these operators, so
their recall behavior cannot remove baseline candidates. The three arms are
batched within the documented 16-query multi-query API limit.
