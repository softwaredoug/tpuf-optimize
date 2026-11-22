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

## Second training-only iteration: product-attribute BM25

This iteration tested whether lexical matches in existing bullet-point, color,
and category attributes could add useful candidates absent from the current
four-channel union. It preserved the `c46c935` channel and all first-round
channels, candidate limits, and fusion settings. Only the new candidate arm
changed: one arm used `Sum` across the three attribute BM25 fields, and a
follow-up used `Max` to avoid accumulating weak matches from multiple fields.
This was a single hypothesis with a narrow remediation, not a tokenizer,
channel, RRF, or candidate-depth sweep.

The fresh baseline and candidate Referee tasks each contained the same 1,000
training queries; query IDs, text, and locales were checked for exact parity.
Queries were partitioned before evaluation using a deterministic SHA-256
ordering into an 800-query development fold and a disjoint 200-query validation
fold. Metrics and confidence intervals below are paired on the same queries;
the percentile bootstrap used 20,000 resamples per fold. Query-level content,
scores, IDs, and results remain temporary.

| Attribute-arm variant | Fold | Baseline NDCG@10 | Candidate NDCG@10 | Paired delta | 95% CI |
| --- | --- | ---: | ---: | ---: | ---: |
| Sum | Development (n=800) | 0.375325 | 0.356906 | -0.018420 | [-0.025687, -0.011308] |
| Sum | Validation (n=200) | 0.407366 | 0.377611 | -0.029756 | [-0.044323, -0.015796] |
| Sum | All training (n=1,000) | 0.381733 | 0.361047 | -0.020687 | [-0.027233, -0.014344] |
| Max | Development (n=800) | 0.375325 | 0.355328 | -0.019997 | [-0.027787, -0.012460] |
| Max | Validation (n=200) | 0.407366 | 0.381178 | -0.026188 | [-0.041266, -0.011701] |
| Max | All training (n=1,000) | 0.381733 | 0.360498 | -0.021235 | [-0.028186, -0.014563] |

Both candidates lost on the development and validation folds, so the new arm
is not part of the default retrieval path. The summed arm had 1,740 zero-result
channel executions out of 5,000 and a mean union of 92.69 documents/query; the
max arm had the same zero-result count and a mean union of 91.06. The added
attribute channel increased candidate volume without improving relevance.

No holdout or final benchmark data was accessed in this iteration.
