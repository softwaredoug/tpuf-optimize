# ESCI lexical intent routing

## Evaluation

Compared the unchanged `c46c935` lexical strategy against six query-dependent
lexical routing hypotheses on the assigned 1,000-query training split. Each
variant used a fresh Referee task; the task query IDs, text, and locales were
verified identical to the baseline task before evaluation. Retrieval remained
lexical-only and used the existing index. No query-level results or query
content are retained here.

The confidence intervals are paired, percentile-bootstrap 95% intervals over
per-query NDCG deltas (20,000 resamples). Referee's aggregate NDCG is the
arithmetic mean over the split.

| Candidate | NDCG@10 | Delta vs. baseline | Paired 95% CI |
| --- | ---: | ---: | ---: |
| c46 baseline: passage + title + fuzzy | 0.367401 | — | — |
| Attribute fields on specification-like queries; description field on long queries | 0.359254 | -0.008147 | [-0.012148, -0.004218] |
| Attribute fields on specification-like queries | 0.365168 | -0.002233 | [-0.005202, +0.000748] |
| Whitespace title/brand fields on identifier-like queries | 0.366248 | -0.001153 | [-0.002327, -0.000102] |
| Brand BM25 on one-token queries | 0.366703 | -0.000698 | [-0.001694, +0.000215] |
| Suppress fuzzy clauses for digit-bearing queries | 0.367076 | -0.000325 | [-0.000875, -0.000011] |
| Locale-specific color-field boost | 0.366977 | -0.000424 | [-0.001274, +0.000426] |

## Decision

None of the tested routes improved aggregate NDCG@10 over the c46 baseline.
The fixed candidate is therefore the unchanged baseline strategy; the
experimental routing changes were discarded rather than committing a
regression. No reindexing was performed.
