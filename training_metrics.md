# Lexical pseudo-relevance feedback: training metrics

All metrics below use only the assigned 1,000-query training split. The
candidate uses two feedback terms from the top five lexical results, requiring
each term in at least two feedback documents, and fuses the expanded lexical
ranking at weight 0.25 (RRF rank constant 60). Results were compared against
the exact `c46c935` baseline on paired queries.

| Evaluation set | Queries | Baseline NDCG@10 | Feedback NDCG@10 | Paired delta | 95% bootstrap CI for delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full training split | 1,000 | 0.367528 | 0.368429 | +0.000901 | [-0.002248, +0.004095] |
| Settings-selection fold | 214 | — | — | +0.001702 | [-0.004288, +0.008071] |
| Disjoint confirmation fold | 786 | — | — | +0.000683 | [-0.003041, +0.004417] |

The selection fold was fixed by query-ID ordinal modulo five; the complementary
fold was not used to choose settings. Neither fold nor the full-split paired
bootstrap establishes a reliable improvement. No query text, IDs, or
per-query metrics are included here. The feedback option is therefore
experimental and not enabled by default.
