# Lexical field-fusion training comparison

Matched Referee training-split comparison on 1,000 ESCI queries. Each run used
the same query IDs, query text, and locales, with NDCG@10. Deltas and 95% CIs
are paired query-level bootstrap estimates (20,000 resamples) versus the
unchanged `c46c935` lexical baseline.

| Added lexical channel | NDCG@10 | Delta vs. baseline | Paired 95% CI |
| --- | ---: | ---: | ---: |
| Baseline (`none`) | 0.367785 | — | — |
| Brand BM25 | 0.366967 | -0.000818 | [-0.003232, +0.001703] |
| Description BM25, b=0.25 | 0.352002 | -0.015783 | [-0.021981, -0.009564] |
| Title word_v2 BM25 | 0.368759 | +0.000974 | [-0.001935, +0.003839] |
| Passage word_v2 BM25 | 0.370241 | +0.002456 | [-0.000677, +0.005716] |
| Passage BM25, b=0.25 | 0.371449 | +0.003664 | [-0.000004, +0.007345] |
| **Passage ASCII-folded BM25** | **0.371560** | **+0.003776** | **[+0.000723, +0.007043]** |
| Brand + description BM25 | 0.353208 | -0.014576 | [-0.020804, -0.008344] |
| Passage b=0.25 + word_v2 BM25 | 0.368736 | +0.000951 | [-0.004168, +0.006132] |

The selected fixed candidate adds `passage__word_v4__fold` BM25 at weight
`0.5` to the original lexical ranking. It does not change indexing, use
embeddings, or use a reranker.

## Holdout confirmation

The orchestrator reports matched holdout NDCG@10 of `0.364953` for baseline
and `0.367718` for this candidate, a delta of `+0.002765`. The candidate was
accepted. Only aggregate metrics are recorded.
