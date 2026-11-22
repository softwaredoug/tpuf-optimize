# Lexical candidate union + position reranking

## Strategy

Keep the c46 passage/title/fuzzy lexical channel and add three filter-only
candidate channels: all-token title, exact-phrase passage, and exact-phrase
brand. Fuse their top-50 result lists with reciprocal-rank fusion (constant
20), cap the union at 50 candidates, then apply the deterministic lexical
position reranker with a maximum boost of `0.18`. The reranker uses only
returned title, description, and brand text. The final result is truncated to
10. This uses existing indexed data and no embedding or external model.

## Matched Referee results

| Variant | Training NDCG@10 | Holdout NDCG@10 |
| --- | ---: | ---: |
| c46 baseline | 0.367528 | 0.365288 |
| Candidate union + RRF | 0.381733 | 0.378523 |
| Candidate union + RRF + position reranker | 0.382969 | 0.380279 |
| Candidate union + position + passage-fold BM25 (0.5) | 0.386081 | 0.382376 |

The combined candidate's training delta over baseline was `+0.015441`; paired
bootstrap 95% CI: `[+0.010990, +0.020080]`. It added `+0.001236` over union
alone on training. On matched holdout tasks it improved `+0.014991` over
baseline and `+0.001757` over union alone. The private holdout result is
aggregate-only; no query text, query IDs, result rows, or per-query judgments
are stored here.

The passage-fold arm added `+0.003112` over union+position on training
(paired bootstrap 95% CI: `[+0.000926, +0.005379]`) and `+0.002097` on matched
holdout tasks. The final stacked candidate improved `+0.018680` over the
matched training baseline and `+0.017423` over the matched holdout baseline.

No latency benchmark was performed. Phrase filters may add query latency.
