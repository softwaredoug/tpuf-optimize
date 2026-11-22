# Lexical candidate union + position reranking

## Strategy

Keep the c46 passage/title/fuzzy lexical channel and add three filter-only
candidate channels: all-token title, exact-phrase passage, and exact-phrase
brand. Fuse their top-50 result lists with reciprocal-rank fusion (constant
20), cap the union at 50 candidates, then apply the deterministic lexical
position reranker with a maximum boost of `0.18`. The reranker uses only
returned title, description, and brand text. An opt-in `0.15` fuzzy positional
bonus further rewards one-edit matches of long query tokens in returned title
and brand text; it is disabled for Japanese. The final result is truncated to
10. This uses existing indexed data and no embedding or external model.

## Reproduction

Run the current holdout leader against a query CSV with:

```sh
uv run query --queries queries.csv --output results.csv \
  --lexical-candidate-fusion --lexical-rerank \
  --lexical-cross-field-boost 0.03 \
  --lexical-position-aggregation rms \
  --lexical-fuzzy-position-boost 0.15 \
  --field-fusion passage_fold_brand_whitespace
```

## Matched Referee results

| Variant | Training NDCG@10 | Holdout NDCG@10 |
| --- | ---: | ---: |
| c46 baseline | 0.367528 | 0.365288 |
| Candidate union + RRF | 0.381733 | 0.378523 |
| Candidate union + RRF + position reranker | 0.382969 | 0.380279 |
| Candidate union + position + passage-fold BM25 (0.5) | 0.386081 | 0.382376 |
| Previous stack + whitespace-brand BM25 (0.5) | 0.386874 | 0.383537 |
| Previous stack + RMS position aggregation | 0.387749 | 0.383781 |
| RMS stack + one-edit fuzzy positional bonus (0.15) | 0.388490 | 0.384293 |

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

A bounded cross-field coverage boost of `0.03` further improved the stack by
`+0.000209` on the disjoint training fold (paired bootstrap 95% CI:
`[+0.000023, +0.000527]`) and by `+0.000158` on matched holdout tasks. It
rewards terms split across title/description/brand only beyond the best single
field's coverage. Holdout results are aggregate-only.

Adding `brand__whitespace__default` BM25 at weight `0.5` to the accepted
cross-field stack improved full training NDCG@10 by `+0.000600`; the disjoint
validation delta was `+0.001110` (95% CI `[-0.000775, +0.003035]`). On matched
holdout tasks it improved by `+0.001003`. Only aggregate holdout metrics are
recorded.

RMS position aggregation added `+0.001253` over weighted mean on a disjoint
training validation fold (paired bootstrap 95% CI: `[+0.000020, +0.003156]`)
and `+0.000244` on matched holdout tasks. The `weighted_mean` default remains
unchanged for compatibility; RMS is an opt-in experimental mode.

The one-edit fuzzy positional bonus at weight `0.15` raised full training by
`+0.000741` and the disjoint training validation fold by `+0.000832` (paired
95% CI `[-0.001036, +0.002667]`). On matched private holdout it changed
NDCG@10 by `+0.000512` (paired 95% CI `[-0.000896, +0.001856]`). The measured
holdout result is positive but uncertain; the feature remains opt-in. Only
aggregate holdout results are recorded.

No latency benchmark was performed. Phrase filters may add query latency.
