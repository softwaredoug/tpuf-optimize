# Typed evidence reranker experiment

Compared with the exact `cc5fc8a` parent behavior: dense/lexical weights 2:1,
RRF constant 10, 100 candidates per arm, top 10 output, and stable dense-first
tie ordering. The cache and labels came only from the assigned training split.
The 1,000 training queries were separated by a fixed query-ID hash into 748
fit and 252 query-disjoint validation queries. Query/document IDs were used
only to join and split; neither IDs nor labels are model features. No raw
queries, candidate rows, labels, or per-query metrics are included here.

NDCG@10 results:

| Variant | Fit queries | Validation queries |
| --- | ---: | ---: |
| Parent RRF | 0.40576 | 0.39512 |
| Candidate-set ceiling | 0.96658 | — |
| Selected typed evidence | 0.40906 | 0.39910 |

The candidate-set ceiling is computed on the training candidates only. The
selected configuration improved held-out validation by `+0.00398` NDCG@10;
a query-paired bootstrap 95% interval was `[+0.00057, +0.00773]`. The feature
scales and cap were selected on fit queries only; validation was used for
reporting and ablation checking, not tuning.

Validation ablations using selected scales:

| Features | NDCG@10 |
| --- | ---: |
| Parent RRF | 0.39512 |
| Typed slots + support/conflict interaction | 0.39564 |
| Title distinct-term coverage + proximity | 0.39607 |
| Channel agreement + normalized gaps | 0.39746 |
| Slots + title | 0.39694 |
| Slots + channels | 0.39672 |
| Title + channels | 0.39784 |
| All features | 0.39910 |

The gains are complementary: the all-feature configuration exceeded each
single-block and two-block ablation on validation. The default in
`TypedEvidenceConfig` pins this fit-selected model. Adjustments remain capped
at 0.02, and unknown slot values remain neutral. Passage text is not scored:
the available title is used once for title features and explicit
brand/color/model/number/unit checks, with no duplicated title-in-passage
evidence.
