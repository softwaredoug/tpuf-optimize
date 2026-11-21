# ESCI cached score-fusion training trial

## Pinned candidate

`minmax-0.5`: convex fusion of per-query min-max normalized scores with dense
weight `0.50` and lexical weight `0.50`. Dense distances are reversed so lower
distance is better. Missing-arm candidates receive normalized score `0`.

This is a training-only nomination, not a production default or holdout result.

## Comparator and provenance

- Candidate inputs: the approved cached ESCI training candidates (1,000 queries;
  up to 100 dense and lexical results per query, with 4 empty lexical lists).
  Cache query IDs and query text were checked against each Referee training
  task before completion.
- Referee access: the assigned training bearer token only. No Turbopuffer or
  holdout calls were made.
- Parent comparator: weighted RRF matching `cc5fc8a`: `2 / (10 + dense_rank)`
  plus `1 / (10 + lexical_rank)`, one-based arm ranks, top 100 per arm, stable
  dense-first insertion for equal fusion scores, and final output truncated to
  10.
- Query-disjoint selection/evaluation: SHA-256 of each opaque query ID; first
  digest byte `< 128` selected the configuration (512 queries), and all others
  formed independent internal validation (488 queries). Query text, IDs, and
  document IDs are intentionally not recorded here.
- The configuration sweep used the same selection set for RRF, three min-max
  dense weights, three percentile dense weights, and mean-imputed missing-score
  variants at weight 0.50. After selection, fresh Referee tasks compared parent
  RRF and the nominated configuration on the same validation IDs.
- Metrics below are arithmetic means of Referee per-query NDCG@10. Validation
  differences are paired by query.

## Selection scores

| Configuration | Selection queries | NDCG@10 |
| --- | ---: | ---: |
| RRF parent comparator | 512 | 0.376604 |
| minmax-0.25 | 512 | 0.336945 |
| **minmax-0.5 (nominated)** | **512** | **0.383510** |
| minmax-0.75 | 512 | 0.376595 |
| percentile-0.25 | 512 | 0.332886 |
| percentile-0.5 | 512 | 0.374317 |
| percentile-0.75 | 512 | 0.372790 |
| minmax-0.5-mean | 512 | 0.375693 |
| percentile-0.5-mean | 512 | 0.370159 |

## Independent paired internal validation

| Configuration | Validation queries | NDCG@10 |
| --- | ---: | ---: |
| RRF parent comparator | 488 | 0.369445 |
| **minmax-0.5** | **488** | **0.378217** |

Paired delta: **+0.008771 NDCG**. Per-query outcomes: 173 wins, 130 ties, 185
losses. The parent RRF's mean across all 1,000 training queries was 0.373111,
consistent with the cached reference of approximately 0.3731.
