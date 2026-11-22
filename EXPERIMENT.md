# Single-request composite lexical BM25

## Variant

One Turbopuffer request per query, with no candidate union or subsequent
reranking. For locale analyzer suffix `s` and query `q`, the candidate's
`rank_by` is:

```text
Sum(
  Product(1.00, BM25(title__s, q)),
  Product(0.50, BM25(passage__s, q)),
  Product(0.25, BM25(brand__s, q)),
)
```

The `passage` is the pre-existing title-plus-description field. The matched
`cdb2d64` baseline is the same single-request expression without the brand
term:

```text
Sum(
  Product(1.00, BM25(title__s, q)),
  Product(0.50, BM25(passage__s, q)),
)
```

Both variants filter on the query locale and request the same top-10 output.
They use only existing indexed fields and locale-specific analyzers. No
reindexing or schema changes are involved.

## Training evaluation

Matched Referee ESCI training tasks contained **500 queries** each. Both variants
ran one serial Turbopuffer query request per task query (500 requests per run),
with no local result cache, no 429 retries, and the same top-10 result depth.
The tasks had identical query payloads. Rankings were submitted to Referee;
aggregate metrics only are recorded here.

| Variant | NDCG@10 | p50 request latency | p95 request latency |
| --- | ---: | ---: | ---: |
| `cdb2d64` passage baseline | 0.36793 | 60.06 ms | 68.59 ms |
| title + passage + brand (`brand` weight 0.25) | 0.36992 | 61.05 ms | 77.92 ms |

Paired mean NDCG@10 delta: **+0.00199**. Paired-query bootstrap 95% interval:
**[-0.00232, +0.00644]** (10,000 resamples). The point estimate is positive,
but the interval includes zero; p50 latency increased by 0.98 ms and p95 by
9.33 ms. This is a small, uncertain training gain, so private holdout evaluation
is useful to determine whether the brand contribution generalizes.
