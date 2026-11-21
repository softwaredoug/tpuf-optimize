# Locale-specific dense/lexical fusion experiment

## Frozen comparison

The parent uses weighted reciprocal-rank fusion over the cached arms:

`score = 2 / (10 + dense_rank) + 1 / (10 + lexical_rank)`

Ranks are one-based, up to 100 candidates are read from each arm, output is
limited to 10, and equal scores retain dense-first insertion order. The fixed
alternative set was `(dense, lexical) = (3,1), (1,1), (1,2), (1,3)`; it was
not expanded after seeing results. Candidate document `score` values are not
used by RRF.

The saved candidates were evaluated only with the assigned training Referee
token. The parent reproduced the supplied training NDCG (`0.373111`, reported
as `~0.3731` by the orchestrator), which also checks that candidate IDs and
training task IDs were aligned.

| Global weights | Training NDCG |
| --- | ---: |
| `(2,1)` parent | 0.373111 |
| `(3,1)` | 0.369866 |
| `(1,1)` | 0.372056 |
| `(1,2)` | 0.340282 |
| `(1,3)` | 0.324757 |

## Grouped internal fit and validation

Query text was normalized (case-folded and whitespace-collapsed), then assigned
deterministically to fit or validation by a salted SHA-256 bucket. Identical
normalized queries stayed together: 799 fit queries and 201 validation queries,
with no query-text group overlap. The validation split was not used to choose
weights. Locale counts in the complete training split were `us=752`, `es=119`,
`jp=129`; fit support, counted as distinct locale-specific query groups, was
`us=614`, `es=86`, `jp=99`.

The per-locale fit selected the highest mean from the fixed candidate set, but
accepted a non-parent option only when its paired mean gain over the parent was
greater than one paired standard error. Local weights were then shrunk toward
`(2,1)` with two fit query groups as prior strength. Local support below 20
groups falls back to `(2,1)`.

| Locale | Best unshrunk fit option | Fit gain vs parent | Paired SE | Decision |
| --- | --- | ---: | ---: | --- |
| `es` | `(1,1)` | +0.010864 | 0.013986 | retain parent; gain is within 1 SE |
| `jp` | `(1,1)` | +0.002671 | 0.012406 | retain parent; gain is within 1 SE |
| `us` | `(2,1)` | 0.000000 | 0.000000 | retain parent |

Thus the one frozen configuration was the parent everywhere: `us=(2,1)`,
`es=(2,1)`, `jp=(2,1)`. There is no locale-specific change to nominate.

The untouched grouped validation set was evaluated with that frozen
configuration. `NDCG mean ± standard error` is shown below; the paired
candidate-minus-parent deltas are exactly zero because the frozen configuration
is the parent.

| Split / locale | Queries | Parent NDCG ± SE | Frozen NDCG ± SE | Paired delta ± SE |
| --- | ---: | ---: | ---: | ---: |
| Validation / overall | 201 | 0.369041 ± 0.022303 | 0.369041 ± 0.022303 | 0.000000 ± 0.000000 |
| Validation / `es` | 33 | 0.408859 ± 0.061737 | 0.408859 ± 0.061737 | 0.000000 ± 0.000000 |
| Validation / `jp` | 30 | 0.396157 ± 0.057818 | 0.396157 ± 0.057818 | 0.000000 ± 0.000000 |
| Validation / `us` | 138 | 0.353624 ± 0.026177 | 0.353624 ± 0.026177 | 0.000000 ± 0.000000 |

## Analyzer decision

No analyzer alternative is shortlisted. The cached rows contain the already
retrieved dense and lexical rankings plus title/brand/color/category metadata,
but do not contain counterfactual rankings for other analyzer configurations.
Changing tokenizers cannot be evaluated from these saved ranks, and no separate
live queries were issued. The existing schema's locale analyzers are therefore
left unchanged.

`src/tpuf_optimize/locale_hybrid.py` contains the offline rank-fusion,
shrinkage, grouped-split, and uncertainty helpers. It does not issue Turbopuffer
requests or alter the production query strategy.
