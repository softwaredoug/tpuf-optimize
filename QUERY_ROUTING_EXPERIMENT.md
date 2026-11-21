# Query-shape RRF routing experiment

The offline `routing.py` helper fuses the existing dense and lexical result
lists over their shared candidate union. It uses one-based list ranks, a
deterministic dense-first tie break, RRF `k=10`, and returns the top 10.
The candidate arms contain up to 100 results each.

## Training-only evaluation

Evaluated the 1,000-query training cache with Referee. For internal
query-disjoint selection, query IDs were assigned to development or validation
by `SHA256(query_id)[0] < 179` (713 development, 287 validation). Query IDs
were used only for this internal partition, never by the router. A fixed dense
2:1 lexical RRF profile scored 0.3731 overall, consistent with the parent
baseline.

Development NDCG:

| Rule | NDCG |
| --- | ---: |
| Fixed dense:lexical 2:1 | 0.374589 |
| Mixed alphanumeric identifier tokens only: dense:lexical 1:3 | 0.372060 |
| At least 4 tokens: dense:lexical 3:1; otherwise 2:1 | 0.374205 |
| Three-way identifier / descriptive / baseline routing | 0.371676 |

The best routing-only rule on development was semantic-heavy routing for
non-identifier queries with at least four tokens. Its disjoint validation NDCG
was 0.368741 versus 0.369439 for the fixed 2:1 baseline. The selected routing
rule therefore did not beat baseline on either split; no holdout candidate is
recommended. `QueryRouter` retains the best-tested routing-only candidate as
an experiment, not as a proven relevance improvement.
