# Lexical position reranker: training evaluation

Evaluated on the assigned 1,000-query ESCI training split using matched Referee
tasks. The baseline is the `c46c935` title/passage BM25 plus fuzzy title/brand
retrieval, with the same top-50 lexical candidates as the reranker.

The reranker uses only the returned title, description, and brand strings.
Its deterministic feature score combines query-term coverage, in-order
coverage, minimum term span, exact normalized phrase, and early field
occurrence. Japanese text is tokenized as Unicode CJK characters; absent
attributes contribute no evidence. The reranker adds at most 0.18 to a
normalized score derived from the existing rank, retaining original order for
ties. It adds no candidate source and uses no embeddings.

| Run | NDCG@10 |
| --- | ---: |
| c46 baseline | 0.367528 |
| lexical position reranker, selected boost 0.18 | 0.370771 |

The boost was selected from 0.06, 0.12, and 0.18 using a deterministic,
query-disjoint tuning fold (492 queries). Mean paired delta on that fold was
+0.000907, +0.002515, and +0.003748, respectively. The selected boost improved
the held-aside 508-query fold by +0.002753 NDCG@10. Its paired delta over all
1,000 training queries was +0.003243, with a paired bootstrap 95% percentile
interval of [+0.001706, +0.004800].

No query text, query IDs, result rows, or per-query judgments are stored here.
