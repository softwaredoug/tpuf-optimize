"""Frozen retrieval and ranking adapters for the ESCI benchmark."""

import json
import subprocess
import sys
import types
from dataclasses import dataclass

from turbopuffer.types import Row

from tpuf_optimize.searcher import Searcher
from tpuf_optimize.strategy import TPufStrategy


@dataclass(frozen=True)
class Variant:
    slug: str
    name: str
    revision: str
    branch: str
    historical_ndcg: float
    uses_voyage4_query_embedding: bool
    rerank_model: str | None = None
    rerank_tokens_proxy_per_query: float = 0.0
    category: str = "Unclassified"


# Previous holdout scores are selection evidence, not scores on the new split.
VARIANTS = (
    Variant(
        "typed", "Typed + computed reranker", "6508e87",
        "innovate-typed-computed-ltr", .42305, True,
        category="LTR",
    ),
    Variant("computed", "Computed-feature reranker", "e64100d",
            "innovate-computed-ltr-features", .41592, True,
            category="LTR"),
    Variant(
        "learned", "Learned pairwise reranker", "adf9768",
        "research/learned-rank", .41094, True,
        category="LTR",
    ),
    Variant(
        "identifier", "Identifier-gated RRF", "7e22661",
        "confidence-aware-fusion", .40316, True,
        category="Hybrid (lexical + embedding)",
    ),
    Variant("product", "Deterministic product reranker", "7ffd772",
            "research-product-rerank", .39691, True,
            category="Hybrid (lexical + embedding)"),
    Variant(
        "minmax", "Min-max score fusion", "41e64d4",
        "research-score-fusion", .39538, True,
        category="Hybrid (lexical + embedding)",
    ),
    Variant(
        "hybrid", "2:1 Voyage-4 + BM25 RRF", "cc5fc8a",
        "experiment-6-hybrid-rrf", .39500, True,
        category="Hybrid (lexical + embedding)",
    ),
    Variant("four", "Four-channel RRF", "3f4c9bc", "exp7-rank-fusion", .39064, True,
            category="Hybrid (lexical + embedding)"),
    Variant("fuzzy", "Passage BM25 + fuzzy title/brand", "c46c935",
            "exp3-bm25-passage-fuzzy", .37768, False,
            category="Lexical"),
    Variant(
        "rms-stack",
        "RMS lexical candidate stack",
        "038e845",
        "experiment/lexical-union-position",
        .383781,
        False,
        category="Lexical",
    ),
    Variant(
        "rms-stack-fuzzy015",
        "RMS stack + fuzzy-position boost 0.15",
        "d663a92",
        "experiment/lexical-union-position",
        .384293,
        False,
        category="Lexical",
    ),
    Variant(
        "single-composite",
        "One-request composite BM25 (title + passage + brand)",
        "911ac84cabb89bfc276c74faab123c5d0505c7a4",
        "lexical-one-query-composite",
        .372057,
        False,
        category="Lexical",
    ),
    Variant(
        "single-rerank",
        "One-request BM25 + local lexical reranker",
        "079cac2a8d71109a53c78369668024fd3ac2efb0",
        "experiment/lexical-one-query-rerank",
        .375447,
        False,
        category="Lexical",
    ),
    Variant("passage", "Title + locale-aware passage BM25", "cdb2d64",
            "relevance/field-weights-passage", .37658, False,
            category="Lexical"),
    Variant("voyage", "Baseline: Voyage-4 ANN (Turbopuffer)", "cc5fc8a",
            "experiment-6-hybrid-rrf", .37443, True,
            category="Embedding"),
    Variant(
        "bm25", "Baseline: weighted title/description BM25", "1768df7",
        "main", .32220, False,
        category="Lexical",
    ),
    Variant(
        "voyage-rerank",
        "Voyage rerank-3 over ANN + BM25",
        "0a5816e",
        "experiment/tpuf-native-rerankers",
        .45505,
        True,
        "voyage/rerank-3",
        25_380,
        category="Third-party reranker",
    ),
)

MODEL_RERANK_PRICES_USD_PER_MILLION = {
    "cohere/rerank-v3.5": 0.10,
    "voyage/rerank-3": 0.05,
}


def source(revision, path):
    return subprocess.check_output(["git", "show", f"{revision}:{path}"], text=True)


def load(revision, file, alias=None):
    path = f"src/tpuf_optimize/{file}.py"
    name = alias or f"_esci_bench_{revision}_{file}"
    module = types.ModuleType(name)
    sys.modules[name] = module
    exec(compile(source(revision, path), f"{revision}:{path}", "exec"), module.__dict__)
    return module


def _rows(pairs):
    return [Row.from_dict({"id": str(doc), "$dist": score}) for doc, score in pairs]


class FrozenSearcher(Searcher):
    """Run one pinned variant through the shared uncached timer and 429 policy."""

    CANDIDATE_REVISION = "0e92863"

    def __init__(self, namespace, variant):
        super().__init__(namespace, TPufStrategy(), batch_size=1)
        self.variant = variant
        self.slug = slug = variant.slug
        rev = variant.revision
        self.direct = slug in {"voyage", "bm25", "fuzzy", "passage"}
        if slug in {"rms-stack", "rms-stack-fuzzy015"}:
            strategy = load(rev, "strategy")
            self.frozen = strategy.TPufStrategy(
                field_fusion="passage_fold_brand_whitespace"
            )
            return
        if slug == "single-composite":
            self.frozen = load(rev, "strategy").TPufStrategy(
                variant="composite",
            )
            return
        if slug == "single-rerank":
            self.single_request = load(rev, "single_request")
            self.frozen = self.single_request.SingleRequestLexicalStrategy()
            return
        if slug == "voyage-rerank":
            self.native_rerank = load(rev, "native_rerank")
            self.frozen = load("cc5fc8a", "strategy").TPufStrategy()
            return
        if self.direct or slug in {"hybrid", "four"}:
            self.frozen = load(rev, "strategy").TPufStrategy()
            return
        self.collector = load(
            self.CANDIDATE_REVISION, "training_collector", "tpuf_optimize.training_collector"
        )
        # Candidate generation uses the same BM25 expression and embedding as
        # the collector's original parent, not the selected ranking branch.
        self.arms = [self.collector.CandidateQueryStrategy(arm) for arm in ("dense", "lexical")]
        if slug in {"computed", "typed", "learned"}:
            self.ranker = load(
                rev, "learned_rank" if slug == "learned" else "computed_feature_rank",
            )
            model = {
                "learned": "esci_pairwise_ranker.json",
                "computed": "esci_pairwise_ranker_computed_features_training.json",
                "typed": "esci_pairwise_ranker_typed_computed_no_unknown_training.json",
            }[slug]
            self.model = self.ranker.LinearRanker.from_json(
                json.loads(source(rev, f"models/{model}")),
            )
            if slug != "learned":
                self.features = load(rev, "computed_feature_collector")
                self.feature_strategy = self.features.ComputedFeatureQueryStrategy("dense")
        else:
            self.ranker = load(rev, {
                "identifier": "confidence_fusion",
                "product": "product_rerank",
                "minmax": "score_fusion",
            }[slug])
            if slug == "identifier":
                self.frozen = load("cc5fc8a", "strategy").TPufStrategy()

    def _retrieve(self, queries):
        response = self.namespace.multi_query(queries=queries)
        if len(response.results) != len(queries):
            raise ValueError("Incomplete Turbopuffer subquery response")
        return [result.rows or [] for result in response.results]

    def _add_features(self, text, locale, arms):
        ids = list(dict.fromkeys(str(candidate["doc_id"]) for arm in arms for candidate in arm))
        if not ids:
            return
        names = ("title_bm25", "passage_bm25", "brand_bm25")
        query = {
            "filters": ("And", (("locale", "Eq", locale), ("id", "In", ids))),
            "rank_by": ("id", "asc"),
            "top_k": len(ids),
            "include_attributes": ["title", "brand"],
            "compute_attributes": self.feature_strategy.computed_attributes(text, locale),
        }
        [rows] = self._retrieve([query])
        features = {}
        for row in rows:
            data = row.model_dump()
            values = {name: self.features._number(data.get(name)) for name in names}
            values["title_full_coverage"] = self.features._title_full_coverage(
                text, data.get("title"),
            )
            features[str(row.id)] = values
        if len(rows) != len(ids) or features.keys() != set(ids):
            raise ValueError("Computed features did not cover the retrieved candidate union")
        for arm in arms:
            for candidate in arm:
                candidate["features"] = features[str(candidate["doc_id"])]

    def _retrieve_measured_batch(self, requests, top_k):
        [request] = requests
        text, locale = request.text, request.locale
        slug = self.slug
        if slug == "voyage-rerank":
            queries = self.native_rerank.candidate_queries(
                self.frozen, text, locale
            )
            response = self.namespace.multi_query(
                queries=queries,
                rerank_by=self.native_rerank.native_rerank_by(
                    "voyage/rerank-3", text
                ),
                limit=top_k,
            )
            if len(response.results) != 1:
                raise ValueError("Voyage reranker returned an incomplete result.")
            return [(response.results[0].rows or [])[:top_k]]
        if slug in {"rms-stack", "rms-stack-fuzzy015"}:
            queries = [
                self.frozen.query(text, locale, 50),
                *self.frozen.lexical_candidate_queries(text, locale, 50),
            ]
            arm_rows = self._retrieve(queries)
            fused_scores = {}
            documents = {}
            for rows in arm_rows:
                for rank, row in enumerate(rows, 1):
                    document_id = str(row.id)
                    fused_scores[document_id] = (
                        fused_scores.get(document_id, 0.0) + 1.0 / (20 + rank)
                    )
                    documents.setdefault(document_id, row)
            ordered_ids = sorted(
                fused_scores,
                key=lambda document_id: (
                    -fused_scores[document_id],
                    document_id,
                ),
            )[:50]
            candidates = [documents[document_id] for document_id in ordered_ids]
            if slug == "rms-stack-fuzzy015":
                ranked = self.frozen.rerank_lexically(
                    text, candidates, .18, .03, "rms", .15, locale,
                )
            else:
                ranked = self.frozen.rerank_lexically(
                    text, candidates, .18, .03, "rms",
                )
            return [ranked[:top_k]]
        if slug == "single-composite":
            query = self.frozen.query(text, locale, top_k)
            response = self.namespace.query(**query)
            return [(response.rows or [])[:top_k]]
        if slug == "single-rerank":
            query = self.frozen.query(text, locale, 100)
            response = self.namespace.query(**query)
            ranked = self.single_request.rerank_candidates(
                text, response.rows or [], .35,
            )
            return [ranked[:top_k]]
        if self.direct:
            if slug == "voyage":
                ann_query = self.frozen.hybrid_queries(text, locale, top_k)[0]
                ann_query["top_k"] = top_k
                return self._retrieve([ann_query])
            return self._retrieve([self.frozen.query(text, locale, top_k)])
        if slug == "hybrid":
            rows = self._retrieve(self.frozen.hybrid_queries(text, locale, top_k))
            return [self.frozen.reciprocal_rank_fusion(rows, (2, 1), top_k=top_k)]
        if slug == "four":
            rows = self._retrieve(self.frozen.fusion_queries(text, locale, top_k))
            scores = {}
            for arm in rows:
                for rank, row in enumerate(arm, 1):
                    doc = str(row.id)
                    scores[doc] = scores.get(doc, 0) + 1 / (self.frozen.RRF_RANK_CONSTANT + rank)
            return [_rows(sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))[:top_k])]

        rows = self._retrieve([
            arm.query(text, locale, self.collector.HYBRID_DEPTH) for arm in self.arms
        ])
        dense, lexical = [self.collector._candidate_list(hits) for hits in rows]
        if slug == "identifier":
            weights = self.ranker.confidence_aware_weights(text, dense, lexical, gate="identifier")
            return [self.frozen.reciprocal_rank_fusion(rows, weights, top_k=top_k)]
        if slug == "product":
            hits = self.ranker.rerank_candidates(
                text, locale, dense, lexical, top_k=top_k, max_adjustment=.05,
            )
            return [_rows((hit["doc_id"], hit["score"]) for hit in hits)]
        if slug == "minmax":
            hits = self.ranker.fuse_convex(dense, lexical, dense_weight=.5, normalization="minmax")
            return [_rows(hits[:top_k])]
        if not (dense or lexical):
            return [[]]
        if slug != "learned":
            self._add_features(text, locale, (dense, lexical))
        record = self.ranker.parse_query_record({
            "query_id": request.query_id, "query_text": text, "locale": locale,
            "dense": dense, "lexical": lexical,
        })
        hits = self.ranker.model_ranking(record, self.model)[:top_k]
        return [_rows((hit.doc_id, score) for hit, score in hits)]
