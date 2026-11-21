from __future__ import annotations

from tpuf_optimize.typed_evidence import (
    EvidenceState,
    TypedEvidenceConfig,
    candidate_features,
    rerank_candidates,
)


def test_typed_slots_distinguish_support_contradiction_and_unknown() -> None:
    query = "Acme blue model X100 12 oz"
    candidates = [
        {
            "doc_id": "match",
            "title": "Acme blue model X100 bottle 12 oz",
            "brand": "Acme",
            "color": "Blue",
        },
        {
            "doc_id": "mismatch",
            "title": "Other red model Z200 bottle 16 lb",
            "brand": "Other",
            "color": "Red",
        },
        {
            "doc_id": "unknown",
            "title": "Insulated bottle",
            "brand": None,
            "color": None,
        },
    ]
    dense_ranks = {"match": 1, "mismatch": 2, "unknown": 3}
    lexical_ranks = {"match": 2, "mismatch": 1}
    gaps = {"match": 1.0, "mismatch": 0.5, "unknown": 0.0}
    matching = candidate_features(
        query, candidates[0], candidates, dense_ranks, lexical_ranks, gaps, gaps
    )
    mismatching = candidate_features(
        query, candidates[1], candidates, dense_ranks, lexical_ranks, gaps, gaps
    )
    unknown = candidate_features(
        query, candidates[2], candidates, dense_ranks, lexical_ranks, gaps, gaps
    )

    assert (
        matching.brand,
        matching.color,
        matching.model,
        matching.number,
        matching.unit,
    ) == (EvidenceState.SUPPORT,) * 5
    assert matching.typed_support_interaction == 1.0
    assert mismatching.typed_support_interaction == 0.0
    assert (
        mismatching.brand,
        mismatching.color,
        mismatching.model,
        mismatching.number,
        mismatching.unit,
    ) == (EvidenceState.CONTRADICTION,) * 5
    assert (
        unknown.brand,
        unknown.color,
        unknown.model,
        unknown.number,
        unknown.unit,
    ) == (EvidenceState.UNKNOWN,) * 5


def test_title_evidence_rewards_distinct_term_coverage_and_proximity() -> None:
    dense = [
        {"doc_id": "near", "score": 0.1, "title": "blue running shoes"},
        {"doc_id": "far", "score": 0.2, "title": "shoes for running in blue"},
    ]
    lexical = [
        {"doc_id": "near", "score": 10.0, "title": "blue running shoes"},
        {"doc_id": "far", "score": 9.0, "title": "shoes for running in blue"},
    ]
    results = rerank_candidates(
        "blue running shoes blue",
        dense,
        lexical,
        config=TypedEvidenceConfig(
            title_coverage=1.0,
            title_proximity=1.0,
            max_adjustment=0.2,
            brand=0,
            color=0,
            model=0,
            number=0,
            unit=0,
            typed_support_interaction=0,
            typed_conflict_interaction=0,
            channel_agreement=0,
            dense_gap=0,
            lexical_gap=0,
            gap_agreement=0,
        ),
    )
    assert results[0]["doc_id"] == "near"
    assert results[0]["evidence_score"] > results[1]["evidence_score"]


def test_missing_slot_metadata_is_neutral_and_rrf_ties_are_dense_first() -> None:
    dense = [
        {"doc_id": "first", "score": 0.2, "title": "plain item"},
        {"doc_id": "second", "score": 0.2, "title": "plain item"},
    ]
    ranked = rerank_candidates(
        "unmatched blue 12 oz",
        dense,
        [],
        config=TypedEvidenceConfig(max_adjustment=0.0),
    )
    assert [result["doc_id"] for result in ranked] == ["first", "second"]
    assert all(result["score"] == result["rrf_score"] for result in ranked)


def test_parent_weighted_rrf_scores_and_dense_first_union_order_are_preserved() -> None:
    dense = [
        {"doc_id": "shared", "score": 0.1, "title": "item"},
        {"doc_id": "dense-only", "score": 0.2, "title": "item"},
    ]
    lexical = [
        {"doc_id": "lexical-only", "score": 10.0, "title": "item"},
        {"doc_id": "shared", "score": 9.0, "title": "item"},
    ]
    ranked = rerank_candidates(
        "item",
        dense,
        lexical,
        config=TypedEvidenceConfig(max_adjustment=0.0),
    )
    by_id = {row["doc_id"]: row["rrf_score"] for row in ranked}
    assert by_id["shared"] == 2 / 11 + 1 / 12
    assert by_id["dense-only"] == 2 / 12
    assert by_id["lexical-only"] == 1 / 11
    assert [row["doc_id"] for row in ranked] == [
        "shared",
        "dense-only",
        "lexical-only",
    ]


def test_typed_slot_interaction_can_rescue_consistent_product_evidence() -> None:
    dense = [
        {
            "doc_id": "wrong",
            "score": 0.1,
            "title": "Other red model Z200 16 lb",
            "brand": "Other",
            "color": "Red",
        },
        {
            "doc_id": "right",
            "score": 0.2,
            "title": "Acme blue model X100 12 oz",
            "brand": "Acme",
            "color": "Blue",
        },
    ]
    lexical = [
        {
            "doc_id": "wrong",
            "score": 10.0,
            "title": "Other red model Z200 16 lb",
            "brand": "Other",
            "color": "Red",
        },
        {
            "doc_id": "right",
            "score": 9.0,
            "title": "Acme blue model X100 12 oz",
            "brand": "Acme",
            "color": "Blue",
        },
    ]
    ranked = rerank_candidates(
        "Acme blue model X100 12 oz",
        dense,
        lexical,
        config=TypedEvidenceConfig(
            brand=1,
            color=1,
            model=1,
            number=1,
            unit=1,
            typed_support_interaction=1,
            typed_conflict_interaction=-1,
            title_coverage=0,
            title_proximity=0,
            channel_agreement=0,
            dense_gap=0,
            lexical_gap=0,
            gap_agreement=0,
            max_adjustment=0.08,
        ),
    )
    assert ranked[0]["doc_id"] == "right"
    assert all(abs(row["rerank_adjustment"]) <= 0.08 for row in ranked)
