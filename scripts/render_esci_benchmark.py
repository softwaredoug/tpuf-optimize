"""Render both Pareto charts and refresh the README from aggregate-only data."""

import argparse
import json
from pathlib import Path

from esci_benchmark_variants import VARIANTS
from plot_pareto import Point, render_svg

ROOT = Path(__file__).resolve().parents[1]
START = "<!-- esci-benchmark:start -->"
END = "<!-- esci-benchmark:end -->"
CATEGORY_LABELS = {
    "Lexical": "Lexical",
    "Hybrid (lexical + embedding)": "Hybrid",
    "LTR": "LTR",
    "Third-party reranker": "Third-party reranker",
    "Embedding": "Embedding",
}
FRESH_LEXICAL_SLUGS = {"fuzzy", "rms-stack", "rms-stack-fuzzy015"}


def render_block(aggregate, assets, fresh_lexical=None):
    scores = aggregate.get("variants", {})
    fresh_scores = (fresh_lexical or {}).get("variants", {})
    costs = aggregate.get("cost_estimates", {}).get(
        "variant_cost_per_run_usd", {}
    )
    query_count = aggregate.get("query_count", 5000)
    known_slugs = {variant.slug for variant in VARIANTS}
    unknown = (scores.keys() | fresh_scores.keys()) - known_slugs
    if unknown:
        raise ValueError("Unexpected variant in benchmark aggregate")
    if fresh_scores.keys() - FRESH_LEXICAL_SLUGS:
        raise ValueError("Fresh lexical aggregate contains an unexpected variant")
    count = aggregate.get("query_count")
    disjoint = aggregate.get("disjoint_count")
    fresh_count = (fresh_lexical or {}).get("query_count")
    fresh_disjoint = (fresh_lexical or {}).get("disjoint_count")
    lines = [START, ""]
    if count is not None:
        lines += [
            f"Historical leaderboard variants used the same {count:,} Referee queries. "
            f"The second NDCG column covers the {disjoint:,} queries not found in "
            f"{aggregate['checked_artifacts']} available historical task artifacts.",
            "",
        ]
    if fresh_count is not None:
        lines += [
            f"Fresh lexical rows use a separate {fresh_count:,}-query Referee session. "
            f"Their disjoint column covers {fresh_disjoint:,} queries after "
            f"excluding {fresh_count - fresh_disjoint} overlaps with "
            f"{fresh_lexical['checked_artifacts']} historical task artifacts. "
            "Compare values within a cohort, not across cohorts.",
            "",
        ]
    if count is not None:
        lines += [
            "Classes: lexical; hybrid (lexical + embedding, including "
            "rule-based reranking on both); LTR (Learning to Rank), a classic "
            "ML reranker using query-dependent, query, and document features; "
            "third-party reranker. Pure embedding retrieval is listed separately."
            "",
        ]
    lines += [
        "| Variant | Cohort | Class | NDCG@10 (full) | NDCG@10 (disjoint) | "
        f"p50 (ms) | p95 (ms) | Cost* (USD/{query_count:,} queries) | Commit |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    ordered = sorted(
        (variant for variant in VARIANTS if variant.slug in scores),
        key=lambda v: scores.get(v.slug, {}).get("ndcg_at_10", -1),
        reverse=True,
    )
    fresh_ordered = sorted(
        (variant for variant in VARIANTS if variant.slug in fresh_scores),
        key=lambda variant: fresh_scores[variant.slug]["ndcg_at_10"],
        reverse=True,
    )
    table_rows = []
    for variant in ordered:
        result = scores[variant.slug]
        if not result["commit"].startswith(variant.revision):
            raise ValueError("Aggregate revision does not match the pinned variant")
        table_rows.append((
            float(result["ndcg_at_10"]),
            variant,
            "Historical leaderboard",
            result,
            costs.get(variant.slug),
        ))
    for variant in fresh_ordered:
        result = fresh_scores[variant.slug]
        if not result["commit"].startswith(variant.revision):
            raise ValueError("Fresh lexical aggregate revision does not match")
        table_rows.append((
            float(result["ndcg_at_10"]),
            variant,
            "Fresh lexical",
            result,
            0.0,
        ))
    for _ndcg, variant, cohort, result, estimated_cost in sorted(
        table_rows,
        key=lambda row: (-row[0], row[2], row[1].name),
    ):
        values = (
            f"{result['ndcg_at_10']:.5f}",
            f"{result['disjoint_ndcg_at_10']:.5f}",
            f"{result['p50_ms']:.2f}",
            f"{result['p95_ms']:.2f}",
        )
        cost_cell = (
            "—" if estimated_cost is None else f"${float(estimated_cost):.6f}*"
        )
        branch_url = f"https://github.com/softwaredoug/tpuf-optimize/tree/{variant.branch}"
        commit_url = f"https://github.com/softwaredoug/tpuf-optimize/commit/{variant.revision}"
        lines.append(
            f"| [{variant.name}]({branch_url}) | {cohort} | "
            f"{CATEGORY_LABELS.get(variant.category, variant.category)} | "
            f"{' | '.join(values)} | {cost_cell} | "
            f"[`{variant.revision}`]({commit_url}) |"
        )
    if costs:
        cost_method = aggregate.get("cost_estimates", {})
        rerank_rate = cost_method.get(
            "native_rerank_prices_usd_per_million_tokens", {}
        ).get("voyage/rerank-3")
        rerank_proxy = cost_method.get(
            "native_rerank_token_proxy_per_query"
        )
        rerank_note = (
            f" Voyage rerank-3 uses ${rerank_rate:g}/M and a "
            f"{rerank_proxy:,.0f}-token-per-query training proxy."
            if isinstance(rerank_rate, (int, float))
            and isinstance(rerank_proxy, (int, float))
            else ""
        )
        lines.extend([
            "",
            (
                f"*Estimated external model inference per {query_count:,}-query run. "
                "Voyage-4 "
                f"query embedding uses a UTF-8-bytes/4 token proxy "
                f"({cost_method.get('voyage4_embedding_usd_per_million_tokens', 0):g} "
                "USD/million tokens); lexical-only and local CPU rankers add no "
                "model inference charge. Excludes Turbopuffer database/query "
                f"charges and index-time embedding.{rerank_note}"
            ),
            "",
            "- **Lexical** — BM25 and other lexical matching only.",
            "- **Hybrid** — combines lexical + embedding retrieval/fusion, including "
            "rule-based reranking over those candidates.",
            "- **LTR** — classic ML reranker using query-dependent, query, and "
            "document features.",
            "- **Third-party reranker** — hosted reranking model such as Voyage rerank-3.",
            "- **Embedding** — pure embedding retrieval; listed separately from hybrid.",
        ])
    pending = len(VARIANTS) - len(scores.keys() | fresh_scores.keys())
    for percentile in ("p50", "p95"):
        points = [
            *[
                Point(
                    variant.name,
                    scores[variant.slug][f"{percentile}_ms"],
                    scores[variant.slug]["ndcg_at_10"],
                    "Historical leaderboard",
                )
                for variant in ordered
            ],
            *[
                Point(
                    variant.name,
                    fresh_scores[variant.slug][f"{percentile}_ms"],
                    fresh_scores[variant.slug]["ndcg_at_10"],
                    "Fresh lexical",
                )
                for variant in fresh_ordered
            ],
        ]
        filename = f"esci-pareto-{percentile}.svg"
        (assets / filename).write_text(
            render_svg(points, pending, percentile), encoding="utf-8",
        )
        lines.extend(["", f"![ESCI NDCG@10 versus {percentile} latency](assets/{filename})"])
    cost_points = [
        Point(
            variant.name,
            float(costs[variant.slug]),
            float(scores[variant.slug]["ndcg_at_10"]),
            "Historical leaderboard",
        )
        for variant in ordered
        if variant.slug in scores and variant.slug in costs
    ]
    cost_points.extend(
        Point(
            variant.name,
            0.0,
            float(fresh_scores[variant.slug]["ndcg_at_10"]),
            "Fresh lexical",
        )
        for variant in fresh_ordered
    )
    cost_filename = "esci-pareto-cost.svg"
    query_count = aggregate.get("query_count", 5000)
    (assets / cost_filename).write_text(
        render_svg(
            cost_points,
            pending,
            "cost",
            chart_title="ESCI relevance vs. model inference cost",
            x_axis_label=(
                f"Estimated model inference cost per {query_count:,} queries (USD)"
            ),
            x_description="Estimated model inference cost per benchmark query set in USD",
            minimum_x_span=0.001,
        ),
        encoding="utf-8",
    )
    lines.extend([
        "",
        (
            f"![ESCI NDCG@10 versus estimated model cost per "
            f"{query_count:,} queries](assets/{cost_filename})"
        ),
    ])
    return "\n".join([*lines, "", END])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, default=ROOT / "assets/esci-benchmark.json")
    parser.add_argument(
        "--lexical-aggregate",
        type=Path,
        default=ROOT / "assets/esci-lexical-fresh-5k.json",
    )
    parser.add_argument("--readme", type=Path, default=ROOT / "README.md")
    args = parser.parse_args()
    readme = args.readme.read_text(encoding="utf-8")
    first, last = readme.index(START), readme.index(END) + len(END)
    aggregate = (
        json.loads(args.aggregate.read_text(encoding="utf-8")) if args.aggregate.exists() else {}
    )
    fresh_lexical = (
        json.loads(args.lexical_aggregate.read_text(encoding="utf-8"))
        if args.lexical_aggregate.exists() else None
    )
    replacement = render_block(
        aggregate, args.readme.parent / "assets", fresh_lexical
    )
    args.readme.write_text(readme[:first] + replacement + readme[last:], encoding="utf-8")


if __name__ == "__main__":
    main()
