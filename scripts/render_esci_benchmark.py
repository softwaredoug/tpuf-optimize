"""Render both Pareto charts and refresh the README from aggregate-only data."""

import argparse
import json
from pathlib import Path

from esci_benchmark_variants import VARIANTS
from plot_pareto import Point, render_svg

ROOT = Path(__file__).resolve().parents[1]
START = "<!-- esci-benchmark:start -->"
END = "<!-- esci-benchmark:end -->"


def render_block(aggregate, assets):
    scores = aggregate.get("variants", {})
    costs = aggregate.get("cost_estimates", {}).get(
        "variant_cost_per_run_usd", {}
    )
    query_count = aggregate.get("query_count", 5000)
    unknown = scores.keys() - {v.slug for v in VARIANTS}
    if unknown:
        raise ValueError("Unexpected variant in benchmark aggregate")
    count = aggregate.get("query_count")
    disjoint = aggregate.get("disjoint_count")
    lines = [START, ""]
    if count is not None:
        lines += [
            f"Scored variants used the same {count:,} Referee queries. "
            f"The second NDCG column covers the {disjoint:,} queries not found in "
            f"{aggregate['checked_artifacts']} available historical task artifacts.",
            "",
        ]
    lines += [
        "| Variant | NDCG@10 (full) | NDCG@10 (disjoint) | p50 (ms) | p95 (ms) | "
        f"Cost* (USD/{query_count:,} queries) | Commit |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    ordered = sorted(
        VARIANTS,
        key=lambda v: scores.get(v.slug, {}).get("ndcg_at_10", -1),
        reverse=True,
    )
    for variant in ordered:
        result = scores.get(variant.slug)
        if result is not None and not result["commit"].startswith(variant.revision):
            raise ValueError("Aggregate revision does not match the pinned variant")
        values = (
            ("Pending", "—", "—", "—") if result is None else
            (f"{result['ndcg_at_10']:.5f}", f"{result['disjoint_ndcg_at_10']:.5f}",
             f"{result['p50_ms']:.2f}", f"{result['p95_ms']:.2f}")
        )
        estimated_cost = costs.get(variant.slug)
        cost_cell = (
            "—" if estimated_cost is None else f"${float(estimated_cost):.6f}*"
        )
        commit_url = f"https://github.com/softwaredoug/tpuf-optimize/commit/{variant.revision}"
        branch_url = f"https://github.com/softwaredoug/tpuf-optimize/tree/{variant.branch}"
        lines.append(
            f"| [{variant.name}]({branch_url}) | {' | '.join(values)} | {cost_cell} | "
            f"[`{variant.revision}`]({commit_url}) |"
        )
    if costs:
        cost_method = aggregate.get("cost_estimates", {})
        lines.extend([
            "",
            (
                f"*Estimated external model inference per {query_count:,}-query run. "
                "Voyage-4 "
                f"query embedding uses a UTF-8-bytes/4 token proxy "
                f"({cost_method.get('voyage4_embedding_usd_per_million_tokens', 0):g} "
                "USD/million tokens); lexical-only and local CPU rankers add no "
                "model inference charge. Excludes Turbopuffer database/query "
                "charges and index-time embedding."
            ),
        ])
    for percentile in ("p50", "p95"):
        points = [
            Point(v.name, scores[v.slug][f"{percentile}_ms"], scores[v.slug]["ndcg_at_10"])
            for v in ordered if v.slug in scores
        ]
        filename = f"esci-pareto-{percentile}.svg"
        (assets / filename).write_text(
            render_svg(points, len(VARIANTS) - len(points), percentile), encoding="utf-8",
        )
        lines.extend(["", f"![ESCI NDCG@10 versus {percentile} latency](assets/{filename})"])
    return "\n".join([*lines, "", END])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, default=ROOT / "assets/esci-benchmark.json")
    parser.add_argument("--readme", type=Path, default=ROOT / "README.md")
    args = parser.parse_args()
    readme = args.readme.read_text(encoding="utf-8")
    first, last = readme.index(START), readme.index(END) + len(END)
    aggregate = (
        json.loads(args.aggregate.read_text(encoding="utf-8")) if args.aggregate.exists() else {}
    )
    replacement = render_block(aggregate, args.readme.parent / "assets")
    args.readme.write_text(readme[:first] + replacement + readme[last:], encoding="utf-8")


if __name__ == "__main__":
    main()
