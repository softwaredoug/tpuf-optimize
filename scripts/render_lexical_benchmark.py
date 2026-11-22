"""Render separate Pareto charts for the fresh matched lexical benchmark."""

import argparse
import json
from pathlib import Path

from plot_pareto import Point, render_svg

ROOT = Path(__file__).resolve().parents[1]
VARIANT_NAMES = {
    "fuzzy": "c46 lexical",
    "rms-stack": "RMS stack",
    "rms-stack-fuzzy015": "RMS + fuzzy position 0.15",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "assets")
    args = parser.parse_args()

    try:
        aggregate = json.loads(args.aggregate.read_text(encoding="utf-8"))
        if aggregate.get("query_count") != 5000:
            raise ValueError("Expected an aggregate for 5,000 queries")
        variants = aggregate["variants"]
        missing = set(VARIANT_NAMES) - variants.keys()
        if missing:
            raise ValueError(
                "Aggregate is missing lexical variants: "
                + ", ".join(sorted(missing))
            )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        for percentile in ("p50", "p95"):
            latency_key = f"{percentile}_ms"
            points = [
                Point(
                    VARIANT_NAMES[slug],
                    float(variants[slug][latency_key]),
                    float(variants[slug]["ndcg_at_10"]),
                )
                for slug in VARIANT_NAMES
            ]
            output = args.output_dir / f"esci-lexical-fresh-{percentile}.svg"
            output.write_text(
                render_svg(
                    points,
                    0,
                    percentile,
                    chart_title=(
                        "Fresh 5,000-query lexical benchmark: "
                        f"NDCG@10 vs. {percentile} latency"
                    ),
                    x_axis_label=f"{percentile} successful query latency (ms)",
                    x_description=(
                        f"{percentile} successful query latency in milliseconds"
                    ),
                    minimum_x_span=2.0,
                ),
                encoding="utf-8",
            )
            print(f"Saved {output}")
    except (OSError, KeyError, TypeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
