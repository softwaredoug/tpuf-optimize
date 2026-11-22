#!/usr/bin/env python3
"""Generate a latency/relevance Pareto chart from the README leaderboard."""

import argparse
import math
from dataclasses import dataclass
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECTION = "## ESCI 5,000-query leaderboard"
HEADERS = (
    ("Variant", "p50 (ms)", "NDCG@10 (full)"),
    ("Variant", "Latency p50 / p95 (ms)", "NDCG@10"),
)


@dataclass(frozen=True)
class Point:
    name: str
    latency: float
    ndcg: float


def parse_leaderboard(markdown: str) -> tuple[list[Point], int]:
    """Read complete rows and count rows awaiting either measurement."""
    lines = markdown.splitlines()
    try:
        start = lines.index(SECTION) + 1
    except ValueError as error:
        raise ValueError("ESCI leaderboard section not found") from error
    points = []
    pending = 0
    columns = None
    for line in lines[start:]:
        if line.startswith("## "):
            break
        if not line.startswith("|"):
            if columns is not None:
                break
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if columns is None:
            for header in HEADERS:
                if all(name in cells for name in header):
                    columns = [cells.index(name) for name in header]
                    break
            continue
        if all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        if len(cells) <= max(columns):
            raise ValueError(f"Malformed leaderboard row: {line}")
        name, latency, ndcg = (cells[index] for index in columns)
        if name.startswith("[") and "](" in name:
            name = name[1:name.index("](")]
        if any(value.casefold() in {"pending", "", "—"} for value in (latency, ndcg)):
            pending += 1
            continue
        p50 = float(latency.split("/", 1)[0])
        quality = float(ndcg)
        if not math.isfinite(p50) or p50 < 0 or not 0 <= quality <= 1:
            raise ValueError(f"Invalid latency or NDCG for {name}")
        points.append(Point(name, p50, quality))
    if columns is None or not (points or pending):
        raise ValueError("ESCI leaderboard has no readable rows")
    return points, pending


def pareto_frontier(points: list[Point]) -> list[Point]:
    """Keep points with no competitor at least as fast and as relevant."""
    return sorted(
        [
            a for a in points
            if not any(
                b.latency <= a.latency and b.ndcg >= a.ndcg
                and (b.latency < a.latency or b.ndcg > a.ndcg)
                for b in points
            )
        ],
        key=lambda point: (point.latency, point.ndcg, point.name),
    )


def padded_range(values: list[float], minimum_span: float) -> tuple[float, float]:
    low, high = min(values), max(values)
    pad = max((high - low) * 0.12, minimum_span / 2)
    return low - pad, high + pad


def render_svg(
    points: list[Point],
    pending: int,
    percentile: str = "p50",
    *,
    chart_title: str | None = None,
    x_axis_label: str | None = None,
    x_description: str | None = None,
    minimum_x_span: float = 1.0,
) -> str:
    axis_label = x_axis_label or f"End-to-end {percentile} latency (ms)"
    height = max(390, 135 + 25 * len(points))
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1060 {height}"'
        ' role="img" aria-labelledby="title desc">',
        f'<title id="title">{escape(chart_title or "ESCI relevance vs. latency")}</title>',
        f'<desc id="desc">{escape(x_description or f"{percentile} latency in milliseconds")} '
        'versus NDCG at 10.'
        ' Lower latency and higher NDCG are better.</desc>',
        '<style>text{font:14px system-ui,sans-serif;fill:#172438}'
        '.small{font-size:12px;fill:#43546b}.heading{font-size:23px;'
        'font-weight:bold}.axis{stroke:#73829a}.grid{stroke:#dfe6ee}</style>',
        '<rect width="100%" height="100%" rx="12" fill="#f6f8fb"/>',
        f'<text x="32" y="40" class="heading">'
        f'{escape(chart_title or "ESCI relevance vs. latency")}</text>',
        f'<text x="32" y="64" class="small">{len(points)} measured; '
        f'{pending} pending. Faster and more relevant is upper left.</text>',
    ]
    if not points:
        svg.append(
            '<text x="530" y="195" text-anchor="middle">'
            'Awaiting both NDCG@10 and latency measurements.</text>'
        )
        return "\n".join([*svg, "</svg>"]) + "\n"

    left, right, top, bottom = 86, 615, 94, height - 70
    x_min, x_max = padded_range([p.latency for p in points], minimum_x_span)
    y_min, y_max = padded_range([p.ndcg for p in points], 0.001)
    x_min, y_min, y_max = max(0, x_min), max(0, y_min), min(1, y_max)

    def x(latency: float) -> float:
        return left + (latency - x_min) / (x_max - x_min) * (right - left)

    def y(ndcg: float) -> float:
        return bottom - (ndcg - y_min) / (y_max - y_min) * (bottom - top)

    for step in range(6):
        fraction = step / 5
        latency = x_min + fraction * (x_max - x_min)
        ndcg = y_min + fraction * (y_max - y_min)
        px, py = x(latency), y(ndcg)
        svg.extend([
            f'<path class="grid" d="M{px:.1f} {top}V{bottom} M{left} {py:.1f}H{right}"/>',
            f'<text class="small" x="{px:.1f}" y="{bottom + 20}" '
            f'text-anchor="middle">{latency:.1f}</text>',
            f'<text class="small" x="{left - 10}" y="{py + 4:.1f}" '
            f'text-anchor="end">{ndcg:.4f}</text>',
        ])
    svg.extend([
        f'<path class="axis" fill="none" d="M{left} {top}V{bottom}H{right}"/>',
        f'<text x="{(left + right) / 2}" y="{height - 19}" '
        f'text-anchor="middle">{escape(axis_label)}</text>',
        f'<text transform="translate(24 {(top + bottom) / 2}) rotate(-90)" '
        'text-anchor="middle">NDCG@10</text>',
        '<text x="657" y="100">Variants · green = Pareto frontier</text>',
    ])
    frontier = pareto_frontier(points)
    line = " ".join(f"{x(p.latency):.1f},{y(p.ndcg):.1f}" for p in frontier)
    svg.append(f'<polyline points="{line}" fill="none" stroke="#087859" stroke-width="2"/>')
    labels: dict[tuple[float, float], list[int]] = {}
    for number, point in enumerate(points, 1):
        px, py = x(point.latency), y(point.ndcg)
        color = "#087859" if point in frontier else "#43628b"
        name = escape(point.name)
        svg.extend([
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{color}">'
            f'<title>{name}: {point.latency:g} ms, {point.ndcg:g} NDCG</title></circle>',
            f'<circle cx="664" cy="{100 + 25 * number - 4}" r="4" fill="{color}"/>',
            f'<text x="677" y="{100 + 25 * number}">{number}. {name}</text>',
        ])
        labels.setdefault((px, py), []).append(number)
    for (px, py), numbers in labels.items():
        label = ", ".join(str(number) for number in numbers)
        svg.append(f'<text class="small" x="{px + 7:.1f}" y="{py - 7:.1f}">{label}</text>')
    return "\n".join([*svg, "</svg>"]) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readme", type=Path, default=ROOT / "README.md")
    parser.add_argument("--output", type=Path, default=ROOT / "assets/esci-pareto.svg")
    args = parser.parse_args()
    try:
        points, pending = parse_leaderboard(args.readme.read_text(encoding="utf-8"))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(render_svg(points, pending), encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(f"Saved {args.output}: {len(points)} measured, {pending} pending.")


if __name__ == "__main__":
    main()
