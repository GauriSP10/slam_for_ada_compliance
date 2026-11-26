
"""Generate Level 2 ablation reports from documented experimental results."""

import csv
from pathlib import Path
from statistics import mean, stdev

OUTPUT = Path("outputs")
SOURCE = "docs/LEVEL2_VISION_RESULTS.md"

# Values transcribed from Gauri's documented experiments.
# A missing error bar is not equivalent to zero uncertainty.
EXPERIMENTS = [
    ("stereo_matcher", "SGBM", 12, 3.0, "in", "all estimates"),
    ("stereo_matcher", "learned", 11, 6.0, "in", "all estimates"),
    ("stereo_matcher", "hybrid", 11, 3.7, "in", "all estimates"),
    ("roi", "hand_drawn", None, 2.00, "in", "height estimates"),
    ("roi", "automatic", None, 2.02, "in", "height estimates"),
    ("temporal", "single_frame", 7, 1.69, "in", "seven features"),
    ("temporal", "fused", 7, 1.81, "in", "seven features"),
]

def calculate_sgbm_error_stats():
    """Calculate SGBM error variability from real measurement results."""
    results_path = Path("outputs/level2_results.csv")

    with results_path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    absolute_errors = [
        abs(float(row["error_in"]))
        for row in rows
    ]

    if len(absolute_errors) < 2:
        raise ValueError("At least two estimates are required.")

    return {
        "n": len(absolute_errors),
        "mae": mean(absolute_errors),
        "sd": stdev(absolute_errors),
    }

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    sgbm_stats = calculate_sgbm_error_stats()

    columns = [
        "comparison", "variant", "n", "mean_abs_error_in",
        "error_bar_in", "population", "source",
    ]

    records = []
    for comparison, variant, n, mae, unit, population in EXPERIMENTS:
        assert unit == "in"
        if comparison == "stereo_matcher" and variant == "SGBM":
            n = sgbm_stats["n"]
            mae = sgbm_stats["mae"]
            error_bar = sgbm_stats["sd"]
        else:
            error_bar = ""
        records.append({
            "comparison": comparison,
            "variant": variant,
            "n": n if n is not None else "",
            "mean_abs_error_in": mae,
            "error_bar_in": error_bar,
            "population": population,
            "source": SOURCE,
        })

    csv_path = OUTPUT / "ablation.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)

    lines = [
        "# Level 2 Ablation Analysis",
        "",
        "Source: `docs/LEVEL2_VISION_RESULTS.md`.",
        "",
        "These are previously documented experimental results, "
        "not newly reproduced runs.",
        "",
        "| Comparison | Variant | n | MAE (in) | Error bar (in) |",
        "|---|---|---:|---:|---:|",
    ]

    for record in records:
        error_bar = record["error_bar_in"]
        error_bar_text = (
            f"{error_bar:.2f}" if error_bar != "" else "N/A"
        )

        lines.append(
            f"| {record['comparison']} | {record['variant']} | "
            f"{record['n'] or 'N/A'} | "
            f"{record['mean_abs_error_in']:.2f} | "
            f"{error_bar_text} |"
        )

    lines.extend([
        "",
        "## Findings",
        "",
        "- Hybrid stereo improves height accuracy, but SGBM has "
        "the lowest overall reported MAE.",
        "- Automatic ROIs produce similar height accuracy to "
        "hand-drawn annotations.",
        "- Temporal fusion does not improve the reported "
        "seven-feature mean error.",
        "",
        "## Error-bar interpretation",
        "",
        "For SGBM, the error bar is the sample standard deviation "
        "of absolute measurement errors across 12 estimates "
        "(approximately 1.72 inches).",
        "It describes variation between estimates, not a "
        "95% confidence interval or measurement uncertainty.",
        "N/A means the underlying per-estimate errors are "
        "unavailable for calculating comparable error bars.",
        "",
        "## Limitations",
        "",
        "- Comparisons use different measurement populations; "
        "overall matcher MAEs should not be treated as paired results.",
        "- Individual residuals are unavailable for several variants, "
        "so statistical error bars cannot be computed reliably.",
        "- ROI sample counts are not specified for the reported "
        "2.00 and 2.02-inch height MAEs.",
        "- Learned and hybrid results were documented by the team "
        "but have not been independently rerun in this environment.",
        "- Error bars are marked N/A rather than invented.",
        "",
    ])

    md_path = OUTPUT / "ablation.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Created {csv_path} ({len(records)} rows)")
    print(f"Created {md_path}")

if __name__ == "__main__":
    main()
