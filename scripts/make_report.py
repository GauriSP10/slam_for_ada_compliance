
"""Build the complete Level 2 ADA compliance report and figures."""

from __future__ import annotations

import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from math import sqrt

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs"
RESULTS = OUTPUT / "level2_results.csv"
ERRORS = OUTPUT / "errors_by_method.csv"
ABLATION = OUTPUT / "ablation.csv"
ABLATION_MD = OUTPUT / "ablation.md"
FAILURES = OUTPUT / "failure_analysis.md"
REPORT = OUTPUT / "report.md"
FIGURES = OUTPUT / "figures"

def load_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Required file missing: {path}")

    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def run_script(path: Path) -> None:
    subprocess.run(
        [sys.executable, str(path)],
        cwd=ROOT,
        check=True,
    )

def generate_dependencies() -> None:
    """Regenerate existing analysis outputs."""
    if not RESULTS.is_file():
        raise FileNotFoundError(
            "Level 2 results are missing. Run "
            "'python scripts/run_level2.py' first."
        )

    run_script(ROOT / "scripts" / "make_ablation.py")

    from slam_for_ada.eval.report import write_failure_analysis

    write_failure_analysis(
        results_csv=RESULTS,
        output_md=FAILURES,
    )

def calculate_stats(rows: list[dict]) -> dict:
    errors = [float(row["error_in"]) for row in rows]
    absolute = [abs(error) for error in errors]

    scored = [
        row for row in rows
        if row["decision_correct"].strip().lower()
        in {"yes", "no", "true", "false", "1", "0"}
    ]
    correct = sum(
        row["decision_correct"].strip().lower()
        in {"yes", "true", "1"}
        for row in scored
    )

    return {
        "n": len(rows),
        "mae": mean(absolute),
        "median": median(absolute),
        "maximum": max(absolute),
        "rmse": sqrt(mean([error**2 for error in errors])),
        "scored": len(scored),
        "correct": correct,
    }

def create_error_plot(rows: list[dict]) -> Path:
    """Plot signed measurement errors for individual estimates."""
    labels = [
        f"{row['feature']} ({row['method']})"
        for row in rows
    ]
    errors = [float(row["error_in"]) for row in rows]

    fig, ax = plt.subplots(figsize=(11, max(6, len(rows) * 0.48)))
    positions = list(range(len(rows)))

    ax.barh(positions, errors)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Signed measurement error (inches)")
    ax.set_title("Level 2 Measurement Errors")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()

    path = FIGURES / "measurement_errors.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path

def create_method_plot(rows: list[dict]) -> Path:
    """Compare mean absolute errors by measurement method."""
    groups = defaultdict(list)

    for row in rows:
        groups[row["method"]].append(abs(float(row["error_in"])))

    methods = sorted(groups)
    values = [mean(groups[method]) for method in methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(methods, values)
    ax.set_ylabel("Mean absolute error (inches)")
    ax.set_title("Error by Measurement Method")
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    path = FIGURES / "errors_by_measurement_method.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path

def create_ablation_plot(rows: list[dict]) -> Path:
    """Plot documented ablations without inventing missing error bars."""
    labels = [
        f"{row['comparison']}: {row['variant']}"
        for row in rows
    ]
    values = [
        float(row["mean_abs_error_in"])
        for row in rows
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    positions = list(range(len(rows)))

    ax.barh(positions, values)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Reported mean absolute error (inches)")
    ax.set_title("Documented Level 2 Ablation Results")
    ax.grid(axis="x", alpha=0.25)

    fig.tight_layout()

    path = FIGURES / "ablation_comparison.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path

def build_report(
    measurements: list[dict],
    ablations: list[dict],
) -> str:
    stats = calculate_stats(measurements)

    lines = [
        "# Level 2: Stereo Visual SLAM for ADA Compliance",
        "",
        "## Executive Summary",
        "",
        "This report summarizes the available Level 2 measurements, "
        "ADA decisions, documented ablation experiments, and "
        "failure analysis.",
        "",
        "## Measurement Performance",
        "",
        "| Metric | Result |",
        "|---|---:|",
        f"| Number of measurement estimates | {stats['n']} |",
        f"| Mean absolute error | {stats['mae']:.3f} in |",
        f"| Median absolute error | {stats['median']:.3f} in |",
        f"| Maximum absolute error | {stats['maximum']:.3f} in |",
        f"| RMSE | {stats['rmse']:.3f} in |",
        f"| Correct ADA decisions | "
        f"{stats['correct']} / {stats['scored']} |",
        "",
        "These values were calculated from the local "
        "`outputs/level2_results.csv` file.",
        "",
        "## Individual Measurement Errors",
        "",
        "![Measurement errors](figures/measurement_errors.png)",
        "",
        "## Error by Measurement Method",
        "",
        "![Errors by method](figures/errors_by_measurement_method.png)",
        "",
        "## Ablation Analysis",
        "",
        "![Ablation comparisons](figures/ablation_comparison.png)",
        "",
        "The ablation measurements include experiments documented "
        "by the team. They were not all independently reproduced "
        "in this Ubuntu environment.",
        "",
        "Different variants may have different measurement "
        "populations, so their overall MAEs should not be "
        "interpreted as paired comparisons.",
        "",
        "| Comparison | Variant | n | MAE (in) | "
        "Error bar (in) |",
        "|---|---|---:|---:|---:|",
    ]

    for row in ablations:
        error_bar = row.get("error_bar_in", "")
        lines.append(
            f"| {row['comparison']} | {row['variant']} | "
            f"{row.get('n') or 'N/A'} | "
            f"{float(row['mean_abs_error_in']):.3f} | "
            f"{float(error_bar):.3f} "
            f"|" if error_bar else
            f"| {row['comparison']} | {row['variant']} | "
            f"{row.get('n') or 'N/A'} | "
            f"{float(row['mean_abs_error_in']):.3f} | N/A |"
        )

    lines.extend([
        "",
        "The available SGBM error bar represents the sample "
        "standard deviation of absolute measurement errors, "
        "not a confidence interval. Unavailable comparable "
        "error bars are marked N/A.",
        "",
        "## Failure Analysis",
        "",
        "See [Failure Analysis](failure_analysis.md) for "
        "individual problematic measurements, documented "
        "quality-gate refusals, and limitations.",
        "",
        "## Sources and Reproducibility",
        "",
        "- `outputs/level2_results.csv`: local measurements",
        "- `outputs/errors_by_method.csv`: grouped error statistics",
        "- `outputs/ablation.csv`: ablation comparisons",
        "- `outputs/ablation.md`: detailed ablation interpretation",
        "- `outputs/failure_analysis.md`: failure cases",
        "- `docs/LEVEL2_VISION_RESULTS.md`: team experiments",
        "- `data/ground_truth/ground_truth.yaml`: tape ground truth",
        "",
        "The report was generated using:",
        "",
        "```bash",
        "python scripts/make_report.py",
        "```",
        "",
        "## Limitations",
        "",
        "- Only features with available estimates were scored.",
        "- Some features were not annotated or were not visible "
        "in usable frames.",
        "- Some stereo and detection experiments are documented "
        "results rather than locally reproduced runs.",
        "- Automatic object recognition is incomplete.",
        "- Missing experimental uncertainty values were "
        "not fabricated.",
        "",
    ])

    return "\n".join(lines) + "\n"

def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    generate_dependencies()

    measurements = load_csv(RESULTS)
    ablations = load_csv(ABLATION)

    if not measurements:
        raise ValueError("The measurement results CSV is empty.")

    create_error_plot(measurements)
    create_method_plot(measurements)
    create_ablation_plot(ablations)

    REPORT.write_text(
        build_report(measurements, ablations),
        encoding="utf-8",
    )

    print(f"Created {REPORT}")
    print(f"Created {FAILURES}")
    print(f"Created {ABLATION}")
    print(f"Created {FIGURES / 'measurement_errors.png'}")
    print(f"Created {FIGURES / 'errors_by_measurement_method.png'}")
    print(f"Created {FIGURES / 'ablation_comparison.png'}")

if __name__ == "__main__":
    main()
