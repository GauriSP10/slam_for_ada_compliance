
"""Failure analysis for Level 2 ADA compliance measurements."""

from __future__ import annotations

import csv
from pathlib import Path

RESULTS_CSV = Path("outputs/level2_results.csv")
OPEN_VOCAB_CSV = Path("outputs/open_vocab_results.csv")
OUTPUT_MD = Path("outputs/failure_analysis.md")
VISION_NOTES = Path("docs/LEVEL2_VISION_RESULTS.md")
ROI_NOTES = Path("data/annotations/rois.yaml")

def load_measurement_results(
    path: Path = RESULTS_CSV,
) -> list[dict]:
    """Load actual measurement results and validate required columns."""
    if not path.is_file():
        raise FileNotFoundError(f"Measurement results not found: {path}")

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = set(reader.fieldnames or [])
        rows = list(reader)

    required = {
        "feature",
        "method",
        "error_in",
        "measured_verdict",
        "tape_verdict",
        "decision_correct",
    }

    missing = required - columns
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")

    if not rows:
        raise ValueError("The measurement results CSV is empty.")

    return rows

def identify_problematic_measurements(
    rows: list[dict],
    error_threshold_in: float = 3.0,
) -> list[dict]:
    """Flag borderline, incorrect, and high-error measurements."""
    problems = []

    for row in rows:
        error = abs(float(row["error_in"]))
        verdict = row["measured_verdict"].strip().lower()
        correct = row["decision_correct"].strip().lower()

        reasons = []

        if verdict == "borderline":
            reasons.append("Borderline ADA decision")

        if correct in {"no", "false", "0"}:
            reasons.append("ADA verdict differs from tape verdict")

        if error >= error_threshold_in:
            reasons.append(
                f"Absolute measurement error >= {error_threshold_in:g} in"
            )

        if reasons:
            problems.append(
                {
                    "feature": row["feature"],
                    "method": row["method"],
                    "abs_error_in": error,
                    "measured_verdict": row["measured_verdict"],
                    "tape_verdict": row["tape_verdict"],
                    "reasons": reasons,
                }
            )

    return problems

def load_open_vocab_results(
    path: Path = OPEN_VOCAB_CSV,
) -> tuple[list[dict] | None, list[str]]:
    """Read optional detection evidence without assuming its schema."""
    if not path.is_file():
        return None, []

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        rows = list(reader)

    return rows, columns

def build_failure_report(rows: list[dict]) -> str:
    """Create a Markdown report from measured and documented evidence."""
    problems = identify_problematic_measurements(rows)

    borderline_count = sum(
        row["measured_verdict"].strip().lower() == "borderline"
        for row in rows
    )

    scored = [
        row for row in rows
        if row["decision_correct"].strip().lower()
        in {"yes", "no", "true", "false", "1", "0"}
    ]

    correct_count = sum(
        row["decision_correct"].strip().lower()
        in {"yes", "true", "1"}
        for row in scored
    )

    confidently_wrong = [
        row for row in rows
        if row["decision_correct"].strip().lower()
        in {"no", "false", "0"}
        and row["measured_verdict"].strip().lower()
        in {"pass", "fail"}
        and row["tape_verdict"].strip().lower()
        in {"pass", "fail"}
    ]

    lines = [
        "# Level 2 Failure Analysis",
        "",
        "## Evidence and scope",
        "",
        f"- Measured estimates in local CSV: **{len(rows)}**",
        f"- Scored ADA decisions: **{len(scored)}**",
        f"- Matching ADA decisions: **{correct_count}**",
        f"- Borderline measured verdicts: **{borderline_count}**",
        f"- Confidently wrong PASS/FAIL decisions: "
        f"**{len(confidently_wrong)}**",
        "",
        "Sources:",
        "",
        "- `outputs/level2_results.csv` (local measurements)",
        "- `docs/LEVEL2_VISION_RESULTS.md` (team experiment notes)",
        "- `data/annotations/rois.yaml` (ROI annotations)",
        "",
        "A borderline result is not a confidently wrong PASS/FAIL "
        "decision. A large numerical error does not by itself "
        "prove an ADA classification failure.",
        "",
        "## Measurements requiring investigation",
        "",
        "Flagging rule: borderline verdict, incorrect ADA verdict, "
        "or absolute error of at least 3 inches.",
        "",
        "| Feature | Method | Absolute error (in) | "
        "Measured verdict | Tape verdict | Reason |",
        "|---|---|---:|---|---|---|",
    ]

    for problem in problems:
        lines.append(
            f"| {problem['feature']} | "
            f"{problem['method']} | "
            f"{problem['abs_error_in']:.2f} | "
            f"{problem['measured_verdict']} | "
            f"{problem['tape_verdict']} | "
            f"{'; '.join(problem['reasons'])} |"
        )

    if not problems:
        lines.append("| None identified | - | - | - | - | - |")

    lines.extend([
        "",
        "## Documented refusals and failure mechanisms",
        "",
        "The following cases are documented in the team's "
        "Level 2 vision results. They are not new experiments "
        "performed by this report generator.",
        "",
        "| Case | Documented observation | Interpretation / mitigation |",
        "|---|---|---|",
        "| Frame quality | 9 keyframes rejected for blur and "
        "29 for people | Quality filter avoids poor or "
        "obstructed measurements |",
        "| Floor fitting | 5 of 60 frames fitted another "
        "surface | Reject inconsistent floor planes |",
        "| Glass sidelight | Depth matched a background "
        "at about 12 m rather than 1.5 m | Reject scattered "
        "or cross-camera-inconsistent depth |",
        "| Oblique door | An implausible 69-inch width "
        "was produced before quality gating | Reject "
        "grazing geometry and widths outside 20-60 in |",
        "| Door-edge stereo | Depth is unreliable near "
        "vertical depth discontinuities | Direct stereo "
        "widths remain less stable |",
        "| Hallway geometry | Floor homography "
        "overestimated the curved hallway | Straight-line "
        "boundary approximations introduce bias |",
        "| Entrance door detector | Dark leaf edges "
        "were weak; the outer frame was selected | "
        "Automatic door detection is incomplete |",
        "| Learned depth / glass handrail | A plausible "
        "but incorrect thin-rail depth was produced | "
        "Hybrid SGBM consistency checks helped |",
        "| Temporal fusion | Handrail point tracking "
        "drifted, with approximately 9-inch scatter | "
        "Thin structures are difficult to track |",
        "",
        "These documented cases are not all incorrect ADA "
        "classifications. Some were refused by quality "
        "gates before producing a final measurement.",
        "",
        "## Possible sources of measurement error",
        "",
        "- Door-width tape measurements use the full opening; "
        "some vision measurements refer to leaf edges. "
        "The measurement bases are not identical.",
        "- Several heights have uncertain measurement "
        "reference points, such as the center versus "
        "top or bottom of a device.",
        "- Strongly oblique views and glass or thin "
        "structures can cause poor depth estimates.",
        "- Most measured features have only one observation; "
        "single-frame estimates are fragile.",
        "- Feature annotations are mostly manual. "
        "A missing annotation is not evidence that "
        "the measurement system attempted and failed.",
        "",
        "## Open-vocabulary detection evidence",
        "",
    ])

    vocab_rows, vocab_columns = load_open_vocab_results()

    if vocab_rows is None:
        lines.extend([
            "`outputs/open_vocab_results.csv` is unavailable locally.",
            "",
            "The team's vision-results document contains "
            "open-vocabulary findings, but individual "
            "detection failures cannot be independently "
            "reconstructed from a missing local CSV.",
        ])
    else:
        lines.extend([
            f"- Local open-vocabulary CSV rows: **{len(vocab_rows)}**",
            "- Available columns: "
            + ", ".join(f"`{column}`" for column in vocab_columns),
            "",
            "The available schema has not been assumed to "
            "represent successful or failed detections. "
            "A detailed failure count requires verified "
            "labels or outcome fields.",
        ])

    lines.extend([
        "",
        "## Limitations",
        "",
        "- This report analyzes the local Level 2 results "
        "and team-documented experimental observations.",
        "- Not all 23 ground-truth features have measured "
        "estimates in the local results CSV.",
        "- Quality-gate refusal counts are included only "
        "where explicitly documented; unavailable counts "
        "are not inferred.",
        "- The causes listed above describe documented "
        "mechanisms, not necessarily proven causes of "
        "each individual CSV measurement error.",
        "- Some Tier 2 experiments were conducted by "
        "teammates and were not independently reproduced "
        "in this Ubuntu environment.",
        "",
    ])

    return "\n".join(lines) + "\n"

def write_failure_analysis(
    results_csv: Path = RESULTS_CSV,
    output_md: Path = OUTPUT_MD,
) -> Path:
    """Generate the required failure-analysis Markdown file."""
    rows = load_measurement_results(results_csv)
    report = build_failure_report(rows)

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(report, encoding="utf-8")

    return output_md

def main() -> None:
    output = write_failure_analysis()
    print(f"Created {output}")

if __name__ == "__main__":
    main()
