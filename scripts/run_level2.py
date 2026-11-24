"""CLI: run the whole Level 2 vision pipeline, score it against the tape measurements, write results to outputs/.

Usage: python scripts/run_level2.py [--stereo sgbm|learned|hybrid] [--auto]
  --stereo  the matcher used for measurements (learned is slow, about 14 s per camera pair per frame, but cached)
  --auto    measure height features from open-vocabulary detection masks instead of hand-drawn depth boxes
"""

from __future__ import annotations

import sys
import warnings

from slam_for_ada.ada.rules import load_rules
from slam_for_ada.config import load_config
from slam_for_ada.eval.ground_truth import load_ground_truth
from slam_for_ada.eval.minimal_scorer import format_table, score, write_csv
from slam_for_ada.pipeline import run_vision


def main(argv: list[str]) -> None:
    warnings.filterwarnings("ignore")
    method = argv[argv.index("--stereo") + 1] if "--stereo" in argv else None
    auto = "--auto" in argv
    cfg = load_config()
    measurements, estimates, floor, stats = run_vision(cfg, method, auto=auto)
    tag = (f"_{method}" if method and method != "sgbm" else "") + ("_auto" if auto else "")
    print(f"stereo: {method or cfg.stereo.method}; floor: camera height {floor.d:.3f} m, {floor.source}")
    if auto:
        print(
            f"height features: {stats.from_detection} measured from a detection mask, {stats.from_fallback} from the hand-drawn "
            f"fallback, {stats.unmeasured} not measurable (of {stats.total})"
        )
    print(f"{len(measurements)} measurements -> {len(estimates)} estimates\n")
    results = score(estimates, load_ground_truth(cfg.path("ground_truth")), load_rules(cfg.path("rules")))
    print(format_table(results))
    out = cfg.path("outputs")
    write_csv(results, out / f"level2_results{tag}.csv")
    (out / f"level2_results{tag}.md").write_text(format_table(results), encoding="utf-8")
    print(f"\nwrote {out / f'level2_results{tag}.csv'}")


if __name__ == "__main__":
    main(sys.argv[1:])
