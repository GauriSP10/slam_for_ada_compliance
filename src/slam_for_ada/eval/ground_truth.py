"""Load the typed-up tape-measure ground truth (data/ground_truth/ground_truth.yaml)."""

from __future__ import annotations

from pathlib import Path

import yaml

from slam_for_ada.datatypes import FeatureKind, GroundTruthFeature


def load_ground_truth(path: Path) -> dict[str, GroundTruthFeature]:
    """Ground-truth features keyed by id."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["features"]
    out = {}
    for f in raw:
        gt = GroundTruthFeature(
            f["id"],
            FeatureKind(f["kind"]),
            float(f["value_in"]),
            bool(f["confirmed"]),
            f["photo"],
            f.get("description", ""),
            tuple(f.get("bags", ())),
            f.get("note", ""),
        )
        if gt.id in out:
            raise ValueError(f"duplicate ground-truth id {gt.id}")
        out[gt.id] = gt
    return out
