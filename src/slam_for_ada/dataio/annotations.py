"""Load the hand-annotated feature locations (data/annotations/rois.yaml)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from slam_for_ada.dataio.bag_reader import list_bags
from slam_for_ada.datatypes import FeatureKind


@dataclass(frozen=True)
class Annotation:
    """One observation of one ground-truth feature in one keyframe, with its pixel geometry."""

    id: str
    feature_id: str
    kind: FeatureKind
    bag: str  # full bag stem
    key: int  # keyframe number in the cache
    geometry: Mapping[str, Any]  # everything else from the YAML entry (jambs, boxes, points, ...)
    note: str = ""


def load_annotations(path: Path, bag_dir: Path) -> list[Annotation]:
    """Read the YAML and resolve the short bag suffix ("32") to the full bag stem."""
    stems = [b.stem for b in list_bags(bag_dir)]
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["annotations"]
    out, seen = [], set()
    for r in raw:
        if r["id"] in seen:
            raise ValueError(f"duplicate annotation id {r['id']}")
        seen.add(r["id"])
        suffix = str(r["bag"])
        matches = [s for s in stems if s.endswith("_" + suffix) or s == suffix]
        if len(matches) != 1:
            raise ValueError(f"{r['id']}: bag '{suffix}' matches {matches}")
        geom = {k: v for k, v in r.items() if k not in ("id", "feature_id", "kind", "bag", "key", "note")}
        out.append(Annotation(r["id"], r["feature_id"], FeatureKind(r["kind"]), matches[0], int(r["key"]), geom, r.get("note", "")))
    return out
