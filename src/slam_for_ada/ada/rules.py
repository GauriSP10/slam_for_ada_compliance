"""Load the ADA thresholds (config/ada_rules.yaml) and find the rule that applies to a feature."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import yaml

from slam_for_ada.datatypes import AdaRule, FeatureKind, Verification


def load_rules(path: Path) -> list[AdaRule]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["rules"]
    rules = [
        AdaRule(
            id=r["id"],
            kind=FeatureKind(r["kind"]),
            applies_to=tuple(r.get("applies_to", ())),
            min_in=r.get("min_in"),
            max_in=r.get("max_in"),
            borderline_sigmas=r.get("borderline_sigmas", 2.0),
            citation=r.get("citation", ""),
            verification=Verification(r["verification"]),
            note=str(r.get("note", "")).strip(),
        )
        for r in raw
    ]
    if len({r.id for r in rules}) != len(rules):
        raise ValueError("duplicate rule ids")
    return rules


def rule_for(feature_id: str, rules: Sequence[AdaRule]) -> AdaRule | None:
    """The rule whose ``applies_to`` lists this ground-truth id, if any."""
    for r in rules:
        if feature_id in r.applies_to:
            return r
    return None
