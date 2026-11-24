"""Choosing which keyframes an evaluation runs on."""

from __future__ import annotations

from slam_for_ada.config import Config
from slam_for_ada.dataio.bag_reader import list_bags
from slam_for_ada.dataio.frame_cache import load_index


def sample_keyframes(cfg: Config, n_frames: int) -> list[tuple[str, int]]:
    """About ``n_frames`` usable keyframes, spread evenly through every bag, as (bag stem, keyframe number)."""
    bags = list_bags(cfg.path("bags"))
    picks: list[tuple[str, int]] = []
    for bag in bags:
        keys = [r["key"] for r in load_index(cfg.path("cache"), bag.stem) if r["usable"]]
        picks += [(bag.stem, k) for k in keys[:: max(1, len(keys) * len(bags) // n_frames)]]
    return picks[:n_frames]
