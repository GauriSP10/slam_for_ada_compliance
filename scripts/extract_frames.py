"""CLI: read the bags once and fill data/cache with keyframes.  Usage: python scripts/extract_frames.py [bag_stem ...]"""

from __future__ import annotations

import sys
import time

from slam_for_ada.config import load_config
from slam_for_ada.dataio.bag_reader import list_bags
from slam_for_ada.dataio.frame_cache import extract_bag
from slam_for_ada.vision.frame_quality import PersonDetector


def main(argv: list[str]) -> None:
    cfg = load_config()
    detector = PersonDetector(cfg.path("models"), cfg.cache.person_model, cfg.cache.person_conf)
    wanted = set(argv)
    for bag in list_bags(cfg.path("bags")):
        if wanted and bag.stem not in wanted and not any(w in bag.stem for w in wanted):
            continue
        t0 = time.time()
        recs = extract_bag(bag, cfg.path("cache"), cfg.cache, detector)
        usable = sum(r["usable"] for r in recs)
        print(f"{bag.stem}: {len(recs)} keyframes, {usable} usable, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
