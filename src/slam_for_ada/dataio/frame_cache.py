"""Extract keyframes from the bags once and cache them, so the 10 GB bags are read a single time.

Layout:  <cache>/frames/<bag>/<index:05d>_cam<k>.png   one PNG per camera
         <cache>/frames/<bag>/index.json               one record per keyframe (stamps, quality)
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import cv2

from slam_for_ada.config import CacheConfig
from slam_for_ada.dataio.bag_reader import CAMERA_IDS, iter_camera_sets
from slam_for_ada.datatypes import CameraSet, FrameRef
from slam_for_ada.vision.frame_quality import PersonDetector, assess, set_sharpness


def _bag_dir(cache_root: Path, bag: str) -> Path:
    return Path(cache_root) / "frames" / bag


def select_keyframes(bag_path: Path, cfg: CacheConfig) -> Iterator[tuple[CameraSet, dict[int, float]]]:
    """From each window of ``keyframe_window`` consecutive camera sets, keep the sharpest one.

    Walking blur is the main enemy of corner and edge detection, so picking the sharpest set per
    half second beats taking every Nth set.
    """
    best: tuple[float, CameraSet, dict[int, float]] | None = None
    in_window = 0
    for cam_set in iter_camera_sets(bag_path, max_spread_ns=int(cfg.max_sync_spread_ms * 1e6)):
        sharp = set_sharpness(cam_set, cfg.sharpness_scale)
        score = min(sharp.values())
        if best is None or score > best[0]:
            best = (score, cam_set, sharp)
        in_window += 1
        if in_window == cfg.keyframe_window:
            yield best[1], best[2]
            best, in_window = None, 0
    if best is not None:
        yield best[1], best[2]


def reassess(records: list[dict], cfg: CacheConfig) -> list[dict]:
    """Recompute ``usable`` / ``reasons`` from the stored sharpness and person fractions.

    Cheap, so thresholds can be tuned without touching the bags or the PNGs.
    """
    import numpy as np

    ref = float(np.median([min(r["sharpness"].values()) for r in records])) if records else 0.0
    out = []
    for r in records:
        q = assess(None, {int(c): v for c, v in r["sharpness"].items()}, r["person_fraction"], cfg, ref)
        out.append({**r, "usable": q.usable, "reasons": list(q.reasons)})
    return out


def extract_bag(bag_path: Path, cache_root: Path, cfg: CacheConfig, detector: PersonDetector, overwrite: bool = False) -> list[dict]:
    """Write one bag's keyframes and index.json. Returns the index records.

    If the cache exists it is not re-extracted, but the usable/blur decision is re-evaluated
    against the current config.
    """
    bag_path = Path(bag_path)
    out = _bag_dir(cache_root, bag_path.stem)
    index_file = out / "index.json"
    if index_file.exists() and not overwrite:
        records = reassess(json.loads(index_file.read_text()), cfg)
        index_file.write_text(json.dumps(records, indent=1))
        return records
    out.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for k, (cam_set, sharp) in enumerate(select_keyframes(bag_path, cfg)):
        frac = float(detector.mask(cam_set.images[0]).mean())
        for cam, img in cam_set.images.items():
            cv2.imwrite(str(out / f"{k:05d}_cam{cam}.png"), img, [cv2.IMWRITE_PNG_COMPRESSION, cfg.png_compress])
        records.append(
            {
                "key": k,
                "set_index": cam_set.ref.index,
                "t_ns": cam_set.ref.t_ns,
                "stamps_ns": {str(c): t for c, t in cam_set.stamps_ns.items()},
                "sharpness": {str(c): v for c, v in sharp.items()},
                "person_fraction": frac,
                "usable": True,
                "reasons": [],
            }
        )
    records = reassess(records, cfg)
    index_file.write_text(json.dumps(records, indent=1))
    return records


def load_index(cache_root: Path, bag: str) -> list[dict]:
    """The keyframe records for one bag."""
    return json.loads((_bag_dir(cache_root, bag) / "index.json").read_text())


def load_camera_set(cache_root: Path, bag: str, key: int, cameras=CAMERA_IDS) -> CameraSet:
    """Read one cached keyframe back as a CameraSet. ``key`` is the keyframe number, not the raw set index."""
    rec = load_index(cache_root, bag)[key]
    d = _bag_dir(cache_root, bag)
    images = {}
    for c in cameras:
        path = d / f"{key:05d}_cam{c}.png"
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"missing or unreadable cached frame: {path}")
        images[c] = img
    stamps = {c: int(rec["stamps_ns"][str(c)]) for c in cameras}
    return CameraSet(FrameRef(bag, key, int(rec["t_ns"])), images, stamps)
