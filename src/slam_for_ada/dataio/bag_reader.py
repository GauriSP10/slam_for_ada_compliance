"""Read the five front cameras and the IMU from ROS1 .bag files without ROS (rosbags library)."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from rosbags.highlevel import AnyReader

from slam_for_ada.datatypes import CameraId, CameraSet, FrameRef, ImageBGR, ImuSample

CAMERA_IDS: tuple[CameraId, ...] = (0, 1, 2, 3, 4)
IMU_TOPIC = "/imu/imu_compensated"


def camera_topic(cam: CameraId) -> str:
    return f"/camera_array/cam{cam}/image_raw"


def list_bags(bag_dir: Path) -> list[Path]:
    """All .bag files in a directory, in recording order."""
    return sorted(Path(bag_dir).glob("*.bag"))


def stamp_ns(msg) -> int:
    """ROS header stamp as integer nanoseconds."""
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


def decode_bgr(msg) -> ImageBGR:
    """sensor_msgs/Image (bgr8) -> contiguous (H, W, 3) uint8 array. Copies, so the bag buffer can be freed."""
    if msg.encoding != "bgr8":
        raise ValueError(f"expected bgr8 images, got {msg.encoding}")
    rows = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    return np.ascontiguousarray(rows[:, : msg.width * 3].reshape(msg.height, msg.width, 3))


def iter_camera_sets(bag_path: Path, cameras: Sequence[CameraId] = CAMERA_IDS, max_spread_ns: int = 5_000_000) -> Iterator[CameraSet]:
    """Yield time-matched camera sets in order.

    The cameras are hardware-synchronised (stamps agree to about 1.6 ms), but messages arrive
    interleaved and a camera can drop a frame. We keep one queue per camera; whenever every
    queue has a head we compare the head stamps. Heads within ``max_spread_ns`` form a set;
    otherwise the oldest head is a stray from a dropped frame and is discarded.
    """
    bag_path = Path(bag_path)
    topics = {camera_topic(c): c for c in cameras}
    queues: dict[CameraId, deque[tuple[int, ImageBGR]]] = {c: deque() for c in cameras}
    index = 0
    with AnyReader([bag_path]) as reader:
        conns = [c for c in reader.connections if c.topic in topics]
        for conn, _t, raw in reader.messages(connections=conns):
            cam = topics[conn.topic]
            msg = reader.deserialize(raw, conn.msgtype)
            queues[cam].append((stamp_ns(msg), decode_bgr(msg)))
            while all(queues[c] for c in cameras):
                heads = {c: queues[c][0][0] for c in cameras}
                if max(heads.values()) - min(heads.values()) <= max_spread_ns:
                    items = {c: queues[c].popleft() for c in cameras}
                    ref = FrameRef(bag_path.stem, index, items[cameras[0]][0])
                    yield CameraSet(ref, {c: items[c][1] for c in cameras}, {c: items[c][0] for c in cameras})
                    index += 1
                else:
                    queues[min(heads, key=lambda c: heads[c])].popleft()


def iter_imu(bag_path: Path, topic: str = IMU_TOPIC) -> Iterator[ImuSample]:
    """Yield IMU samples in time order."""
    with AnyReader([Path(bag_path)]) as reader:
        conns = [c for c in reader.connections if c.topic == topic]
        for conn, _t, raw in reader.messages(connections=conns):
            m: Any = reader.deserialize(raw, conn.msgtype)
            a, w = m.linear_acceleration, m.angular_velocity
            yield ImuSample(stamp_ns(m), (float(a.x), float(a.y), float(a.z)), (float(w.x), float(w.y), float(w.z)))
