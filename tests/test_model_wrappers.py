"""The three model wrappers, tested with fake models: no weights, no GPU, no network."""

from __future__ import annotations

import numpy as np
import pytest

from slam_for_ada.config import StereoConfig
from slam_for_ada.datatypes import CameraPair, DepthMethod, FeatureKind, RectifiedPair
from slam_for_ada.vision.learned_stereo import DIVISIBLE_BY, OnnxStereoDepth
from slam_for_ada.vision.mono_depth import MetricMonoDepth
from slam_for_ada.vision.open_vocab import OpenVocabDetector, RawDetection, to_object_detections
from tests.helpers import REF, F, H, W


# ----------------------------------------------------------------------------------------------- learned stereo
class FakeSession:
    """Stands in for onnxruntime: records its inputs and returns a constant disparity (with a few non-positive pixels)."""

    def __init__(self, disparity: float = 60.0):
        self.disparity, self.calls, self.last_feed = disparity, 0, {}

    def run(self, _outputs, feed):
        self.calls += 1
        self.last_feed = feed
        n, _, h, w = feed["left_image"].shape
        out = np.full((n, 1, h, w), self.disparity, np.float32)
        out[0, 0, :5, :] = 0.0  # the model reports nothing usable in the top rows
        return [out]


def _pair(K) -> RectifiedPair:
    img = np.random.default_rng(0).integers(0, 255, (H, W, 3), dtype=np.uint8)
    P1 = np.hstack([K, np.zeros((3, 1))])
    P2 = P1.copy()
    P2[0, 3] = -F * 0.33
    return RectifiedPair(REF, CameraPair(0, 2), img, img, np.eye(3), np.eye(3), P1, P2, np.eye(4), 0.33, F)


def test_onnx_wrapper_pads_normalises_and_crops(K):
    sess = FakeSession(60.0)
    est = OnnxStereoDepth(StereoConfig(), K, session=sess)
    d = est.disparity(_pair(K))
    h, w = sess.last_feed["left_image"].shape[2:]
    assert h % DIVISIBLE_BY == 0 and w % DIVISIBLE_BY == 0 and h >= H and w >= W  # padded to the model's grid
    x = sess.last_feed["left_image"]
    assert x.dtype == np.float32 and -3.0 < x.min() and x.max() < 3.0  # ImageNet-normalised, not raw 0-255
    assert d.shape == (H, W) and np.isnan(d[:5]).all() and np.allclose(d[10:], 60.0)  # cropped back; invalid -> NaN


def test_onnx_wrapper_depth_matches_geometry(K):
    est = OnnxStereoDepth(StereoConfig(), K, session=FakeSession(60.0))
    depth = est(_pair(K))
    assert depth.method is DepthMethod.LEARNED
    assert np.allclose(depth.depth_m[200:400, 100:600], F * 0.33 / 60, rtol=1e-4)  # f * B / d


def test_onnx_wrapper_caches_to_disk(K, tmp_path):
    sess = FakeSession()
    est = OnnxStereoDepth(StereoConfig(), K, cache_dir=tmp_path, session=sess)
    a = est.disparity(_pair(K))
    b = est.disparity(_pair(K))
    assert sess.calls == 1 and np.array_equal(a, b, equal_nan=True)  # second call read the cache
    assert len(list(tmp_path.glob("*.npy"))) == 1


def test_onnx_wrapper_needs_a_model_or_a_session(K):
    with pytest.raises(ValueError):
        OnnxStereoDepth(StereoConfig(), K)


# ----------------------------------------------------------------------------------------------- open vocabulary
class FakeBackend:
    def __init__(self, candidates):
        self.candidates, self.mask_calls = candidates, 0

    def boxes(self, image, prompts, threshold):
        return self.candidates

    def masks(self, image, boxes):
        self.mask_calls += 1
        out = []
        for x0, y0, x1, y1 in boxes:
            m = np.zeros(image.shape[:2], bool)
            m[int(y0) : int(y1), int(x0) : int(x1)] = True
            out.append(m)
        return out


def _img():
    return np.zeros((H, W, 3), np.uint8)


def test_detector_ranks_filters_clips_and_masks():
    cands = [
        ("a wall thermostat", 0.30, (10, 10, 40, 60)),
        ("a wall thermostat", 0.60, (100, 100, 140, 160)),  # best first
        ("a wall thermostat", 0.10, (200, 200, 240, 260)),  # below the threshold
        ("a handrail", 0.50, (-20, 400, 800, 450)),  # sticks out of the image: clipped
        ("a card reader", 0.90, (50, 50, 52, 80)),
    ]  # 2 px wide: dropped
    backend = FakeBackend(cands)
    dets = OpenVocabDetector(backend).detect(_img(), ["x"], threshold=0.2)
    assert [d.prompt for d in dets] == ["a wall thermostat", "a handrail", "a wall thermostat"]
    assert dets[0].score == 0.60 and dets[1].box[0] == 0.0 and dets[1].box[2] == W - 1.0
    assert all(d.mask is not None and d.mask.any() for d in dets) and backend.mask_calls == 1


def test_detector_caps_detections_per_prompt_and_can_skip_masks():
    cands = [("a door handle", 0.9 - 0.05 * i, (10 * i, 10, 10 * i + 20, 40)) for i in range(10)]
    backend = FakeBackend(cands)
    dets = OpenVocabDetector(backend).detect(_img(), ["a door handle"], threshold=0.1, with_masks=False, max_per_prompt=4)
    assert len(dets) == 4 and all(d.mask is None for d in dets) and backend.mask_calls == 0
    assert OpenVocabDetector(FakeBackend([])).detect(_img(), ["a door handle"]) == []


def test_detections_convert_to_the_shared_contract():
    dets = [RawDetection("a handrail", 0.5, (0, 0, 10, 10)), RawDetection("an unknown prompt", 0.9, (0, 0, 5, 5))]
    out = to_object_detections(dets, REF)
    assert len(out) == 1 and out[0].kind is FeatureKind.HANDRAIL_HEIGHT


# ----------------------------------------------------------------------------------------------- monocular depth
def test_mono_wrapper_resamples_to_the_image_grid():
    low = np.full((54, 72), 2.5, np.float32)  # model output at a tenth of the resolution
    out = MetricMonoDepth(lambda image: low)(_img())
    assert out.shape == (H, W) and out.dtype == np.float32 and np.allclose(out, 2.5, atol=1e-4)


def test_mono_wrapper_rejects_a_bad_backend():
    with pytest.raises(ValueError):
        MetricMonoDepth(lambda image: np.zeros((2, 3, 4)))(_img())
