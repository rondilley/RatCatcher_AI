"""Tests for the native-resolution detection windows.

The geometry is pure, so these run anywhere.  The point they defend is
that a window keeps its full size: the whole reason for cutting one is
that rescaling destroys small animals, so a window that gets clipped to a
sliver at a frame edge, or shrunk to a bounding box, would give back the
thing it was built to avoid.
"""

from __future__ import annotations

import numpy as np

from datetime import datetime

from ratcatcher.detection.roi_crop import plan_windows
from ratcatcher.motion.detector import MotionRegion


def _region(x, y, w, h):
    return MotionRegion(x=x, y=y, width=w, height=h, area=(w * h) / (4056 * 3040))


def test_no_regions_yields_no_windows():
    assert plan_windows([], 4056, 3040, window=640) == []


def test_small_region_gets_a_full_size_window_centred_on_it():
    win = plan_windows([_region(2000, 1500, 40, 30)], 4056, 3040, window=640)
    assert len(win) == 1
    w = win[0]
    assert (w.width, w.height) == (640, 640)
    # Centre of the window sits on the centre of the motion.
    assert abs((w.x + w.width / 2) - 2020) <= 1
    assert abs((w.y + w.height / 2) - 1515) <= 1


def test_window_at_the_frame_edge_slides_in_rather_than_shrinking():
    """A finch in the corner must still get 640x640, not a sliver."""
    for region in (_region(0, 0, 30, 30), _region(4026, 3010, 30, 30)):
        win = plan_windows([region], 4056, 3040, window=640)[0]
        assert (win.width, win.height) == (640, 640)
        assert 0 <= win.x <= 4056 - 640
        assert 0 <= win.y <= 3040 - 640


def test_overlapping_regions_merge_into_one_window():
    """Two contours on one animal must not spend two inferences."""
    regions = [_region(2000, 1500, 100, 100), _region(2050, 1550, 100, 100)]
    assert len(plan_windows(regions, 4056, 3040, window=640, max_windows=2)) == 1


def test_separate_regions_get_separate_windows_up_to_the_cap():
    regions = [
        _region(200, 200, 60, 60),
        _region(3000, 2000, 60, 60),
        _region(1500, 900, 60, 60),
    ]
    assert len(plan_windows(regions, 4056, 3040, window=640, max_windows=2)) == 2
    assert len(plan_windows(regions, 4056, 3040, window=640, max_windows=3)) == 3


def test_largest_motion_wins_when_the_cap_bites():
    small = _region(100, 100, 40, 40)
    large = _region(3000, 2000, 400, 400)
    win = plan_windows([small, large], 4056, 3040, window=640, max_windows=1)[0]
    assert win.x <= 3200 <= win.x + win.width
    assert win.y <= 2200 <= win.y + win.height


def test_region_larger_than_the_window_is_not_cropped_away():
    """A cat close to the lens should not be cut into a 640 box."""
    win = plan_windows([_region(1000, 800, 1200, 900)], 4056, 3040, window=640)[0]
    assert win.width >= 1200
    assert win.height >= 900


def test_windows_always_lie_inside_the_frame():
    rng = np.random.default_rng(0)
    for _ in range(200):
        x, y = int(rng.integers(0, 4000)), int(rng.integers(0, 3000))
        w, h = int(rng.integers(10, 900)), int(rng.integers(10, 900))
        for win in plan_windows([_region(x, y, w, h)], 4056, 3040, window=640):
            assert win.x >= 0 and win.y >= 0
            assert win.x + win.width <= 4056
            assert win.y + win.height <= 3040


def test_region_scale_maps_motion_coords_into_capture_coords():
    """Motion measured on the 1080p frame, windows cut from the capture."""
    win = plan_windows(
        [_region(960, 540, 20, 20)],
        4056,
        3040,
        window=640,
        region_scale=4056 / 1920,
    )[0]
    # 960 * 2.1125 = 2028, the horizontal centre of the capture frame.
    assert abs((win.x + win.width / 2) - 2049) <= 4


# -- End to end through the engine ---------------------------------------------
#
# Hardware-gated: needs the NPU and the compiled HEF.  This is the
# regression the whole change exists for -- a bird-sized animal in a
# full-resolution frame, which the whole-frame path cannot see and the
# native window can.

from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import pytest  # noqa: E402

from ratcatcher.config import load_config  # noqa: E402
from ratcatcher.detection.hailo_detector import _HAILO_AVAILABLE  # noqa: E402
from ratcatcher.pipeline.event import DetectionEvent  # noqa: E402

_HEF = Path(__file__).parent.parent / "models" / "ratcatcher_best.hef"
_VAL = Path(__file__).parent.parent / "datasets" / "ratcatcher" / "val"

requires_npu = pytest.mark.skipif(
    not (_HAILO_AVAILABLE and _HEF.exists() and _VAL.exists()),
    reason="needs a Hailo device, the compiled HEF and the val set",
)


def _animal_crops(limit):
    """Real labelled animals, cropped to their ground-truth boxes."""
    out = []
    for img_path in sorted(_VAL.glob("images/*.jpg"))[:400]:
        lab = _VAL / "labels" / (img_path.stem + ".txt")
        if not lab.exists():
            continue
        lines = [l.split() for l in lab.read_text().split("\n") if l.strip()]
        if len(lines) != 1:
            continue
        cx, cy, w, h = map(float, lines[0][1:5])
        img = cv2.imread(str(img_path))
        ih, iw = img.shape[:2]
        x0, y0 = max(0, int((cx - w / 2) * iw)), max(0, int((cy - h / 2) * ih))
        x1, y1 = min(iw, int((cx + w / 2) * iw)), min(ih, int((cy + h / 2) * ih))
        if (y1 - y0) >= 200 and (x1 - x0) >= 200:
            out.append(img[y0:y1, x0:x1])
        if len(out) >= limit:
            break
    if len(out) < limit:
        pytest.skip("not enough single-object val crops")
    return out


def _compose(crop, obj_h, cap_w=4056, cap_h=3040, at=(1500, 2000)):
    """Put one animal at a given on-screen height in a full-res capture."""
    capture = np.full((cap_h, cap_w, 3), 70, dtype=np.uint8)
    capture[::11, ::11] = 150                       # texture, not flat grey
    obj_w = max(2, int(crop.shape[1] * obj_h / crop.shape[0]))
    oy, ox = at
    capture[oy : oy + obj_h, ox : ox + obj_w] = cv2.resize(
        crop, (obj_w, obj_h), interpolation=cv2.INTER_AREA
    )
    return capture, ox, oy, obj_w


@requires_npu
def test_native_window_finds_what_the_whole_frame_misses():
    """The regression this whole change exists for.

    One sample would prove nothing -- recall at bird scale is a fraction,
    not a certainty -- so this compares both paths over a set and asserts
    the separation, which is the actual claim.
    """
    from ratcatcher.detection.hailo_detector import HailoDetector
    from ratcatcher.pipeline.engine import PipelineEngine

    CAP_W, CAP_H = 4056, 3040
    SCALE = CAP_W / 1920
    OBJ_H = 62          # a House Finch at this feeder's measured framing

    crops = _animal_crops(20)
    engine = PipelineEngine(load_config())
    engine._detector = HailoDetector(str(_HEF), confidence_threshold=0.45)

    whole_hits = window_hits = placed_ok = 0
    try:
        for crop in crops:
            capture, ox, oy, obj_w = _compose(crop, OBJ_H, CAP_W, CAP_H)
            frame = cv2.resize(capture, (1920, 1080), interpolation=cv2.INTER_AREA)

            whole = DetectionEvent(
                timestamp=datetime.now(), camera_id=0, frame=frame,
                frame_width=1920, frame_height=1080,
            )
            if engine._detect_for(whole):
                whole_hits += 1

            win = plan_windows(
                [_region(ox, oy, obj_w, OBJ_H)], CAP_W, CAP_H, window=640
            )[0]
            wx, wy, ww, wh = win.bounds
            cropped = DetectionEvent(
                timestamp=datetime.now(), camera_id=0, frame=frame,
                frame_width=1920, frame_height=1080,
                crops=[(capture[wy : wy + wh, wx : wx + ww].copy(), wx, wy)],
                capture_scale=SCALE,
            )
            found = engine._detect_for(cropped)
            if not found:
                continue
            window_hits += 1

            # Every box must come back in frame coordinates, and at least
            # one must land on the animal.
            cx = (ox + obj_w / 2) / SCALE
            cy = (oy + OBJ_H / 2) / SCALE
            for d in found:
                bx, by, bw, bh = d.bbox
                assert 0 <= bx <= 1920 and 0 <= by <= 1080, f"box off-frame: {d.bbox}"
            if any(
                d.bbox[0] <= cx <= d.bbox[0] + d.bbox[2]
                and d.bbox[1] <= cy <= d.bbox[1] + d.bbox[3]
                for d in found
            ):
                placed_ok += 1
    finally:
        engine._detector.close()

    # Measured on this Pi: 0/20 whole-frame, 15/20 native window.
    assert whole_hits <= 1, (
        f"whole-frame path found {whole_hits}/20 at {OBJ_H}px; the premise "
        "of the ROI crop is that it finds almost none"
    )
    assert window_hits >= 10, f"native window found only {window_hits}/20"
    assert placed_ok >= window_hits - 2, (
        f"only {placed_ok} of {window_hits} boxes landed on the animal -- "
        "the window-to-frame coordinate mapping is wrong"
    )
