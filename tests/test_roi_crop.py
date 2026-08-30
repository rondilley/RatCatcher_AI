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
    # 16:9 deliberately, against the sensor's 4:3.  The two axes then
    # scale differently, which is the case a single scale gets wrong.
    FRAME_W, FRAME_H = 1920, 1080
    SCALE = (CAP_W / FRAME_W, CAP_H / FRAME_H)
    OBJ_H = 62          # a House Finch at this feeder's measured framing

    crops = _animal_crops(20)
    engine = PipelineEngine(load_config())
    engine._detector = HailoDetector(str(_HEF), confidence_threshold=0.45)

    whole_hits = window_hits = placed_ok = 0
    try:
        for crop in crops:
            capture, ox, oy, obj_w = _compose(crop, OBJ_H, CAP_W, CAP_H)
            frame = cv2.resize(
                capture, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA
            )

            whole = DetectionEvent(
                timestamp=datetime.now(), camera_id=0, frame=frame,
                frame_width=FRAME_W, frame_height=FRAME_H,
            )
            if engine._detect_for(whole):
                whole_hits += 1

            win = plan_windows(
                [_region(ox, oy, obj_w, OBJ_H)], CAP_W, CAP_H, window=640
            )[0]
            wx, wy, ww, wh = win.bounds
            cropped = DetectionEvent(
                timestamp=datetime.now(), camera_id=0, frame=frame,
                frame_width=FRAME_W, frame_height=FRAME_H,
                crops=[(capture[wy : wy + wh, wx : wx + ww].copy(), wx, wy)],
                capture_scale=SCALE,
            )
            found = engine._detect_for(cropped)
            if not found:
                continue
            window_hits += 1

            # Every box must come back in frame coordinates, and at least
            # one must land on the animal.  Each axis by its own scale:
            # deriving cy from the width ratio would restate whatever
            # _detect_for did rather than check it.
            cx = (ox + obj_w / 2) / SCALE[0]
            cy = (oy + OBJ_H / 2) / SCALE[1]
            for d, _detail, _dbox in found:
                bx, by, bw, bh = d.bbox
                assert 0 <= bx <= FRAME_W and 0 <= by <= FRAME_H, (
                    f"box off-frame: {d.bbox}"
                )
            if any(
                d.bbox[0] <= cx <= d.bbox[0] + d.bbox[2]
                and d.bbox[1] <= cy <= d.bbox[1] + d.bbox[3]
                for d, _detail, _dbox in found
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


# -- Coordinate mapping and the detail patch -----------------------------------
#
# The mapping test above needs the NPU.  This one does not: the ONNX is
# committed and the OpenCV DNN backend is a real detector on the CPU, so
# the coordinate arithmetic is checked on any machine and without
# contending with the running service for the Hailo device.

_ONNX = Path(__file__).parent.parent / "models" / "ratcatcher_best.onnx"

requires_cpu_model = pytest.mark.skipif(
    not (_ONNX.exists() and _VAL.exists()),
    reason="needs the ONNX model and the val set",
)


def test_cut_detail_floors_a_small_box_at_the_minimum():
    """A finch-sized box must not yield a finch-sized patch."""
    from ratcatcher.pipeline.engine import _DETAIL_MIN, _cut_detail

    window = np.full((640, 640, 3), 70, dtype=np.uint8)
    patch, box = _cut_detail(window, (300, 300, 50, 62))

    assert patch.shape[:2] == (_DETAIL_MIN, _DETAIL_MIN)
    bx, by, bw, bh = box
    # The box keeps its native size and stays inside the patch.
    assert (bw, bh) == (50, 62)
    assert 0 <= bx and bx + bw <= _DETAIL_MIN
    assert 0 <= by and by + bh <= _DETAIL_MIN


def test_cut_detail_pads_a_large_box_and_stays_inside_the_window():
    """Padding is proportional above the floor, and clamped at the edge."""
    from ratcatcher.pipeline.engine import _DETAIL_PAD, _cut_detail

    window = np.full((640, 640, 3), 70, dtype=np.uint8)

    patch, _box = _cut_detail(window, (200, 200, 200, 160))
    assert patch.shape[0] == int(200 * _DETAIL_PAD)

    # A box against the corner still yields a full square, not a sliver.
    corner, box = _cut_detail(window, (600, 600, 40, 40))
    assert corner.shape[:2] == (256, 256)
    bx, by, _bw, _bh = box
    assert bx >= 0 and by >= 0


def test_cut_detail_does_not_hold_the_window_alive():
    """The patch is copied: a view would keep 1.2 MB per queued event."""
    from ratcatcher.pipeline.engine import _cut_detail

    window = np.full((640, 640, 3), 70, dtype=np.uint8)
    patch, _box = _cut_detail(window, (300, 300, 50, 62))
    assert patch.base is None


@requires_cpu_model
def test_box_from_a_window_lands_on_the_animal_near_the_frame_bottom():
    """The regression for the single-scale mapping.

    An animal in the bottom quarter of the sensor is where a
    width-derived scale sends the box off the frame entirely: capture
    y=2800 maps to 1325 in a 1080-tall frame instead of 995.  The animal
    is deliberately large, because this asserts the arithmetic, not
    recall -- recall at bird scale is the NPU test above.
    """
    from ratcatcher.detection.opencv_detector import OpenCVDetector
    from ratcatcher.pipeline.engine import PipelineEngine

    CAP_W, CAP_H = 4056, 3040
    FRAME_W, FRAME_H = 1920, 1080          # 16:9 against the sensor's 4:3
    SCALE = (CAP_W / FRAME_W, CAP_H / FRAME_H)
    OBJ_H = 400
    AT = (2600, 2000)                      # (y, x), bottom quarter

    engine = PipelineEngine(load_config())
    engine._detector = OpenCVDetector(str(_ONNX), confidence_threshold=0.25)

    checked = 0
    for crop in _animal_crops(6):
        capture, ox, oy, obj_w = _compose(crop, OBJ_H, CAP_W, CAP_H, at=AT)
        frame = cv2.resize(
            capture, (FRAME_W, FRAME_H), interpolation=cv2.INTER_AREA
        )

        win = plan_windows(
            [_region(ox, oy, obj_w, OBJ_H)], CAP_W, CAP_H, window=640
        )[0]
        wx, wy, ww, wh = win.bounds
        event = DetectionEvent(
            timestamp=datetime.now(), camera_id=0, frame=frame,
            frame_width=FRAME_W, frame_height=FRAME_H,
            crops=[(capture[wy : wy + wh, wx : wx + ww].copy(), wx, wy)],
            capture_scale=SCALE,
        )

        found = engine._detect_for(event)
        if not found:
            continue
        checked += 1

        cx = (ox + obj_w / 2) / SCALE[0]
        cy = (oy + OBJ_H / 2) / SCALE[1]
        assert any(
            d.bbox[0] <= cx <= d.bbox[0] + d.bbox[2]
            and d.bbox[1] <= cy <= d.bbox[1] + d.bbox[3]
            for d, _detail, _dbox in found
        ), (
            f"no box landed on the animal at frame ({cx:.0f}, {cy:.0f}); "
            f"got {[d.bbox for d, _, _ in found]} -- a box near y="
            f"{(oy + OBJ_H / 2) / SCALE[0]:.0f} means the vertical scale "
            "came from the frame width"
        )

        for d, detail, detail_bbox in found:
            bx, by, bw, bh = d.bbox
            assert 0 <= bx and bx + bw <= FRAME_W, f"box off-frame: {d.bbox}"
            assert 0 <= by and by + bh <= FRAME_H, f"box off-frame: {d.bbox}"

            # Every windowed detection carries native pixels for the
            # thumbnail, with its box expressed against them.
            assert detail is not None and detail_bbox is not None
            dh, dw = detail.shape[:2]
            assert dw >= 256 and dh >= 256
            # The box may spill past the window -- YOLO returns boxes
            # that run off the edge of what it was shown -- so what
            # matters is that it overlaps the patch, which is all
            # cv2.rectangle can draw anyway.
            dbx, dby, dbw, dbh = detail_bbox
            assert dbw > 0 and dbh > 0
            assert dbx < dw and dby < dh
            assert dbx + dbw > 0 and dby + dbh > 0

    assert checked, "the CPU detector found nothing to check the mapping with"


@requires_cpu_model
def test_whole_frame_detections_carry_no_detail_patch():
    """No window, no native pixels -- the thumbnail falls back to the frame."""
    from ratcatcher.detection.opencv_detector import OpenCVDetector
    from ratcatcher.pipeline.engine import PipelineEngine

    engine = PipelineEngine(load_config())
    engine._detector = OpenCVDetector(str(_ONNX), confidence_threshold=0.25)

    crop = _animal_crops(1)[0]
    frame = cv2.resize(crop, (1920, 1080), interpolation=cv2.INTER_AREA)
    event = DetectionEvent(
        timestamp=datetime.now(), camera_id=0, frame=frame,
        frame_width=1920, frame_height=1080,
    )

    found = engine._detect_for(event)
    assert found, "a full-frame animal should be detectable"
    for _det, detail, detail_bbox in found:
        assert detail is None and detail_bbox is None
