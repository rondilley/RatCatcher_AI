"""Tests for motion detection and ROI masking.

Exercises MotionDetector (background subtraction, area filtering, cooldown,
coordinate scaling) and ROIMask (full-frame default, polygon masking,
point containment) using real numpy frames and real OpenCV operations.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from ratcatcher.config import MotionConfig
from ratcatcher.motion.detector import MotionDetector, MotionRegion
from ratcatcher.motion.regions import ROIMask


# -- helpers ----------------------------------------------------------------

def _make_config(**overrides) -> MotionConfig:
    """Return a MotionConfig with sane test defaults.

    Cooldown is zeroed so most tests are not time-sensitive.
    """
    defaults = dict(
        cooldown_seconds=0.0,
        process_width=320,
        process_height=240,
        min_area_pct=0.005,
        history=500,
        var_threshold=16,
        detect_shadows=False,
        erode_kernel=3,
        dilate_kernel=7,
        learning_rate=-1.0,
    )
    defaults.update(overrides)
    return MotionConfig(**defaults)


def _black_frame(height: int = 480, width: int = 640) -> np.ndarray:
    """Return a solid black BGR frame."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def _warmup(detector: MotionDetector, frame: np.ndarray, n: int = 30) -> None:
    """Feed *n* identical frames so MOG2 learns the background."""
    for _ in range(n):
        detector.detect(frame)


# ---------------------------------------------------------------------------
# MotionDetector tests
# ---------------------------------------------------------------------------

class TestMotionDetector:

    def test_no_motion_on_static_frames(self):
        """Identical frames after warmup should produce no detections."""
        config = _make_config()
        detector = MotionDetector(config)
        frame = _black_frame()

        _warmup(detector, frame, n=40)

        # Several more identical frames -- none should trigger motion.
        detections_total = 0
        for _ in range(10):
            detections_total += len(detector.detect(frame))

        assert detections_total == 0

    def test_motion_detected_on_change(self):
        """A large bright rectangle injected after warmup must be detected."""
        config = _make_config()
        detector = MotionDetector(config)
        bg = _black_frame()

        _warmup(detector, bg, n=40)

        # Draw a large white rectangle in the centre of the frame.
        changed = bg.copy()
        changed[190:290, 270:370] = 255  # 100x100 white block

        regions = detector.detect(changed)
        assert len(regions) >= 1
        # The detected region should be a MotionRegion with positive area.
        assert all(isinstance(r, MotionRegion) for r in regions)
        assert all(r.area > 0 for r in regions)

    def test_motion_region_coordinates_scaled(self):
        """Returned bounding boxes must be in original-frame coordinates."""
        orig_w, orig_h = 1280, 960
        config = _make_config(process_width=320, process_height=240)
        detector = MotionDetector(config)

        bg = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
        _warmup(detector, bg, n=40)

        # Place a 200x200 white block in the lower-right quadrant.
        changed = bg.copy()
        changed[600:800, 900:1100] = 255

        regions = detector.detect(changed)
        assert len(regions) >= 1

        for r in regions:
            # Coordinates must be within the original frame bounds, not
            # within the smaller processing resolution.
            assert 0 <= r.x < orig_w
            assert 0 <= r.y < orig_h
            assert r.x + r.width <= orig_w
            assert r.y + r.height <= orig_h
            # At least one coordinate should exceed the processing width,
            # proving the values were scaled back up.
            assert r.x >= config.process_width or r.y >= config.process_height

    def test_cooldown_suppresses_repeat(self):
        """Back-to-back detections in the same grid cell should be suppressed."""
        config = _make_config(cooldown_seconds=10.0)
        detector = MotionDetector(config)
        bg = _black_frame()

        _warmup(detector, bg, n=40)

        # Warmup may have triggered transient detections that populated
        # the cooldown grid.  Clear it so the first real detection fires.
        detector._cooldown_grid.clear()

        changed = bg.copy()
        changed[190:290, 270:370] = 255

        first = detector.detect(changed)
        assert len(first) >= 1, "First detection should fire"

        # Immediately detect again with the same changed frame -- the grid
        # cell cooldown (10 s) should suppress it.
        second = detector.detect(changed)
        assert len(second) == 0, "Cooldown should suppress the repeat detection"

    def test_reset_clears_state(self):
        """After reset() the background model is re-initialised."""
        config = _make_config()
        detector = MotionDetector(config)
        bg = _black_frame()

        _warmup(detector, bg, n=40)

        # Inject motion so the detector has internal state.
        changed = bg.copy()
        changed[190:290, 270:370] = 255
        detector.detect(changed)

        detector.reset()

        # After reset, warmup on the *changed* frame so that it becomes
        # the new background -- then the original black frame is the
        # novelty.  But the key assertion is simply that reset() does not
        # raise and that the detector is usable afterwards.
        _warmup(detector, changed, n=40)

        # Now the changed frame is the learned background.  Feeding the
        # original black frame should look like a whole-frame change.
        regions = detector.detect(bg)
        # We cannot predict exact count, but it should produce at least
        # one region because the background changed everywhere.
        assert len(regions) >= 1

    def test_min_area_filters_small_changes(self):
        """A tiny change (2x2 pixels) should be below min_area_pct."""
        config = _make_config(min_area_pct=0.005)
        detector = MotionDetector(config)
        bg = _black_frame()

        _warmup(detector, bg, n=40)

        # A 2x2 white dot is far below 0.5% of 320x240 = 384 pixels.
        changed = bg.copy()
        changed[240:242, 320:322] = 255

        regions = detector.detect(changed)
        assert len(regions) == 0


# ---------------------------------------------------------------------------
# ROIMask tests
# ---------------------------------------------------------------------------

class TestROIMask:

    def test_roi_mask_full_frame_when_empty(self):
        """No polygons means the entire mask is 255 (fully active)."""
        roi = ROIMask(polygons=[], frame_width=640, frame_height=480)
        mask = roi.mask
        assert mask.shape == (480, 640)
        assert mask.dtype == np.uint8
        assert np.all(mask == 255)

    def test_roi_mask_applies_correctly(self):
        """A polygon ROI should zero-out foreground pixels outside the ROI."""
        # Define a polygon covering roughly the centre quarter of the frame.
        polygon = [
            [0.25, 0.25],
            [0.75, 0.25],
            [0.75, 0.75],
            [0.25, 0.75],
        ]
        roi = ROIMask(
            polygons=[polygon], frame_width=640, frame_height=480
        )

        # Create an all-white foreground mask (motion everywhere).
        fg = np.full((480, 640), 255, dtype=np.uint8)
        result = roi.apply(fg)

        # Pixels clearly inside the polygon should be 255.
        assert result[300, 400] == 255  # centre-ish
        # Pixels clearly outside the polygon should be 0.
        assert result[10, 10] == 0
        assert result[470, 630] == 0

    def test_roi_contains_point(self):
        """contains_point should respect polygon boundaries."""
        polygon = [
            [0.25, 0.25],
            [0.75, 0.25],
            [0.75, 0.75],
            [0.25, 0.75],
        ]
        roi = ROIMask(
            polygons=[polygon], frame_width=640, frame_height=480
        )

        # Centre of the polygon -- clearly inside.
        assert roi.contains_point(0.5, 0.5) is True

        # Corners of the frame -- clearly outside.
        assert roi.contains_point(0.05, 0.05) is False
        assert roi.contains_point(0.95, 0.95) is False

        # Edge case: a point right on the polygon boundary counts as inside
        # (cv2.pointPolygonTest returns 0 for on-edge, and the code uses >=0).
        assert roi.contains_point(0.25, 0.25) is True

    def test_roi_contains_point_full_frame(self):
        """With no polygons every point should be considered inside."""
        roi = ROIMask(polygons=[], frame_width=640, frame_height=480)

        assert roi.contains_point(0.0, 0.0) is True
        assert roi.contains_point(0.5, 0.5) is True
        assert roi.contains_point(1.0, 1.0) is True


# -- Whole-frame illumination changes ------------------------------------------
#
# A cloud crossing the sun, or an auto-exposure step, changes every pixel at
# once and MOG2 reports it as one region covering nearly the whole frame.
# Measured on this Pi, that was the only "motion" the cameras produced in 680
# frames -- so without a ceiling the ROI-crop window degenerates to the full
# frame, which is exactly what it exists to avoid.
#
# cooldown_seconds=0 throughout: settling MOG2 fires motion and arms the
# cooldown grid, so with the default 2 s these tests would pass on the
# cooldown rather than on the thing they claim to measure.


def _settle(det, frame, n=40):
    """Let MOG2 learn a stable background."""
    for _ in range(n):
        det.detect(frame)


def _textured_base():
    base = np.full((480, 640, 3), 90, dtype=np.uint8)
    base[::7, ::7] = 200          # texture, so MOG2 has structure to model
    return base


def _brighter(frame, by=60):
    return np.clip(frame.astype(np.int16) + by, 0, 255).astype(np.uint8)


def test_whole_frame_brightness_step_passes_without_a_ceiling():
    """Establishes what the ceiling is actually suppressing."""
    det = MotionDetector(MotionConfig(cooldown_seconds=0.0))
    base = _textured_base()
    _settle(det, base)

    regions = det.detect(_brighter(base))
    assert regions != []
    assert max(r.area for r in regions) > 0.5


def test_whole_frame_brightness_step_is_rejected_when_a_ceiling_is_set():
    det = MotionDetector(MotionConfig(cooldown_seconds=0.0, max_area_pct=0.5))
    base = _textured_base()
    _settle(det, base)

    assert det.detect(_brighter(base)) == []


def test_a_small_moving_object_still_survives_the_ceiling():
    det = MotionDetector(MotionConfig(cooldown_seconds=0.0, max_area_pct=0.5))
    base = _textured_base()
    _settle(det, base)

    moved = base.copy()
    cv2.rectangle(moved, (300, 220), (360, 280), (255, 255, 255), -1)
    assert det.detect(moved) != []
