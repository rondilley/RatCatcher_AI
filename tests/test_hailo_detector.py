"""Tests for the Hailo NPU detection backend.

Split into two groups:

* **Parsing tests** run anywhere.  They exercise the pure output-decoding
  logic against fabricated tensors whose layout was captured from a real
  Hailo-8 running the Model Zoo ``yolov8n.hef``.
* **Hardware tests** are skipped unless ``hailo_platform`` imports and a
  compiled HEF is present under ``models/``.  They cover device lifecycle
  and the repeated-inference path that regressed with
  ``HAILO_NETWORK_GROUP_NOT_ACTIVATED``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ratcatcher.detection.detector import Detection, RATCATCHER_CLASSES
from ratcatcher.detection.hailo_detector import _HAILO_AVAILABLE, HailoDetector


MODEL_PATH = Path(__file__).parent.parent / "models" / "yolov8n.hef"

requires_hailo = pytest.mark.skipif(
    not (_HAILO_AVAILABLE and MODEL_PATH.is_file()),
    reason="requires hailo_platform and a compiled HEF in models/",
)


# -- helpers ----------------------------------------------------------------

def _parser(confidence_threshold: float = 0.25) -> HailoDetector:
    """Build a detector shell that can parse output but owns no device.

    ``_parse_nms_output`` touches only ``_confidence_threshold``, so the
    decode logic can be tested without a Hailo attached.
    """
    det = HailoDetector.__new__(HailoDetector)
    det._confidence_threshold = confidence_threshold
    return det


def _nms_output(per_class: dict[int, list[list[float]]]) -> list:
    """Build a HAILO_NMS_BY_CLASS output from {coco_id: [[y0,x0,y1,x1,s]]}.

    The real output is a single-element batch list holding 80 arrays of
    shape ``(n, 5)`` -- one per COCO class, empty where nothing survived
    the on-chip NMS.
    """
    classes = [
        np.asarray(per_class.get(i, []), dtype=np.float32).reshape(-1, 5)
        for i in range(80)
    ]
    return [classes]


# ---------------------------------------------------------------------------
# On-chip NMS output parsing
# ---------------------------------------------------------------------------

class TestParseNmsOutput:

    def test_rows_are_ymin_xmin_ymax_xmax_score(self):
        """Coordinates are normalised (y0, x0, y1, x1) -- not (x0, y0, ...)."""
        # Box covering the right half, vertically centred.
        out = _nms_output({14: [[0.25, 0.5, 0.75, 1.0, 0.9]]})
        dets = _parser()._parse_nms_output(out, orig_w=800, orig_h=400)

        assert len(dets) == 1
        assert dets[0].bbox == (400, 100, 400, 200)

    def test_coco_class_is_mapped_to_ratcatcher_class(self):
        out = _nms_output({
            14: [[0.1, 0.1, 0.2, 0.2, 0.9]],   # bird  -> bird
            15: [[0.3, 0.3, 0.4, 0.4, 0.9]],   # cat   -> cat
            16: [[0.5, 0.5, 0.6, 0.6, 0.9]],   # dog   -> unknown_animal
        })
        dets = _parser()._parse_nms_output(out, orig_w=640, orig_h=640)

        names = sorted(d.class_name for d in dets)
        assert names == ["bird", "cat", "unknown_animal"]
        for d in dets:
            assert d.class_name == RATCATCHER_CLASSES[d.class_id]

    def test_non_animal_classes_are_dropped(self):
        """person and bus have no RatCatcher equivalent."""
        out = _nms_output({
            0: [[0.1, 0.1, 0.9, 0.9, 0.99]],   # person
            5: [[0.2, 0.2, 0.8, 0.8, 0.95]],   # bus
        })
        assert _parser()._parse_nms_output(out, 640, 640) == []

    def test_confidence_threshold_is_applied_on_host(self):
        """The HEF's compiled-in threshold (0.2) is looser than ours."""
        out = _nms_output({14: [
            [0.1, 0.1, 0.5, 0.5, 0.9],
            [0.2, 0.2, 0.6, 0.6, 0.21],
        ]})
        dets = _parser(confidence_threshold=0.25)._parse_nms_output(out, 640, 640)

        assert [round(d.confidence, 2) for d in dets] == [0.9]

    def test_out_of_range_coords_are_clamped(self):
        """The NPU emits values marginally outside [0, 1]."""
        out = _nms_output({14: [[-0.002, -0.0, 1.01, 1.004, 0.9]]})
        dets = _parser()._parse_nms_output(out, orig_w=320, orig_h=240)

        x, y, w, h = dets[0].bbox
        assert (x, y) == (0, 0)
        assert x + w <= 320
        assert y + h <= 240

    def test_multiple_boxes_per_class(self):
        out = _nms_output({14: [
            [0.0, 0.0, 0.2, 0.2, 0.9],
            [0.5, 0.5, 0.7, 0.7, 0.8],
            [0.8, 0.8, 0.9, 0.9, 0.7],
        ]})
        dets = _parser()._parse_nms_output(out, 640, 640)

        assert len(dets) == 3
        assert all(d.class_name == "bird" for d in dets)

    def test_empty_output_returns_no_detections(self):
        assert _parser()._parse_nms_output(_nms_output({}), 640, 640) == []


# ---------------------------------------------------------------------------
# Pixel box conversion
# ---------------------------------------------------------------------------

class TestToPixelBox:

    def test_rounds_to_integers(self):
        assert HailoDetector._to_pixel_box(10.4, 20.6, 30.5, 40.4, 640, 480) == (
            10, 21, 30, 40
        )

    def test_clamps_box_to_frame(self):
        box = HailoDetector._to_pixel_box(600.0, 400.0, 200.0, 200.0, 640, 480)
        x, y, w, h = box
        assert x + w <= 640
        assert y + h <= 480

    def test_negative_origin_clamped_to_zero(self):
        x, y, _, _ = HailoDetector._to_pixel_box(-5.0, -8.0, 50.0, 50.0, 640, 480)
        assert (x, y) == (0, 0)

    def test_degenerate_box_gets_minimum_size(self):
        """A zero-area box is widened to 1px rather than dropped."""
        assert HailoDetector._to_pixel_box(10.0, 10.0, 0.0, 0.0, 640, 480) == (
            10, 10, 1, 1
        )


# ---------------------------------------------------------------------------
# Live device
# ---------------------------------------------------------------------------

@requires_hailo
class TestHailoDevice:

    @pytest.fixture
    def detector(self):
        """A detector owning the NPU for the duration of one test.

        Deliberately function-scoped: the Hailo-8 exposes a single
        physical device, so a longer-lived fixture would starve any test
        that opens its own detector (HAILO_OUT_OF_PHYSICAL_DEVICES).
        """
        with HailoDetector(MODEL_PATH, confidence_threshold=0.25) as det:
            yield det

    def test_model_zoo_hef_uses_on_chip_nms(self, detector):
        assert detector._nms_on_chip is True

    def test_input_size_is_read_from_the_hef(self, detector):
        assert detector._input_size == (640, 640)

    def test_backend_name_and_classes(self, detector):
        assert detector.backend_name == "hailo"
        assert detector.get_classes() == list(RATCATCHER_CLASSES)

    def test_detect_returns_detections(self, detector):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        result = detector.detect(frame)

        assert isinstance(result, list)
        assert all(isinstance(d, Detection) for d in result)

    def test_repeated_inference_reuses_the_activated_network_group(self, detector):
        """Regression: every frame after the first raised
        HAILO_NETWORK_GROUP_NOT_ACTIVATED when the vstreams were rebuilt
        per call instead of being held open."""
        rng = np.random.default_rng(0)
        frame = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)

        for _ in range(5):
            assert isinstance(detector.detect(frame), list)

    @pytest.mark.parametrize("shape", [(240, 320, 3), (1080, 1920, 3), (500, 500, 3)])
    def test_detect_accepts_arbitrary_frame_sizes(self, detector, shape):
        dets = detector.detect(np.zeros(shape, dtype=np.uint8))

        h, w = shape[:2]
        for d in dets:
            x, y, bw, bh = d.bbox
            assert 0 <= x < w and 0 <= y < h
            assert x + bw <= w and y + bh <= h

    def test_close_is_idempotent_and_detect_then_raises(self):
        det = HailoDetector(MODEL_PATH)
        det.close()
        det.close()

        with pytest.raises(RuntimeError, match="closed"):
            det.detect(np.zeros((480, 640, 3), dtype=np.uint8))

    def test_missing_model_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            HailoDetector(tmp_path / "nope.hef")

    def test_only_one_detector_may_hold_the_device(self, detector):
        """The NPU exposes a single VDevice, so a second detector cannot
        be opened alongside the first -- callers must share one instance
        across camera pipelines rather than constructing per stream."""
        with pytest.raises(RuntimeError, match="HAILO_OUT_OF_PHYSICAL_DEVICES"):
            HailoDetector(MODEL_PATH)

    def test_device_is_reusable_after_close(self, detector):
        """Closing must actually release the VDevice, not just the vstreams."""
        detector.close()

        with HailoDetector(MODEL_PATH) as second:
            assert isinstance(
                second.detect(np.zeros((480, 640, 3), dtype=np.uint8)), list
            )


# ---------------------------------------------------------------------------
# Raw-tensor output parsing (custom-compiled HEFs, host-side NMS)
# ---------------------------------------------------------------------------

def _raw_parser(
    confidence_threshold: float = 0.25,
    nms_threshold: float = 0.45,
    input_size: tuple[int, int] = (640, 640),
) -> HailoDetector:
    """Detector shell for the raw-tensor decode path (no device needed)."""
    det = HailoDetector.__new__(HailoDetector)
    det._confidence_threshold = confidence_threshold
    det._nms_threshold = nms_threshold
    det._input_size = input_size
    return det


def _raw_output(
    boxes: list[tuple[float, float, float, float]],
    scores: list[tuple[int, float]],
    num_classes: int,
    n_anchors: int = 100,
    transposed: bool = False,
) -> np.ndarray:
    """Build a YOLOv8 head tensor of shape ``(1, 4 + num_classes, N)``.

    ``boxes`` are centre-format ``(cx, cy, w, h)`` in network pixels and
    ``scores`` are ``(class_id, score)`` pairs, positionally paired with
    ``boxes``.  Remaining anchors are left as all-zero background.

    ``n_anchors`` stays comfortably above the channel count because
    ``_parse_raw_output`` picks the anchor axis as the longer one.
    """
    channels = 4 + num_classes
    assert n_anchors > channels, "anchor axis must be the longer one"

    preds = np.zeros((n_anchors, channels), dtype=np.float32)
    for i, ((cx, cy, w, h), (cls, score)) in enumerate(zip(boxes, scores)):
        preds[i, :4] = (cx, cy, w, h)
        preds[i, 4 + cls] = score

    tensor = preds if transposed else preds.T
    return np.expand_dims(tensor, axis=0)


class TestParseRawOutput:

    def test_coco_tensor_maps_classes_and_converts_centre_boxes(self):
        """84 channels means a stock COCO head -> COCO ids need mapping."""
        out = _raw_output(
            boxes=[(320.0, 240.0, 100.0, 80.0)],
            scores=[(14, 0.9)],  # bird
            num_classes=80,
        )
        dets = _raw_parser()._parse_raw_output(out, orig_w=640, orig_h=640)

        assert len(dets) == 1
        assert dets[0].class_name == "bird"
        # centre (320, 240) size 100x80 -> top-left (270, 200)
        assert dets[0].bbox == (270, 200, 100, 80)

    def test_non_animal_coco_class_is_dropped(self):
        out = _raw_output(
            boxes=[(320.0, 240.0, 100.0, 80.0)],
            scores=[(0, 0.99)],  # person
            num_classes=80,
        )
        assert _raw_parser()._parse_raw_output(out, 640, 640) == []

    def test_custom_head_uses_ratcatcher_ids_directly(self):
        """A 5-class head is our own model -- no COCO mapping applies."""
        out = _raw_output(
            boxes=[(320.0, 240.0, 100.0, 80.0)],
            scores=[(2, 0.9)],  # rat
            num_classes=len(RATCATCHER_CLASSES),
        )
        dets = _raw_parser()._parse_raw_output(out, 640, 640)

        assert len(dets) == 1
        assert dets[0].class_id == 2
        assert dets[0].class_name == "rat"

    def test_boxes_are_scaled_to_the_original_frame(self):
        """Network coords are 640x640; the frame here is 1280x480."""
        out = _raw_output(
            boxes=[(320.0, 320.0, 64.0, 64.0)],
            scores=[(14, 0.9)],
            num_classes=80,
        )
        dets = _raw_parser()._parse_raw_output(out, orig_w=1280, orig_h=480)

        # x doubles, y halves.
        assert dets[0].bbox == (576, 216, 128, 48)

    def test_below_threshold_predictions_are_dropped(self):
        out = _raw_output(
            boxes=[(320.0, 240.0, 100.0, 80.0), (100.0, 100.0, 50.0, 50.0)],
            scores=[(14, 0.9), (14, 0.1)],
            num_classes=80,
        )
        dets = _raw_parser(confidence_threshold=0.25)._parse_raw_output(
            out, 640, 640
        )

        assert [round(d.confidence, 2) for d in dets] == [0.9]

    def test_nms_suppresses_overlapping_duplicates(self):
        """Two near-identical boxes on one class collapse to the best one."""
        out = _raw_output(
            boxes=[(320.0, 240.0, 100.0, 80.0), (322.0, 242.0, 100.0, 80.0)],
            scores=[(14, 0.9), (14, 0.8)],
            num_classes=80,
        )
        dets = _raw_parser(nms_threshold=0.45)._parse_raw_output(out, 640, 640)

        assert len(dets) == 1
        assert round(dets[0].confidence, 2) == 0.9

    def test_distant_boxes_both_survive_nms(self):
        out = _raw_output(
            boxes=[(100.0, 100.0, 50.0, 50.0), (500.0, 500.0, 50.0, 50.0)],
            scores=[(14, 0.9), (14, 0.8)],
            num_classes=80,
        )
        dets = _raw_parser()._parse_raw_output(out, 640, 640)

        assert len(dets) == 2

    @pytest.mark.parametrize("transposed", [False, True])
    def test_both_channel_layouts_decode_identically(self, transposed):
        """(1, 84, N) and (1, N, 84) must give the same answer."""
        out = _raw_output(
            boxes=[(320.0, 240.0, 100.0, 80.0)],
            scores=[(14, 0.9)],
            num_classes=80,
            transposed=transposed,
        )
        dets = _raw_parser()._parse_raw_output(out, 640, 640)

        assert len(dets) == 1
        assert dets[0].bbox == (270, 200, 100, 80)

    def test_all_background_returns_no_detections(self):
        out = _raw_output(boxes=[], scores=[], num_classes=80)
        assert _raw_parser()._parse_raw_output(out, 640, 640) == []
