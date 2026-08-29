"""Tests for detection backend selection.

Every test drives the real ``create_detector`` against the real model
files in ``models/`` -- no doubles.  What that exercises depends on the
machine: on a Pi with a working NPU the auto path returns the Hailo
backend, and on any other machine (or a Pi whose driver did not survive
a kernel upgrade) it returns a CPU backend.  Both are passes.  The
property under test is that *something usable comes back*, because the
regression being guarded against is a hard crash at pipeline startup.

Background: ``_auto_select`` originally caught only ImportError and
FileNotFoundError, but HailoDetector wraps device failures in
RuntimeError.  When the Hailo kernel module went missing after a kernel
upgrade on 2026-08-24, ``backend: "auto"`` therefore raised out of
``create_detector`` and took the whole video pipeline down instead of
degrading to the CPU.  For an unattended field deployment that is worse
than slow detection.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ratcatcher.config import DetectionConfig
from ratcatcher.detection.detector import Detection, RATCATCHER_CLASSES
from ratcatcher.detection.factory import create_detector


MODELS_DIR = Path(__file__).parent.parent / "models"
CUSTOM_ONNX = MODELS_DIR / "ratcatcher_best.onnx"

requires_custom_model = pytest.mark.skipif(
    not CUSTOM_ONNX.is_file(),
    reason="requires models/ratcatcher_best.onnx (gitignored; see README)",
)

# CPU backends are the fallbacks under test.  Whichever one wins, it has
# to satisfy the ObjectDetector protocol.
CPU_BACKENDS = {"ncnn", "opencv_dnn"}


@pytest.fixture
def detector_factory():
    """Create detectors and release them when the test ends.

    The Hailo backend owns a VDevice, a configured network group and its
    vstreams.  Dropping the last reference and leaving that to the
    garbage collector segfaults the interpreter: on a Pi with a working
    NPU the auto path returns Hailo here, and this file took the whole
    suite down with SIGSEGV as the first test returned.  The same defect
    killed the daemon on every shutdown; see PipelineEngine.stop().

    Only the Hailo backend has a close(); the CPU backends hold no
    device and the ObjectDetector protocol does not declare one.
    """
    created = []

    def make(*args, **kwargs):
        detector = create_detector(*args, **kwargs)
        created.append(detector)
        return detector

    yield make

    for detector in created:
        close = getattr(detector, "close", None)
        if close is not None:
            close()


def _config(**overrides) -> DetectionConfig:
    base = {
        "backend": "auto",
        "model_path": "ratcatcher_best.onnx",
        "confidence_threshold": 0.45,
        "nms_threshold": 0.45,
        "input_size": (640, 640),
    }
    base.update(overrides)
    return DetectionConfig(**base)


@requires_custom_model
class TestAutoSelection:

    def test_auto_always_returns_a_usable_detector(self, detector_factory):
        """The regression test: auto must never raise, whatever the state
        of the NPU."""
        detector = detector_factory(_config(), models_dir=MODELS_DIR)

        assert detector.backend_name in CPU_BACKENDS | {"hailo"}
        assert detector.get_classes() == list(RATCATCHER_CLASSES)

    def test_auto_selected_detector_runs_a_real_frame(self, detector_factory):
        """A backend that constructs but cannot infer is no use."""
        detector = detector_factory(_config(), models_dir=MODELS_DIR)

        rng = np.random.default_rng(0)
        frame = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
        result = detector.detect(frame)

        assert isinstance(result, list)
        assert all(isinstance(d, Detection) for d in result)

    def test_auto_falls_back_when_no_hef_exists(self, detector_factory):
        """``model_path`` with no matching .hef must land on a CPU backend
        rather than raising FileNotFoundError out of the factory."""
        detector = detector_factory(_config(), models_dir=MODELS_DIR)

        if not (MODELS_DIR / "ratcatcher_best.hef").is_file():
            assert detector.backend_name in CPU_BACKENDS


@requires_custom_model
class TestExplicitBackend:

    def test_explicit_opencv_dnn_is_honoured(self, detector_factory):
        detector = detector_factory(
            _config(backend="opencv_dnn"), models_dir=MODELS_DIR
        )
        assert detector.backend_name == "opencv_dnn"

    def test_explicit_hailo_fails_loudly_when_unavailable(self):
        """Unlike auto, an explicit backend choice must not silently
        degrade -- the caller asked for the NPU specifically."""
        hef = MODELS_DIR / "ratcatcher_best.hef"
        if hef.is_file():
            pytest.skip("a custom HEF exists, so hailo may legitimately load")

        with pytest.raises((RuntimeError, ImportError, FileNotFoundError)):
            create_detector(_config(backend="hailo"), models_dir=MODELS_DIR)

    def test_unknown_backend_is_rejected(self):
        with pytest.raises(RuntimeError, match="Unknown detection backend"):
            create_detector(
                _config(backend="tpu"), models_dir=MODELS_DIR
            )


class TestMissingModel:

    def test_auto_raises_when_no_backend_can_load_anything(self, tmp_path):
        """With an empty models directory every backend fails, and the
        aggregate error must name each one that was tried."""
        with pytest.raises(RuntimeError) as exc_info:
            create_detector(_config(), models_dir=tmp_path)

        message = str(exc_info.value)
        assert "No detection backend available" in message
        for backend in ("hailo", "ncnn", "opencv_dnn"):
            assert backend in message
