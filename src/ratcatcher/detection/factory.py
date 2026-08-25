"""Detector factory for RatCatcher AI.

Selects and constructs the appropriate detection backend based on the
``DetectionConfig.backend`` setting and the packages / hardware available
at runtime.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ratcatcher.config import DetectionConfig
from ratcatcher.detection.detector import ObjectDetector

logger = logging.getLogger(__name__)


def create_detector(
    config: DetectionConfig,
    models_dir: str | Path = "models",
) -> ObjectDetector:
    """Build an ``ObjectDetector`` from a ``DetectionConfig``.

    Parameters
    ----------
    config:
        Detection configuration section.
    models_dir:
        Base directory that contains model files.  ``config.model_path``
        is resolved relative to this directory.

    Returns
    -------
    A fully-initialised detector instance.

    Raises
    ------
    RuntimeError
        If no usable backend could be found.
    """
    models_dir = Path(models_dir)
    model_base = models_dir / config.model_path
    backend = config.backend.lower()

    if backend == "auto":
        return _auto_select(config, model_base)

    if backend == "hailo":
        return _make_hailo(config, model_base)

    if backend == "ncnn":
        return _make_ncnn(config, model_base)

    if backend in ("opencv_dnn", "opencv"):
        return _make_opencv(config, model_base)

    raise RuntimeError(
        f"Unknown detection backend '{config.backend}'.  "
        f"Supported values: auto, hailo, ncnn, opencv_dnn"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _auto_select(
    config: DetectionConfig, model_base: Path
) -> ObjectDetector:
    """Try backends in priority order: hailo -> ncnn -> opencv_dnn."""
    errors: list[str] = []

    # --- Hailo ---
    # RuntimeError is caught alongside the import/file errors because
    # HailoDetector wraps every device-level failure in one: the NPU is
    # absent, already claimed by another process, or -- the case seen in
    # the field on 2026-08-24 -- its kernel module did not survive a
    # kernel upgrade, leaving no /dev/hailo0 for libhailort to open.
    # Under "auto" that must degrade to a CPU backend, not take the whole
    # pipeline down.  An explicit backend: "hailo" still fails hard, since
    # there the caller asked for the NPU specifically.
    try:
        return _make_hailo(config, model_base)
    except (ImportError, FileNotFoundError, RuntimeError) as exc:
        errors.append(f"hailo: {exc}")
        # Logged at warning, not debug: silently running detection on the
        # CPU at a fraction of the frame rate is exactly the kind of
        # degradation that should be visible in the journal.
        logger.warning(
            "Auto-detect: Hailo NPU unavailable, falling back to CPU -- %s",
            exc,
        )

    # --- NCNN ---
    try:
        return _make_ncnn(config, model_base)
    except (ImportError, FileNotFoundError, RuntimeError) as exc:
        errors.append(f"ncnn: {exc}")
        logger.debug("Auto-detect: NCNN unavailable -- %s", exc)

    # --- OpenCV DNN (always available if opencv is installed) ---
    try:
        return _make_opencv(config, model_base)
    except (ImportError, FileNotFoundError, RuntimeError) as exc:
        errors.append(f"opencv_dnn: {exc}")
        logger.debug("Auto-detect: OpenCV DNN unavailable -- %s", exc)

    raise RuntimeError(
        "No detection backend available.  Tried (in order):\n  "
        + "\n  ".join(errors)
    )


def _make_hailo(
    config: DetectionConfig, model_base: Path
) -> ObjectDetector:
    """Construct a HailoDetector.

    Looks for a ``.hef`` file with the same stem as the configured model
    path (e.g. ``yolov8n.onnx`` -> ``yolov8n.hef``).
    """
    from ratcatcher.detection.hailo_detector import HailoDetector

    hef_path = model_base.with_suffix(".hef")
    detector: ObjectDetector = HailoDetector(
        model_path=hef_path,
        confidence_threshold=config.confidence_threshold,
        nms_threshold=config.nms_threshold,
        input_size=config.input_size,
    )
    logger.info("Detection backend: hailo (%s)", hef_path.name)
    return detector


def _make_ncnn(
    config: DetectionConfig, model_base: Path
) -> ObjectDetector:
    """Construct an NCNNDetector.

    Expects ``.param`` and ``.bin`` files with the same stem as the
    configured model path.
    """
    from ratcatcher.detection.ncnn_detector import NCNNDetector

    detector: ObjectDetector = NCNNDetector(
        model_path=model_base,
        confidence_threshold=config.confidence_threshold,
        nms_threshold=config.nms_threshold,
        input_size=config.input_size,
    )
    logger.info("Detection backend: ncnn (%s)", model_base.stem)
    return detector


def _make_opencv(
    config: DetectionConfig, model_base: Path
) -> ObjectDetector:
    """Construct an OpenCVDetector using the ``.onnx`` model directly."""
    from ratcatcher.detection.opencv_detector import OpenCVDetector

    onnx_path = model_base
    # If the caller's model_path didn't already end in .onnx, try adding it.
    if not onnx_path.suffix:
        onnx_path = onnx_path.with_suffix(".onnx")

    detector: ObjectDetector = OpenCVDetector(
        model_path=onnx_path,
        confidence_threshold=config.confidence_threshold,
        nms_threshold=config.nms_threshold,
        input_size=config.input_size,
    )
    logger.info("Detection backend: opencv_dnn (%s)", onnx_path.name)
    return detector
