"""Platform-aware camera factory.

Selects the appropriate CameraSource implementation based on the
``source_type`` field of a ``CameraConfig`` and the current platform.
"""

from __future__ import annotations

import logging

from ratcatcher.camera.capture import (
    CameraSource,
    FileSource,
    WebcamSource,
    _PICAMERA2_AVAILABLE,
)
from ratcatcher.config import CameraConfig

logger = logging.getLogger(__name__)


def create_camera(config: CameraConfig) -> CameraSource:
    """Instantiate the correct camera back-end for *config*.

    Parameters
    ----------
    config : CameraConfig
        Camera configuration loaded from the project YAML.

    Returns
    -------
    CameraSource
        A camera source ready to be ``start()``-ed.

    Raises
    ------
    RuntimeError
        If the requested source type is unavailable on this platform.
    ValueError
        If ``source_type`` is not recognised.
    """
    source_type = config.source_type.lower().strip()

    if source_type == "file":
        return _create_file_source(config)

    if source_type == "webcam":
        return _create_webcam_source(config)

    if source_type == "picamera":
        return _create_picamera_source(config)

    if source_type == "auto":
        return _create_auto_source(config)

    raise ValueError(f"Unknown camera source_type: {config.source_type!r}")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _create_file_source(config: CameraConfig) -> FileSource:
    if config.file_path is None:
        raise ValueError(
            "CameraConfig.file_path must be set when source_type is 'file'"
        )
    return FileSource(
        path=config.file_path,
        fps=config.fps,
        resolution=config.resolution,
    )


def _create_webcam_source(config: CameraConfig) -> WebcamSource:
    return WebcamSource(
        device_index=config.device_index,
        resolution=config.resolution,
        fps=config.fps,
    )


def _create_picamera_source(config: CameraConfig) -> CameraSource:
    if not _PICAMERA2_AVAILABLE:
        raise RuntimeError(
            "PicameraSource is not available on this platform. "
            "Install picamera2 on a Raspberry Pi to use source_type='picamera'."
        )
    # Import here so the class name is only resolved when we know it exists.
    from ratcatcher.camera.capture import PicameraSource  # type: ignore[attr-defined]

    return PicameraSource(
        camera_num=config.device_index,
        resolution=config.capture_resolution or config.resolution,
        fps=config.fps,
    )


def _create_auto_source(config: CameraConfig) -> CameraSource:
    """Try picamera first (if available), then fall back to webcam."""
    if _PICAMERA2_AVAILABLE:
        logger.info(
            "Auto-detect: Picamera2 available -- using PicameraSource "
            "(camera_num=%d)",
            config.device_index,
        )
        return _create_picamera_source(config)

    logger.info(
        "Auto-detect: Picamera2 not available -- falling back to "
        "WebcamSource (device_index=%d)",
        config.device_index,
    )
    return _create_webcam_source(config)
