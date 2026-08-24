"""Audio source factory for RatCatcher AI.

Selects and constructs the appropriate audio backend based on the
``AudioConfig.source_type`` setting and the hardware available at
runtime, mirroring ``detection.factory`` and ``camera.platform_camera``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ratcatcher.audio.capture import ArecordSource, AudioSource, WavFileSource
from ratcatcher.config import AudioConfig

logger = logging.getLogger(__name__)


def create_audio_source(config: AudioConfig) -> AudioSource:
    """Build an ``AudioSource`` from an ``AudioConfig``.

    Parameters
    ----------
    config:
        Audio configuration section.

    Returns
    -------
    A constructed but not yet started audio source.

    Raises
    ------
    RuntimeError
        If the requested backend is unknown or unusable.
    """
    source_type = config.source_type.lower()

    if source_type == "auto":
        return _auto_select(config)

    if source_type == "alsa":
        return _make_alsa(config)

    if source_type == "file":
        return _make_file(config)

    raise RuntimeError(
        f"Unknown audio source type '{config.source_type}'. "
        f"Supported values: auto, alsa, file"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _auto_select(config: AudioConfig) -> AudioSource:
    """Try ALSA hardware first, then fall back to a configured file."""
    errors: list[str] = []

    try:
        return _make_alsa(config)
    except RuntimeError as exc:
        errors.append(f"alsa: {exc}")
        logger.debug("Auto-detect: ALSA capture unavailable -- %s", exc)

    if config.file_path:
        try:
            return _make_file(config)
        except (RuntimeError, FileNotFoundError, ValueError) as exc:
            errors.append(f"file: {exc}")
            logger.debug("Auto-detect: file source unavailable -- %s", exc)
    else:
        errors.append("file: no audio.file_path configured")

    raise RuntimeError(
        "No audio source available. Tried (in order):\n  " + "\n  ".join(errors)
    )


def _make_alsa(config: AudioConfig) -> AudioSource:
    """Construct an ArecordSource, verifying capture hardware exists."""
    if not ArecordSource.is_available():
        raise RuntimeError(
            "arecord not found. Install it with: sudo apt-get install alsa-utils"
        )

    devices = ArecordSource.list_capture_devices()
    if not devices:
        raise RuntimeError(
            "No ALSA capture devices found. The I2S microphones are not "
            "enabled yet. Run: sudo ./scripts/enable_i2s_mics.sh and reboot."
        )

    source: AudioSource = ArecordSource(
        device=config.device,
        sample_rate=config.sample_rate,
        channels=config.channels,
    )
    logger.info(
        "Audio source: alsa (device=%s, %d channels)",
        config.device,
        config.channels,
    )
    return source


def _make_file(config: AudioConfig) -> AudioSource:
    """Construct a WavFileSource from the configured path."""
    if not config.file_path:
        raise RuntimeError(
            "audio.source_type is 'file' but audio.file_path is not set"
        )

    source: AudioSource = WavFileSource(Path(config.file_path), loop=config.file_loop)
    logger.info("Audio source: file (%s)", config.file_path)
    return source
