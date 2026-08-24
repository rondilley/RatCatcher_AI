"""WAV clip writing for audio detections.

Bird song clips are written as plain 16-bit PCM WAV using the standard
library. Unlike video, there is no reason to shell out to FFmpeg here:
the clips are short, WAV needs no encoder, and keeping the write in
process means a detection is never lost to a subprocess failure.
"""

from __future__ import annotations

import logging
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def write_wav(
    path: str | Path,
    samples: np.ndarray,
    sample_rate: int,
) -> Path:
    """Write float audio to a 16-bit PCM WAV file.

    Parameters
    ----------
    path:
        Destination file. Parent directories are created as needed.
    samples:
        Float audio in roughly [-1.0, 1.0]. Either 1-D mono or 2-D
        ``(frames, channels)``.
    sample_rate:
        Sample rate in Hz.

    Returns
    -------
    The path written.

    Raises
    ------
    ValueError
        If the sample rate is not positive or the array is not 1-D or 2-D.
    OSError
        If the file could not be written.
    """
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    if samples.ndim not in (1, 2):
        raise ValueError(
            f"samples must be 1-D or 2-D, got shape {samples.shape}"
        )

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    channels = 1 if samples.ndim == 1 else samples.shape[1]

    # Clip before scaling. A signal that overshoots full scale would
    # otherwise wrap around on cast and turn a loud call into harsh noise.
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")

    try:
        with wave.open(str(destination), "wb") as handle:
            handle.setnchannels(channels)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(pcm.tobytes())
    except (wave.Error, OSError) as exc:
        raise OSError(f"Could not write WAV file {destination}: {exc}") from exc

    logger.debug(
        "Wrote %s (%.2f s, %d channel(s))",
        destination,
        samples.shape[0] / sample_rate,
        channels,
    )
    return destination


def clip_filename(timestamp: datetime, channel: int, species: str | None) -> str:
    """Build a sortable, filesystem-safe clip name.

    Leading with the timestamp means a directory listing is chronological
    without any extra sorting.
    """
    stamp = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
    label = _sanitise(species) if species else "unknown"
    return f"{stamp}_ch{channel}_{label}.wav"


def _sanitise(name: str) -> str:
    """Reduce a species name to characters that are safe in a filename."""
    safe = [
        char if (char.isalnum() or char in "-_") else "_"
        for char in name.strip()
    ]
    collapsed = "".join(safe)
    while "__" in collapsed:
        collapsed = collapsed.replace("__", "_")
    return collapsed.strip("_") or "unknown"
