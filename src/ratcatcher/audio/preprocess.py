"""Signal conditioning for the SPH0645 I2S MEMS microphones.

The SPH0645 needs two corrections before its output is usable:

1. A large DC offset. The part has no output coupling capacitor, so every
   sample sits on a constant bias. Left in place, that bias dominates any
   energy measurement and shows up as a huge spike at 0 Hz in the
   spectrum, which wrecks both the activity gate and the classifier.

2. Low-frequency rumble. Wind, traffic and enclosure vibration live below
   the range of bird song. Removing them costs nothing and materially
   improves the signal-to-noise ratio outdoors.

Both are handled by the same one-pole high-pass filter. It is applied
per channel and keeps its state between blocks so a continuous stream is
filtered without discontinuities at block boundaries.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class DCBlocker:
    """Stateful high-pass filter built from moving-average subtraction.

    Computes ``y[n] = x[n] - mean(x[n-N+1 .. n])``. Subtracting a running
    mean is a high-pass whose minus-3 dB corner falls at approximately
    ``0.443 * sample_rate / N``, so the window length is derived from the
    requested cutoff.

    A recursive one-pole filter would give a cleaner stopband, but it
    cannot be vectorised in numpy: the per-sample Python loop it requires
    measured 155 ms per second of stereo audio on this Pi 5, or about 15
    percent of a core burnt continuously alongside two cameras and YOLO.
    The moving average is computed with a single cumulative sum, which is
    three orders of magnitude cheaper and entirely adequate for the job
    here. Stopband ripple does not matter when the goal is removing DC
    and rumble that carry no signal in the first place.

    State (the trailing ``N-1`` samples) is retained between calls, so a
    stream split into blocks filters identically to the same stream
    filtered whole.

    Parameters
    ----------
    sample_rate:
        Sample rate in Hz.
    cutoff_hz:
        Approximate minus-3 dB corner frequency. The default of 150 Hz
        sits well below the fundamental of essentially every North
        American songbird while still removing DC and wind rumble.
    channels:
        Number of independent channels to track state for.
    """

    def __init__(
        self,
        sample_rate: int,
        cutoff_hz: float = 150.0,
        channels: int = 2,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")
        if not 0.0 < cutoff_hz < sample_rate / 2.0:
            raise ValueError(
                f"cutoff_hz must be between 0 and the Nyquist frequency "
                f"({sample_rate / 2.0} Hz), got {cutoff_hz}"
            )
        if channels < 1:
            raise ValueError(f"channels must be >= 1, got {channels}")

        self._sample_rate = sample_rate
        self._cutoff_hz = cutoff_hz
        self._channels = channels

        # Window length for the requested corner frequency, floored at 3
        # so the filter always spans a meaningful average.
        self._window = max(3, int(round(0.443 * sample_rate / cutoff_hz)))

        self._history = np.zeros((self._window - 1, channels), dtype=np.float64)
        self._primed = False

    def reset(self) -> None:
        """Clear filter state, as when a capture is restarted."""
        self._history[:] = 0.0
        self._primed = False

    def process(self, block: np.ndarray) -> np.ndarray:
        """Filter a block of shape ``(frames, channels)``.

        Returns a new float32 array of the same shape. The input is not
        modified.
        """
        if block.ndim != 2:
            raise ValueError(
                f"block must be 2-D (frames, channels), got shape {block.shape}"
            )
        if block.shape[1] != self._channels:
            raise ValueError(
                f"block has {block.shape[1]} channels, filter configured "
                f"for {self._channels}"
            )
        if block.shape[0] == 0:
            return block.astype(np.float32, copy=True)

        samples = block.astype(np.float64)

        # On the very first block, prime the history with the opening
        # sample instead of zeros. Starting from silence would otherwise
        # make the filter read the microphone's DC bias as a step change
        # and emit a large transient across the first window.
        if not self._primed:
            self._history[:] = samples[0]
            self._primed = True

        padded = np.concatenate((self._history, samples), axis=0)

        # Cumulative sum with a leading zero row, so the sum of the window
        # ending at output index i is cumulative[i + N] - cumulative[i].
        cumulative = np.concatenate(
            (np.zeros((1, self._channels), dtype=np.float64), np.cumsum(padded, axis=0)),
            axis=0,
        )
        count = samples.shape[0]
        window_sums = cumulative[self._window : self._window + count] - cumulative[:count]
        means = window_sums / float(self._window)

        self._history = padded[-(self._window - 1):].copy()

        return (samples - means).astype(np.float32)

    @property
    def cutoff_hz(self) -> float:
        return self._cutoff_hz

    @property
    def window(self) -> int:
        """Length of the moving-average window in samples."""
        return self._window

    @property
    def channels(self) -> int:
        return self._channels


def split_channels(block: np.ndarray) -> list[np.ndarray]:
    """Split an interleaved block into a list of 1-D per-channel arrays.

    With the documented wiring, index 0 is the left microphone (SEL tied
    to ground) and index 1 is the right (SEL tied to 3.3V).
    """
    if block.ndim != 2:
        raise ValueError(
            f"block must be 2-D (frames, channels), got shape {block.shape}"
        )
    return [np.ascontiguousarray(block[:, ch]) for ch in range(block.shape[1])]


def to_mono(block: np.ndarray) -> np.ndarray:
    """Average all channels into a single 1-D array."""
    if block.ndim == 1:
        return block.astype(np.float32, copy=True)
    if block.ndim != 2:
        raise ValueError(
            f"block must be 1-D or 2-D, got shape {block.shape}"
        )
    return block.mean(axis=1).astype(np.float32)


def rms(signal: np.ndarray) -> float:
    """Root-mean-square amplitude of a signal."""
    if signal.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))


def rms_dbfs(signal: np.ndarray) -> float:
    """RMS level in dB relative to full scale.

    Returns -inf for digital silence. Useful for reporting microphone
    level in ``ratcatcher test-mic`` and for logging capture health.
    """
    level = rms(signal)
    if level <= 0.0:
        return float("-inf")
    return float(20.0 * np.log10(level))


def peak_dbfs(signal: np.ndarray) -> float:
    """Peak absolute level in dB relative to full scale."""
    if signal.size == 0:
        return float("-inf")
    peak = float(np.max(np.abs(signal)))
    if peak <= 0.0:
        return float("-inf")
    return float(20.0 * np.log10(peak))


def normalise_peak(signal: np.ndarray, target_peak: float = 0.7) -> np.ndarray:
    """Scale a signal so its loudest sample reaches ``target_peak``.

    Digital silence is returned unchanged rather than amplified, which
    would otherwise turn a dead channel into full-scale noise.
    """
    if not 0.0 < target_peak <= 1.0:
        raise ValueError(f"target_peak must be in (0, 1], got {target_peak}")

    if signal.size == 0:
        return signal.astype(np.float32, copy=True)

    peak = float(np.max(np.abs(signal)))
    if peak <= 0.0:
        return signal.astype(np.float32, copy=True)

    return (signal * (target_peak / peak)).astype(np.float32)
