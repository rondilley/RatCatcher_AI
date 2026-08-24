"""Sound activity gate for RatCatcher AI.

Plays the same role for audio that motion detection plays for video: a
cheap pre-filter that keeps expensive inference off windows which
obviously contain nothing of interest. BirdNET costs far more than an
FFT, so rejecting silence and broadband noise up front is what makes
continuous listening affordable on a Pi.

A window is judged on two measurements:

* **Signal-to-noise ratio against a tracked noise floor.** How far the
  loudest moment rises above the ambient level at this location. An
  absolute threshold cannot work here: the level a bird produces at the
  microphone depends on how far away it is, and a threshold tuned in a
  quiet yard silences the gate entirely in a noisy one. The floor is
  estimated continuously from the quiet part of each window, falling
  quickly when conditions get quieter and rising slowly so a sustained
  call cannot drag it up behind itself.

* **Spectral flatness.** The ratio of the geometric to the arithmetic
  mean of the power spectrum, sometimes called Wiener entropy. It runs
  from 0 for a pure tone to 1 for white noise. Bird song is tonal and
  scores low; rain and wind score high.

The gate is deliberately permissive. BirdNET costs about 60 ms per
three-second window on a Pi 5, or roughly four percent of one core for
two microphones, so the saving from rejecting a window is small and the
cost of wrongly rejecting a real bird is total. Species filtering is
BirdNET's confidence threshold's job; this gate only turns away what is
plainly silence or plainly noise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# Analysis frame geometry. At 48 kHz a 2048-sample frame spans 43 ms, long
# enough to resolve the low end of the bird band and short enough that a
# frequency-sweeping call stays quasi-stationary within it. Half-overlap
# ensures a short call cannot straddle a frame boundary and be missed.
_FRAME_SIZE = 2048
_FRAME_HOP = 1024

# Frames within this many dB of the loudest are considered part of the
# same event and are the ones judged for tonality.
_TONALITY_RANGE_DB = 12.0

# Amplitude floor before taking a logarithm, roughly -300 dBFS. Keeps a
# digitally silent frame finite so it can be averaged and compared.
_LEVEL_FLOOR = 1e-15


@dataclass(frozen=True)
class ActivityResult:
    """Outcome of evaluating one window of audio."""

    triggered: bool
    band_rms_dbfs: float
    noise_floor_dbfs: float
    snr_db: float
    spectral_flatness: float
    peak_frequency_hz: float

    def __str__(self) -> str:
        state = "ACTIVE" if self.triggered else "quiet"
        return (
            f"{state} band={self.band_rms_dbfs:+.1f} dBFS "
            f"floor={self.noise_floor_dbfs:+.1f} snr={self.snr_db:+.1f} dB "
            f"flatness={self.spectral_flatness:.3f} "
            f"peak={self.peak_frequency_hz:.0f} Hz"
        )


class SoundActivityDetector:
    """Decide whether a window of audio plausibly contains bird song.

    Stateful: the noise floor is tracked across successive windows, so a
    detector instance belongs to one microphone channel and must not be
    shared between them.

    Parameters
    ----------
    sample_rate:
        Sample rate in Hz.
    band_low_hz, band_high_hz:
        Frequency range searched for activity. The default 1000-10000 Hz
        span covers North American songbird vocalisations while excluding
        most mechanical and weather noise.
    snr_margin_db:
        How far the loudest moment must rise above the tracked noise
        floor to trigger. This is the primary criterion.

        The default of 2 dB is deliberately low. Measured against a
        two-minute field soundscape with BirdNET identifications as
        ground truth, 2 dB kept 100 percent of the 29 windows containing
        a real bird, 4 dB lost four of them, and 6 dB lost fourteen. In a
        dawn chorus there is no quiet baseline to measure against, since
        the birds are the ambient sound, so a margin large enough to
        reject anything also rejects the subject. At 2 dB the gate still
        turns away digital silence and steady broadband noise, which is
        where it actually saves work: quiet nights and steady rain, not
        a busy morning.
    absolute_floor_dbfs:
        Hard minimum level. Guards against triggering on digital silence
        or on the noise floor of a disconnected microphone, where the SNR
        test alone can be satisfied by rounding noise.
    max_flatness:
        Maximum spectral flatness for a window to trigger. Windows above
        this are noise-like rather than tonal.
    floor_rise_rate, floor_fall_rate:
        Smoothing applied to the noise floor estimate per window. The
        floor falls quickly and rises slowly on purpose: a bird that
        sings for a minute must not raise the floor to its own level and
        gate itself out.
    """

    def __init__(
        self,
        sample_rate: int,
        band_low_hz: float = 1000.0,
        band_high_hz: float = 10000.0,
        snr_margin_db: float = 2.0,
        absolute_floor_dbfs: float = -85.0,
        max_flatness: float = 0.6,
        floor_rise_rate: float = 0.05,
        floor_fall_rate: float = 0.5,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")

        nyquist = sample_rate / 2.0
        if not 0.0 <= band_low_hz < band_high_hz:
            raise ValueError(
                f"require 0 <= band_low_hz < band_high_hz, got "
                f"{band_low_hz} and {band_high_hz}"
            )
        if band_high_hz > nyquist:
            raise ValueError(
                f"band_high_hz ({band_high_hz}) exceeds the Nyquist "
                f"frequency ({nyquist}) for a {sample_rate} Hz stream"
            )
        if not 0.0 <= max_flatness <= 1.0:
            raise ValueError(
                f"max_flatness must be in [0, 1], got {max_flatness}"
            )
        for name, rate in (
            ("floor_rise_rate", floor_rise_rate),
            ("floor_fall_rate", floor_fall_rate),
        ):
            if not 0.0 < rate <= 1.0:
                raise ValueError(f"{name} must be in (0, 1], got {rate}")

        self._sample_rate = sample_rate
        self._band_low_hz = band_low_hz
        self._band_high_hz = band_high_hz
        self._snr_margin_db = snr_margin_db
        self._absolute_floor_dbfs = absolute_floor_dbfs
        self._max_flatness = max_flatness
        self._floor_rise_rate = floor_rise_rate
        self._floor_fall_rate = floor_fall_rate

        self._noise_floor_dbfs: float | None = None

    def reset(self) -> None:
        """Forget the tracked noise floor, as when a capture restarts."""
        self._noise_floor_dbfs = None

    def evaluate(self, window: np.ndarray) -> ActivityResult:
        """Measure one mono window and decide whether it triggers.

        The window is analysed in short overlapping frames rather than as
        a whole. Measuring flatness across a full three-second window
        fails on exactly the signal this is meant to catch: real bird song
        sweeps in frequency, so over three seconds its energy is spread
        across many bins and it scores as noise-like. Within a 43 ms frame
        the same warble is quasi-stationary and reads as strongly tonal.

        Judging the loudest frame rather than the window average also
        means a brief call is not diluted by the silence around it.

        Parameters
        ----------
        window:
            1-D float32 audio, already DC-blocked.

        Returns
        -------
        An ``ActivityResult`` carrying the decision and every measurement
        behind it, so thresholds can be tuned from real recordings.
        """
        if window.ndim != 1:
            raise ValueError(
                f"window must be 1-D mono audio, got shape {window.shape}"
            )

        if window.size < _FRAME_SIZE:
            return ActivityResult(
                False, float("-inf"), float("-inf"), 0.0, 1.0, 0.0
            )

        frames = _frame_signal(window.astype(np.float64), _FRAME_SIZE, _FRAME_HOP)

        # Hann taper suppresses spectral leakage, which would otherwise
        # smear a loud tone across the spectrum and inflate flatness.
        taper = np.hanning(_FRAME_SIZE)
        spectra = np.fft.rfft(frames * taper, axis=1)
        power = np.abs(spectra) ** 2
        freqs = np.fft.rfftfreq(_FRAME_SIZE, d=1.0 / self._sample_rate)

        band = (freqs >= self._band_low_hz) & (freqs <= self._band_high_hz)
        if not np.any(band):
            return ActivityResult(
                False, float("-inf"), float("-inf"), 0.0, 1.0, 0.0
            )

        band_power = power[:, band]
        band_freqs = freqs[band]

        # Parseval: recover each frame's in-band time-domain RMS. The 8/3
        # factor compensates for the Hann window's power loss.
        frame_rms = np.sqrt(np.sum(band_power, axis=1) * (8.0 / 3.0)) / _FRAME_SIZE
        with np.errstate(divide="ignore"):
            frame_dbfs = 20.0 * np.log10(np.maximum(frame_rms, _LEVEL_FLOOR))

        loudest = int(np.argmax(frame_rms))
        band_rms_dbfs = float(frame_dbfs[loudest])
        peak_frequency = float(band_freqs[int(np.argmax(band_power[loudest]))])

        noise_floor = self._update_noise_floor(frame_dbfs)
        snr_db = band_rms_dbfs - noise_floor

        # Judge tonality on the loudest frames only. Quiet frames are
        # dominated by background and say nothing about the call.
        threshold = band_rms_dbfs - _TONALITY_RANGE_DB
        candidates = band_power[frame_dbfs >= threshold]
        flatness = float(
            np.min([_spectral_flatness(frame) for frame in candidates])
        )

        triggered = (
            band_rms_dbfs >= self._absolute_floor_dbfs
            and snr_db >= self._snr_margin_db
            and flatness <= self._max_flatness
        )

        return ActivityResult(
            triggered=triggered,
            band_rms_dbfs=band_rms_dbfs,
            noise_floor_dbfs=noise_floor,
            snr_db=snr_db,
            spectral_flatness=flatness,
            peak_frequency_hz=peak_frequency,
        )

    def _update_noise_floor(self, frame_dbfs: np.ndarray) -> float:
        """Track the ambient level using the quiet part of this window.

        The 25th percentile is used rather than the mean or minimum: a
        call occupying part of the window leaves the majority of frames at
        background level, and a percentile ignores both the call and any
        single anomalously quiet frame.
        """
        observed = float(np.percentile(frame_dbfs, 25))

        if self._noise_floor_dbfs is None:
            self._noise_floor_dbfs = observed
            return observed

        previous = self._noise_floor_dbfs
        rate = (
            self._floor_fall_rate if observed < previous else self._floor_rise_rate
        )
        self._noise_floor_dbfs = previous + rate * (observed - previous)
        return self._noise_floor_dbfs

    # -- properties --------------------------------------------------------

    @property
    def snr_margin_db(self) -> float:
        return self._snr_margin_db

    @property
    def absolute_floor_dbfs(self) -> float:
        return self._absolute_floor_dbfs

    @property
    def max_flatness(self) -> float:
        return self._max_flatness

    @property
    def noise_floor_dbfs(self) -> float | None:
        """Current ambient estimate, or None before the first window."""
        return self._noise_floor_dbfs

    @property
    def band_hz(self) -> tuple[float, float]:
        return (self._band_low_hz, self._band_high_hz)


def _frame_signal(signal: np.ndarray, frame_size: int, hop: int) -> np.ndarray:
    """Split a 1-D signal into overlapping frames without copying.

    Returns a 2-D array of shape ``(frames, frame_size)``. Uses a strided
    view, so the frames share memory with the input; callers must not
    write to the result. The FFT reads it only, and avoiding the copy
    keeps a three-second window off the allocator entirely.
    """
    if signal.size < frame_size:
        return np.empty((0, frame_size), dtype=signal.dtype)

    count = 1 + (signal.size - frame_size) // hop
    stride = signal.strides[0]
    return np.lib.stride_tricks.as_strided(
        signal,
        shape=(count, frame_size),
        strides=(stride * hop, stride),
        writeable=False,
    )


def _spectral_flatness(power: np.ndarray) -> float:
    """Ratio of geometric to arithmetic mean of a power spectrum.

    Returns 0.0 for a pure tone and approaches 1.0 for white noise. The
    geometric mean is computed in the log domain because the direct
    product of thousands of small values underflows to zero.
    """
    if power.size == 0:
        return 1.0

    arithmetic = float(np.mean(power))
    if arithmetic <= 0.0:
        return 1.0

    # Floor the spectrum before taking logs. Exact zeros occur in silent
    # or synthetic signals and would send the geometric mean to zero,
    # reporting a pure tone where there is no signal at all.
    floor = arithmetic * 1e-12
    geometric = float(np.exp(np.mean(np.log(np.maximum(power, floor)))))

    return float(np.clip(geometric / arithmetic, 0.0, 1.0))
