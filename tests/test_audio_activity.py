"""Tests for the sound activity gate.

The gate's failure mode that matters is rejecting a real bird, so these
tests lead with signals shaped like real song: frequency-sweeping calls,
short calls surrounded by silence, and calls buried in background noise.

The regression these lock down is concrete. An earlier version measured
spectral flatness across the whole three-second window and used an
absolute dBFS threshold. Against a real field soundscape it rejected 28
of the 29 windows that contained a confidently identified bird, because
a warbling call spread across many bins over three seconds and because
field recordings sit far below any absolute threshold picked in advance.
"""

from __future__ import annotations

import numpy as np
import pytest

from ratcatcher.audio.activity import SoundActivityDetector

SAMPLE_RATE = 48000
WINDOW = SAMPLE_RATE * 3


def _seconds(count: float) -> np.ndarray:
    return np.arange(int(SAMPLE_RATE * count)) / SAMPLE_RATE


def _warble(seconds: float = 3.0, amplitude: float = 0.05) -> np.ndarray:
    """A frequency-sweeping tone, which is what real bird song looks like."""
    t = _seconds(seconds)
    instantaneous = 3500 + 700 * np.sin(2 * np.pi * 10 * t)
    return (amplitude * np.sin(2 * np.pi * instantaneous * t)).astype(np.float32)


def _noise(seconds: float = 3.0, amplitude: float = 0.02, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(SAMPLE_RATE * seconds)) * amplitude).astype(
        np.float32
    )


def _settled_gate(background: np.ndarray, **kwargs) -> SoundActivityDetector:
    """Return a detector whose noise floor has converged on a background."""
    gate = SoundActivityDetector(SAMPLE_RATE, **kwargs)
    for _ in range(6):
        gate.evaluate(background)
    return gate


class TestRejectsWhatHasNoSignal:
    def test_digital_silence_is_rejected(self):
        gate = SoundActivityDetector(SAMPLE_RATE)
        assert not gate.evaluate(np.zeros(WINDOW, dtype=np.float32)).triggered

    def test_steady_broadband_noise_is_rejected(self):
        """Rain and a dead preamp both look like this and must not trigger."""
        gate = SoundActivityDetector(SAMPLE_RATE)
        result = None
        for _ in range(6):
            result = gate.evaluate(_noise(seed=1))
        assert not result.triggered

    def test_low_frequency_rumble_is_rejected(self):
        """Wind sits below the bird band and carries no in-band energy."""
        rumble = (0.3 * np.sin(2 * np.pi * 60 * _seconds(3.0))).astype(np.float32)
        gate = SoundActivityDetector(SAMPLE_RATE)
        assert not gate.evaluate(rumble).triggered


class TestAcceptsRealSong:
    def test_warbling_call_triggers(self):
        """The regression case: an FM sweep must not read as noise."""
        gate = _settled_gate(_noise(amplitude=0.0005, seed=2))
        assert gate.evaluate(_warble()).triggered

    def test_short_call_in_silence_triggers(self):
        """A 0.4 s call must not be diluted by the surrounding quiet."""
        window = _noise(amplitude=0.0005, seed=3)
        call = _warble(seconds=0.4, amplitude=0.05)
        window[SAMPLE_RATE : SAMPLE_RATE + call.size] = call

        gate = _settled_gate(_noise(amplitude=0.0005, seed=3))
        assert gate.evaluate(window).triggered

    def test_call_over_background_noise_triggers(self):
        gate = _settled_gate(_noise(amplitude=0.005, seed=4))
        mixed = _warble() + _noise(amplitude=0.005, seed=5)
        assert gate.evaluate(mixed.astype(np.float32)).triggered

    def test_song_is_tonal(self):
        """Flatness must separate song from noise, not merely pass it."""
        gate = SoundActivityDetector(SAMPLE_RATE)
        song = gate.evaluate(_warble())
        noise = gate.evaluate(_noise(seed=6))
        assert song.spectral_flatness < 0.2
        assert noise.spectral_flatness > song.spectral_flatness


class TestNoiseFloorTracking:
    def test_floor_is_unset_before_first_window(self):
        assert SoundActivityDetector(SAMPLE_RATE).noise_floor_dbfs is None

    def test_floor_converges_on_background_level(self):
        gate = _settled_gate(_noise(amplitude=0.01, seed=7))
        assert gate.noise_floor_dbfs is not None
        assert -80.0 < gate.noise_floor_dbfs < -20.0

    def test_loud_sustained_call_does_not_gate_itself_out(self):
        """The floor must rise slowly or a long song silences itself."""
        gate = _settled_gate(_noise(amplitude=0.0005, seed=8))
        triggered = [gate.evaluate(_warble()).triggered for _ in range(10)]
        assert all(triggered)

    def test_floor_adapts_to_a_louder_location(self):
        """A gate tuned in silence must still work in a noisy place."""
        quiet_gate = _settled_gate(_noise(amplitude=0.0005, seed=9))
        loud_gate = _settled_gate(_noise(amplitude=0.02, seed=10))

        assert loud_gate.noise_floor_dbfs > quiet_gate.noise_floor_dbfs

        # The same call, scaled to each environment, triggers in both.
        assert quiet_gate.evaluate(_warble(amplitude=0.005)).triggered
        assert loud_gate.evaluate(_warble(amplitude=0.2)).triggered

    def test_reset_forgets_the_floor(self):
        gate = _settled_gate(_noise(seed=11))
        gate.reset()
        assert gate.noise_floor_dbfs is None


class TestMeasurementsAndValidation:
    def test_peak_frequency_locates_a_pure_tone(self):
        tone = (0.05 * np.sin(2 * np.pi * 4000 * _seconds(3.0))).astype(np.float32)
        result = SoundActivityDetector(SAMPLE_RATE).evaluate(tone)
        assert result.peak_frequency_hz == pytest.approx(4000, abs=50)

    def test_result_renders_every_measurement(self):
        text = str(SoundActivityDetector(SAMPLE_RATE).evaluate(_warble()))
        for field in ("band=", "floor=", "snr=", "flatness=", "peak="):
            assert field in text

    def test_window_shorter_than_a_frame_is_rejected_not_crashed(self):
        gate = SoundActivityDetector(SAMPLE_RATE)
        assert not gate.evaluate(np.zeros(100, dtype=np.float32)).triggered

    def test_stereo_input_is_rejected(self):
        """The gate is per channel; passing both would silently misreport."""
        gate = SoundActivityDetector(SAMPLE_RATE)
        with pytest.raises(ValueError, match="1-D"):
            gate.evaluate(np.zeros((WINDOW, 2), dtype=np.float32))

    def test_band_above_nyquist_is_rejected(self):
        with pytest.raises(ValueError, match="Nyquist"):
            SoundActivityDetector(SAMPLE_RATE, band_high_hz=SAMPLE_RATE)

    def test_inverted_band_is_rejected(self):
        with pytest.raises(ValueError, match="band_low_hz"):
            SoundActivityDetector(SAMPLE_RATE, band_low_hz=9000, band_high_hz=1000)

    def test_out_of_range_flatness_is_rejected(self):
        with pytest.raises(ValueError, match="max_flatness"):
            SoundActivityDetector(SAMPLE_RATE, max_flatness=1.5)
