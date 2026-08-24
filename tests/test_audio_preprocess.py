"""Tests for SPH0645 signal conditioning.

Real numpy signals throughout, no test doubles. The DC blocker is the
one piece of the audio path that every later stage depends on, because
the SPH0645 always emits a large DC bias and every energy measurement
downstream is wrong until it is removed.
"""

from __future__ import annotations

import numpy as np
import pytest

from ratcatcher.audio.preprocess import (
    DCBlocker,
    normalise_peak,
    peak_dbfs,
    rms,
    rms_dbfs,
    split_channels,
    to_mono,
)

SAMPLE_RATE = 48000


def _tone(frequency: float, seconds: float, amplitude: float = 0.05) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


class TestDCBlocker:
    def test_removes_constant_offset(self):
        """The SPH0645's DC bias must be gone after filtering."""
        signal = _tone(2000, 1.0) + 0.4
        block = np.stack([signal, signal], axis=1)

        output = DCBlocker(SAMPLE_RATE, 150.0, 2).process(block)

        assert abs(float(block.mean()) - 0.4) < 0.01
        assert abs(float(output.mean())) < 1e-3

    def test_preserves_bird_band_signal(self):
        """A 2 kHz tone must survive with its amplitude essentially intact."""
        signal = _tone(2000, 1.0) + 0.4
        block = np.stack([signal, signal], axis=1)

        output = DCBlocker(SAMPLE_RATE, 150.0, 2).process(block)

        # Skip the settling region at the very start of the stream.
        settled = output[SAMPLE_RATE // 2 :, 0]
        assert rms(settled) == pytest.approx(0.05 / np.sqrt(2), rel=0.05)

    def test_attenuates_low_frequency_rumble(self):
        """Wind rumble below the corner must be strongly reduced."""
        rumble = _tone(30, 1.0, amplitude=0.3)
        block = rumble.reshape(-1, 1)

        output = DCBlocker(SAMPLE_RATE, 150.0, 1).process(block)

        settled = output[SAMPLE_RATE // 2 :, 0]
        assert rms(settled) < rms(rumble) * 0.3

    def test_block_boundaries_are_seamless(self):
        """Filtering in chunks must equal filtering the whole stream."""
        signal = _tone(2000, 1.0) + 0.4
        block = np.stack([signal, signal], axis=1)

        whole = DCBlocker(SAMPLE_RATE, 150.0, 2).process(block)

        chunked_filter = DCBlocker(SAMPLE_RATE, 150.0, 2)
        chunks = [
            chunked_filter.process(block[start : start + 4800])
            for start in range(0, block.shape[0], 4800)
        ]
        chunked = np.concatenate(chunks, axis=0)

        assert np.allclose(whole, chunked, atol=1e-6)

    def test_window_length_follows_cutoff(self):
        """A lower corner frequency needs a longer averaging window."""
        low = DCBlocker(SAMPLE_RATE, 50.0, 1)
        high = DCBlocker(SAMPLE_RATE, 500.0, 1)

        assert low.window > high.window

    def test_channel_mismatch_is_rejected(self):
        blocker = DCBlocker(SAMPLE_RATE, 150.0, 2)
        with pytest.raises(ValueError, match="channels"):
            blocker.process(np.zeros((100, 1), dtype=np.float32))

    def test_rejects_cutoff_above_nyquist(self):
        with pytest.raises(ValueError, match="Nyquist"):
            DCBlocker(SAMPLE_RATE, SAMPLE_RATE, 1)

    def test_empty_block_is_returned_unchanged(self):
        blocker = DCBlocker(SAMPLE_RATE, 150.0, 2)
        output = blocker.process(np.zeros((0, 2), dtype=np.float32))
        assert output.shape == (0, 2)

    def test_reset_clears_state(self):
        """After reset, the filter behaves as if freshly constructed."""
        signal = (_tone(2000, 0.2) + 0.4).reshape(-1, 1)

        blocker = DCBlocker(SAMPLE_RATE, 150.0, 1)
        first = blocker.process(signal)
        blocker.reset()
        second = blocker.process(signal)

        assert np.allclose(first, second, atol=1e-6)


class TestChannelHelpers:
    def test_split_channels_separates_microphones(self):
        left = _tone(2000, 0.1)
        right = _tone(4000, 0.1)
        block = np.stack([left, right], axis=1)

        channels = split_channels(block)

        assert len(channels) == 2
        assert np.allclose(channels[0], left)
        assert np.allclose(channels[1], right)

    def test_to_mono_averages_channels(self):
        block = np.stack(
            [np.full(100, 0.5, dtype=np.float32), np.full(100, 0.1, dtype=np.float32)],
            axis=1,
        )
        assert np.allclose(to_mono(block), 0.3)

    def test_to_mono_passes_through_1d(self):
        signal = _tone(2000, 0.1)
        assert np.allclose(to_mono(signal), signal)


class TestLevels:
    def test_rms_of_sine_is_amplitude_over_root_two(self):
        assert rms(_tone(1000, 1.0, amplitude=0.5)) == pytest.approx(
            0.5 / np.sqrt(2), rel=0.01
        )

    def test_silence_reports_negative_infinity(self):
        silence = np.zeros(1000, dtype=np.float32)
        assert rms_dbfs(silence) == float("-inf")
        assert peak_dbfs(silence) == float("-inf")

    def test_full_scale_peak_is_zero_dbfs(self):
        signal = np.array([1.0, -1.0, 0.5], dtype=np.float32)
        assert peak_dbfs(signal) == pytest.approx(0.0, abs=0.01)

    def test_normalise_peak_scales_to_target(self):
        scaled = normalise_peak(_tone(1000, 0.1, amplitude=0.01), target_peak=0.7)
        assert float(np.max(np.abs(scaled))) == pytest.approx(0.7, rel=0.01)

    def test_normalise_peak_leaves_silence_alone(self):
        """Amplifying silence would turn a dead channel into full-scale noise."""
        silence = np.zeros(100, dtype=np.float32)
        assert np.allclose(normalise_peak(silence), 0.0)
