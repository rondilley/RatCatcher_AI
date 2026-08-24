"""Tests for audio capture and WAV clip writing.

Real WAV files written to real temporary directories, matching the
project's no-test-doubles policy.

The live ALSA path is exercised against ALSA's "null" PCM, which
supports capture without any hardware. That covers the subprocess
handling, S32_LE format negotiation, byte parsing and shutdown --
everything in the capture path except the microphones themselves.

What still cannot be tested here, and needs the I2S overlay enabled and
the Pi rebooted: whether the googlevoicehat overlay binds on this
kernel, whether the SPH0645 really is left-justified in its 32-bit slot
as raw_to_float32 assumes, and whether the SEL wiring puts one mic on
each channel.
"""

from __future__ import annotations

import wave
from datetime import datetime

import numpy as np
import pytest

from ratcatcher.audio.capture import (
    ArecordSource,
    AudioSource,
    WavFileSource,
    raw_to_float32,
)
from ratcatcher.audio.clip_writer import clip_filename, write_wav

SAMPLE_RATE = 48000


def _write_pcm_wav(path, samples: np.ndarray, sample_width: int = 2) -> None:
    """Write a real WAV file at a given bit depth."""
    channels = 1 if samples.ndim == 1 else samples.shape[1]
    clipped = np.clip(samples, -1.0, 1.0)

    if sample_width == 2:
        raw = (clipped * 32767).astype("<i2").tobytes()
    elif sample_width == 4:
        raw = (clipped * (2**31 - 1)).astype("<i4").tobytes()
    elif sample_width == 3:
        values = (clipped * (2**23 - 1)).astype("<i4").flatten()
        raw = b"".join(int(v).to_bytes(4, "little", signed=True)[:3] for v in values)
    else:
        raise ValueError(f"unsupported width {sample_width}")

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(sample_width)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(raw)


def _stereo_tone(seconds: float = 1.0) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    left = 0.5 * np.sin(2 * np.pi * 1000 * t)
    right = 0.25 * np.sin(2 * np.pi * 2000 * t)
    return np.stack([left, right], axis=1).astype(np.float32)


class TestRawConversion:
    def test_full_scale_maps_to_unity(self):
        """The SPH0645 leaves data left-justified, so 2**31 is full scale."""
        raw = np.array([2**31 - 1, -(2**31)], dtype="<i4").tobytes()
        block = raw_to_float32(raw, channels=1)
        assert block[0, 0] == pytest.approx(1.0, abs=1e-6)
        assert block[1, 0] == pytest.approx(-1.0, abs=1e-6)

    def test_channels_are_deinterleaved(self):
        raw = np.array([100, 200, 300, 400], dtype="<i4").tobytes()
        block = raw_to_float32(raw, channels=2)
        assert block.shape == (2, 2)
        assert block[0, 0] < block[0, 1]  # 100 < 200 within frame 0

    def test_partial_frame_is_discarded(self):
        """ALSA can return a partial frame when a capture stops mid-block."""
        raw = np.array([1, 2, 3], dtype="<i4").tobytes()
        block = raw_to_float32(raw, channels=2)
        assert block.shape == (1, 2)

    def test_empty_input_yields_empty_block(self):
        assert raw_to_float32(b"", channels=2).shape == (0, 2)

    def test_zero_channels_is_rejected(self):
        with pytest.raises(ValueError, match="channels"):
            raw_to_float32(b"", channels=0)


class TestWavFileSource:
    def test_reads_stereo_at_correct_rate(self, tmp_path):
        path = tmp_path / "stereo.wav"
        _write_pcm_wav(path, _stereo_tone())

        source = WavFileSource(path)
        assert source.sample_rate == SAMPLE_RATE
        assert source.channels == 2

        source.start()
        ok, block = source.read(1000)
        source.stop()

        assert ok
        assert block.shape == (1000, 2)

    def test_channels_are_not_swapped(self, tmp_path):
        """Channel 0 is the left microphone and must stay that way."""
        path = tmp_path / "stereo.wav"
        _write_pcm_wav(path, _stereo_tone())

        source = WavFileSource(path)
        source.start()
        _, block = source.read(SAMPLE_RATE // 2)
        source.stop()

        left_rms = float(np.sqrt(np.mean(block[:, 0] ** 2)))
        right_rms = float(np.sqrt(np.mean(block[:, 1] ** 2)))
        assert left_rms > right_rms * 1.5

    @pytest.mark.parametrize("width", [2, 3, 4])
    def test_supported_bit_depths_round_trip(self, tmp_path, width):
        path = tmp_path / f"depth{width}.wav"
        signal = _stereo_tone(0.2)
        _write_pcm_wav(path, signal, sample_width=width)

        source = WavFileSource(path)
        source.start()
        ok, block = source.read(1000)
        source.stop()

        assert ok
        assert np.allclose(block, signal[:1000], atol=1e-3)

    def test_end_of_file_reports_failure(self, tmp_path):
        path = tmp_path / "short.wav"
        _write_pcm_wav(path, _stereo_tone(0.05))

        source = WavFileSource(path)
        source.start()
        ok, block = source.read(SAMPLE_RATE)
        source.stop()

        assert not ok
        assert block is None

    def test_looping_returns_a_full_block_at_the_wrap(self, tmp_path):
        """Looping must not emit a short block every time it wraps."""
        path = tmp_path / "loop.wav"
        _write_pcm_wav(path, _stereo_tone(0.05))

        source = WavFileSource(path, loop=True)
        source.start()
        for _ in range(5):
            ok, block = source.read(1000)
            assert ok
            assert block.shape == (1000, 2)
        source.stop()

    def test_missing_file_is_rejected_at_construction(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            WavFileSource(tmp_path / "nope.wav")

    def test_non_wav_content_is_rejected(self, tmp_path):
        path = tmp_path / "bogus.wav"
        path.write_bytes(b"this is not a wav file")
        with pytest.raises(ValueError, match="Could not read"):
            WavFileSource(path)

    def test_file_source_is_not_realtime(self, tmp_path):
        """Drives backpressure: a file must not silently drop windows."""
        path = tmp_path / "s.wav"
        _write_pcm_wav(path, _stereo_tone(0.1))
        assert WavFileSource(path).is_realtime is False

    def test_satisfies_the_audio_source_protocol(self, tmp_path):
        path = tmp_path / "s.wav"
        _write_pcm_wav(path, _stereo_tone(0.1))
        assert isinstance(WavFileSource(path), AudioSource)


class TestArecordSource:
    def test_live_capture_is_realtime(self):
        """Drives dropping: stalling the reader causes ALSA overruns."""
        assert ArecordSource().is_realtime is True

    def test_satisfies_the_audio_source_protocol(self):
        assert isinstance(ArecordSource(), AudioSource)

    def test_device_enumeration_never_raises(self):
        """Must return an empty list, not blow up, before the mics exist."""
        assert isinstance(ArecordSource.list_capture_devices(), list)

    def test_reading_before_start_fails_cleanly(self):
        ok, block = ArecordSource().read(1000)
        assert not ok
        assert block is None

    def test_non_positive_read_is_rejected(self):
        with pytest.raises(ValueError, match="num_frames"):
            ArecordSource().read(0)

    def test_unknown_device_raises_rather_than_hanging(self):
        """A wrong device name must surface the ALSA error immediately."""
        if not ArecordSource.is_available():
            pytest.skip("arecord not installed")
        with pytest.raises(RuntimeError, match="arecord"):
            ArecordSource(device="ratcatcher_no_such_device").start()

    def test_capture_against_the_alsa_null_device(self):
        """Exercises the real subprocess, format negotiation and parsing.

        ALSA's 'null' PCM supports capture without any hardware, so this
        covers everything in the live path except the microphones
        themselves.
        """
        if not ArecordSource.is_available():
            pytest.skip("arecord not installed")

        source = ArecordSource(device="null", sample_rate=SAMPLE_RATE, channels=2)
        try:
            source.start()
            ok, block = source.read(4096)
        finally:
            source.stop()

        assert ok
        assert block.shape == (4096, 2)
        assert block.dtype == np.float32
        assert np.all(np.isfinite(block))
        assert not source.is_running

    def test_stop_does_not_stall_on_a_full_pipe(self):
        """Regression: arecord blocks on a full stdout pipe and then
        never reaches its SIGTERM handler. Closing the read end first
        makes it exit on EPIPE. Without that fix this took 5 seconds and
        needed SIGKILL on every single shutdown.
        """
        if not ArecordSource.is_available():
            pytest.skip("arecord not installed")

        import time

        source = ArecordSource(device="null", sample_rate=SAMPLE_RATE, channels=2)
        source.start()
        source.read(4096)          # leave the pipe filling behind us

        started = time.perf_counter()
        source.stop()
        assert time.perf_counter() - started < 2.0

    def test_repeated_start_stop_cycles_are_clean(self):
        if not ArecordSource.is_available():
            pytest.skip("arecord not installed")

        source = ArecordSource(device="null", sample_rate=SAMPLE_RATE, channels=2)
        for _ in range(3):
            source.start()
            ok, _ = source.read(1024)
            assert ok
            source.stop()
            assert not source.is_running


class TestClipWriter:
    def test_round_trips_through_a_real_file(self, tmp_path):
        signal = _stereo_tone(0.5)
        path = write_wav(tmp_path / "clip.wav", signal, SAMPLE_RATE)

        assert path.exists()
        source = WavFileSource(path)
        source.start()
        ok, block = source.read(1000)
        source.stop()

        assert ok
        assert np.allclose(block, signal[:1000], atol=1e-3)

    def test_creates_missing_parent_directories(self, tmp_path):
        path = write_wav(
            tmp_path / "a" / "b" / "clip.wav",
            _stereo_tone(0.05),
            SAMPLE_RATE,
        )
        assert path.exists()

    def test_mono_is_written_as_one_channel(self, tmp_path):
        mono = _stereo_tone(0.1)[:, 0]
        path = write_wav(tmp_path / "mono.wav", mono, SAMPLE_RATE)

        with wave.open(str(path), "rb") as handle:
            assert handle.getnchannels() == 1

    def test_overshoot_is_clipped_not_wrapped(self, tmp_path):
        """Wrapping on cast would turn a loud call into harsh noise."""
        loud = np.full(1000, 2.0, dtype=np.float32)
        path = write_wav(tmp_path / "loud.wav", loud, SAMPLE_RATE)

        source = WavFileSource(path)
        source.start()
        _, block = source.read(100)
        source.stop()

        assert float(np.min(block)) > 0.9

    def test_invalid_sample_rate_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="sample_rate"):
            write_wav(tmp_path / "x.wav", _stereo_tone(0.05), 0)

    def test_three_dimensional_input_is_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="1-D or 2-D"):
            write_wav(tmp_path / "x.wav", np.zeros((10, 2, 2), dtype=np.float32), SAMPLE_RATE)


class TestClipFilename:
    def test_leads_with_a_sortable_timestamp(self):
        name = clip_filename(datetime(2026, 8, 23, 14, 5, 9), 0, "Sialia mexicana")
        assert name.startswith("20260823_140509")
        assert name.endswith(".wav")

    def test_records_the_channel(self):
        assert "_ch1_" in clip_filename(datetime(2026, 8, 23), 1, "Corvus corax")

    def test_species_spaces_become_safe_characters(self):
        name = clip_filename(datetime(2026, 8, 23), 0, "Sialia mexicana")
        assert "Sialia_mexicana" in name
        assert " " not in name

    def test_unidentified_audio_is_labelled_unknown(self):
        assert "unknown" in clip_filename(datetime(2026, 8, 23), 0, None)

    def test_path_separators_cannot_escape_the_directory(self):
        name = clip_filename(datetime(2026, 8, 23), 0, "../../etc/passwd")
        assert "/" not in name
        assert ".." not in name
