"""Tests for the threaded audio pipeline and its storage.

Drives the real engine over real WAV files written to real temporary
directories, and reads results back out of a real SQLite database. No
test doubles anywhere, matching the rest of the suite.

BirdNET itself is not required: when the model is absent the engine
degrades to gate-only operation, and these tests assert that degraded
behaviour explicitly, since that is what a fresh checkout does before
``scripts/download_models.sh`` has run.
"""

from __future__ import annotations

import dataclasses
import logging
import wave

import numpy as np
import pytest

from ratcatcher.config import load_config
from ratcatcher.pipeline.audio_engine import AudioEngine, WindowAccumulator
from ratcatcher.storage.database import DetectionDatabase

SAMPLE_RATE = 48000


def _write_wav(path, samples: np.ndarray) -> None:
    channels = 1 if samples.ndim == 1 else samples.shape[1]
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())


def _call(seconds: float, amplitude: float = 0.2) -> np.ndarray:
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    return amplitude * np.sin(2 * np.pi * (3500 + 700 * np.sin(2 * np.pi * 10 * t)) * t)


def _stereo_recording(seconds: float = 9.0, dc_offset: float = 0.35) -> np.ndarray:
    """Calls on the left microphone only, over a realistic DC bias."""
    total = int(SAMPLE_RATE * seconds)
    rng = np.random.default_rng(0)
    left = rng.standard_normal(total) * 0.0005
    right = rng.standard_normal(total) * 0.0005

    call = _call(0.5)
    for start in (1.0, 4.5):
        begin = int(start * SAMPLE_RATE)
        if begin + call.size > total:
            break
        left[begin : begin + call.size] += call

    return np.stack([left + dc_offset, right + dc_offset], axis=1).astype(np.float32)


def _config(tmp_path, wav_path, **overrides):
    base = load_config("config/default.yaml")
    settings = {
        "enabled": True,
        "source_type": "file",
        "file_path": str(wav_path),
        "cooldown_seconds": 0.0,
        "block_seconds": 0.5,
    }
    settings.update(overrides)
    audio = dataclasses.replace(base.audio, **settings)
    return dataclasses.replace(
        base,
        audio=audio,
        system=dataclasses.replace(base.system, data_dir=str(tmp_path / "data")),
    )


def _run_to_completion(engine: AudioEngine, timeout: float = 60.0) -> None:
    """Start the engine, let the file drain, then stop it."""
    engine.start()
    capture = next(t for t in engine._threads if t.name == "audio-capture")
    capture.join(timeout=timeout)
    engine._window_queue.join()
    engine.stop()


class TestWindowAccumulator:
    def test_emits_a_window_once_enough_samples_arrive(self):
        accumulator = WindowAccumulator(1000, 1000, 2)

        assert accumulator.push(np.zeros((400, 2), dtype=np.float32)) == []
        assert accumulator.push(np.zeros((400, 2), dtype=np.float32)) == []

        windows = accumulator.push(np.zeros((400, 2), dtype=np.float32))
        assert len(windows) == 1
        assert windows[0].shape == (1000, 2)

    def test_one_block_can_yield_several_windows(self):
        accumulator = WindowAccumulator(100, 100, 1)
        assert len(accumulator.push(np.zeros((350, 1), dtype=np.float32))) == 3

    def test_leftover_samples_are_retained(self):
        accumulator = WindowAccumulator(100, 100, 1)
        accumulator.push(np.zeros((150, 1), dtype=np.float32))
        assert accumulator.pending_samples == 50

    def test_content_is_preserved_in_order(self):
        accumulator = WindowAccumulator(10, 10, 1)
        block = np.arange(10, dtype=np.float32).reshape(-1, 1)
        window = accumulator.push(block)[0]
        assert np.allclose(window[:, 0], np.arange(10))

    def test_overlapping_hop_reuses_samples(self):
        accumulator = WindowAccumulator(10, 5, 1)
        windows = accumulator.push(np.arange(20, dtype=np.float32).reshape(-1, 1))
        assert len(windows) == 3
        assert np.allclose(windows[1][:, 0], np.arange(5, 15))

    def test_channel_mismatch_is_rejected(self):
        accumulator = WindowAccumulator(100, 100, 2)
        with pytest.raises(ValueError, match="block must be"):
            accumulator.push(np.zeros((100, 1), dtype=np.float32))

    def test_hop_larger_than_window_is_rejected(self):
        with pytest.raises(ValueError, match="hop_samples"):
            WindowAccumulator(100, 200, 1)


class TestAudioEngine:
    def test_processes_a_recording_end_to_end(self, tmp_path):
        wav = tmp_path / "in.wav"
        _write_wav(wav, _stereo_recording())
        config = _config(tmp_path, wav)

        engine = AudioEngine(config)
        _run_to_completion(engine)

        stats = engine.stats
        assert stats["blocks_captured"] > 0
        assert stats["windows_analysed"] == 3  # 9 seconds at 3-second windows

    def test_file_source_drops_nothing(self, tmp_path):
        """Backpressure regression: a file outruns analysis and must wait."""
        wav = tmp_path / "in.wav"
        _write_wav(wav, _stereo_recording(seconds=30.0))
        config = _config(tmp_path, wav)

        engine = AudioEngine(config)
        _run_to_completion(engine)

        assert engine.stats["windows_dropped"] == 0
        assert engine.stats["windows_analysed"] == 10

    def test_dc_offset_does_not_trigger_the_gate(self, tmp_path):
        """A silent recording with the SPH0645's bias must stay silent."""
        wav = tmp_path / "flat.wav"
        total = int(SAMPLE_RATE * 9)
        flat = np.full((total, 2), 0.35, dtype=np.float32)
        _write_wav(wav, flat)

        engine = AudioEngine(_config(tmp_path, wav))
        _run_to_completion(engine)

        assert engine.stats["windows_gated"] == 0

    def test_writes_detections_to_the_database(self, tmp_path):
        wav = tmp_path / "in.wav"
        _write_wav(wav, _stereo_recording())
        config = _config(tmp_path, wav)

        engine = AudioEngine(config)
        _run_to_completion(engine)

        with DetectionDatabase(config.db_full_path) as db:
            rows = db.get_audio_detections()

        assert rows
        assert all(row["band_rms_dbfs"] is not None for row in rows)

    def test_only_the_channel_with_a_call_triggers(self, tmp_path):
        """Channel identity must survive: the calls are on the left mic."""
        wav = tmp_path / "in.wav"
        _write_wav(wav, _stereo_recording())
        config = _config(tmp_path, wav)

        engine = AudioEngine(config)
        _run_to_completion(engine)

        with DetectionDatabase(config.db_full_path) as db:
            channels = {row["channel"] for row in db.get_audio_detections()}

        assert channels == {0}

    def test_writes_wav_clips(self, tmp_path):
        wav = tmp_path / "in.wav"
        _write_wav(wav, _stereo_recording())
        config = _config(tmp_path, wav, save_clips=True)

        engine = AudioEngine(config)
        _run_to_completion(engine)

        assert list(config.audio_clip_full_path.glob("*.wav"))

    def test_clips_can_be_disabled(self, tmp_path):
        wav = tmp_path / "in.wav"
        _write_wav(wav, _stereo_recording())
        config = _config(tmp_path, wav, save_clips=False)

        engine = AudioEngine(config)
        _run_to_completion(engine)

        with DetectionDatabase(config.db_full_path) as db:
            rows = db.get_audio_detections()

        assert rows
        assert all(row["clip_path"] is None for row in rows)

    def test_cooldown_suppresses_repeat_events(self, tmp_path):
        wav = tmp_path / "in.wav"
        _write_wav(wav, _stereo_recording(seconds=30.0))

        busy = _config(tmp_path / "a", wav, cooldown_seconds=0.0)
        quiet = _config(tmp_path / "b", wav, cooldown_seconds=3600.0)

        busy_engine = AudioEngine(busy)
        _run_to_completion(busy_engine)
        quiet_engine = AudioEngine(quiet)
        _run_to_completion(quiet_engine)

        # windows_stored, not identifications: cooldown suppresses rows
        # whether or not BirdNET could name what was in them, and a
        # synthetic tone is not guaranteed to be named at all.
        assert quiet_engine.stats["windows_stored"] < busy_engine.stats["windows_stored"]

    def test_mono_mode_collapses_the_channels(self, tmp_path):
        wav = tmp_path / "in.wav"
        _write_wav(wav, _stereo_recording())
        config = _config(tmp_path, wav, per_channel=False)

        engine = AudioEngine(config)
        _run_to_completion(engine)

        with DetectionDatabase(config.db_full_path) as db:
            channels = {row["channel"] for row in db.get_audio_detections()}

        assert channels <= {0}

    def test_runs_without_birdnet(self, tmp_path):
        """A fresh checkout has no model and must still capture and gate."""
        wav = tmp_path / "in.wav"
        _write_wav(wav, _stereo_recording())
        config = _config(tmp_path, wav, model_path="does_not_exist.tflite")

        engine = AudioEngine(config)
        _run_to_completion(engine)

        with DetectionDatabase(config.db_full_path) as db:
            rows = db.get_audio_detections()

        assert rows
        assert all(row["species"] is None for row in rows)

    def test_unmatched_windows_are_not_counted_as_identifications(self, tmp_path):
        """A stored row is not an identification.

        With no model every gated window is stored and none names a
        species, which is the case that used to report one identification
        per window on a night the microphones named nothing.
        """
        wav = tmp_path / "in.wav"
        _write_wav(wav, _stereo_recording())
        config = _config(tmp_path, wav, model_path="does_not_exist.tflite")

        engine = AudioEngine(config)
        _run_to_completion(engine)

        assert engine.stats["windows_stored"] > 0
        assert engine.stats["identifications"] == 0

    def test_unmatched_windows_are_not_logged_at_info(self, tmp_path, caplog):
        """A window that named nothing is not a sighting.

        The gate is permissive by design, so on a quiet night the windows
        that match nothing are most of the stream and at info level they
        bury the identifications. The measurement still goes out at debug,
        which is what the gate is retuned against.
        """
        wav = tmp_path / "in.wav"
        _write_wav(wav, _stereo_recording())
        config = _config(tmp_path, wav, model_path="does_not_exist.tflite")

        with caplog.at_level(
            logging.DEBUG, logger="ratcatcher.pipeline.audio_engine"
        ):
            engine = AudioEngine(config)
            _run_to_completion(engine)

        unmatched = [
            record
            for record in caplog.records
            if "no species match" in record.getMessage()
        ]
        assert unmatched, "recording must contain a window that matched nothing"
        assert all(record.levelno < logging.INFO for record in unmatched)

    def test_missing_source_raises_rather_than_hanging(self, tmp_path):
        config = _config(tmp_path, tmp_path / "absent.wav")
        with pytest.raises((RuntimeError, FileNotFoundError)):
            AudioEngine(config).start()

    def test_stop_is_safe_before_start(self, tmp_path):
        wav = tmp_path / "in.wav"
        _write_wav(wav, _stereo_recording(seconds=3.0))
        AudioEngine(_config(tmp_path, wav)).stop()

    def test_gate_state_is_per_channel(self, tmp_path):
        """Each microphone tracks its own noise floor."""
        wav = tmp_path / "in.wav"
        _write_wav(wav, _stereo_recording())
        config = _config(tmp_path, wav)

        engine = AudioEngine(config)
        _run_to_completion(engine)

        assert set(engine._gates) == {0, 1}
        assert engine._gates[0] is not engine._gates[1]


class TestUnifiedStorage:
    def test_view_interleaves_both_modalities(self, tmp_path):
        with DetectionDatabase(tmp_path / "t.db") as db:
            db.insert_detection(
                timestamp="2026-08-23T10:00:00",
                camera_id=0,
                stage="classification",
                class_name="bird",
                species="Sialia mexicana",
                confidence=0.9,
            )
            db.insert_audio_detection(
                timestamp="2026-08-23T10:01:00",
                channel=1,
                species="Poecile gambeli",
                confidence=0.7,
            )

            rows = db.get_all_detections()
            assert [row["modality"] for row in rows] == ["audio", "video"]
            assert rows[0]["source_id"] == 1
            assert rows[1]["source_id"] == 0

    def test_modality_filter(self, tmp_path):
        with DetectionDatabase(tmp_path / "t.db") as db:
            db.insert_detection(
                timestamp="2026-08-23T10:00:00",
                camera_id=0,
                stage="detection",
                class_name="squirrel",
            )
            db.insert_audio_detection(
                timestamp="2026-08-23T10:01:00", channel=0, species="Corvus corax"
            )

            assert len(db.get_all_detections(modality="audio")) == 1
            assert len(db.get_all_detections(modality="video")) == 1

    def test_unknown_modality_is_rejected(self, tmp_path):
        with DetectionDatabase(tmp_path / "t.db") as db:
            with pytest.raises(ValueError, match="modality"):
                db.get_all_detections(modality="acoustic")

    def test_silent_channel_level_is_stored_as_null(self, tmp_path):
        """SQLite has no infinity; a silent channel measures -inf dBFS."""
        with DetectionDatabase(tmp_path / "t.db") as db:
            db.insert_audio_detection(
                timestamp="2026-08-23T10:00:00",
                channel=0,
                band_rms_dbfs=float("-inf"),
            )
            assert db.get_audio_detections()[0]["band_rms_dbfs"] is None

    def test_alternatives_round_trip_through_metadata(self, tmp_path):
        with DetectionDatabase(tmp_path / "t.db") as db:
            db.insert_audio_detection(
                timestamp="2026-08-23T10:00:00",
                channel=0,
                species="Corvus corax",
                metadata={"alternatives": [{"species": "Corvus corone", "conf": 0.2}]},
            )
            row = db.get_audio_detections()[0]
            assert row["metadata"]["alternatives"][0]["species"] == "Corvus corone"

    def test_channel_filter(self, tmp_path):
        with DetectionDatabase(tmp_path / "t.db") as db:
            for channel in (0, 1):
                db.insert_audio_detection(
                    timestamp=f"2026-08-23T10:0{channel}:00",
                    channel=channel,
                    species="Corvus corax",
                )
            assert len(db.get_audio_detections(channel=0)) == 1
