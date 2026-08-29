"""Threaded bird song detection pipeline for RatCatcher AI.

Runs as an independent detector alongside the camera pipeline. It shares
the SQLite database but nothing else: no queues, no locks and no shared
state cross the boundary, so audio keeps working when the cameras or the
NPU are unavailable, and vice versa.

Two threads, mirroring the camera pipeline's separation of capture from
inference::

    capture thread    device -> DC blocker -> window accumulator -> queue
    analysis thread   queue -> activity gate -> BirdNET -> SQLite + WAV

The activity gate sits in front of BirdNET the way motion detection sits
in front of YOLO, but it earns much less and is tuned accordingly.
BirdNET costs about 60 ms per three-second window here, so two channels
running continuously use roughly four percent of one core. With so
little to save, the gate is set to reject only what plainly has no
signal: digital silence, a dead microphone, steady rain. Measured
against a field soundscape it keeps every window containing a real bird.
"""

from __future__ import annotations

import logging
import queue
import threading
from datetime import datetime
from pathlib import Path

import numpy as np

from ratcatcher.audio.activity import ActivityResult, SoundActivityDetector
from ratcatcher.audio.birdnet import BirdNetClassifier, SongDetection
from ratcatcher.audio.capture import AudioSource
from ratcatcher.audio.clip_writer import clip_filename, write_wav
from ratcatcher.audio.factory import create_audio_source
from ratcatcher.audio.preprocess import DCBlocker, split_channels, to_mono
from ratcatcher.config import Config
from ratcatcher.monitoring.events import log_audio_detection
from ratcatcher.storage.database import DetectionDatabase

logger = logging.getLogger(__name__)

# Fallback window length used when BirdNET is not loaded, so the gate and
# the capture path can still be exercised. BirdNET itself reports the
# real value, which is normally the same three seconds.
_DEFAULT_WINDOW_SECONDS = 3.0


class WindowAccumulator:
    """Collect streamed blocks into fixed-length analysis windows.

    The capture device delivers blocks sized for latency, not for the
    classifier. This buffers them until a full window is available and
    then advances by ``hop`` samples, so consecutive windows can overlap
    if configured to.
    """

    def __init__(self, window_samples: int, hop_samples: int, channels: int) -> None:
        if window_samples <= 0:
            raise ValueError(
                f"window_samples must be positive, got {window_samples}"
            )
        if not 0 < hop_samples <= window_samples:
            raise ValueError(
                f"hop_samples must be in (0, window_samples], got {hop_samples}"
            )
        if channels < 1:
            raise ValueError(f"channels must be >= 1, got {channels}")

        self._window_samples = window_samples
        self._hop_samples = hop_samples
        self._channels = channels
        self._buffer = np.zeros((0, channels), dtype=np.float32)

    def push(self, block: np.ndarray) -> list[np.ndarray]:
        """Add a block and return every complete window it produced."""
        if block.ndim != 2 or block.shape[1] != self._channels:
            raise ValueError(
                f"block must be (frames, {self._channels}), got shape {block.shape}"
            )

        self._buffer = np.concatenate((self._buffer, block), axis=0)

        windows: list[np.ndarray] = []
        while self._buffer.shape[0] >= self._window_samples:
            windows.append(self._buffer[: self._window_samples].copy())
            self._buffer = self._buffer[self._hop_samples :]

        return windows

    def reset(self) -> None:
        self._buffer = np.zeros((0, self._channels), dtype=np.float32)

    @property
    def pending_samples(self) -> int:
        return int(self._buffer.shape[0])


class AudioEngine:
    """Capture, gate and identify bird song from the I2S microphones."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._audio_config = config.audio
        self._stop_event = threading.Event()

        # Bounded so a stalled classifier drops windows rather than
        # growing the queue until the Pi runs out of memory.
        self._window_queue: queue.Queue = queue.Queue(maxsize=16)

        self._source: AudioSource | None = None
        self._classifier: BirdNetClassifier | None = None
        self._database: DetectionDatabase | None = None
        self._threads: list[threading.Thread] = []

        # One detector per channel. The gate tracks a noise floor across
        # windows, and the two microphones face different directions with
        # different ambient levels, so sharing one instance would let the
        # louder side raise the floor on the quieter one.
        self._gates: dict[int, SoundActivityDetector] = {}

        self._last_detection: dict[int, datetime] = {}

        self._stats = {
            "blocks_captured": 0,
            "windows_analysed": 0,
            "windows_gated": 0,
            "identifications": 0,
            "windows_dropped": 0,
        }
        self._stats_lock = threading.Lock()

    # -- properties --------------------------------------------------------

    @property
    def stats(self) -> dict[str, int]:
        with self._stats_lock:
            return dict(self._stats)

    @property
    def is_running(self) -> bool:
        return not self._stop_event.is_set() and bool(self._threads)

    # -- lifecycle ---------------------------------------------------------

    def start(self, database: DetectionDatabase | None = None) -> None:
        """Open the device, load BirdNET and start both threads.

        Parameters
        ----------
        database:
            An open database to write into. When omitted the engine opens
            its own at the configured path and closes it on ``stop()``.

        Raises
        ------
        RuntimeError
            If no audio source could be opened.
        """
        if self._threads:
            logger.warning("Audio engine already started")
            return

        self._stop_event.clear()

        self._source = create_audio_source(self._audio_config)
        self._source.start()

        self._owns_database = database is None
        if database is not None:
            self._database = database
        else:
            self._database = DetectionDatabase(self._config.db_full_path)

        self._init_classifier()

        window_samples = int(
            self._audio_config.sample_rate * _DEFAULT_WINDOW_SECONDS
        )
        if self._classifier is not None:
            window_samples = self._classifier.window_samples

        self._accumulator = WindowAccumulator(
            window_samples=window_samples,
            hop_samples=window_samples,
            channels=self._source.channels,
        )
        self._blocker = DCBlocker(
            sample_rate=self._source.sample_rate,
            cutoff_hz=self._audio_config.highpass_hz,
            channels=self._source.channels,
        )

        for name, target in (
            ("audio-capture", self._capture_loop),
            ("audio-analysis", self._analysis_loop),
        ):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)

        logger.info(
            "Audio engine started: %d channel(s) @ %d Hz, %.1f s windows",
            self._source.channels,
            self._source.sample_rate,
            window_samples / self._source.sample_rate,
        )

    def stop(self) -> None:
        """Stop both threads and release the device."""
        if not self._threads:
            return

        self._stop_event.set()

        for thread in self._threads:
            thread.join(timeout=10.0)
            if thread.is_alive():
                logger.warning("Audio thread %s did not stop cleanly", thread.name)
        self._threads.clear()

        if self._source is not None:
            self._source.stop()
            self._source = None

        if self._database is not None and self._owns_database:
            self._database.close()
        self._database = None

        logger.info("Audio engine stopped: %s", self.stats)

    def _init_classifier(self) -> None:
        """Load BirdNET, degrading to gate-only operation if unavailable.

        A missing model must not take the whole engine down. Without it
        the capture path and activity gate still run and still log that
        something was heard, which is enough to tune thresholds and to
        confirm the microphones work before the weights are downloaded.
        """
        models_dir = Path("models")
        try:
            self._classifier = BirdNetClassifier(
                model_path=models_dir / self._audio_config.model_path,
                labels_path=models_dir / self._audio_config.labels_path,
                min_confidence=self._audio_config.min_confidence,
                top_k=self._audio_config.top_k,
                num_threads=self._audio_config.num_threads,
            )
        except (ImportError, FileNotFoundError, ValueError, RuntimeError) as exc:
            self._classifier = None
            logger.warning(
                "BirdNET unavailable, running gate-only (no species ID): %s", exc
            )

    # -- threads -----------------------------------------------------------

    def _capture_loop(self) -> None:
        """Read blocks, remove DC, and emit complete windows."""
        assert self._source is not None

        block_frames = max(
            1,
            int(self._audio_config.sample_rate * self._audio_config.block_seconds),
        )

        while not self._stop_event.is_set():
            ok, block = self._source.read(block_frames)
            if not ok or block is None:
                if not self._stop_event.is_set():
                    logger.info("Audio source ended")
                break

            with self._stats_lock:
                self._stats["blocks_captured"] += 1

            filtered = self._blocker.process(block)

            for window in self._accumulator.push(filtered):
                self._enqueue_window(window, realtime=self._source.is_realtime)

    def _enqueue_window(self, window: np.ndarray, realtime: bool) -> None:
        """Hand a window to the analysis thread.

        Live capture drops when the queue is full: blocking the reader
        would stall the sound card and produce ALSA overruns, and losing
        the occasional window is preferable to losing the stream. A file
        source instead waits, because it delivers far faster than
        realtime and dropping would silently discard most of a recording
        being analysed offline.
        """
        item = (datetime.now(), window)

        if not realtime:
            while not self._stop_event.is_set():
                try:
                    self._window_queue.put(item, timeout=0.5)
                    return
                except queue.Full:
                    continue
            return

        try:
            self._window_queue.put_nowait(item)
        except queue.Full:
            with self._stats_lock:
                self._stats["windows_dropped"] += 1
            logger.debug("Analysis queue full, dropped a window")

    def _analysis_loop(self) -> None:
        """Gate each window, then identify what survives."""
        while not self._stop_event.is_set():
            try:
                timestamp, window = self._window_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            try:
                self._analyse_window(timestamp, window)
            except Exception:
                # One malformed window must not kill the analysis thread
                # and silence the microphones for the rest of the run.
                logger.exception("Audio analysis failed for one window")
            finally:
                self._window_queue.task_done()

    # -- analysis ----------------------------------------------------------

    def _analyse_window(self, timestamp: datetime, window: np.ndarray) -> None:
        """Evaluate one multi-channel window and store any identifications."""
        with self._stats_lock:
            self._stats["windows_analysed"] += 1

        if self._audio_config.per_channel:
            signals = list(enumerate(split_channels(window)))
        else:
            signals = [(0, to_mono(window))]

        for channel, signal in signals:
            # Always evaluate, so the noise floor keeps tracking even with
            # gating disabled and the measurements are always recorded.
            activity = self._gate_for(channel).evaluate(signal)
            if self._audio_config.gate_enabled and not activity.triggered:
                continue

            with self._stats_lock:
                self._stats["windows_gated"] += 1

            if self._in_cooldown(channel, timestamp):
                continue

            detections = (
                self._classifier.identify(signal, channel=channel)
                if self._classifier is not None
                else []
            )

            self._store(timestamp, channel, signal, activity, detections)

    def _gate_for(self, channel: int) -> SoundActivityDetector:
        """Return this channel's activity detector, creating it on demand."""
        gate = self._gates.get(channel)
        if gate is None:
            gate = SoundActivityDetector(
                sample_rate=self._audio_config.sample_rate,
                band_low_hz=self._audio_config.gate_band_low_hz,
                band_high_hz=self._audio_config.gate_band_high_hz,
                snr_margin_db=self._audio_config.gate_snr_margin_db,
                absolute_floor_dbfs=self._audio_config.gate_absolute_floor_dbfs,
                max_flatness=self._audio_config.gate_max_flatness,
            )
            self._gates[channel] = gate
        return gate

    def _in_cooldown(self, channel: int, timestamp: datetime) -> bool:
        """True if this channel fired too recently to report again.

        A single bird sings continuously for many windows. Without this,
        one chickadee produces a detection every three seconds for as long
        as it stays in the tree.
        """
        previous = self._last_detection.get(channel)
        if previous is None:
            return False
        elapsed = (timestamp - previous).total_seconds()
        return elapsed < self._audio_config.cooldown_seconds

    def _store(
        self,
        timestamp: datetime,
        channel: int,
        signal: np.ndarray,
        activity: ActivityResult,
        detections: list[SongDetection],
    ) -> None:
        """Write a detection to the database, with an optional WAV clip."""
        if self._database is None:
            return

        best = detections[0] if detections else None
        self._last_detection[channel] = timestamp

        clip_path: str | None = None
        if self._audio_config.save_clips:
            species_label = best.scientific_name if best else None
            destination = self._config.audio_clip_full_path / clip_filename(
                timestamp, channel, species_label
            )
            try:
                write_wav(destination, signal, self._audio_config.sample_rate)
                clip_path = str(destination)
            except (OSError, ValueError) as exc:
                logger.error("Could not write audio clip: %s", exc)

        alternatives = [
            {
                "species": entry.scientific_name,
                "common": entry.common_name,
                "conf": round(entry.confidence, 4),
            }
            for entry in detections[1:]
        ]

        try:
            self._database.insert_audio_detection(
                timestamp=timestamp.isoformat(),
                channel=channel,
                species=best.scientific_name if best else None,
                common_name=best.common_name if best else None,
                confidence=best.confidence if best else None,
                duration_seconds=signal.size / self._audio_config.sample_rate,
                band_rms_dbfs=activity.band_rms_dbfs,
                spectral_flatness=activity.spectral_flatness,
                peak_frequency_hz=activity.peak_frequency_hz,
                clip_path=clip_path,
                metadata={"alternatives": alternatives} if alternatives else None,
            )
        except Exception as exc:
            logger.error("Could not store audio detection: %s", exc)
            return

        with self._stats_lock:
            self._stats["identifications"] += 1

        if best is not None:
            # Only windows that named a species. One that passed the
            # gate and matched nothing identifies no bird, and at gate
            # rates those would bury the real songs in a forwarded log.
            log_audio_detection(
                channel=channel,
                species=best.scientific_name,
                common_name=best.common_name,
                confidence=best.confidence,
                rms_dbfs=activity.band_rms_dbfs,
                flatness=activity.spectral_flatness,
                peak_hz=activity.peak_frequency_hz,
                clip_path=clip_path,
            )
            logger.info(
                "Song identified on channel %d: %s (%.1f%%)",
                channel,
                best.common_name,
                best.confidence * 100.0,
            )
        else:
            logger.info(
                "Sound activity on channel %d, no species match (%s)",
                channel,
                activity,
            )
