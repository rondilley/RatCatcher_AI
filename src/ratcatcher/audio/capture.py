"""Audio source abstraction layer for RatCatcher AI.

Provides a Protocol for audio sources and concrete implementations for
ALSA capture devices (the dual SPH0645 I2S microphones) and WAV files.

The two SPH0645 microphones share a single I2S bus and are separated by
their SEL pin, so the operating system presents them as one stereo
capture device: channel 0 is the left microphone, channel 1 is the right.
Both channels are therefore read together and split downstream rather
than being opened as two independent devices.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import wave
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


# The SPH0645 emits 24 bits of data left-justified in a 32-bit slot, so the
# capture format is always 32-bit signed little-endian regardless of how
# many of those bits carry signal.
_SAMPLE_FORMAT = "S32_LE"
_BYTES_PER_SAMPLE = 4


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class AudioSource(Protocol):
    """Minimal interface every audio back-end must satisfy."""

    def start(self) -> None:
        """Begin producing audio."""
        ...

    def stop(self) -> None:
        """Release resources and stop producing audio."""
        ...

    def read(self, num_frames: int) -> tuple[bool, np.ndarray | None]:
        """Return ``(success, block)``.

        ``block`` is a float32 array of shape ``(num_frames, channels)``
        normalised to roughly [-1.0, 1.0], or None on failure or EOF.
        """
        ...

    @property
    def sample_rate(self) -> int:
        """Sample rate in Hz."""
        ...

    @property
    def channels(self) -> int:
        """Number of captured channels."""
        ...

    @property
    def is_running(self) -> bool:
        """True between successful ``start()`` and ``stop()``."""
        ...

    @property
    def is_realtime(self) -> bool:
        """True if the source produces audio at wall-clock speed.

        Live capture is realtime: a consumer that falls behind must drop
        data, because blocking the reader stalls the device and causes
        ALSA overruns. A file is not realtime: it delivers as fast as it
        is read, so a consumer must be allowed to apply backpressure or
        it will silently lose most of the recording.
        """
        ...


# ---------------------------------------------------------------------------
# Shared conversion helper
# ---------------------------------------------------------------------------


def raw_to_float32(raw: bytes, channels: int) -> np.ndarray:
    """Convert interleaved signed 32-bit PCM bytes to a float32 array.

    Parameters
    ----------
    raw:
        Interleaved little-endian signed 32-bit samples.
    channels:
        Number of interleaved channels.

    Returns
    -------
    Array of shape ``(frames, channels)`` scaled to roughly [-1.0, 1.0].

    Notes
    -----
    Scaling divides by 2**31 rather than by the SPH0645's true 24-bit
    range. The microphone leaves its data left-justified in the 32-bit
    slot, so the full-scale value really is 2**31 and no shift is needed.
    Trailing bytes that do not complete a frame are discarded; ALSA can
    return a partial frame when a capture is stopped mid-block.
    """
    if channels < 1:
        raise ValueError(f"channels must be >= 1, got {channels}")

    samples = np.frombuffer(raw, dtype="<i4")
    usable = (samples.size // channels) * channels
    if usable != samples.size:
        samples = samples[:usable]

    if usable == 0:
        return np.zeros((0, channels), dtype=np.float32)

    block = samples.reshape(-1, channels).astype(np.float32)
    block /= float(1 << 31)
    return block


# ---------------------------------------------------------------------------
# ALSA capture (production: dual SPH0645 over I2S)
# ---------------------------------------------------------------------------


class ArecordSource:
    """Continuous ALSA capture driven by a long-running ``arecord`` process.

    Uses a subprocess rather than a Python ALSA binding for the same
    reason ``ClipWriter`` shells out to FFmpeg: it adds no dependency, it
    is trivial to reproduce by hand when debugging hardware, and a wedged
    capture can be killed without taking the interpreter with it.

    The process writes raw PCM to stdout and is read in fixed-size blocks.
    """

    def __init__(
        self,
        device: str = "default",
        sample_rate: int = 48000,
        channels: int = 2,
        read_timeout: float = 5.0,
    ) -> None:
        self._device = device
        self._sample_rate = sample_rate
        self._channels = channels
        self._read_timeout = read_timeout

        self._process: subprocess.Popen[bytes] | None = None
        self._lock = threading.Lock()
        self._running = False

    # -- lifecycle ---------------------------------------------------------

    @staticmethod
    def is_available() -> bool:
        """True if the ``arecord`` binary is present on this system."""
        return shutil.which("arecord") is not None

    @staticmethod
    def list_capture_devices() -> list[str]:
        """Return the ALSA capture card names reported by ``arecord -l``.

        Returns an empty list if ``arecord`` is missing or no capture
        hardware is present, which is the expected state before the I2S
        overlay has been enabled and the Pi rebooted.
        """
        if not ArecordSource.is_available():
            return []

        try:
            result = subprocess.run(
                ["arecord", "-l"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            logger.error("Could not enumerate capture devices: %s", exc)
            return []

        devices = [
            line.strip()
            for line in result.stdout.splitlines()
            if line.startswith("card ")
        ]
        return devices

    def start(self) -> None:
        """Launch the capture process.

        Raises
        ------
        RuntimeError
            If ``arecord`` is unavailable or the process exits immediately,
            which usually means the device name is wrong or the I2S
            overlay has not been loaded.
        """
        with self._lock:
            if self._running:
                return

            if not self.is_available():
                raise RuntimeError(
                    "arecord not found. Install it with: "
                    "sudo apt-get install alsa-utils"
                )

            command = [
                "arecord",
                "-D", self._device,
                "-f", _SAMPLE_FORMAT,
                "-r", str(self._sample_rate),
                "-c", str(self._channels),
                "-t", "raw",
                "--quiet",
            ]

            try:
                self._process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except OSError as exc:
                raise RuntimeError(
                    f"Failed to launch arecord on device '{self._device}': {exc}"
                ) from exc

            # A bad device name makes arecord exit straight away. Catch that
            # here so the caller gets the real ALSA error rather than a
            # confusing stream of empty reads later on.
            try:
                self._process.wait(timeout=0.3)
            except subprocess.TimeoutExpired:
                pass
            else:
                stderr = b""
                if self._process.stderr is not None:
                    stderr = self._process.stderr.read()
                raise RuntimeError(
                    f"arecord exited immediately for device "
                    f"'{self._device}': {stderr.decode(errors='replace').strip()}"
                )

            self._running = True

        logger.info(
            "Audio capture started: device=%s rate=%d channels=%d",
            self._device,
            self._sample_rate,
            self._channels,
        )

    def stop(self) -> None:
        """Terminate the capture process."""
        with self._lock:
            self._running = False
            process = self._process
            self._process = None

        if process is None:
            return

        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("arecord did not terminate, killing it")
            process.kill()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.error("arecord could not be killed")

        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()

        logger.info("Audio capture stopped")

    # -- reading -----------------------------------------------------------

    def read(self, num_frames: int) -> tuple[bool, np.ndarray | None]:
        """Read exactly ``num_frames`` frames, blocking until they arrive.

        Returns ``(False, None)`` if the capture process has stopped or
        the stream ended early.
        """
        if num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}")

        process = self._process
        if not self._running or process is None or process.stdout is None:
            return False, None

        want = num_frames * self._channels * _BYTES_PER_SAMPLE

        try:
            raw = process.stdout.read(want)
        except (OSError, ValueError) as exc:
            logger.error("Audio read failed: %s", exc)
            return False, None

        if raw is None or len(raw) < want:
            # Short read means the process died. Surface its stderr, which
            # is where ALSA reports overruns and device errors.
            if process.poll() is not None and process.stderr is not None:
                try:
                    message = process.stderr.read().decode(errors="replace")
                except (OSError, ValueError):
                    message = ""
                if message.strip():
                    logger.error("arecord: %s", message.strip())
            return False, None

        return True, raw_to_float32(raw, self._channels)

    # -- properties --------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_realtime(self) -> bool:
        return True

    @property
    def device(self) -> str:
        return self._device


# ---------------------------------------------------------------------------
# WAV file capture (development and tests)
# ---------------------------------------------------------------------------


class WavFileSource:
    """Read audio from a WAV file, for development and tests.

    Accepts 16, 24 and 32-bit PCM. Playback is not rate-limited: ``read``
    returns immediately, so a test can pull an entire file through the
    pipeline without waiting in real time.
    """

    def __init__(self, path: str | Path, loop: bool = False) -> None:
        self._path = Path(path)
        self._loop = loop

        if not self._path.exists():
            raise FileNotFoundError(f"Audio file not found: {self._path}")

        self._wave: wave.Wave_read | None = None
        self._running = False

        # Read the header up front so sample_rate and channels are known
        # before start(), matching how the camera sources expose resolution.
        try:
            with wave.open(str(self._path), "rb") as handle:
                self._sample_rate = handle.getframerate()
                self._channels = handle.getnchannels()
                self._sample_width = handle.getsampwidth()
        except (wave.Error, OSError, EOFError) as exc:
            raise ValueError(f"Could not read WAV file {self._path}: {exc}") from exc

        if self._sample_width not in (2, 3, 4):
            raise ValueError(
                f"Unsupported sample width {self._sample_width * 8}-bit in "
                f"{self._path}. Supported: 16, 24 or 32-bit PCM."
            )

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._running:
            return
        try:
            self._wave = wave.open(str(self._path), "rb")
        except (wave.Error, OSError, EOFError) as exc:
            raise RuntimeError(f"Could not open {self._path}: {exc}") from exc
        self._running = True
        logger.info(
            "Audio file source started: %s (rate=%d channels=%d width=%d-bit)",
            self._path.name,
            self._sample_rate,
            self._channels,
            self._sample_width * 8,
        )

    def stop(self) -> None:
        self._running = False
        if self._wave is not None:
            self._wave.close()
            self._wave = None
        logger.info("Audio file source stopped")

    # -- reading -----------------------------------------------------------

    def read(self, num_frames: int) -> tuple[bool, np.ndarray | None]:
        if num_frames <= 0:
            raise ValueError(f"num_frames must be positive, got {num_frames}")

        if not self._running or self._wave is None:
            return False, None

        try:
            raw = self._wave.readframes(num_frames)
        except (wave.Error, OSError) as exc:
            logger.error("WAV read failed: %s", exc)
            return False, None

        frames_read = len(raw) // (self._channels * self._sample_width)

        if frames_read < num_frames:
            if not self._loop:
                return False, None
            # Rewind and take the remainder so looping does not emit a
            # short block at every wrap.
            try:
                self._wave.rewind()
                remaining = num_frames - frames_read
                raw += self._wave.readframes(remaining)
            except (wave.Error, OSError) as exc:
                logger.error("WAV rewind failed: %s", exc)
                return False, None

        return True, self._decode(raw)

    def _decode(self, raw: bytes) -> np.ndarray:
        """Convert PCM bytes of this file's width to normalised float32."""
        if self._sample_width == 4:
            return raw_to_float32(raw, self._channels)

        if self._sample_width == 2:
            samples = np.frombuffer(raw, dtype="<i2").astype(np.float32)
            usable = (samples.size // self._channels) * self._channels
            samples = samples[:usable]
            if usable == 0:
                return np.zeros((0, self._channels), dtype=np.float32)
            return samples.reshape(-1, self._channels) / float(1 << 15)

        # 24-bit: three bytes per sample, sign-extended into int32.
        raw_bytes = np.frombuffer(raw, dtype=np.uint8)
        usable_bytes = (raw_bytes.size // (3 * self._channels)) * 3 * self._channels
        raw_bytes = raw_bytes[:usable_bytes]
        if usable_bytes == 0:
            return np.zeros((0, self._channels), dtype=np.float32)

        triplets = raw_bytes.reshape(-1, 3).astype(np.int32)
        values = triplets[:, 0] | (triplets[:, 1] << 8) | (triplets[:, 2] << 16)
        values = np.where(values & 0x800000, values - 0x1000000, values)
        block = values.reshape(-1, self._channels).astype(np.float32)
        return block / float(1 << 23)

    # -- properties --------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_realtime(self) -> bool:
        return False

    @property
    def path(self) -> Path:
        return self._path
