"""Video clip writer that pipes raw frames to FFmpeg for H.264 encoding."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class ClipWriter:
    """Write short video clips from in-memory frames via FFmpeg.

    Frames are piped to an FFmpeg subprocess as raw RGB24 data and
    encoded into an H.264 MP4 file.

    Parameters
    ----------
    clip_dir : Path
        Directory where output clips are written.
    ffmpeg_preset : str
        FFmpeg x264 encoding preset (e.g. "ultrafast", "fast", "medium").
    """

    def __init__(self, clip_dir: Path, ffmpeg_preset: str = "fast") -> None:
        self._clip_dir = Path(clip_dir)
        self._clip_dir.mkdir(parents=True, exist_ok=True)
        self._preset = ffmpeg_preset

    def is_available(self) -> bool:
        """Return True if ffmpeg is found on the system PATH."""
        return shutil.which("ffmpeg") is not None

    def write_clip(
        self,
        frames: list[tuple[np.ndarray, float]],
        output_name: str,
        fps: int = 30,
    ) -> Path | None:
        """Encode *frames* into an H.264 MP4 clip.

        Parameters
        ----------
        frames : list of (ndarray, float)
            Each element is a ``(frame, timestamp)`` pair. Frames are
            expected in BGR (OpenCV) order; they are converted to RGB
            before being piped to FFmpeg.
        output_name : str
            File stem (or full filename) for the output clip.
        fps : int
            Frames-per-second for the output video.

        Returns
        -------
        Path or None
            The path to the written clip file, or ``None`` if FFmpeg is
            not installed or if encoding failed.
        """
        if not frames:
            logger.warning("write_clip called with an empty frame list")
            return None

        if not output_name.endswith(".mp4"):
            output_name = output_name + ".mp4"
        output_path = self._clip_dir / output_name

        # Determine frame dimensions from the first frame.
        first_frame = frames[0][0]
        height, width = first_frame.shape[:2]

        cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264",
            "-preset", self._preset,
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_path),
        ]

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError:
            logger.warning(
                "FFmpeg is not installed or not on PATH; clip not written"
            )
            return None

        try:
            for frame, _ts in frames:
                # OpenCV frames are BGR; FFmpeg expects RGB when pix_fmt=rgb24.
                if frame.ndim == 3 and frame.shape[2] == 3:
                    rgb = frame[:, :, ::-1]
                else:
                    rgb = frame
                proc.stdin.write(rgb.tobytes())  # type: ignore[union-attr]

            proc.stdin.close()  # type: ignore[union-attr]
            proc.wait()

            if proc.returncode != 0:
                stderr_output = proc.stderr.read().decode("utf-8", errors="replace")  # type: ignore[union-attr]
                logger.warning(
                    "FFmpeg exited with code %d for %s: %s",
                    proc.returncode,
                    output_path,
                    stderr_output,
                )
                return None

        except OSError as exc:
            logger.warning("I/O error while writing clip %s: %s", output_path, exc)
            # Ensure the subprocess is cleaned up.
            proc.kill()
            proc.wait()
            return None

        logger.info("Clip written: %s (%d frames)", output_path, len(frames))
        return output_path
