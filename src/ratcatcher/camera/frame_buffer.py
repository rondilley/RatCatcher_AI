"""Thread-safe ring buffer for pre-event video recording.

Stores (frame, timestamp) pairs in a bounded deque so the pipeline can
retrieve the last N seconds of footage when a detection event occurs.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import List, Tuple

import numpy as np


class FrameBuffer:
    """Fixed-capacity ring buffer of (frame, timestamp) tuples.

    Parameters
    ----------
    max_frames : int
        Maximum number of frames the buffer can hold.  When capacity is
        reached the oldest frame is silently dropped.
    """

    def __init__(self, max_frames: int) -> None:
        if max_frames < 1:
            raise ValueError(
                f"max_frames must be at least 1, got {max_frames}"
            )
        self._buf: deque[Tuple[np.ndarray, float]] = deque(maxlen=max_frames)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, frame: np.ndarray, timestamp: float | None = None) -> None:
        """Append a frame with its capture timestamp.

        Parameters
        ----------
        frame : np.ndarray
            The BGR video frame.
        timestamp : float, optional
            POSIX timestamp.  Defaults to ``time.time()`` if omitted.
        """
        if timestamp is None:
            timestamp = time.time()
        with self._lock:
            self._buf.append((frame, timestamp))

    def get_frames(self, seconds: float) -> List[Tuple[np.ndarray, float]]:
        """Return all frames captured within the last *seconds* seconds.

        The returned list is ordered oldest-first.  If the buffer has no
        frames or *seconds* is non-positive an empty list is returned.
        """
        if seconds <= 0.0:
            return []

        with self._lock:
            if not self._buf:
                return []
            cutoff = self._buf[-1][1] - seconds
            return [
                (frame.copy(), ts)
                for frame, ts in self._buf
                if ts >= cutoff
            ]

    def clear(self) -> None:
        """Remove all frames from the buffer."""
        with self._lock:
            self._buf.clear()

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @property
    def count(self) -> int:
        """Number of frames currently stored."""
        with self._lock:
            return len(self._buf)

    @property
    def max_frames(self) -> int:
        """Maximum capacity of the buffer."""
        # deque.maxlen is set at creation and never changes.
        return self._buf.maxlen  # type: ignore[return-value]
