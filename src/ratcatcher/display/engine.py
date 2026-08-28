"""Threaded status panel driver for RatCatcher AI.

One thread. It reads the database and the health module, builds a status
frame, and writes the frame to the panel when the contents change::

    timer -> SQLite + health -> status frame -> serial port

Like the audio engine, this is an independent consumer. It holds its own
database handle and shares no queue, lock or object with the camera
pipeline, so a panel that is unplugged, wedged or absent cannot affect
detection. The pipeline gives it one thing only: the state word in the
header, pushed through ``set_state``.

Two rules govern how often the panel is written to. A frame whose
contents match the last one drawn is not sent, because every refresh
costs power and a small part of the panel's life. A full refresh is
forced every so many frames, because partial refreshes leave the ghost
of the previous image behind and the ghosts accumulate.
"""

from __future__ import annotations

import logging
import threading
import time

from ratcatcher.config import Config
from ratcatcher.display.factory import create_panel
from ratcatcher.display.panel import NullPanel, StatusPanel
from ratcatcher.display.protocol import StatusFrame
from ratcatcher.display.status import build_status_frame
from ratcatcher.storage.database import DetectionDatabase

logger = logging.getLogger(__name__)


class DisplayEngine:
    """Keeps a status panel showing what the system has detected."""

    def __init__(self, config: Config, panel: StatusPanel | None = None) -> None:
        self._config = config
        self._display_config = config.display
        self._stop_event = threading.Event()
        self._wake = threading.Event()

        self._panel = panel
        self._owns_panel = panel is None
        self._database: DetectionDatabase | None = None
        self._owns_database = False
        self._thread: threading.Thread | None = None

        self._state = "RUN"
        self._cameras: int | None = None
        self._audio_active: bool | None = None
        self._state_lock = threading.Lock()

        # One tick at a time. refresh_now is public and the background
        # thread ticks on its own schedule, so without this the two can
        # both be reading the serial port, and each takes the bytes the
        # other was about to read.
        self._tick_lock = threading.Lock()

        self._seq = 0
        self._last_frame: StatusFrame | None = None
        self._last_key: str | None = None
        self._last_sent = 0.0
        self._since_full_refresh = 0
        self._frames_sent = 0
        self._frames_skipped = 0
        self._write_failures = 0

    # -- public API --------------------------------------------------------

    @property
    def stats(self) -> dict[str, int]:
        return {
            "frames_sent": self._frames_sent,
            "frames_skipped": self._frames_skipped,
            "write_failures": self._write_failures,
        }

    @property
    def is_running(self) -> bool:
        return not self._stop_event.is_set() and self._thread is not None

    @property
    def panel(self) -> StatusPanel | None:
        return self._panel

    @property
    def last_frame(self) -> StatusFrame | None:
        """The last frame accepted by the panel, for printing or testing."""
        return self._last_frame

    def set_state(
        self,
        state: str | None = None,
        *,
        cameras: int | None = None,
        audio_active: bool | None = None,
        redraw: bool = False,
    ) -> None:
        """Update what the header reports about the running system.

        ``redraw`` wakes the thread at once instead of waiting for the
        next poll, so a state change that a person needs to see, such as
        the system stopping, reaches the panel immediately.
        """
        with self._state_lock:
            if state is not None:
                self._state = state
            if cameras is not None:
                self._cameras = cameras
            if audio_active is not None:
                self._audio_active = audio_active
        if redraw:
            self._wake.set()

    def open(self, database: DetectionDatabase | None = None) -> None:
        """Acquire the panel and the database, without starting a thread.

        Separate from ``start`` so that a caller wanting one reading, or
        wanting to control when frames are sent, does not have to run a
        background thread it must then stop. ``ratcatcher display
        --once`` takes this path.

        Raises
        ------
        RuntimeError
            If the panel cannot be opened and the configuration named a
            specific transport. The "auto" transport falls back to a
            null panel instead, so it never raises.
        """
        if self._panel is None:
            self._panel = create_panel(self._display_config)

        # Opening is idempotent on every transport, so a panel handed in
        # by the caller and a panel built by the factory are treated the
        # same way. The alternative, opening only what we built, leaves
        # a supplied panel silently unopened.
        self._panel.open()

        if database is not None:
            self._database = database
            self._owns_database = False
        elif self._database is None:
            self._database = DetectionDatabase(self._config.db_full_path)
            self._owns_database = True

        # The panel's memory holds whatever was on it before, including
        # the image left by a previous run. Start from a clean full
        # refresh rather than partially overwriting a stale screen.
        self._since_full_refresh = self._display_config.full_refresh_every

    def close(self, announce_stop: bool = True) -> None:
        """Release the panel, leaving it reporting the truth.

        A final frame marked STOP is sent first. A panel that keeps
        showing RUN after the system has exited is worse than a blank
        one: it reports a system that is not there.

        Pass ``announce_stop=False`` when the panel was driven for one
        reading rather than for the life of the system. There, STOP
        would overwrite the reading that was just asked for, and would
        cost a second full refresh to do it.
        """
        if announce_stop:
            try:
                self._send_final_frame()
            except Exception as exc:  # noqa: BLE001 -- shutdown must not raise
                logger.debug("Could not send the closing frame: %s", exc)

        if self._panel is not None and self._owns_panel:
            self._panel.close()
            self._panel = None

        if self._database is not None and self._owns_database:
            self._database.close()
            self._database = None
            self._owns_database = False

    def start(self, database: DetectionDatabase | None = None) -> None:
        """Open the panel and drive it from a background thread."""
        if self._thread is not None:
            raise RuntimeError("DisplayEngine is already running")

        self.open(database)

        self._stop_event.clear()
        self._wake.clear()
        self._thread = threading.Thread(
            target=self._run_loop, name="display", daemon=True
        )
        self._thread.start()

        assert self._panel is not None
        logger.info(
            "Status panel running: %s, window=%s, refresh=%.0fs",
            self._panel.description,
            self._display_config.window,
            self._display_config.refresh_seconds,
        )

    def stop(self) -> None:
        """Stop the background thread and release the panel."""
        if self._thread is not None:
            self._stop_event.set()
            self._wake.set()
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("Display thread did not stop within 5 seconds")
            self._thread = None

        self.close()
        logger.info("Status panel stopped: %s", self.stats)

    def refresh_now(self, full: bool = False) -> bool:
        """Build and send one frame immediately. Returns True if sent."""
        if full:
            self._since_full_refresh = self._display_config.full_refresh_every
        return self._tick(force=True)

    def refresh_now_if_changed(self) -> bool:
        """Send a frame only if it would change what is on the panel.

        Returns True if a frame was sent. This is what the background
        thread does on each tick; it is public so that a caller driving
        the panel itself gets the same rule without a thread.
        """
        return self._tick(force=False)

    # -- internals ---------------------------------------------------------

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._tick(force=False)
            except Exception as exc:  # noqa: BLE001 -- a panel fault is not fatal
                logger.warning("Status panel update failed: %s", exc)

            self._wake.wait(timeout=self._display_config.refresh_seconds)
            self._wake.clear()

    def _tick(self, force: bool) -> bool:
        with self._tick_lock:
            return self._tick_locked(force)

    def _tick_locked(self, force: bool) -> bool:
        panel = self._panel
        database = self._database
        if panel is None or database is None:
            return False

        for message in panel.poll():
            self._handle_message(message)

        frame = self._build_frame()
        key = frame.content_key()
        heartbeat = self._display_config.heartbeat_seconds
        elapsed = time.monotonic() - self._last_sent

        unchanged = key == self._last_key
        if unchanged and not force and elapsed < heartbeat:
            self._frames_skipped += 1
            return False

        full = self._since_full_refresh >= self._display_config.full_refresh_every
        frame = _with_refresh(frame, self._seq, full)

        if not panel.send(frame):
            self._write_failures += 1
            return False

        self._seq += 1
        self._frames_sent += 1
        self._last_frame = frame
        self._last_key = key
        self._last_sent = time.monotonic()
        self._since_full_refresh = 0 if full else self._since_full_refresh + 1
        return True

    def _build_frame(self) -> StatusFrame:
        with self._state_lock:
            state = self._state
            cameras = self._cameras
            audio_active = self._audio_active

        assert self._database is not None
        return build_status_frame(
            self._database,
            self._config,
            state=state,
            cameras=cameras,
            audio_active=audio_active,
        )

    def _send_final_frame(self) -> None:
        # Held for the same reason _tick holds it: close() may be called
        # while a caller on another thread is mid-refresh.
        with self._tick_lock:
            if self._panel is None or self._database is None:
                return
            if isinstance(self._panel, NullPanel):
                return
            self.set_state("STOP")
            frame = _with_refresh(self._build_frame(), self._seq, True)
            self._panel.send(frame)

    def _handle_message(self, message: dict) -> None:
        kind = message.get("t")
        if kind == "btn":
            logger.info("Status panel button: %s", message.get("id", "?"))
            # A button press asks for a fresh reading, which is the only
            # thing this panel has to offer. It carries no menu.
            self._last_key = None
            self._since_full_refresh = self._display_config.full_refresh_every
        elif kind == "err":
            logger.warning("Status panel reported an error: %s", message.get("msg"))
        elif kind == "hello":
            logger.info("Status panel firmware %s", message.get("fw", "?"))
            self._last_key = None


def _with_refresh(frame: StatusFrame, seq: int, full: bool) -> StatusFrame:
    """Return the frame stamped with its sequence number and refresh mode."""
    from dataclasses import replace

    return replace(frame, seq=seq, full_refresh=full)
