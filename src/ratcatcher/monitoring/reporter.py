"""Periodic status reporting to syslog.

One thread. It reads the database and the health module, and writes a
single logfmt status line::

    timer -> SQLite + health -> status line -> syslog

Like the status panel and the audio engine, this is an independent
consumer. It holds its own database handle and shares no queue, lock or
object with the camera pipeline, so a wedged loghost cannot affect
detection. The pipeline gives it one thing only: what it actually
opened, pushed through ``set_state``.

The counts come from ``get_category_counts``, the same function the
e-paper panel draws from, so the syslog reading and the panel reading
cannot drift apart.
"""

from __future__ import annotations

import logging
import threading

from ratcatcher.config import Config
from ratcatcher.display.status import format_uptime, window_start
from ratcatcher.monitoring.events import log_status
from ratcatcher.monitoring.health import check_health
from ratcatcher.monitoring.stats import get_category_counts
from ratcatcher.storage.database import DetectionDatabase

logger = logging.getLogger(__name__)


class StatusReporter:
    """Emits a periodic status line describing what the system has seen."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._syslog_config = config.syslog
        self._stop_event = threading.Event()

        self._database: DetectionDatabase | None = None
        self._owns_database = False
        self._thread: threading.Thread | None = None

        self._cameras: int | None = None
        self._audio_active: bool | None = None
        self._state_lock = threading.Lock()

        self._reports_sent = 0
        self._report_failures = 0

    # -- public API --------------------------------------------------------

    @property
    def stats(self) -> dict[str, int]:
        return {
            "reports_sent": self._reports_sent,
            "report_failures": self._report_failures,
        }

    @property
    def is_running(self) -> bool:
        return not self._stop_event.is_set() and self._thread is not None

    def set_state(
        self,
        *,
        cameras: int | None = None,
        audio_active: bool | None = None,
    ) -> None:
        """Record what the running pipeline actually opened.

        Without this the reporter can only say what the configuration
        asked for, which is not the same thing: a camera that failed to
        open would still be reported as present.
        """
        with self._state_lock:
            if cameras is not None:
                self._cameras = cameras
            if audio_active is not None:
                self._audio_active = audio_active

    def open(self, database: DetectionDatabase | None = None) -> None:
        """Acquire the database handle without starting a thread.

        Separate from ``start`` so a caller wanting one reading does not
        have to run a background thread it must then stop.
        """
        if database is not None:
            self._database = database
            self._owns_database = False
        elif self._database is None:
            self._database = DetectionDatabase(self._config.db_full_path)
            self._owns_database = True

    def close(self) -> None:
        """Release the database handle if this reporter opened it."""
        if self._database is not None and self._owns_database:
            self._database.close()
            self._database = None
            self._owns_database = False

    def start(self, database: DetectionDatabase | None = None) -> None:
        """Open the database and report from a background thread."""
        if self._thread is not None:
            raise RuntimeError("StatusReporter is already running")

        self.open(database)

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, name="syslog-status", daemon=True
        )
        self._thread.start()

        logger.info(
            "Syslog status reporting every %.0fs, window=%s",
            self._syslog_config.status_interval_seconds,
            self._syslog_config.status_window,
        )

    def stop(self) -> None:
        """Stop the background thread and release the database."""
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("Status reporter did not stop within 5 seconds")
            self._thread = None

        self.close()
        logger.info("Syslog status reporting stopped: %s", self.stats)

    def report_now(self) -> bool:
        """Build and emit one status line. Returns True if it was sent."""
        database = self._database
        if database is None:
            return False

        with self._state_lock:
            cameras = self._cameras
            audio_active = self._audio_active

        if cameras is None:
            cameras = sum(1 for camera in self._config.cameras if camera.enabled)
        if audio_active is None:
            audio_active = self._config.audio.enabled

        since, label = window_start(self._syslog_config.status_window)
        counts = get_category_counts(database, since=since)
        health = check_health(
            data_dir=self._config.data_path,
            temp_warning_c=self._config.monitoring.temp_warning_c,
            temp_critical_c=self._config.monitoring.temp_critical_c,
            battery=self._config.battery,
        )

        log_status(
            window=label,
            counts=counts,
            cameras=cameras,
            npu=health.hailo_available,
            audio=bool(audio_active),
            temp_c=health.cpu_temp_c,
            disk_pct=health.disk_usage_pct,
            uptime=format_uptime(health.uptime_seconds),
            power=health.power_source,
            batt_pct=health.battery_percent,
            batt_v=health.battery_volts,
        )
        self._reports_sent += 1
        return True

    # -- internals ---------------------------------------------------------

    def _run_loop(self) -> None:
        # Report at once rather than after the first interval. Five
        # minutes of silence at startup looks exactly like a system that
        # failed to start.
        while True:
            try:
                self.report_now()
            except Exception as exc:  # noqa: BLE001 -- reporting is not fatal
                self._report_failures += 1
                logger.warning("Status report failed: %s", exc)

            if self._stop_event.wait(
                timeout=self._syslog_config.status_interval_seconds
            ):
                return
