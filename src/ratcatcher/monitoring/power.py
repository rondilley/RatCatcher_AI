"""Battery monitoring and low-charge shutdown.

One thread::

    timer -> UPS HAT -> transition record + sample row -> maybe poweroff

Like ``StatusReporter``, the status panel and the audio engine, this is
an independent consumer. It holds its own database handle and its own
I2C client, and shares no queue, lock or object with the camera
pipeline. A UPS that stops answering must not be able to stop detection.

**Why it does not stop the pipeline itself.** The obvious design gives
this thread a reference to ``PipelineEngine`` so it can shut the cameras
down before halting. That would be the one place in the system where an
independent consumer reaches back into the pipeline, and it is not
needed: ``systemctl poweroff`` takes the units down as part of the
shutdown transition, sending ratcatcher.service the same SIGTERM an
ordinary ``systemctl stop`` does, which the CLI already handles by
stopping the pipeline cleanly. Deferring to systemd keeps the dependency
arrow pointing one way.

**Why the shutdown is issued through systemd rather than the HAT.** The
UPS can cut its own output -- writing 0x55 to register 0x01 does it --
and that would preserve slightly more charge. It would also cut power to
a running kernel with dirty pages, which is precisely the failure a UPS
exists to prevent. The operating system halts first; the HAT is left
alone, and its auto-restart bit brings the Pi back when mains returns.
"""

from __future__ import annotations

import logging
import subprocess
import threading
from datetime import datetime
from typing import Callable, Sequence

from ratcatcher.config import Config
from ratcatcher.monitoring.battery import BatteryReading, UpsBattery
from ratcatcher.monitoring.events import (
    POWER_ON_BATTERY,
    POWER_ON_MAINS,
    POWER_SHUTDOWN,
    log_power,
)
from ratcatcher.storage.database import DetectionDatabase

logger = logging.getLogger(__name__)

# The service runs as the unprivileged "ratcatcher" user with no logind
# session, so polkit's allow_active never applies to it and a bare
# "systemctl poweroff" is refused. The sudoers drop-in installed by the
# postinst grants exactly this one command and nothing else. -n makes a
# missing grant fail immediately rather than blocking on a password
# prompt that nothing will ever answer.
SHUTDOWN_COMMAND: tuple[str, ...] = ("sudo", "-n", "systemctl", "poweroff")

# Long enough for systemd to accept the job, short enough that a wedged
# sudo does not park this thread forever.
_SHUTDOWN_TIMEOUT_SECONDS = 30.0

CommandRunner = Callable[[Sequence[str]], int]


def _run_command(command: Sequence[str]) -> int:
    """Run a command and return its exit status.

    A subprocess is a trust boundary: sudo may be absent, the drop-in
    may be missing, and systemd may not answer. Any of those is reported
    and returns non-zero rather than raising into the monitor thread.
    """
    try:
        completed = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=_SHUTDOWN_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.error("Shutdown command %s failed: %s", " ".join(command), exc)
        return 1

    if completed.returncode != 0:
        logger.error(
            "Shutdown command %s exited %d: %s",
            " ".join(command),
            completed.returncode,
            (completed.stderr or "").strip(),
        )
    return completed.returncode


class BatteryMonitor:
    """Polls the UPS, records what it says, and halts on a flat pack."""

    def __init__(
        self,
        config: Config,
        *,
        runner: CommandRunner | None = None,
        battery: UpsBattery | None = None,
    ) -> None:
        """``runner`` and ``battery`` name the two things that are hardware.

        Both default to the real article. A test supplies its own so that
        readings can be scripted and a shutdown records the command
        rather than halting the machine running the tests; nothing else
        has a reason to pass either.
        """
        self._config = config
        self._battery_config = config.battery
        self._stop_event = threading.Event()
        self._runner = runner if runner is not None else _run_command

        self._database: DetectionDatabase | None = None
        self._owns_database = False
        self._thread: threading.Thread | None = None
        self._ups = battery if battery is not None else UpsBattery(
            bus=config.battery.i2c_bus,
            address=config.battery.i2c_address,
        )

        # None until the first successful reading. The first reading
        # establishes the source rather than reporting a transition into
        # it -- except when it is already on battery, which is worth
        # saying at once.
        self._source: str | None = None
        self._last_sample: datetime | None = None
        self._shutdown_fired = False

        self._samples_stored = 0
        self._transitions = 0
        self._read_failures = 0

    # -- public API --------------------------------------------------------

    @property
    def stats(self) -> dict[str, int]:
        return {
            "samples_stored": self._samples_stored,
            "transitions": self._transitions,
            "read_failures": self._read_failures,
            "shutdown_fired": int(self._shutdown_fired),
        }

    @property
    def is_running(self) -> bool:
        return not self._stop_event.is_set() and self._thread is not None

    @property
    def power_source(self) -> str | None:
        """Last observed power source, or None before the first reading."""
        return self._source

    def open(self, database: DetectionDatabase | None = None) -> None:
        """Acquire the database handle without starting a thread."""
        if database is not None:
            self._database = database
            self._owns_database = False
        elif self._database is None:
            self._database = DetectionDatabase(self._config.db_full_path)
            self._owns_database = True

    def close(self) -> None:
        """Release the database handle and the I2C bus."""
        if self._database is not None and self._owns_database:
            self._database.close()
            self._database = None
            self._owns_database = False
        self._ups.close()

    def start(self, database: DetectionDatabase | None = None) -> None:
        """Poll the UPS from a background thread."""
        if self._thread is not None:
            raise RuntimeError("BatteryMonitor is already running")

        self.open(database)

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, name="battery-monitor", daemon=True
        )
        self._thread.start()

        if self._battery_config.shutdown_enabled:
            logger.info(
                "Battery monitoring every %.0fs, shutdown below %d%% or %d min",
                self._battery_config.poll_interval_seconds,
                self._battery_config.shutdown_percent,
                self._battery_config.shutdown_minutes,
            )
        else:
            logger.info(
                "Battery monitoring every %.0fs, automatic shutdown disabled",
                self._battery_config.poll_interval_seconds,
            )

    def stop(self) -> None:
        """Stop the background thread and release what it holds."""
        if self._thread is not None:
            self._stop_event.set()
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("Battery monitor did not stop within 5 seconds")
            self._thread = None

        self.close()
        logger.info("Battery monitoring stopped: %s", self.stats)

    def poll_once(self, now: datetime | None = None) -> BatteryReading | None:
        """Take one reading and act on it. Returns the reading, or None.

        Separated from the loop so the whole decision path can be driven
        over a scripted sequence of readings in a test without a thread
        or a real clock.
        """
        reading = self._ups.read()
        if reading is None:
            self._read_failures += 1
            return None

        moment = now if now is not None else datetime.now()
        changed = self._note_source(reading)
        self._maybe_store(reading, moment, forced=changed)
        self._maybe_shutdown(reading)
        return reading

    # -- internals ---------------------------------------------------------

    def _note_source(self, reading: BatteryReading) -> bool:
        """Record the power source, reporting a change. True if it moved."""
        source = reading.power_source
        previous = self._source
        self._source = source

        if previous == source:
            return False

        if previous is None:
            # Starting up already on battery means the Pi booted during
            # an outage, which is worth a record. Starting up on mains is
            # the ordinary case and says nothing.
            if not reading.on_battery:
                return False
        else:
            self._transitions += 1

        if reading.on_battery:
            log_power(
                event=POWER_ON_BATTERY,
                percent=reading.percent,
                pack_v=reading.pack_volts,
                minutes_remaining=reading.minutes_to_empty,
            )
        else:
            log_power(
                event=POWER_ON_MAINS,
                percent=reading.percent,
                pack_v=reading.pack_volts,
            )
            # Mains is back, so a shutdown that was armed and failed can
            # arm again for the next outage.
            self._shutdown_fired = False

        return True

    def _maybe_store(
        self, reading: BatteryReading, now: datetime, *, forced: bool
    ) -> None:
        """Write a sample if the interval has elapsed, or the source moved.

        A transition is stored whatever the interval says. It is the one
        sample whose timestamp is the answer to a question -- when did
        the power actually go -- and rounding it to the next five-minute
        boundary would throw that away.
        """
        if self._database is None:
            return

        interval = self._battery_config.sample_interval_seconds
        if not forced and self._last_sample is not None:
            if (now - self._last_sample).total_seconds() < interval:
                return

        try:
            self._database.insert_battery_sample(
                timestamp=now.isoformat(),
                percent=reading.percent,
                pack_mv=reading.pack_mv,
                pack_ma=reading.pack_ma,
                remaining_mah=reading.remaining_mah,
                vbus_mv=reading.vbus_mv,
                vbus_ma=reading.vbus_ma,
                vbus_mw=reading.vbus_mw,
                minutes_to_empty=reading.minutes_to_empty,
                minutes_to_full=reading.minutes_to_full,
                power_source=reading.power_source,
                charge_state=reading.charge_state_name,
                cells_mv=reading.cells_mv,
                gauge_ok=reading.gauge_ok,
                charger_ok=reading.charger_ok,
            )
        except Exception as exc:  # noqa: BLE001 -- storing must not be fatal
            logger.warning("Battery sample not stored: %s", exc)
            return

        self._last_sample = now
        self._samples_stored += 1

    def _shutdown_reason(self, reading: BatteryReading) -> str | None:
        """Name the floor this reading has crossed, or None for neither.

        Both floors are checked whatever the gauge's comms register says.
        A gauge that has stopped answering leaves the last figures it
        managed to report, which may be stale -- but an unnecessary clean
        shutdown costs a restart, while declining to act on a pack that
        really is flat costs a hard power cut into a running filesystem.
        """
        if not reading.on_battery:
            return None

        config = self._battery_config
        if reading.percent <= config.shutdown_percent:
            return "percent"
        minutes = reading.minutes_to_empty
        if minutes is not None and minutes <= config.shutdown_minutes:
            return "minutes"
        return None

    def _maybe_shutdown(self, reading: BatteryReading) -> bool:
        """Halt the system if the pack has fallen past a floor."""
        if self._shutdown_fired or not self._battery_config.shutdown_enabled:
            return False

        reason = self._shutdown_reason(reading)
        if reason is None:
            return False

        self._shutdown_fired = True

        if not reading.healthy:
            reason = f"{reason},gauge-degraded"

        logger.critical(
            "Battery at %d%% on battery power -- halting the system (%s)",
            reading.percent,
            reason,
        )
        log_power(
            event=POWER_SHUTDOWN,
            percent=reading.percent,
            pack_v=reading.pack_volts,
            minutes_remaining=reading.minutes_to_empty,
            reason=reason,
        )

        self._runner(SHUTDOWN_COMMAND)
        return True

    def _run_loop(self) -> None:
        # Read at once rather than after the first interval, so a Pi that
        # boots during an outage says so immediately instead of a minute
        # later.
        while True:
            try:
                self.poll_once()
            except Exception as exc:  # noqa: BLE001 -- monitoring is not fatal
                self._read_failures += 1
                logger.warning("Battery poll failed: %s", exc)

            if self._stop_event.wait(
                timeout=self._battery_config.poll_interval_seconds
            ):
                return
