"""Focus session: both cameras onto the e-paper panel.

Setting these lenses is a two-handed job at the enclosure, where the
panel is the only display. This engine closes that loop -- capture,
measure, draw -- so a ring can be turned until the number stops rising.

Why the panel and not the terminal
----------------------------------
The lenses have no software focus control, so they are set by hand
outdoors. A terminal needs a second device and a spare hand; the panel
is already bolted to the thing being adjusted.

Why peak-hold is the point
--------------------------
The absolute percentage rests on an uncalibrated ceiling (see
``camera.focus``), but the peak does not. A ring turned past the optimum
makes the live number fall away from a peak that stays put, and that
comparison is exact whatever the scale. It is what tells you to turn
back rather than keep going.

Why it does not redraw every reading
------------------------------------
E-paper is built for infrequent updates. The first version of this
engine ignored that and sent every reading unconditionally at 2 Hz:
about 1140 refreshes in a ten-minute session, which left the panel
visibly mottled with ghosting. Most of those refreshes redrew a number
that had not moved, because a lens nobody is touching reads the same
value every time.

``send_if_changed`` withholds a frame whose content matches the one
already on the glass, so the panel redraws when the reading moves --
which is exactly when a ring is being turned -- and rests otherwise.
``DisplayEngine`` had reached the same conclusion for the status screen
first, and its comment there is the better statement of why: every
refresh costs power and a part of the panel's life.

Why this drives the cameras directly
------------------------------------
Deliberately standalone: it holds no queue, thread or lock in common
with ``PipelineEngine``, and changes nothing about detection. The cost
is that the service must be stopped first, because it holds the cameras
and the panel. The CLI says so rather than letting libcamera fail with a
device-busy backtrace.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime
from typing import Any

from ratcatcher.camera.capture import CameraSource
from ratcatcher.camera.focus import DEFAULT_CEILING, FocusReading, analyse_frame
from ratcatcher.display.panel import StatusPanel
from ratcatcher.display.protocol import ScreenFrame, ScreenLine

logger = logging.getLogger(__name__)

# Seconds between readings. The measurement costs about 30 ms for both
# cameras, so this paces the screen rather than the analysis. One second
# is a compromise: fast enough to steer a ring by, slow enough that a
# minute of active adjustment is tens of refreshes and not hundreds.
DEFAULT_INTERVAL = 1.0

# Partial refreshes before a full one. E-paper ghosts, and a focus
# session redraws far more often than the status screen. Kept under the
# firmware's own backstop of 40 so the clearing refresh happens on a
# schedule rather than as a surprise mid-adjustment.
DEFAULT_FULL_REFRESH_EVERY = 20

# Redraw at least this often even when the reading has not moved.
#
# Needed because the firmware declares the host dead after seven minutes
# of silence and replaces the screen with a splash. Deduplication can
# easily produce that much silence: a lens nobody is touching reads the
# same number indefinitely. In practice the clock in the title changes
# first and sends a frame every minute, but relying on that would tie
# panel liveness to what the title happens to contain.
DEFAULT_HEARTBEAT_SECONDS = 120.0

# Percentage points the reading must move before the panel redraws.
#
# The metric repeats to about +/-1% on a still scene, so the last
# digit flickers even when nothing is being touched, and every flicker
# would be a refresh spent on a change nobody can act on. A real turn
# of the ring moves the number far more than this.
DISPLAY_DEADBAND_PCT = 2

# How often the button pins are checked. Fast enough that a press
# feels answered, and cheap because polling the serial port costs
# nothing -- it is the panel refresh that is expensive, and that only
# happens when a press actually arrives.
BUTTON_POLL_SECONDS = 0.1

# Buttons that ask for a fresh reading. Every key except EXIT does,
# because the mapping from the physical controls to these names is
# not yet confirmed and a button that appears dead is worse than one
# that samples when you did not mean it to.
SAMPLE_BUTTONS = frozenset({"ok", "next", "prev", "home"})


class FocusEngine:
    """Measures both cameras and draws the result until asked to stop."""

    def __init__(
        self,
        cameras: Sequence[CameraSource],
        panel: StatusPanel,
        *,
        ceiling: float = DEFAULT_CEILING,
        interval: float = DEFAULT_INTERVAL,
        full_refresh_every: int = DEFAULT_FULL_REFRESH_EVERY,
        heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
        continuous: bool = False,
    ) -> None:
        self._cameras = list(cameras)
        self._panel = panel
        self._ceiling = ceiling
        self._interval = interval
        self._full_refresh_every = max(1, full_refresh_every)
        self._heartbeat_seconds = heartbeat_seconds
        self._continuous = continuous
        self._samples = 0
        self._peaks: dict[int, float] = {}
        self._seq = 0
        self._since_full = 0
        self._force_full = True
        self._shown: dict[int, int] = {}
        self._peak_shown: dict[int, int] = {}
        self._last_key: str | None = None
        self._last_sent = 0.0
        self._refreshes = 0
        self._skipped = 0
        self._stop = threading.Event()

    @property
    def refreshes(self) -> int:
        """Frames actually written to the panel."""
        return self._refreshes

    @property
    def skipped(self) -> int:
        """Frames withheld because they would have changed nothing."""
        return self._skipped

    @property
    def peaks(self) -> dict[int, float]:
        """Best measurable reading per camera since the last reset."""
        return dict(self._peaks)

    def reset_peaks(self) -> None:
        """Forget the best readings.

        Wanted whenever the scene changes -- the light shifts, or the
        camera is re-aimed -- because a peak set against the old view is
        a target the new one may never reach.
        """
        self._peaks.clear()
        self._peak_shown.clear()
        self._shown.clear()
        logger.info("Focus peaks reset")

    def force_full_refresh(self) -> None:
        """Redraw completely on the next frame, changed or not.

        What the HOME button asks for. A full refresh is how accumulated
        ghosting is lifted, so it has to bypass deduplication -- the
        screen needing a clean is not a change in what it says.
        """
        self._force_full = True

    def stop(self) -> None:
        self._stop.set()

    def read_once(self) -> list[FocusReading]:
        """Capture and measure every camera once, updating the peaks."""
        readings: list[FocusReading] = []
        for index, camera in enumerate(self._cameras):
            ok, frame = camera.read()
            if not ok or frame is None:
                logger.debug("Camera %d produced no frame", index)
                continue
            reading = analyse_frame(
                frame,
                index,
                metadata=_metadata_of(camera),
                ceiling=self._ceiling,
                peak_pct=self._peaks.get(index, 0.0),
            )
            self._peaks[index] = reading.peak_pct
            readings.append(reading)
        return readings

    def build_screen(
        self,
        readings: Sequence[FocusReading],
        *,
        clock: str,
        trigger: str | None = None,
    ) -> ScreenFrame:
        """Lay out one screen. Two rows per camera, then the footer."""
        lines: list[ScreenLine] = []
        for reading in readings:
            shown = self._shown_percent(reading)
            lines.append(
                ScreenLine(
                    text=f"CAM{reading.camera}  {_fmt_percent(reading, shown)}",
                    # An unmeasurable frame draws an empty bar. Showing
                    # its percentage as a filled bar would invite someone
                    # to keep turning a lens that is not the problem.
                    bar=shown,
                )
            )
            lines.append(
                ScreenLine(
                    text=_detail_row(reading, self._peak_shown.get(reading.camera)),
                    rule=True,
                )
            )

        if not readings:
            lines.append(ScreenLine(text="no frames from any camera", rule=True))

        if trigger is not None:
            footer = f"sample {self._samples}  key {trigger}"
        elif self._samples:
            footer = f"sample {self._samples}  press to re-read"
        else:
            footer = "press a button to read"
        lines.append(ScreenLine(text=footer))

        # Content only. The sequence number and the refresh mode are
        # decided at send time, because a frame that turns out to be
        # identical to the one already drawn is never sent at all and
        # must not consume either.
        return ScreenFrame(title=f"FOCUS  {clock}", lines=tuple(lines))

    def send_if_changed(self, frame: ScreenFrame) -> bool:
        """Write the frame only if it would change the screen.

        E-paper wears out and ghosts. The first version of this engine
        sent every reading unconditionally at 2 Hz, which is about 1140
        refreshes in a ten-minute session and left the panel mottled --
        most of them redrawing a number that had not moved, because a
        lens nobody is touching reads the same value every time.
        ``DisplayEngine`` had already solved this for the status screen;
        this is the same rule applied to the focus screen.
        """
        key = frame.content_key()
        stale = (time.monotonic() - self._last_sent) >= self._heartbeat_seconds

        if key == self._last_key and not stale and not self._force_full:
            self._skipped += 1
            return False

        return self._send(frame, key)

    def sample_and_draw(self, trigger: str | None = None) -> list[FocusReading]:
        """Take one reading of every camera and put it on the panel.

        A press always redraws, even when every number is identical to
        the one already showing. That redraw is the only evidence the
        press was received, and a button that silently does nothing is
        indistinguishable from a broken one -- which is exactly how the
        panel behaved before the pull-ups were enabled.
        """
        self._samples += 1
        readings = self.read_once()
        frame = self.build_screen(
            readings, clock=datetime.now().strftime("%H:%M"), trigger=trigger
        )

        if trigger is not None:
            # Full waveform, not partial. A partial update on this panel
            # resolves over two writes -- the first press showed a
            # half-formed screen that reads as noise and the second
            # cleaned it up, so readings alternated between garbage and
            # correct. The frames were never the problem: all six of a
            # test burst were acked in order within 0.15 s each.
            #
            # A full refresh costs a couple of seconds and more of the
            # panel's life than a partial one. That is the right trade
            # here and not in DisplayEngine: the status panel redraws
            # unattended for months, while this runs for a few minutes
            # with someone standing in front of it who needs to trust
            # what it says.
            self.force_full_refresh()
            self._send(frame, frame.content_key())
        else:
            self.send_if_changed(frame)
        return readings

    def draw_exit_screen(self) -> None:
        """Say plainly that the tool is no longer running.

        E-paper holds its last image with no power, so without this the
        panel keeps showing live-looking readings long after the process
        is gone -- which is how a stale number gets trusted. Drawn as a
        full refresh, which also leaves the panel clean.
        """
        self.force_full_refresh()
        frame = ScreenFrame(
            title="FOCUS TOOL EXITED",
            lines=(
                ScreenLine(text=f"{self._samples} samples taken", rule=True),
                ScreenLine(text="readings below are not live"),
                ScreenLine(text="restart: ratcatcher focus"),
            ),
        )
        self._send(frame, frame.content_key())

    def run(self) -> None:
        """Sample on demand until EXIT is pressed or stop() is called.

        Press-to-sample rather than free-running: a lens is adjusted and
        then evaluated, so a reading taken while a hand is still in front
        of it is noise. It also all but removes the panel wear, since the
        screen redraws when asked and not on a timer.
        """
        logger.info("Focus session started on %d camera(s)", len(self._cameras))
        self.sample_and_draw()
        last_auto = time.monotonic()

        while not self._stop.is_set():
            pressed = self._handle_buttons()

            if pressed is not None:
                self.sample_and_draw(trigger=pressed)
                last_auto = time.monotonic()
            else:
                since = time.monotonic() - last_auto
                due = self._interval if self._continuous else self._heartbeat_seconds
                if since >= due:
                    # In continuous mode this is the sample rate. Otherwise
                    # it only keeps the firmware from declaring the host
                    # dead after seven minutes of deliberate silence.
                    self.sample_and_draw()
                    last_auto = time.monotonic()

            self._stop.wait(BUTTON_POLL_SECONDS)

        logger.info(
            "Focus session stopped: %d samples, %d panel refreshes, %d skipped",
            self._samples,
            self._refreshes,
            self._skipped,
        )

    # -- internals ---------------------------------------------------------

    def _shown_percent(self, reading: FocusReading) -> int:
        """The percentage as the panel should show it.

        Held steady inside DISPLAY_DEADBAND_PCT so that a still scene
        produces a still screen. Hysteresis rather than plain rounding,
        because rounding only moves the flicker to the boundary between
        two buckets instead of removing it.
        """
        if not reading.measurable:
            self._shown.pop(reading.camera, None)
            return 0

        value = int(round(reading.focus_pct))
        previous = self._shown.get(reading.camera)
        if previous is not None and abs(value - previous) < DISPLAY_DEADBAND_PCT:
            value = previous
        self._shown[reading.camera] = value

        # The peak shown is the best value ever *displayed*, not the best
        # ever measured. Tracking the raw maximum would let noise creep it
        # upward a point at a time, and each creep is a panel refresh --
        # and a target the live number can then never quite reach.
        best = self._peak_shown.get(reading.camera)
        self._peak_shown[reading.camera] = value if best is None else max(best, value)
        return value

    def _send(self, frame: ScreenFrame, key: str) -> bool:
        self._seq += 1
        full = self._force_full or self._since_full >= self._full_refresh_every
        self._force_full = False
        self._since_full = 0 if full else self._since_full + 1

        sent = self._panel.send_screen(
            replace(frame, seq=self._seq, full_refresh=full)
        )
        if sent:
            self._last_key = key
            self._last_sent = time.monotonic()
            self._refreshes += 1
        return sent

    def _handle_buttons(self) -> str | None:
        """Drain the panel. Returns the key that asked for a reading.

        Only the press edge acts. Firmware 1.2.0 reports the release as
        well, so that a pin stuck down is visible as a press with no
        matching release rather than as silence; acting on both would
        take two readings for every push.
        """
        wanted: str | None = None

        for message in self._panel.poll():
            if message.get("t") != "btn":
                continue
            # Absent "down" means firmware older than 1.2.0, which only
            # ever reported presses.
            if not message.get("down", 1):
                continue

            button = message.get("id")
            logger.info("Panel button: %s", button)

            if button == "exit":
                self._stop.set()
            elif button in SAMPLE_BUTTONS:
                if button == "home":
                    # Doubles as the ghosting clear, since a full refresh
                    # is what lifts it.
                    self.force_full_refresh()
                wanted = button

        return wanted


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _fmt_percent(reading: FocusReading, shown: int) -> str:
    """The headline number, or a dash when it would not mean anything."""
    if not reading.measurable:
        return "  --"
    return f"{shown:3d}%"


def _detail_row(reading: FocusReading, peak_shown: int | None) -> str:
    """Peak, light, exposure and state, in one 41-character row."""
    peak = f"{peak_shown:d}" if reading.measurable and peak_shown is not None else "--"
    parts = [
        f"PEAK {peak}",
        f"LUX {_fmt_lux(reading.lux)}",
        f"EXP {_fmt_exposure(reading.exposure_us)}",
        reading.state,
    ]
    return "  ".join(parts)


def _one_significant_figure(value: float) -> float:
    """Round to one significant figure.

    The panel only has to answer "what order of magnitude", and holding
    the telemetry there is what lets an unchanged screen stay unchanged.
    Auto-exposure adjusts lux and exposure slightly on every frame, so at
    full precision these two fields alone would defeat deduplication and
    redraw the panel once a second however still the scene was.
    """
    if value <= 0:
        return 0.0
    step = 10.0 ** math.floor(math.log10(value))
    return round(value / step) * step


def _fmt_lux(lux: float | None) -> str:
    if lux is None:
        return "--"
    value = _one_significant_figure(lux)
    if value >= 1000:
        return f"{value / 1000:.0f}k"
    return f"{value:.0f}"


def _fmt_exposure(exposure_us: int | None) -> str:
    if exposure_us is None:
        return "--"
    value = _one_significant_figure(float(exposure_us))
    if value >= 1000:
        return f"{value / 1000:.0f}ms"
    return f"{value:.0f}u"


def _metadata_of(camera: CameraSource) -> dict[str, Any] | None:
    """Ask a camera for its control values, if it has any to give.

    Duck-typed rather than declared on ``CameraSource``: only a real
    sensor has an exposure time, and requiring the method would stop the
    tool running against a file source with no hardware attached.
    """
    getter = getattr(camera, "capture_metadata", None)
    if getter is None:
        return None
    return getter()
