"""Builds the status frame that the panel draws.

Reads the detection database and the health module. Nothing here writes
to either, so the panel can be built from a second process while the
pipeline runs.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

from ratcatcher.config import Config
from ratcatcher.display.protocol import LastSighting, StatusFrame, SystemStatus
from ratcatcher.monitoring.health import HealthReport, check_health
from ratcatcher.monitoring.stats import categorize, get_category_counts
from ratcatcher.storage.database import DetectionDatabase

logger = logging.getLogger(__name__)

_CATEGORIES = ("bird", "rodent", "other")


def window_start(window: str, now: datetime | None = None) -> tuple[str | None, str]:
    """Return the ISO lower bound for a counting window and its label.

    "today" starts at local midnight. A person reading a feeder log
    thinks in days, and a count that resets at dawn matches what they
    saw out of the window.
    """
    moment = now if now is not None else datetime.now()
    key = window.lower()

    if key == "today":
        start = moment.replace(hour=0, minute=0, second=0, microsecond=0)
        return start.isoformat(), "today"
    if key == "24h":
        return (moment - timedelta(hours=24)).isoformat(), "24h"
    if key == "all":
        return None, "all"

    raise ValueError(
        f"Unknown display window '{window}'. Supported values: today, 24h, all"
    )


def build_status_frame(
    db: DetectionDatabase,
    config: Config,
    *,
    seq: int = 0,
    full_refresh: bool = False,
    state: str | None = None,
    cameras: int | None = None,
    audio_active: bool | None = None,
    health: HealthReport | None = None,
    now: datetime | None = None,
) -> StatusFrame:
    """Assemble one screen of information.

    Parameters
    ----------
    state:
        Overrides the state word in the header. The pipeline passes its
        own state; when None the state is derived from the health
        warnings, which is what the standalone CLI needs.
    cameras, audio_active:
        What the running pipeline actually opened. When None these fall
        back to what the configuration asks for, which is the best the
        CLI can know without starting the pipeline.
    """
    moment = now if now is not None else datetime.now()
    since, label = window_start(config.display.window, now=moment)

    counts = get_category_counts(db, since=since)
    frame_counts = {
        "bird": (counts.bird.seen, counts.bird.heard),
        "rodent": (counts.rodent.seen, counts.rodent.heard),
        "other": (counts.other.seen, counts.other.heard),
    }

    report = health if health is not None else check_health(
        data_dir=config.data_path,
        temp_warning_c=config.monitoring.temp_warning_c,
        temp_critical_c=config.monitoring.temp_critical_c,
    )

    if cameras is None:
        cameras = sum(1 for camera in config.cameras if camera.enabled)
    if audio_active is None:
        audio_active = config.audio.enabled

    system = SystemStatus(
        cameras=cameras,
        npu=report.hailo_available,
        audio=bool(audio_active),
        temp_c=report.cpu_temp_c,
        disk_pct=report.disk_usage_pct,
        uptime=format_uptime(report.uptime_seconds),
    )

    if state is None:
        state = "WARN" if report.warnings else "RUN"

    return StatusFrame(
        clock=moment.strftime("%H:%M"),
        state=state,
        window=label,
        counts=frame_counts,
        system=system,
        last=last_sighting(db, now=moment),
        seq=seq,
        full_refresh=full_refresh,
    )


def last_sighting(
    db: DetectionDatabase,
    now: datetime | None = None,
) -> LastSighting | None:
    """Return the most recent identification from either sense.

    Not restricted to the counting window. After a quiet night the last
    bird of yesterday is the useful thing to show, not a blank line.

    Only rows the counter accepts are eligible. The unified view labels
    every audio row "bird", so a window BirdNET could not identify would
    otherwise fall through to that class name and hold the panel on a
    bird nobody named.
    """
    moment = now if now is not None else datetime.now()

    try:
        rows = db.get_all_detections(limit=32)
    except Exception as exc:  # noqa: BLE001 -- the panel must not crash on this
        logger.debug("Could not read the last sighting: %s", exc)
        return None

    for row in rows:
        if categorize(
            str(row.get("modality", "")), row.get("class_name"), row.get("species")
        ) is None:
            continue
        name = row.get("common_name") or row.get("species") or row.get("class_name")
        if not name:
            continue
        return LastSighting(
            name=str(name),
            sense="ear" if row.get("modality") == "audio" else "eye",
            clock=_short_time(str(row.get("timestamp", "")), moment),
        )
    return None


def format_uptime(seconds: float | None) -> str:
    """Format an uptime as the shortest thing that reads correctly."""
    if seconds is None or seconds < 0:
        return "?"
    if seconds < 3600:
        return f"{int(seconds // 60)}m"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h"
    return f"{int(seconds // 86400)}d"


def _short_time(timestamp: str, now: datetime) -> str:
    """Show a clock time for today and a date for anything older."""
    try:
        moment = datetime.fromisoformat(timestamp)
    except ValueError:
        return "?"
    if moment.date() == now.date():
        return moment.strftime("%H:%M")
    return moment.strftime("%d%b")
