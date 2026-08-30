"""Detection and status reporting to syslog.

Two kinds of record leave this module: one line per detection, and a
periodic status line carrying the same counts the e-paper panel shows.
Both are written in logfmt -- ``key=value`` pairs -- so a log aggregator
can extract fields without a regular expression per message shape::

    ratcatcher[953]: detection modality=video camera=0 class=squirrel
      common="Western Gray Squirrel" confidence=0.87 pest=yes

They travel on their own logger, ``ratcatcher.events``, and the syslog
handler is attached to that logger alone. That is what keeps a forwarded
stream narrow: libcamera's pixel format warnings and picamera2's startup
chatter never reach it. The logger still propagates, so every line also
lands in journald next to the ordinary log output, and no information is
lost by turning syslog off.

Nothing downstream reads these records and nothing in the pipeline waits
on them. A loghost that is down, slow or absent cannot affect detection.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import socket
from typing import Any

from ratcatcher.config import SyslogConfig
from ratcatcher.monitoring.stats import CategoryCounts

logger = logging.getLogger(__name__)

EVENT_LOGGER_NAME = "ratcatcher.events"

event_logger = logging.getLogger(EVENT_LOGGER_NAME)

# The handler this module installed, kept so a second call to
# configure_event_log replaces it rather than stacking a duplicate on a
# logger that would then send every line twice.
_handler: logging.Handler | None = None

# Detection lines are gated separately from the transport. Turning
# syslog off silences the socket, not the record: the lines still reach
# journald through propagation, which is where they are wanted anyway.
_detections_enabled = True


def format_fields(**fields: Any) -> str:
    """Render keyword arguments as a logfmt fragment.

    Fields whose value is None are dropped rather than written as an
    empty token. A detection with no species is one field shorter, not
    one field carrying nothing, which keeps a parser from having to
    distinguish "absent" from "empty".
    """
    parts = []
    for key, value in fields.items():
        if value is None:
            continue
        parts.append(f"{key}={_encode(value)}")
    return " ".join(parts)


def configure_event_log(config: SyslogConfig) -> bool:
    """Attach a syslog handler to the event logger.

    Returns True if the handler was attached. A syslog socket that
    cannot be opened is reported and then ignored: detection must not
    depend on a loghost being reachable, and the lines still reach
    journald without it.
    """
    global _handler, _detections_enabled

    _detections_enabled = config.detections

    if _handler is not None:
        event_logger.removeHandler(_handler)
        _handler.close()
        _handler = None

    if not config.enabled:
        return False

    try:
        address = _parse_address(config.address)
    except ValueError as exc:
        logger.warning("Syslog disabled -- %s", exc)
        return False

    facility = logging.handlers.SysLogHandler.facility_names.get(
        config.facility.lower()
    )
    if facility is None:
        logger.warning(
            "Syslog disabled -- unknown facility '%s'. Known facilities: %s",
            config.facility,
            ", ".join(sorted(logging.handlers.SysLogHandler.facility_names)),
        )
        return False

    socktype = None
    if isinstance(address, tuple):
        protocol = config.protocol.lower()
        if protocol == "tcp":
            socktype = socket.SOCK_STREAM
        elif protocol == "udp":
            socktype = socket.SOCK_DGRAM
        else:
            logger.warning(
                "Syslog disabled -- unknown protocol '%s'. Use udp or tcp.",
                config.protocol,
            )
            return False

    # SysLogHandler defers its connection: since Python 3.11 a bad
    # socket path constructs without complaint and then raises inside
    # emit, where logging catches it and prints a traceback -- once per
    # detection, forever. Probe the destination here so a misconfigured
    # address is one warning at startup instead.
    try:
        _probe(address, socktype)
        handler = logging.handlers.SysLogHandler(
            address=address, facility=facility, socktype=socktype
        )
    except OSError as exc:
        logger.warning("Syslog disabled -- cannot open %s: %s", config.address, exc)
        return False

    # SysLogHandler writes no tag of its own, so the message has to
    # carry one. Without it the lines arrive unattributed and
    # "journalctl -t ratcatcher" matches nothing.
    handler.setFormatter(
        logging.Formatter(f"{config.ident}[{os.getpid()}]: %(message)s")
    )
    handler.setLevel(logging.INFO)

    event_logger.addHandler(handler)
    event_logger.setLevel(logging.INFO)
    _handler = handler

    logger.info(
        "Syslog reporting to %s, facility=%s, tag=%s",
        config.address,
        config.facility,
        config.ident,
    )
    return True


def close_event_log() -> None:
    """Detach and close the syslog handler, if one is attached."""
    global _handler

    if _handler is not None:
        event_logger.removeHandler(_handler)
        _handler.close()
        _handler = None


def log_video_detection(
    *,
    camera: int,
    class_name: str | None,
    species: str | None = None,
    common_name: str | None = None,
    confidence: float | None = None,
    species_confidence: float | None = None,
    pest: bool = False,
    clip_path: str | None = None,
    thumbnail_path: str | None = None,
) -> None:
    """Report one thing seen by a camera."""
    if not _detections_enabled:
        return

    event_logger.info(
        "detection %s",
        format_fields(
            modality="video",
            camera=camera,
            **{"class": class_name},
            species=species,
            common=common_name,
            confidence=confidence,
            species_confidence=species_confidence,
            pest=pest,
            clip=clip_path,
            thumbnail=thumbnail_path,
        ),
    )


def log_audio_detection(
    *,
    channel: int,
    species: str | None = None,
    common_name: str | None = None,
    confidence: float | None = None,
    rms_dbfs: float | None = None,
    flatness: float | None = None,
    peak_hz: float | None = None,
    clip_path: str | None = None,
) -> None:
    """Report one thing heard by a microphone.

    Callers report only windows that named a species, the same rule the
    video path follows: a window that passed the activity gate and
    matched nothing identifies no bird. That the system is still
    listening is what the periodic status line is for.
    """
    if not _detections_enabled:
        return

    event_logger.info(
        "detection %s",
        format_fields(
            modality="audio",
            channel=channel,
            species=species,
            common=common_name,
            confidence=confidence,
            rms_dbfs=rms_dbfs,
            flatness=flatness,
            peak_hz=peak_hz,
            clip=clip_path,
        ),
    )


def log_status(
    *,
    window: str,
    counts: CategoryCounts,
    cameras: int,
    npu: bool,
    audio: bool,
    temp_c: float | None = None,
    disk_pct: float | None = None,
    uptime: str | None = None,
    power: str | None = None,
    batt_pct: int | None = None,
    batt_v: float | None = None,
) -> None:
    """Report the running totals and the state of the hardware.

    ``counts`` is the same CategoryCounts the status panel draws from,
    so the two readings agree by construction rather than by care.

    The battery fields are dropped when there is no UPS HAT to read, so
    a machine without one emits exactly the line it emitted before they
    existed rather than three empty tokens.
    """
    event_logger.info(
        "status %s",
        format_fields(
            window=window,
            bird_seen=counts.bird.seen,
            bird_heard=counts.bird.heard,
            rodent_seen=counts.rodent.seen,
            rodent_heard=counts.rodent.heard,
            other_seen=counts.other.seen,
            other_heard=counts.other.heard,
            total=counts.total,
            cameras=cameras,
            npu=npu,
            audio=audio,
            power=power,
            batt_pct=batt_pct,
            batt_v=batt_v,
            temp_c=temp_c,
            disk_pct=disk_pct,
            uptime=uptime,
        ),
    )


# The three things worth saying about power, as one record shape rather
# than three. A parser learns "power" once and reads the event field.
POWER_ON_BATTERY = "on_battery"
POWER_ON_MAINS = "on_mains"
POWER_SHUTDOWN = "shutdown"


def log_power(
    *,
    event: str,
    percent: int | None = None,
    pack_v: float | None = None,
    minutes_remaining: int | None = None,
    reason: str | None = None,
) -> None:
    """Report a change in how the system is being powered.

    Sent the moment the change is seen rather than folded into the next
    status line. The status interval is five minutes by default, and on
    a pack measured at 5-7 hours that is a fifth of the useful warning
    time spent saying nothing -- and a short outage could begin and end
    entirely between two status lines, leaving no record that it
    happened at all.

    Losing mains and shutting down are warnings; regaining mains is not.
    A stream filtered to warnings should carry the outage and its
    consequence without also carrying the recovery.
    """
    line = format_fields(
        event=event,
        batt_pct=percent,
        batt_v=pack_v,
        minutes=minutes_remaining,
        reason=reason,
    )
    if event == POWER_ON_MAINS:
        event_logger.info("power %s", line)
    else:
        event_logger.warning("power %s", line)


def _encode(value: Any) -> str:
    """Render one logfmt value, quoting it only when it needs quoting."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        text = f"{value:.3f}".rstrip("0").rstrip(".")
        # A value that rounds to nothing reads better as 0 than as "-0"
        # or an empty string.
        text = text if text not in ("", "-", "-0") else "0"
    else:
        text = str(value)

    if text == "" or any(c in text for c in ' "=\\'):
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return text


def _probe(address: str | tuple[str, int], socktype: int | None) -> None:
    """Check that the syslog destination can actually be reached.

    Raises OSError if it cannot. A UDP destination is not probed, since
    there is nothing to connect to and no delivery to confirm; that is
    the trade UDP syslog makes, and it is the operator's to make.
    """
    if isinstance(address, str):
        # The same order SysLogHandler tries: a datagram socket first,
        # then a stream socket, because /dev/log is one or the other
        # depending on the syslog daemon.
        last: OSError = OSError(f"cannot connect to {address}")
        for kind in (socket.SOCK_DGRAM, socket.SOCK_STREAM):
            probe = socket.socket(socket.AF_UNIX, kind)
            try:
                probe.connect(address)
                return
            except OSError as exc:
                last = exc
            finally:
                probe.close()
        raise last

    if socktype == socket.SOCK_STREAM:
        probe = socket.create_connection(address, timeout=5.0)
        probe.close()


def _parse_address(address: str) -> str | tuple[str, int]:
    """Interpret a configured address as a socket path or a host and port.

    A path is anything that looks like one. Everything else must be
    "host:port": a bare hostname is rejected rather than given a default
    port, because silently choosing 514 for a value the operator may
    have mistyped sends detections somewhere they did not ask for.
    """
    text = address.strip()

    if not text:
        raise ValueError("syslog address is empty")

    if text.startswith("/") or text.startswith("."):
        return text

    host, separator, port = text.rpartition(":")
    if not separator or not host:
        raise ValueError(
            f"syslog address '{address}' is neither a socket path nor host:port"
        )

    try:
        return (host, int(port))
    except ValueError:
        raise ValueError(
            f"syslog address '{address}' has a non-numeric port '{port}'"
        ) from None
