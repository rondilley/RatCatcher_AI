"""Wire protocol between the host and the CrowPanel status display.

One JSON object per line, UTF-8, terminated by a newline, at 115200 baud.
A line-delimited format is used rather than a binary frame because the
panel is a development board on a shared USB bus: a corrupt or partial
line is discarded on the next newline, and a person with a terminal
program can read the traffic without a decoder.

Host to panel
-------------
``status``
    The screen contents. Sent whenever the contents change and at least
    once per heartbeat interval.
``ping``
    Asks the panel to identify itself. Used to find the right serial
    port, which matters because the CrowPanel presents a generic CH340
    descriptor that no other field distinguishes from any other board.

Panel to host
-------------
``hello``
    Sent on boot and in answer to a ping. Carries the firmware version
    and the panel model, which is what makes port detection reliable.
``ack``
    Sent after a status frame is drawn, carrying the sequence number.
``btn``
    Sent when a button is pressed.
``err``
    Sent when a line cannot be parsed or drawn.

The host never depends on an answer. If the panel says nothing, frames
keep going out and the link is reported as unconfirmed, because a panel
that draws correctly but has a broken transmit path is still useful.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

# Raised on any change that an older firmware could not draw correctly.
# The firmware refuses a frame whose version it does not know, so a
# mismatch shows on the screen instead of producing a wrong reading.
PROTOCOL_VERSION = 1

PANEL_MODEL = "crowpanel-2.13"

# Firmware from this version understands the generic "screen" frame.
#
# PROTOCOL_VERSION is deliberately NOT bumped alongside it. Older
# firmware refuses every frame whose version it does not recognise, so
# raising it would blank the ordinary status panel on any Pi whose panel
# has not been reflashed. Unknown frame *types* are already ignored
# silently, which makes a new type a backwards-compatible addition and a
# version bump a breaking one.
MIN_SCREEN_FIRMWARE = (1, 1, 0)

# The firmware reads each line into a fixed buffer. Frames longer than
# this are truncated at the source rather than being cut in half by the
# panel, which would leave it parsing an incomplete object.
MAX_LINE_BYTES = 512

# Longest species name that fits the footer line at the 6-pixel font.
MAX_NAME_CHARS = 22

# Bounds on a generic screen frame. Six lines of 41 characters is what
# the 248 by 122 panel holds at the 6-pixel font, and keeping the frame
# inside MAX_LINE_BYTES is what stops the firmware's fixed receive
# buffer from seeing a truncated object.
MAX_SCREEN_LINES = 6
MAX_SCREEN_CHARS = 41
MAX_TITLE_CHARS = 41


@dataclass(frozen=True)
class LastSighting:
    """The most recent identification, whichever sense produced it."""

    name: str
    # "eye" for a camera detection, "ear" for a song identification
    sense: str
    clock: str


@dataclass(frozen=True)
class SystemStatus:
    """Health of the system itself, as the panel reports it."""

    cameras: int = 0
    npu: bool = False
    audio: bool = False
    temp_c: float | None = None
    disk_pct: float | None = None
    uptime: str = "?"


@dataclass(frozen=True)
class StatusFrame:
    """One complete screen of information.

    ``counts`` holds three categories, each a pair of (seen, heard).
    The panel draws them as the EYE and EAR columns.
    """

    clock: str
    state: str
    window: str
    counts: dict[str, tuple[int, int]]
    system: SystemStatus
    last: LastSighting | None = None
    seq: int = 0
    full_refresh: bool = False

    def content_key(self) -> str:
        """A value that changes only when the drawn content changes.

        The sequence number and the refresh mode are excluded: neither
        alters a single pixel, and including them would make every frame
        look new and drive a needless refresh of the panel.
        """
        return json.dumps(
            _payload(self, include_seq=False), sort_keys=True, separators=(",", ":")
        )


@dataclass(frozen=True)
class ScreenLine:
    """One row of a generic screen.

    ``bar`` draws a filled progress bar for a 0-100 value; ``rule``
    draws a horizontal line beneath the row.
    """

    text: str
    bar: int | None = None
    rule: bool = False


@dataclass(frozen=True)
class ScreenFrame:
    """A title and up to ``MAX_SCREEN_LINES`` rows of free text.

    Deliberately generic. The panel draws whatever it is told rather
    than knowing what a camera or a focus reading is, so a new field
    tool is a change to the host alone. Reflashing an ESP32 needs a
    2.3 GB toolchain and a serial port that a person has to identify by
    hand, and that cost should be paid once rather than per tool.
    """

    title: str
    lines: tuple[ScreenLine, ...] = ()
    seq: int = 0
    full_refresh: bool = False

    def content_key(self) -> str:
        """A value that changes only when the drawn content changes."""
        return json.dumps(
            _screen_payload(self, include_seq=False),
            sort_keys=True,
            separators=(",", ":"),
        )


def encode_frame(frame: StatusFrame) -> bytes:
    """Serialise a status frame to one newline-terminated line.

    The species name is shortened until the whole line fits the panel's
    input buffer. Shortening the name is the only lossy step available:
    every other field is a small number the screen must show exactly.
    """
    payload = _payload(frame, include_seq=True)
    line = _dump(payload)

    if len(line) > MAX_LINE_BYTES and payload.get("last") is not None:
        name = payload["last"]["name"]
        while len(line) > MAX_LINE_BYTES and len(name) > 3:
            name = name[: len(name) - 1]
            payload["last"]["name"] = name.rstrip() + "."
            line = _dump(payload)

    if len(line) > MAX_LINE_BYTES:
        # Nothing left to shorten. Drop the sighting rather than send a
        # line the panel will truncate mid-object.
        payload.pop("last", None)
        line = _dump(payload)

    return line + b"\n"


def encode_screen(frame: ScreenFrame) -> bytes:
    """Serialise a generic screen to one newline-terminated line.

    Text is truncated rather than wrapped: a row that ran onto the next
    one would push the last row off a 122-pixel screen, and losing the
    end of a label is easier to read past than losing a whole reading.
    Whole rows are dropped only if truncation alone cannot fit the
    frame, which the bounded layout should never reach.
    """
    payload = _screen_payload(frame, include_seq=True)
    line = _dump(payload)

    while len(line) > MAX_LINE_BYTES and payload["l"]:
        payload["l"].pop()
        line = _dump(payload)

    return line + b"\n"


def encode_ping() -> bytes:
    """Serialise the identification request."""
    return _dump({"v": PROTOCOL_VERSION, "t": "ping"}) + b"\n"


def decode_line(line: bytes | str) -> dict[str, Any] | None:
    """Parse one line sent by the panel.

    Returns None for blank lines and for anything that is not a JSON
    object with a message type. The panel shares its serial port with
    the ESP32 boot loader, which prints its own banner at reset, so
    unparsable lines are normal traffic rather than an error.
    """
    if isinstance(line, bytes):
        try:
            text = line.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return None
    else:
        text = line

    text = text.strip()
    if not text or not text.startswith("{"):
        return None

    try:
        message = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(message, dict) or "t" not in message:
        return None
    return message


def is_hello(message: dict[str, Any]) -> bool:
    """True if a decoded message identifies a RatCatcher panel."""
    return message.get("t") == "hello" and message.get("panel") == PANEL_MODEL


def supports_screen(firmware: str | None) -> bool:
    """True when this firmware version can draw a ScreenFrame.

    An unknown or unparsable version is given the benefit of the doubt.
    A panel that never answered may still draw perfectly well -- the
    transmit path and the display are independent -- and refusing to
    drive it would turn a cosmetic uncertainty into a dead tool.
    """
    if not firmware:
        return True
    try:
        parts = tuple(int(part) for part in firmware.split(".")[:3])
    except ValueError:
        return True
    return parts >= MIN_SCREEN_FIRMWARE


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _dump(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def _payload(frame: StatusFrame, *, include_seq: bool) -> dict[str, Any]:
    """Build the wire dictionary. Keys are short to keep lines small."""
    system = frame.system
    payload: dict[str, Any] = {
        "v": PROTOCOL_VERSION,
        "t": "status",
        "clk": frame.clock,
        "st": frame.state,
        "win": frame.window,
        "n": {
            name: [int(seen), int(heard)]
            for name, (seen, heard) in sorted(frame.counts.items())
        },
        "sys": {
            "cam": int(system.cameras),
            "npu": 1 if system.npu else 0,
            "aud": 1 if system.audio else 0,
            "temp": None if system.temp_c is None else round(system.temp_c),
            "disk": None if system.disk_pct is None else round(system.disk_pct),
            "up": system.uptime,
        },
    }

    if frame.last is not None:
        payload["last"] = {
            "name": frame.last.name[:MAX_NAME_CHARS],
            "sense": frame.last.sense,
            "clk": frame.last.clock,
        }

    if include_seq:
        payload["seq"] = int(frame.seq)
        payload["full"] = 1 if frame.full_refresh else 0

    return payload


def _screen_payload(frame: ScreenFrame, *, include_seq: bool) -> dict[str, Any]:
    """Build the wire dictionary for a generic screen."""
    lines: list[dict[str, Any]] = []
    for line in frame.lines[:MAX_SCREEN_LINES]:
        item: dict[str, Any] = {"t": line.text[:MAX_SCREEN_CHARS]}
        if line.bar is not None:
            item["b"] = max(0, min(100, int(line.bar)))
        if line.rule:
            item["r"] = 1
        lines.append(item)

    payload: dict[str, Any] = {
        "v": PROTOCOL_VERSION,
        "t": "screen",
        "title": frame.title[:MAX_TITLE_CHARS],
        "l": lines,
    }

    if include_seq:
        payload["seq"] = int(frame.seq)
        payload["full"] = 1 if frame.full_refresh else 0

    return payload
