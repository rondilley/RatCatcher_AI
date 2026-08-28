"""Status panel factory.

Selects and constructs the panel transport from ``DisplayConfig`` and
from the hardware present at runtime, mirroring ``audio.factory`` and
``detection.factory``.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ratcatcher.config import DisplayConfig
from ratcatcher.display.panel import FilePanel, NullPanel, SerialPanel, StatusPanel
from ratcatcher.display.protocol import decode_line, encode_ping, is_hello

logger = logging.getLogger(__name__)

# Opening a serial port resets an ESP32 panel, and nothing the host can
# do stops it. The control lines go to EN and IO0 through the usual
# auto-reset transistors, and a Linux tty open asserts DTR and RTS in
# the driver before pyserial can apply dtr=False/rts=False. Measured on
# a CrowPanel 2.13: the first byte of the ROM banner arrives 0.22 s
# after the open, and the firmware hello arrives at 2.42 s, whether or
# not the host sends anything. Anything written before that is lost in
# the ROM loader.
RESET_SETTLE_SECONDS = 3.0
_PING_INTERVAL_SECONDS = 0.5

# USB serial bridges found on ESP32 development boards. Automatic port
# detection is restricted to these because probing writes a byte to
# every port it opens, and the machine may carry serial devices that
# belong to something else entirely.
_KNOWN_BRIDGES: frozenset[tuple[int, int]] = frozenset(
    {
        (0x1A86, 0x7522),  # CH340K, fitted to the CrowPanel 2.13
        (0x1A86, 0x7523),  # CH340G
        (0x1A86, 0x5523),  # CH341 in serial mode
        (0x1A86, 0x55D3),  # CH343
        (0x10C4, 0xEA60),  # CP2102
        (0x303A, 0x1001),  # ESP32-S3 native USB
    }
)


def create_panel(config: DisplayConfig) -> StatusPanel:
    """Build a ``StatusPanel`` from a ``DisplayConfig``.

    Raises
    ------
    RuntimeError
        If the requested transport is unknown or unusable. The "auto"
        transport never raises: it falls back to a null panel, because
        a missing display must not stop the detection pipeline.
    """
    source_type = config.source_type.lower()

    if source_type == "auto":
        return _auto_select(config)
    if source_type == "serial":
        return _make_serial(config)
    if source_type == "file":
        return _make_file(config)
    if source_type == "null":
        return NullPanel()

    raise RuntimeError(
        f"Unknown display source type '{config.source_type}'. "
        f"Supported values: auto, serial, file, null"
    )


def find_panel_port(
    baud_rate: int = 115200,
    probe_seconds: float = 2.0,
    candidates: list[str] | None = None,
) -> str | None:
    """Find the serial port that a RatCatcher panel answers on.

    Sends a ping to each candidate port and waits for the panel to
    identify itself. The identification handshake is necessary: the
    CrowPanel presents a plain CH340 descriptor, which is the same
    descriptor as any other cheap development board, so the USB
    identifiers alone cannot tell one from the other.

    Returns the port name, or None if no panel answered.
    """
    ports = candidates if candidates is not None else _candidate_ports()
    if not ports:
        logger.debug("No candidate serial ports for the status panel")
        return None

    for port in ports:
        if _probe(port, baud_rate, probe_seconds):
            logger.info("Status panel found on %s", port)
            return port
        logger.debug("No panel answered on %s", port)

    return None


def list_candidate_ports() -> list[tuple[str, str]]:
    """Return (port, description) for every port worth probing."""
    try:
        from serial.tools import list_ports
    except ImportError:
        return []

    found: list[tuple[str, str]] = []
    for info in list_ports.comports():
        if info.vid is None or info.pid is None:
            continue
        if (info.vid, info.pid) in _KNOWN_BRIDGES:
            found.append((info.device, f"{info.description} [{info.vid:04x}:{info.pid:04x}]"))
    return found


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _auto_select(config: DisplayConfig) -> StatusPanel:
    """Try the serial panel, and fall back to a null panel."""
    try:
        return _make_serial(config)
    except RuntimeError as exc:
        logger.info("Status panel unavailable, continuing without one -- %s", exc)
        return NullPanel()


def _make_serial(config: DisplayConfig) -> StatusPanel:
    try:
        import serial  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "pyserial is not installed. Install it with: "
            "pip install -e '.[display]'"
        ) from exc

    port = config.port
    if port in ("", "auto", None):
        port = find_panel_port(
            baud_rate=config.baud_rate,
            probe_seconds=config.probe_seconds,
        )
        if port is None:
            raise RuntimeError(
                "No status panel answered on any serial port. Check that the "
                "panel is plugged in, that it carries the RatCatcher firmware "
                "(scripts/build_panel_firmware.sh), and that this account is "
                "in the 'dialout' group."
            )

    panel = SerialPanel(
        port=port,
        baud_rate=config.baud_rate,
        reconnect_seconds=config.reconnect_seconds,
    )
    try:
        panel.open()
    except OSError as exc:
        raise RuntimeError(str(exc)) from exc

    logger.info("Status panel: %s", panel.description)
    return panel


def _make_file(config: DisplayConfig) -> StatusPanel:
    if not config.file_path:
        raise RuntimeError(
            "display.source_type is 'file' but display.file_path is not set"
        )
    panel = FilePanel(config.file_path)
    panel.open()
    logger.info("Status panel: %s", panel.description)
    return panel


def _candidate_ports() -> list[str]:
    return [port for port, _ in list_candidate_ports()]


def _probe(port: str, baud_rate: float, probe_seconds: float) -> bool:
    """Open one port, ping it, and wait for a RatCatcher hello."""
    import serial

    handle = serial.Serial()
    handle.port = port
    handle.baudrate = int(baud_rate)
    handle.timeout = 0.2
    handle.write_timeout = 1.0
    # Set low before the open, which is the most a caller can do. It is
    # not enough on Linux: see the note on RESET_SETTLE_SECONDS.
    handle.dtr = False
    handle.rts = False

    try:
        handle.open()
    except (serial.SerialException, OSError) as exc:
        logger.debug("Cannot probe %s: %s", port, exc)
        return False

    try:
        handle.reset_input_buffer()
        return wait_for_hello(handle, probe_seconds)
    except (serial.SerialException, OSError) as exc:
        logger.debug("Probe of %s failed: %s", port, exc)
        return False
    finally:
        try:
            handle.close()
        except Exception:  # noqa: BLE001 -- closing must never raise
            pass


def wait_for_hello(handle: Any, seconds: float) -> bool:
    """Ping until the panel answers with a hello, or the time runs out.

    The ping repeats rather than being sent once, because opening the
    port resets the board (see RESET_SETTLE_SECONDS). A single ping sent
    at the open lands while the ESP32 is still in its ROM loader and is
    discarded, so the only thing that could ever answer it was the
    hello the firmware sends at the end of its own boot.
    """
    deadline = time.monotonic() + seconds
    next_ping = 0.0
    buffer = bytearray()
    while time.monotonic() < deadline:
        now = time.monotonic()
        if now >= next_ping:
            try:
                handle.write(encode_ping())
            except Exception:  # noqa: BLE001 -- a closed port ends the wait
                return False
            next_ping = now + _PING_INTERVAL_SECONDS
        chunk = handle.read(256)
        if not chunk:
            continue
        buffer.extend(chunk)
        while b"\n" in buffer:
            raw, _, rest = bytes(buffer).partition(b"\n")
            buffer = bytearray(rest)
            message = decode_line(raw)
            if message is not None and is_hello(message):
                return True
    return False
