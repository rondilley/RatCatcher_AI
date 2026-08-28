"""Transports that carry status frames to a display.

``SerialPanel`` drives the real CrowPanel over USB. ``FilePanel`` and
``NullPanel`` stand in for it during development and in the tests, in
the same way that ``WavFileSource`` stands in for the microphones.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ratcatcher.display.protocol import (
    StatusFrame,
    decode_line,
    encode_frame,
    encode_ping,
    is_hello,
)

logger = logging.getLogger(__name__)

# Read buffer ceiling. The panel sends short acknowledgements, so a
# buffer this size can only fill if the far end has become a noise
# source, in which case the oldest bytes are the least useful.
_MAX_RX_BUFFER = 4096


@runtime_checkable
class StatusPanel(Protocol):
    """A destination for status frames."""

    def open(self) -> None:
        """Make the panel ready to receive.

        Idempotent: opening an already open panel does nothing. The
        engine calls this on every panel it drives, including one handed
        to it already open. Raises OSError on failure.
        """

    def send(self, frame: StatusFrame) -> bool:
        """Write one frame. Returns False if the link is down."""

    def poll(self) -> list[dict[str, Any]]:
        """Return any messages the panel has sent since the last call."""

    def close(self) -> None:
        """Release the panel."""

    @property
    def description(self) -> str:
        """One line naming this transport, for logs and the CLI."""


class SerialPanel:
    """The CrowPanel on a USB serial port.

    Reconnects on its own. A panel that is unplugged and plugged back in
    comes back without restarting RatCatcher, which matters because the
    panel is the thing you look at to find out whether RatCatcher is
    running: it must not be able to take the system down with it.
    """

    def __init__(
        self,
        port: str,
        baud_rate: int = 115200,
        *,
        write_timeout: float = 2.0,
        reconnect_seconds: float = 10.0,
    ) -> None:
        self._port = port
        self._baud_rate = baud_rate
        self._write_timeout = write_timeout
        self._reconnect_seconds = reconnect_seconds
        self._serial: Any = None
        self._rx = bytearray()
        self._next_retry = 0.0
        self._confirmed = False

    @property
    def description(self) -> str:
        return f"serial ({self._port} @ {self._baud_rate} baud)"

    @property
    def port(self) -> str:
        return self._port

    @property
    def connected(self) -> bool:
        return self._serial is not None and bool(self._serial.is_open)

    @property
    def confirmed(self) -> bool:
        """True once the panel has identified itself.

        A frame still goes out when this is False. A panel whose
        transmit path is broken draws perfectly well.
        """
        return self._confirmed

    def open(self) -> None:
        if self.connected:
            return

        import serial  # imported here so pyserial stays an optional extra

        # Both control lines are set low across the open. On an ESP32
        # board they are wired to EN and IO0 through the usual
        # auto-reset transistors. This is the most the host can do, and
        # it is not enough on Linux: the tty open asserts DTR and RTS in
        # the driver before pyserial applies these, so the panel reboots
        # anyway. open() therefore waits the board out before the first
        # frame -- see RESET_SETTLE_SECONDS in display.factory.
        handle = serial.Serial()
        handle.port = self._port
        handle.baudrate = self._baud_rate
        handle.bytesize = serial.EIGHTBITS
        handle.parity = serial.PARITY_NONE
        handle.stopbits = serial.STOPBITS_ONE
        handle.timeout = 0.0
        handle.write_timeout = self._write_timeout
        handle.dtr = False
        handle.rts = False

        try:
            handle.open()
        except (serial.SerialException, OSError) as exc:
            raise OSError(f"Cannot open panel port {self._port}: {exc}") from exc

        self._serial = handle
        self._rx.clear()
        self._confirmed = False
        logger.info("Status panel connected on %s", self._port)

        # Wait out the reset that the open just caused. Without this the
        # first frame is written into the ROM loader and vanishes, and
        # the screen stays blank with every layer reporting success.
        from ratcatcher.display.factory import RESET_SETTLE_SECONDS, wait_for_hello

        try:
            handle.reset_input_buffer()
            if wait_for_hello(handle, RESET_SETTLE_SECONDS):
                self._confirmed = True
                logger.info("Status panel answered on %s", self._port)
            else:
                # No answer is not fatal: a panel that draws correctly but
                # cannot transmit is still worth sending frames to.
                logger.warning(
                    "Status panel on %s did not answer within %.1f s; "
                    "sending frames anyway",
                    self._port,
                    RESET_SETTLE_SECONDS,
                )
        except (serial.SerialException, OSError) as exc:
            logger.debug("Panel ping failed on open: %s", exc)

    def send(self, frame: StatusFrame) -> bool:
        if not self.connected and not self._try_reconnect():
            return False

        import serial

        try:
            self._serial.write(encode_frame(frame))
            return True
        except (serial.SerialException, serial.SerialTimeoutException, OSError) as exc:
            logger.warning("Status panel write failed on %s: %s", self._port, exc)
            self._drop()
            return False

    def poll(self) -> list[dict[str, Any]]:
        if not self.connected:
            return []

        import serial

        try:
            waiting = self._serial.in_waiting
            if waiting:
                self._rx.extend(self._serial.read(waiting))
        except (serial.SerialException, OSError) as exc:
            logger.warning("Status panel read failed on %s: %s", self._port, exc)
            self._drop()
            return []

        if len(self._rx) > _MAX_RX_BUFFER:
            del self._rx[: len(self._rx) - _MAX_RX_BUFFER]

        messages: list[dict[str, Any]] = []
        while b"\n" in self._rx:
            raw, _, rest = bytes(self._rx).partition(b"\n")
            self._rx = bytearray(rest)
            message = decode_line(raw)
            if message is None:
                continue
            if is_hello(message):
                self._confirmed = True
                logger.info(
                    "Status panel identified: firmware %s on %s",
                    message.get("fw", "?"),
                    self._port,
                )
            messages.append(message)
        return messages

    def close(self) -> None:
        self._drop()

    # -- internals ---------------------------------------------------------

    def _drop(self) -> None:
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:  # noqa: BLE001 -- closing must never raise
                pass
        self._serial = None
        self._confirmed = False
        self._next_retry = time.monotonic() + self._reconnect_seconds

    def _try_reconnect(self) -> bool:
        if time.monotonic() < self._next_retry:
            return False
        try:
            self.open()
            return True
        except OSError as exc:
            logger.debug("Status panel reconnect failed: %s", exc)
            self._next_retry = time.monotonic() + self._reconnect_seconds
            return False


class FilePanel:
    """Appends encoded frames to a file.

    Lets the whole host path run with no panel attached. The file holds
    exactly the bytes the serial port would have carried, so a frame can
    be replayed into a real panel later.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._handle: Any = None

    @property
    def description(self) -> str:
        return f"file ({self._path})"

    def open(self) -> None:
        if self._handle is not None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self._path, "ab")
        logger.info("Status panel writing to %s", self._path)

    def send(self, frame: StatusFrame) -> bool:
        if self._handle is None:
            return False
        self._handle.write(encode_frame(frame))
        self._handle.flush()
        return True

    def poll(self) -> list[dict[str, Any]]:
        return []

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


class NullPanel:
    """Accepts frames and discards them, keeping only the last one.

    Used when no panel is configured and by the tests, so the frame
    builder and the engine can be exercised with no hardware.
    """

    def __init__(self) -> None:
        self.frames: list[StatusFrame] = []

    @property
    def description(self) -> str:
        return "null (no panel attached)"

    @property
    def last_frame(self) -> StatusFrame | None:
        return self.frames[-1] if self.frames else None

    def open(self) -> None:
        return None

    def send(self, frame: StatusFrame) -> bool:
        self.frames.append(frame)
        return True

    def poll(self) -> list[dict[str, Any]]:
        return []

    def close(self) -> None:
        return None
