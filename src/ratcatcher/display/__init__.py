"""Status panel support for RatCatcher AI.

Drives an Elecrow CrowPanel ESP32 2.13-inch e-paper display over USB
serial. The panel reports what the system has detected today and whether
the system itself is well.

Like the audio path, the panel is an independent consumer. It reads the
SQLite database and the health module and writes to a serial port. It
never touches the pipeline queues, so an unplugged panel cannot stop the
cameras and a stalled camera cannot blank the panel.
"""

from ratcatcher.display.protocol import (
    PANEL_MODEL,
    PROTOCOL_VERSION,
    LastSighting,
    StatusFrame,
    SystemStatus,
    decode_line,
    encode_frame,
)

__all__ = [
    "PANEL_MODEL",
    "PROTOCOL_VERSION",
    "LastSighting",
    "StatusFrame",
    "SystemStatus",
    "decode_line",
    "encode_frame",
]
