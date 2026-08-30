"""Battery telemetry from the Waveshare UPS HAT (E).

The HAT presents a small MCU on the I2C bus at 0x2d. It is not a fuel
gauge itself: it bridges two chips, a BQ4050 gauge and an IP2368 charger,
and republishes their readings as a flat register map. Register 0x03
reports whether it can still talk to each of them, which is why
``gauge_ok`` and ``charger_ok`` are carried through rather than assumed
-- a reading of 0% from a gauge that has stopped answering is not the
same statement as a reading of 0% from one that has.

**This module is read-only, deliberately.** Three registers in the map
are writable and two of them are destructive: writing 0x55 to 0x01 powers
the board off, taking the Pi down with it, and 0x41 reassigns the I2C
address, after which nothing can find the HAT again without a reset. The
write path is simply not implemented, so no bug in a caller can reach
them. The shutdown path in ``monitoring/power.py`` halts the operating
system through systemd instead, and leaves the HAT alone.

The map was verified against this hardware rather than taken on trust.
Two checks pin the decode: the four cell voltages sum to within a
millivolt of the pack register, and under load the VBUS current and power
registers move together at V*I. See ``tests/test_battery.py``.

Register map (firmware V2.0, all 16-bit values little endian)::

    0x00      0x0a fixed          identity
    0x01      0x0b fixed          identity (write 0x55 powers off)
    0x02      status bits         b7 charging, b6 fast charge,
                                  b5 VBUS powered, b2-0 charge state
    0x03      comms health        b1 BQ4050 ok, b0 IP2368 ok
    0x10-15   VBUS  mV / mA / mW
    0x20-21   pack voltage mV
    0x22-23   pack current mA, signed; negative is discharging
    0x24-25   charge percent
    0x26-27   remaining capacity mAh
    0x28-29   minutes to empty, 0xffff when unknown
    0x2a-2b   minutes to full, 0xffff when unknown
    0x30-37   cell 1..4 voltage mV
    0x50      firmware revision, 0x14 = V2.0
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Sequence

logger = logging.getLogger(__name__)

DEFAULT_I2C_BUS = 1
DEFAULT_I2C_ADDRESS = 0x2D

# The two fixed-value registers, checked before anything else is
# believed. An address is a weak identifier -- any board someone adds to
# the bus later could answer at 0x2d -- and decoding a stranger's
# registers as a battery would report confident nonsense rather than
# nothing at all.
_REG_IDENTITY = 0x00
_IDENTITY = (0x0A, 0x0B)

_REG_STATUS = 0x02
_REG_COMMS = 0x03
_REG_VBUS_MV = 0x10
_REG_VBUS_MA = 0x12
_REG_VBUS_MW = 0x14
_REG_PACK_MV = 0x20
_REG_PACK_MA = 0x22
_REG_PERCENT = 0x24
_REG_REMAINING_MAH = 0x26
_REG_MINUTES_EMPTY = 0x28
_REG_MINUTES_FULL = 0x2A
_REG_CELL_MV = 0x30
CELL_COUNT = 4

# Status register 0x02.
_STATUS_CHARGING = 0x80
_STATUS_FAST_CHARGE = 0x40
_STATUS_VBUS_POWERED = 0x20
_STATUS_CHARGE_STATE = 0x07

# Comms register 0x03.
_COMMS_GAUGE_OK = 0x02
_COMMS_CHARGER_OK = 0x01

# 0xffff is the gauge's "I cannot estimate this" sentinel, not a run of
# 45 days. It reads that way whenever the current is near zero, which on
# mains power is most of the time.
_UNKNOWN_MINUTES = 0xFFFF

CHARGE_STATES = (
    "standby",
    "trickle",
    "constant-current",
    "constant-voltage",
    "pending",
    "full",
    "timeout",
)

# Two 32-byte SMBus block reads cover every register this module wants.
# One read cannot: the SMBus block limit is 32 bytes and the cell
# voltages sit at 0x30.
_BLOCK_SIZE = 32
_REGISTER_COUNT = 64


@dataclass(frozen=True)
class BatteryReading:
    """One sample of the UPS state.

    Currents are signed the way the gauge reports them: ``pack_ma`` is
    positive while the pack is charging and negative while it is running
    the system.
    """

    percent: int
    pack_mv: int
    pack_ma: int
    remaining_mah: int
    cells_mv: tuple[int, ...]
    vbus_mv: int
    vbus_ma: int
    vbus_mw: int
    minutes_to_empty: int | None
    minutes_to_full: int | None
    charging: bool
    fast_charge: bool
    vbus_present: bool
    charge_state: int
    gauge_ok: bool
    charger_ok: bool

    @property
    def on_battery(self) -> bool:
        """True when the HAT is running the system from the pack.

        Taken from the VBUS-powered bit rather than from the sign of the
        pack current. A full pack on mains sits at exactly 0 mA, so the
        current alone cannot separate "mains, nothing to charge" from
        "battery, drawing nothing", and the second of those does not
        happen.
        """
        return not self.vbus_present

    @property
    def pack_volts(self) -> float:
        return self.pack_mv / 1000.0

    @property
    def vbus_volts(self) -> float:
        return self.vbus_mv / 1000.0

    @property
    def charge_state_name(self) -> str:
        if 0 <= self.charge_state < len(CHARGE_STATES):
            return CHARGE_STATES[self.charge_state]
        return f"unknown({self.charge_state})"

    @property
    def healthy(self) -> bool:
        """True when the HAT can still reach both of the chips it bridges."""
        return self.gauge_ok and self.charger_ok

    @property
    def power_source(self) -> str:
        return "battery" if self.on_battery else "mains"


def _u16(registers: Sequence[int], offset: int) -> int:
    return registers[offset] | (registers[offset + 1] << 8)


def _s16(registers: Sequence[int], offset: int) -> int:
    value = _u16(registers, offset)
    return value - 0x10000 if value & 0x8000 else value


def _minutes(registers: Sequence[int], offset: int) -> int | None:
    value = _u16(registers, offset)
    return None if value == _UNKNOWN_MINUTES else value


def decode_registers(registers: Sequence[int]) -> BatteryReading:
    """Decode a full 0x00-0x3f register dump into a reading.

    Separated from the I2C transfer so the decode can be tested against
    a dump captured from real hardware, which is the half of this module
    that can be wrong in a way that matters.
    """
    if len(registers) < _REGISTER_COUNT:
        raise ValueError(
            f"Need {_REGISTER_COUNT} registers to decode, got {len(registers)}"
        )

    identity = (registers[_REG_IDENTITY], registers[_REG_IDENTITY + 1])
    if identity != _IDENTITY:
        raise ValueError(
            f"Not a UPS HAT (E): identity registers read "
            f"0x{identity[0]:02x} 0x{identity[1]:02x}, expected "
            f"0x{_IDENTITY[0]:02x} 0x{_IDENTITY[1]:02x}"
        )

    status = registers[_REG_STATUS]
    comms = registers[_REG_COMMS]

    return BatteryReading(
        percent=_u16(registers, _REG_PERCENT),
        pack_mv=_u16(registers, _REG_PACK_MV),
        pack_ma=_s16(registers, _REG_PACK_MA),
        remaining_mah=_u16(registers, _REG_REMAINING_MAH),
        cells_mv=tuple(
            _u16(registers, _REG_CELL_MV + 2 * cell) for cell in range(CELL_COUNT)
        ),
        vbus_mv=_u16(registers, _REG_VBUS_MV),
        vbus_ma=_u16(registers, _REG_VBUS_MA),
        vbus_mw=_u16(registers, _REG_VBUS_MW),
        minutes_to_empty=_minutes(registers, _REG_MINUTES_EMPTY),
        minutes_to_full=_minutes(registers, _REG_MINUTES_FULL),
        charging=bool(status & _STATUS_CHARGING),
        fast_charge=bool(status & _STATUS_FAST_CHARGE),
        vbus_present=bool(status & _STATUS_VBUS_POWERED),
        charge_state=status & _STATUS_CHARGE_STATE,
        gauge_ok=bool(comms & _COMMS_GAUGE_OK),
        charger_ok=bool(comms & _COMMS_CHARGER_OK),
    )


class UpsBattery:
    """Read-only I2C client for the UPS HAT (E).

    Holds the bus open across reads. Constructing one never touches the
    hardware, so a machine with no HAT -- a development box, or a Pi
    whose HAT is not fitted -- can build the object and simply get None
    from every ``read``.
    """

    def __init__(
        self,
        bus: int = DEFAULT_I2C_BUS,
        address: int = DEFAULT_I2C_ADDRESS,
    ) -> None:
        self._bus_number = bus
        self._address = address
        self._bus: object | None = None
        # One transfer at a time on a given handle. The panel thread and
        # the status reporter both take readings, and smbus2 offers no
        # guarantee that two block reads on one descriptor will not
        # interleave.
        self._lock = threading.Lock()
        # A UPS that is absent stays absent, and this is polled on a
        # timer for the life of the process. Report the first failure
        # and then drop to debug rather than writing the same warning
        # every interval until the disk fills.
        self._warned = False

    @property
    def address(self) -> int:
        return self._address

    def _open(self) -> object | None:
        if self._bus is not None:
            return self._bus
        try:
            import smbus2
        except ImportError:
            self._report("python3-smbus2 is not installed")
            return None
        try:
            self._bus = smbus2.SMBus(self._bus_number)
        except (OSError, PermissionError) as exc:
            self._report(f"cannot open /dev/i2c-{self._bus_number}: {exc}")
            return None
        return self._bus

    def read(self) -> BatteryReading | None:
        """Take one sample, or None if the HAT cannot be read.

        The I2C bus is a hardware boundary: an absent HAT NAKs, a
        contended bus can return EREMOTEIO, and a HAT that has been
        unplugged does both. None means "no reading", which every caller
        already has to handle for the same reason ``_read_cpu_temp``
        can return None.
        """
        with self._lock:
            return self._read_locked()

    def _read_locked(self) -> BatteryReading | None:
        bus = self._open()
        if bus is None:
            return None

        try:
            registers: list[int] = []
            for base in range(0, _REGISTER_COUNT, _BLOCK_SIZE):
                registers += bus.read_i2c_block_data(  # type: ignore[attr-defined]
                    self._address, base, _BLOCK_SIZE
                )
        except OSError as exc:
            self._report(f"I2C read from 0x{self._address:02x} failed: {exc}")
            # Drop the handle so the next poll reopens it. A HAT that was
            # unplugged and put back needs a fresh descriptor.
            self.close()
            return None

        try:
            reading = decode_registers(registers)
        except ValueError as exc:
            self._report(str(exc))
            return None

        self._warned = False
        return reading

    def close(self) -> None:
        """Release the bus. Safe to call more than once."""
        if self._bus is None:
            return
        try:
            self._bus.close()  # type: ignore[attr-defined]
        except OSError:
            pass
        self._bus = None

    def __enter__(self) -> UpsBattery:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _report(self, message: str) -> None:
        if self._warned:
            logger.debug("Battery telemetry unavailable -- %s", message)
            return
        logger.warning("Battery telemetry unavailable -- %s", message)
        self._warned = True


def read_battery(
    bus: int = DEFAULT_I2C_BUS,
    address: int = DEFAULT_I2C_ADDRESS,
) -> BatteryReading | None:
    """Take a single reading, opening and closing the bus around it.

    For callers that want one sample and no object to own --
    ``check_health`` and the ``health`` CLI. A caller polling on its own
    timer should hold a ``UpsBattery``, as ``BatteryMonitor`` does, so
    that a shutdown decision never waits on the panel.

    Opening the device is not what a reading costs. Measured on this Pi:
    open and close together are 0.01 ms, while the two block reads are
    6.76 ms, because 64 bytes at the default 100 kHz bus speed is 6.7 ms
    of wire time. Holding the descriptor open between calls was tried and
    saved nothing worth the shared state it needed.
    """
    with UpsBattery(bus=bus, address=address) as ups:
        return ups.read()
