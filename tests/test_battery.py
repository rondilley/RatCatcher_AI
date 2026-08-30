"""Tests for UPS HAT (E) telemetry and the low-battery shutdown.

No test doubles, following the rest of the suite. Every reading in this
file comes out of the real decoder, applied to a register dump captured
from the hardware on 2026-08-30 and then edited byte by byte to describe
the states a running system passes through -- an outage, a falling pack,
a gauge that stops answering. Nothing constructs a ``BatteryReading``
directly, so the decode path is exercised by every test that depends on
a reading rather than only by the ones that name it.

Two collaborators are substituted, both for the same reason ``FileSource``
substitutes for a camera elsewhere in this suite: the thing behind them
is hardware. ``ScriptedBattery`` replaces the I2C client and the monitor
takes an injected command runner, so a shutdown test records the command
instead of halting the machine running the tests. Both go in through
constructor arguments that exist for the purpose; no test reaches into a
private attribute. The database is real SQLite, and the monitor under
test is the real one.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Sequence

import pytest

from ratcatcher.config import BatteryConfig, Config, StorageConfig, SystemConfig
from ratcatcher.monitoring.battery import (
    CHARGE_STATES,
    UpsBattery,
    decode_registers,
    read_battery,
)
from ratcatcher.monitoring.power import SHUTDOWN_COMMAND, BatteryMonitor
from ratcatcher.storage.database import DetectionDatabase

# Captured from the UPS HAT (E) on this Pi with two block reads of
# /dev/i2c-1 at 0x2d. Fully charged, on mains, in constant-voltage
# charge. Every expected value below was read off the hardware, not
# invented: pack 16385 mV, cells 4097/4100/4124/4063, VBUS 20251 mV at
# 368 mA and 7484 mW, 4792 mAh remaining.
LIVE_DUMP = (
    0x0A, 0x0B, 0xE3, 0x03, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x1B, 0x4F, 0x70, 0x01, 0x3C, 0x1D, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x01, 0x40, 0x00, 0x00, 0x64, 0x00, 0xB8, 0x12,
    0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00,
    0x01, 0x10, 0x04, 0x10, 0x1C, 0x10, 0xDF, 0x0F,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
)


# -- helpers ---------------------------------------------------------------


def _u16(registers: list[int], offset: int, value: int) -> None:
    registers[offset] = value & 0xFF
    registers[offset + 1] = (value >> 8) & 0xFF


def dump(
    *,
    percent: int | None = None,
    on_battery: bool = False,
    pack_ma: int | None = None,
    minutes_to_empty: int | None = None,
    gauge_ok: bool = True,
    charger_ok: bool = True,
) -> list[int]:
    """Edit the captured dump into the state being described.

    Byte edits rather than a hand-built reading, so the decoder is what
    turns these into numbers and a test that passes here says something
    about the code that will run on the Pi.
    """
    registers = list(LIVE_DUMP)

    if percent is not None:
        _u16(registers, 0x24, percent)
    if pack_ma is not None:
        _u16(registers, 0x22, pack_ma & 0xFFFF)
    if minutes_to_empty is not None:
        _u16(registers, 0x28, minutes_to_empty)

    if on_battery:
        # Clear VBUS-powered, charging and fast-charge, and drop the
        # charge state to standby: the HAT reports all four together
        # when the input goes away.
        registers[0x02] = 0x00
        _u16(registers, 0x10, 0)
        _u16(registers, 0x12, 0)
        _u16(registers, 0x14, 0)

    comms = 0
    if gauge_ok:
        comms |= 0x02
    if charger_ok:
        comms |= 0x01
    registers[0x03] = comms

    return registers


class ScriptedBattery:
    """A UpsBattery that returns a prepared sequence of register dumps.

    Stands in for the I2C client only. The dumps go through the real
    decoder, so what is scripted is the hardware, not the reading.
    A ``None`` entry in the script is a failed read.
    """

    def __init__(self, script: Sequence[list[int] | None]) -> None:
        self._script = list(script)
        self.reads = 0
        self.closed = False

    def read(self):
        if not self._script:
            return None
        registers = self._script.pop(0)
        self.reads += 1
        return None if registers is None else decode_registers(registers)

    def close(self) -> None:
        self.closed = True


def make_config(tmp_path: Path, **battery: object) -> Config:
    settings: dict[str, object] = {
        "enabled": True,
        "poll_interval_seconds": 0.01,
        "sample_interval_seconds": 300.0,
        "warn_percent": 40,
        "shutdown_enabled": True,
        "shutdown_percent": 15,
        "shutdown_minutes": 10,
    }
    settings.update(battery)
    return Config(
        system=SystemConfig(data_dir=str(tmp_path)),
        storage=StorageConfig(db_path="detections.db"),
        battery=BatteryConfig(**settings),  # type: ignore[arg-type]
    )


@pytest.fixture
def monitor(tmp_path: Path):
    """A real BatteryMonitor with a scripted battery and a recording runner."""

    def build(script: Sequence[list[int] | None], **battery: object):
        config = make_config(tmp_path, **battery)
        commands: list[Sequence[str]] = []
        engine = BatteryMonitor(
            config,
            runner=lambda cmd: commands.append(cmd) or 0,
            battery=ScriptedBattery(script),
        )
        engine.open()
        return engine, commands

    built: list[BatteryMonitor] = []

    def factory(script, **battery):
        engine, commands = build(script, **battery)
        built.append(engine)
        return engine, commands

    yield factory
    for engine in built:
        engine.close()


# -- decoding --------------------------------------------------------------


def test_decodes_the_captured_hardware_dump() -> None:
    reading = decode_registers(LIVE_DUMP)

    assert reading.percent == 100
    assert reading.pack_mv == 16385
    assert reading.pack_ma == 0
    assert reading.remaining_mah == 4792
    assert reading.cells_mv == (4097, 4100, 4124, 4063)
    assert reading.vbus_mv == 20251
    assert reading.vbus_ma == 368
    assert reading.vbus_mw == 7484
    assert reading.charge_state_name == "constant-voltage"
    assert reading.charging and reading.fast_charge and reading.vbus_present
    assert reading.healthy


def test_cell_voltages_sum_to_the_pack_voltage() -> None:
    """The invariant that pins the decode.

    Two unrelated registers agreeing to within a millivolt is what says
    the byte order and the offsets are right. A transposed pair or a
    big-endian read fails this by hundreds of millivolts.
    """
    reading = decode_registers(LIVE_DUMP)

    assert abs(sum(reading.cells_mv) - reading.pack_mv) <= 2


def test_vbus_power_is_the_product_of_vbus_voltage_and_current() -> None:
    """The second invariant, across the other register block."""
    reading = decode_registers(LIVE_DUMP)

    expected_mw = reading.vbus_mv * reading.vbus_ma / 1000
    assert abs(expected_mw - reading.vbus_mw) / reading.vbus_mw < 0.05


def test_discharge_current_is_signed() -> None:
    """A discharging pack reports a negative current, not 64 amps.

    0x22 is the one signed register in the map. Read as unsigned, a
    modest 1.5 A discharge decodes as 64036 mA.
    """
    reading = decode_registers(dump(on_battery=True, pack_ma=-1500))

    assert reading.pack_ma == -1500


def test_unknown_runtime_decodes_as_none_not_forty_five_days() -> None:
    """0xffff is the gauge declining to estimate, not 65535 minutes."""
    reading = decode_registers(LIVE_DUMP)

    assert reading.minutes_to_empty is None
    assert reading.minutes_to_full is None

    running = decode_registers(dump(on_battery=True, minutes_to_empty=284))
    assert running.minutes_to_empty == 284


def test_power_source_comes_from_the_vbus_bit_not_the_current() -> None:
    """A full pack on mains sits at 0 mA, exactly like an idle one.

    The sign of the pack current cannot separate the two, which is why
    on_battery reads the charger's VBUS-powered bit instead.
    """
    on_mains = decode_registers(dump(percent=100, pack_ma=0))
    assert on_mains.pack_ma == 0
    assert not on_mains.on_battery
    assert on_mains.power_source == "mains"

    on_battery = decode_registers(dump(percent=100, pack_ma=0, on_battery=True))
    assert on_battery.pack_ma == 0
    assert on_battery.on_battery
    assert on_battery.power_source == "battery"


def test_lost_gauge_is_reported_rather_than_assumed() -> None:
    assert decode_registers(dump(gauge_ok=False)).healthy is False
    assert decode_registers(dump(charger_ok=False)).healthy is False
    assert decode_registers(dump()).healthy is True


def test_every_charge_state_has_a_name() -> None:
    for state in range(len(CHARGE_STATES)):
        registers = list(LIVE_DUMP)
        registers[0x02] = (registers[0x02] & ~0x07) | state
        assert decode_registers(registers).charge_state_name == CHARGE_STATES[state]


def test_a_stranger_at_the_address_is_rejected() -> None:
    """An I2C address is a weak identifier.

    Decoding whatever answers at 0x2d as a battery would report
    confident nonsense; the fixed-value registers are what make that
    impossible.
    """
    registers = list(LIVE_DUMP)
    registers[0x00] = 0x55

    with pytest.raises(ValueError, match="Not a UPS HAT"):
        decode_registers(registers)


def test_a_short_read_is_rejected() -> None:
    with pytest.raises(ValueError, match="Need 64 registers"):
        decode_registers(LIVE_DUMP[:32])


# -- the I2C client --------------------------------------------------------


def test_absent_hardware_reads_as_none_rather_than_raising() -> None:
    """A bus that does not exist is the development machine's normal state."""
    with UpsBattery(bus=99, address=0x2D) as ups:
        assert ups.read() is None
        assert ups.read() is None


def test_read_battery_on_a_missing_bus_returns_none() -> None:
    assert read_battery(bus=99) is None


# -- the monitor -----------------------------------------------------------


def test_losing_mains_is_recorded_at_once(monitor, caplog) -> None:
    """The transition sample is stored whatever the sample interval says.

    Its timestamp is the answer to "when did the power go", so rounding
    it up to the next five-minute boundary would throw away the only
    thing it is for.
    """
    engine, _ = monitor([dump(percent=100), dump(percent=99, on_battery=True)])

    engine.poll_once(now=datetime(2026, 8, 30, 12, 0, 0))
    assert engine.power_source == "mains"
    first = engine.stats["samples_stored"]

    with caplog.at_level("WARNING", logger="ratcatcher.events"):
        engine.poll_once(now=datetime(2026, 8, 30, 12, 0, 30))

    assert engine.power_source == "battery"
    assert engine.stats["transitions"] == 1
    # 30 seconds after the last one, against a 300 second interval.
    assert engine.stats["samples_stored"] == first + 1
    assert "event=on_battery" in caplog.text


def test_samples_are_rate_limited_between_transitions(monitor) -> None:
    engine, _ = monitor([dump(percent=100)] * 4)

    start = datetime(2026, 8, 30, 12, 0, 0)
    for offset in (0, 60, 120, 180):
        engine.poll_once(now=start + timedelta(seconds=offset))

    # The first is stored, the other three fall inside 300 seconds.
    assert engine.stats["samples_stored"] == 1


def test_regaining_mains_is_reported_once(monitor, caplog) -> None:
    engine, _ = monitor(
        [
            dump(percent=100),
            dump(percent=90, on_battery=True),
            dump(percent=90, on_battery=True),
            dump(percent=91),
        ]
    )

    start = datetime(2026, 8, 30, 12, 0, 0)
    with caplog.at_level("INFO", logger="ratcatcher.events"):
        for offset in (0, 10, 20, 30):
            engine.poll_once(now=start + timedelta(seconds=offset))

    assert engine.stats["transitions"] == 2
    assert caplog.text.count("event=on_battery") == 1
    assert caplog.text.count("event=on_mains") == 1


def test_booting_on_battery_is_reported_but_booting_on_mains_is_not(
    monitor, caplog
) -> None:
    """A Pi that came up during an outage should say so on its first read."""
    on_mains, _ = monitor([dump(percent=100)])
    with caplog.at_level("INFO", logger="ratcatcher.events"):
        on_mains.poll_once()
    assert "power event=" not in caplog.text
    assert on_mains.stats["transitions"] == 0

    caplog.clear()
    on_battery, _ = monitor([dump(percent=80, on_battery=True)])
    with caplog.at_level("INFO", logger="ratcatcher.events"):
        on_battery.poll_once()
    assert "event=on_battery" in caplog.text


def test_a_flat_pack_halts_the_system_once(monitor) -> None:
    engine, commands = monitor(
        [
            dump(percent=14, on_battery=True),
            dump(percent=13, on_battery=True),
            dump(percent=12, on_battery=True),
        ]
    )

    engine.poll_once()
    engine.poll_once()
    engine.poll_once()

    assert commands == [SHUTDOWN_COMMAND]
    assert engine.stats["shutdown_fired"] == 1


def test_the_minutes_floor_fires_on_a_pack_that_is_still_charged(monitor) -> None:
    """A heavy load empties a half-full pack faster than the percent floor sees.

    35% is well clear of the 15% floor; nine minutes of runtime is not.
    """
    engine, commands = monitor(
        [dump(percent=35, on_battery=True, minutes_to_empty=9)]
    )

    engine.poll_once()

    assert commands == [SHUTDOWN_COMMAND]


def test_a_flat_pack_on_mains_does_not_halt_the_system(monitor) -> None:
    """Charging from empty must not be read as an outage.

    A pack at 5% that is plugged in and charging is the ordinary state
    after an outage ends, and halting there would make recovery a loop.
    """
    engine, commands = monitor([dump(percent=5), dump(percent=6)])

    engine.poll_once()
    engine.poll_once()

    assert commands == []
    assert engine.stats["shutdown_fired"] == 0


def test_shutdown_is_not_armed_unless_it_is_enabled(monitor) -> None:
    """The default. A conffile upgrade must not acquire this behaviour."""
    engine, commands = monitor(
        [dump(percent=2, on_battery=True)], shutdown_enabled=False
    )

    engine.poll_once()

    assert commands == []


def test_the_default_configuration_leaves_shutdown_off() -> None:
    assert BatteryConfig().shutdown_enabled is False
    assert BatteryConfig().enabled is True


def test_a_degraded_gauge_still_halts_and_says_so(monitor, caplog) -> None:
    """An unnecessary clean shutdown costs a restart; not acting costs the disk."""
    engine, commands = monitor(
        [dump(percent=8, on_battery=True, gauge_ok=False)]
    )

    with caplog.at_level("WARNING", logger="ratcatcher.events"):
        engine.poll_once()

    assert commands == [SHUTDOWN_COMMAND]
    assert "gauge-degraded" in caplog.text


def test_a_failed_read_is_counted_and_not_fatal(monitor) -> None:
    engine, commands = monitor([None, None, dump(percent=100)])

    assert engine.poll_once() is None
    assert engine.poll_once() is None
    assert engine.poll_once() is not None
    assert engine.stats["read_failures"] == 2
    assert commands == []


def test_samples_reach_the_database_with_their_cells(tmp_path: Path) -> None:
    """Real SQLite, and the row the monitor actually wrote."""
    config = make_config(tmp_path)
    engine = BatteryMonitor(
        config,
        runner=lambda cmd: 0,
        battery=ScriptedBattery([dump(percent=77, on_battery=True)]),
    )
    engine.open()
    try:
        engine.poll_once(now=datetime(2026, 8, 30, 12, 0, 0))
    finally:
        engine.close()

    with DetectionDatabase(config.db_full_path) as db:
        rows = db.get_battery_samples()
        assert len(rows) == 1
        row = rows[0]
        assert row["percent"] == 77
        assert row["power_source"] == "battery"
        assert row["cells_mv"] == "[4097, 4100, 4124, 4063]"
        assert row["gauge_ok"] == 1

        assert db.get_battery_samples(power_source="mains") == []
        assert len(db.get_battery_samples(power_source="battery")) == 1
        with pytest.raises(ValueError):
            db.get_battery_samples(power_source="wall socket")


def test_the_thread_starts_polls_and_stops(tmp_path: Path) -> None:
    """The real threaded monitor, driven to completion and joined."""
    config = make_config(tmp_path, poll_interval_seconds=0.01)
    scripted = ScriptedBattery([dump(percent=100)] * 50)
    engine = BatteryMonitor(config, runner=lambda cmd: 0, battery=scripted)
    engine.open()

    engine.start()
    try:
        deadline = datetime.now() + timedelta(seconds=5)
        while scripted.reads < 3 and datetime.now() < deadline:
            pass
    finally:
        engine.stop()

    assert scripted.reads >= 3
    assert not engine.is_running
