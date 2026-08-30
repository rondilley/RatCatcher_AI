"""System health monitoring for RatCatcher AI."""

from __future__ import annotations

import logging
import platform
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ratcatcher.config import BatteryConfig
from ratcatcher.monitoring.battery import read_battery

logger = logging.getLogger(__name__)


@dataclass
class HealthReport:
    """Snapshot of system health metrics."""

    timestamp: str
    platform: str
    cpu_temp_c: float | None = None
    cpu_usage_pct: float | None = None
    memory_used_mb: float | None = None
    memory_total_mb: float | None = None
    disk_used_gb: float | None = None
    disk_total_gb: float | None = None
    hailo_available: bool = False
    cameras_connected: int = 0
    uptime_seconds: float | None = None
    # None on a machine with no UPS HAT fitted, which is how every
    # other optional reading here reports its absence.
    battery_percent: int | None = None
    battery_volts: float | None = None
    battery_minutes_remaining: int | None = None
    power_source: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def memory_usage_pct(self) -> float | None:
        if self.memory_used_mb and self.memory_total_mb:
            return (self.memory_used_mb / self.memory_total_mb) * 100
        return None

    @property
    def disk_usage_pct(self) -> float | None:
        if self.disk_used_gb and self.disk_total_gb:
            return (self.disk_used_gb / self.disk_total_gb) * 100
        return None


def check_health(
    data_dir: Path | None = None,
    temp_warning_c: float = 75.0,
    temp_critical_c: float = 82.0,
    battery: BatteryConfig | None = None,
) -> HealthReport:
    """Collect system health metrics.

    Works on both RPi (full metrics) and Windows (partial metrics).
    """
    report = HealthReport(
        timestamp=datetime.now().isoformat(),
        platform=f"{platform.system()} {platform.machine()}",
    )

    report.cpu_temp_c = _read_cpu_temp()
    if report.cpu_temp_c is not None:
        if report.cpu_temp_c >= temp_critical_c:
            report.warnings.append(
                f"CPU temperature CRITICAL: {report.cpu_temp_c:.1f} C"
            )
        elif report.cpu_temp_c >= temp_warning_c:
            report.warnings.append(
                f"CPU temperature WARNING: {report.cpu_temp_c:.1f} C"
            )

    mem = _read_memory()
    if mem is not None:
        report.memory_used_mb, report.memory_total_mb = mem
        pct = report.memory_usage_pct
        if pct is not None and pct > 90:
            report.warnings.append(f"Memory usage high: {pct:.1f}%")

    if data_dir is not None:
        disk = _read_disk_usage(data_dir)
        if disk is not None:
            report.disk_used_gb, report.disk_total_gb = disk
            pct = report.disk_usage_pct
            if pct is not None and pct > 90:
                report.warnings.append(f"Disk usage high: {pct:.1f}%")

    report.hailo_available = _check_hailo()
    report.uptime_seconds = _read_uptime()

    if battery is not None and battery.enabled:
        _add_battery(report, battery)

    return report


def _add_battery(report: HealthReport, config: BatteryConfig) -> None:
    """Fill in the UPS readings and warn on what they say.

    Running from the pack is a warning in its own right, separate from
    the charge level: at 100% on battery nothing is wrong with the
    battery and everything is wrong with the mains, and that is the
    condition worth surfacing first.
    """
    reading = read_battery(bus=config.i2c_bus, address=config.i2c_address)
    if reading is None:
        return

    report.battery_percent = reading.percent
    report.battery_volts = reading.pack_volts
    report.power_source = reading.power_source
    report.battery_minutes_remaining = reading.minutes_to_empty

    if reading.on_battery:
        remaining = (
            f", {reading.minutes_to_empty} min remaining"
            if reading.minutes_to_empty is not None
            else ""
        )
        report.warnings.append(
            f"Running on battery: {reading.percent}%{remaining}"
        )
    elif reading.percent < config.warn_percent:
        report.warnings.append(f"Battery low: {reading.percent}%")

    # The HAT is a bridge, not the gauge. When it says it has lost the
    # chip behind it, the numbers above are the last ones it managed to
    # read rather than the current state, and saying so is the whole
    # value of the register.
    if not reading.healthy:
        lost = []
        if not reading.gauge_ok:
            lost.append("fuel gauge")
        if not reading.charger_ok:
            lost.append("charger")
        report.warnings.append(
            f"UPS HAT has lost contact with its {' and '.join(lost)}"
        )


def _read_cpu_temp() -> float | None:
    """Read CPU temperature. Only works on Linux with thermal zones."""
    thermal_path = Path("/sys/class/thermal/thermal_zone0/temp")
    if not thermal_path.exists():
        return None
    try:
        with open(thermal_path, "r") as f:
            return int(f.read().strip()) / 1000.0
    except (ValueError, PermissionError, OSError):
        return None


def _read_memory() -> tuple[float, float] | None:
    """Read memory usage in MB."""
    try:
        meminfo_path = Path("/proc/meminfo")
        if meminfo_path.exists():
            info: dict[str, int] = {}
            with open(meminfo_path, "r") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0].rstrip(":")
                        info[key] = int(parts[1])
            total_kb = info.get("MemTotal", 0)
            available_kb = info.get("MemAvailable", 0)
            if total_kb > 0:
                used_mb = (total_kb - available_kb) / 1024.0
                total_mb = total_kb / 1024.0
                return used_mb, total_mb
    except (ValueError, PermissionError, OSError):
        pass

    if platform.system() == "Windows":
        try:
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            total_mb = stat.ullTotalPhys / (1024 * 1024)
            used_mb = (stat.ullTotalPhys - stat.ullAvailPhys) / (1024 * 1024)
            return used_mb, total_mb
        except (AttributeError, OSError):
            pass

    return None


def _read_disk_usage(path: Path) -> tuple[float, float] | None:
    """Read disk usage for the partition containing path."""
    try:
        usage = shutil.disk_usage(str(path))
        used_gb = usage.used / (1024 ** 3)
        total_gb = usage.total / (1024 ** 3)
        return used_gb, total_gb
    except (OSError, ValueError):
        return None


def _check_hailo() -> bool:
    """Check if Hailo runtime is available."""
    try:
        import hailo_platform  # type: ignore[import-untyped]
        return True
    except ImportError:
        return False


def _read_uptime() -> float | None:
    """Read system uptime in seconds (Linux only)."""
    uptime_path = Path("/proc/uptime")
    if not uptime_path.exists():
        return None
    try:
        with open(uptime_path, "r") as f:
            return float(f.read().split()[0])
    except (ValueError, PermissionError, OSError):
        return None
