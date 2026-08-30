"""Configuration loading and validation for RatCatcher AI."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_DIR = Path(__file__).parent.parent.parent / "config"


@dataclass(frozen=True)
class CameraConfig:
    id: int
    enabled: bool = True
    source_type: str = "auto"
    resolution: tuple[int, int] = (1920, 1080)
    # Sensor readout size, when it should differ from the frame size used
    # downstream.  The detector needs native-resolution pixels to see a
    # small animal, but a 4056x3040 BGR frame is 37 MB and must never
    # reach a queue or the pre-event ring buffer.  None means capture at
    # ``resolution``, which is what every install did before this existed.
    capture_resolution: tuple[int, int] | None = None
    fps: int = 30
    file_path: str | None = None
    device_index: int = 0
    roi: list[list[float]] = field(default_factory=list)


@dataclass(frozen=True)
class MotionConfig:
    enabled: bool = True
    history: int = 500
    var_threshold: int = 16
    detect_shadows: bool = False
    process_width: int = 320
    process_height: int = 240
    erode_kernel: int = 3
    dilate_kernel: int = 7
    # Upper bound on a region, as a fraction of frame area.  A cloud
    # shadow or an auto-exposure step changes the whole frame at once and
    # MOG2 reports it as one region covering ~99% of the image; that is an
    # illumination change, not an animal.  1.0 keeps the old behaviour of
    # accepting anything.
    max_area_pct: float = 1.0
    min_area_pct: float = 0.005
    cooldown_seconds: float = 2.0
    learning_rate: float = -1.0


@dataclass(frozen=True)
class DetectionConfig:
    enabled: bool = True
    backend: str = "auto"
    model_path: str = "yolov8n.onnx"
    confidence_threshold: float = 0.45
    nms_threshold: float = 0.45
    input_size: tuple[int, int] = (640, 640)
    # Run the detector on native-resolution windows cut around motion
    # rather than on the whole frame squeezed to input_size.  A frame
    # stretched from 1920x1080 to 640x640 shrinks a finch to ~17 px, which
    # is below the detector's floor; a native window leaves it at ~62 px.
    roi_crop: bool = False
    roi_crop_window: int = 640
    roi_crop_max_windows: int = 2


@dataclass(frozen=True)
class ClassificationConfig:
    enabled: bool = True
    model_path: str = "efficientnet_b3_birds.tflite"
    species_config: str = "species.yaml"
    input_size: tuple[int, int] = (300, 300)
    top_k: int = 5
    min_confidence: float = 0.70
    use_xnnpack: bool = True


@dataclass(frozen=True)
class StorageConfig:
    db_path: str = "detections.db"
    clip_dir: str = "clips"
    thumbnail_dir: str = "thumbnails"
    clip_pre_seconds: int = 5
    clip_post_seconds: int = 10
    retention_days: int = 30
    max_disk_gb: float = 10.0
    ffmpeg_preset: str = "fast"


@dataclass(frozen=True)
class AlertConfig:
    enabled: bool = False
    alert_classes: list[str] = field(
        default_factory=lambda: ["squirrel", "rat", "cat", "unknown_animal"]
    )
    cooldown_seconds: int = 300
    methods: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class AudioConfig:
    """Dual SPH0645 I2S microphone capture and bird song identification.

    The two microphones share one I2S bus and are separated by their SEL
    pin, so the operating system presents them as a single stereo capture
    device. ``channels`` is therefore 2 for one stereo device, not two
    independent sources.
    """

    enabled: bool = False
    # "auto", "alsa", or "file"
    source_type: str = "auto"
    # ALSA device name. "default" follows the system default card.
    device: str = "default"
    sample_rate: int = 48000
    channels: int = 2
    # Seconds of audio pulled from the device per read
    block_seconds: float = 0.5
    # For file source: path to a WAV file, and whether to loop it
    file_path: str | None = None
    file_loop: bool = False

    # -- signal conditioning --
    # High-pass corner removing the SPH0645 DC offset and wind rumble
    highpass_hz: float = 150.0

    # -- activity gate --
    gate_enabled: bool = True
    gate_band_low_hz: float = 1000.0
    gate_band_high_hz: float = 10000.0
    # How far above the tracked noise floor a sound must rise to trigger
    gate_snr_margin_db: float = 2.0
    # Hard minimum level, guarding against triggering on digital silence
    gate_absolute_floor_dbfs: float = -85.0
    gate_max_flatness: float = 0.6
    # Seconds to suppress repeated detections on the same channel
    cooldown_seconds: float = 3.0

    # -- BirdNET identification --
    model_path: str = "BirdNET_v2.4_audio-model.tflite"
    labels_path: str = "BirdNET_v2.4_labels_en_us.txt"
    min_confidence: float = 0.25
    top_k: int = 3
    num_threads: int = 2
    # Analyse each microphone separately, or average them to mono first.
    # Per-channel costs one inference per microphone but tells you which
    # side of the feeder the bird was on.
    per_channel: bool = True

    # -- clip retention --
    # Save a WAV alongside each identification
    save_clips: bool = True
    clip_dir: str = "audio"


@dataclass(frozen=True)
class DisplayConfig:
    """Elecrow CrowPanel ESP32 2.13-inch e-paper status panel over USB.

    The panel is an output only. It reads nothing from the pipeline and
    holds nothing the pipeline needs, so it is safe to unplug at any
    time and safe to leave unconfigured.
    """

    enabled: bool = False
    # "auto" (serial, falling back to none), "serial", "file", or "null"
    source_type: str = "auto"
    # Serial device path. "auto" probes every known USB serial bridge
    # and keeps the port whose panel answers the identification request.
    port: str = "auto"
    baud_rate: int = 115200
    # Seconds to wait for a panel to identify itself during a probe
    # Must exceed the ESP32 boot time; opening the port resets the
    # panel and its hello arrives about 2.4 s later.
    probe_seconds: float = 6.0
    # Seconds between reconnection attempts after the panel goes away
    reconnect_seconds: float = 10.0

    # For file source: where the encoded frames are written
    file_path: str | None = None

    # -- what the panel shows --
    # Counting window: "today" (since local midnight), "24h", or "all"
    window: str = "today"

    # -- how often it is redrawn --
    # Seconds between readings. A reading whose contents match the last
    # one drawn is not sent, so this is an upper bound on refresh rate,
    # not a refresh rate.
    refresh_seconds: float = 30.0
    # Redraw at least this often even when nothing has changed, so a
    # stopped host shows as a stale clock rather than as a live screen.
    heartbeat_seconds: float = 300.0
    # Partial refreshes leave a ghost of the previous image. Clear it
    # with a full refresh every this many frames.
    full_refresh_every: int = 15


@dataclass(frozen=True)
class SyslogConfig:
    """Detection and status reporting to syslog.

    An output only, like the status panel. Nothing downstream reads it
    and nothing in the pipeline waits on it, so a loghost that is down,
    slow or absent cannot affect detection.

    Only two kinds of record are sent: one line per detection and a
    periodic status line. Ordinary INFO chatter -- libcamera, picamera2,
    model loading -- stays in journald. A forwarded stream is worth
    keeping narrow: it is the one view of the system an operator has
    when the Pi itself is out of reach.
    """

    # On by default, unlike the panel and the microphones. Those need
    # hardware that may not be attached; this needs a local socket that
    # every Raspberry Pi OS install already has. An upgrade that keeps
    # its old conffile, and so has no syslog block at all, still starts
    # reporting rather than staying silent until someone notices.
    enabled: bool = True
    # A path is a local syslog socket; "host:port" is a remote loghost.
    address: str = "/dev/log"
    # Transport for a remote loghost. Ignored for a socket path.
    protocol: str = "udp"
    # local0 rather than daemon: this is application output, and keeping
    # it off daemon lets rsyslog route feeder detections to their own
    # file without catching every other service on the Pi.
    facility: str = "local0"
    # Tag the lines carry, which is what "journalctl -t" matches on.
    ident: str = "ratcatcher"

    # One line per stored detection.
    detections: bool = True

    # -- periodic status --
    # Far longer than the panel's refresh. The panel redraws to stay
    # readable at a glance; this exists to prove the system is alive and
    # to carry counts, and one line every five minutes does that without
    # filling a loghost with a mostly unchanging record.
    status_interval_seconds: float = 300.0
    # Counting window: "today" (since local midnight), "24h", or "all".
    # Matches display.window so the two agree by default.
    status_window: str = "today"


@dataclass(frozen=True)
class MonitoringConfig:
    health_interval_seconds: int = 60
    temp_warning_c: float = 75.0
    temp_critical_c: float = 82.0


@dataclass(frozen=True)
class SystemConfig:
    data_dir: str = "data"
    log_level: str = "INFO"
    platform: str = "auto"


@dataclass(frozen=True)
class Config:
    system: SystemConfig = field(default_factory=SystemConfig)
    cameras: list[CameraConfig] = field(default_factory=list)
    motion: MotionConfig = field(default_factory=MotionConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    classification: ClassificationConfig = field(
        default_factory=ClassificationConfig
    )
    storage: StorageConfig = field(default_factory=StorageConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    syslog: SyslogConfig = field(default_factory=SyslogConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    @property
    def data_path(self) -> Path:
        return Path(self.system.data_dir)

    @property
    def db_full_path(self) -> Path:
        return self.data_path / self.storage.db_path

    @property
    def clip_full_path(self) -> Path:
        return self.data_path / self.storage.clip_dir

    @property
    def thumbnail_full_path(self) -> Path:
        return self.data_path / self.storage.thumbnail_dir

    @property
    def audio_clip_full_path(self) -> Path:
        return self.data_path / self.audio.clip_dir


def _coerce_tuple(value: Any, length: int = 2) -> tuple[int, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(int(v) for v in value[:length])
    raise ValueError(f"Expected list/tuple of {length} values, got {type(value)}")


def _build_camera_config(raw: dict[str, Any]) -> CameraConfig:
    kwargs: dict[str, Any] = {}
    for key in (
        "id", "enabled", "source_type", "fps", "file_path", "device_index", "roi",
    ):
        if key in raw:
            kwargs[key] = raw[key]
    if "resolution" in raw:
        kwargs["resolution"] = _coerce_tuple(raw["resolution"])
    if raw.get("capture_resolution") is not None:
        kwargs["capture_resolution"] = _coerce_tuple(raw["capture_resolution"])
    return CameraConfig(**kwargs)


def _build_section(cls: type, raw: dict[str, Any] | None) -> Any:
    if raw is None:
        return cls()
    kwargs: dict[str, Any] = {}
    valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
    for key, value in raw.items():
        if key in valid_fields:
            field_obj = cls.__dataclass_fields__[key]
            if field_obj.type in ("tuple[int, int]",):
                kwargs[key] = _coerce_tuple(value)
            else:
                kwargs[key] = value
    return cls(**kwargs)


def load_config(config_path: str | Path | None = None) -> Config:
    """Load configuration from a YAML file.

    If config_path is None, looks for default.yaml in the config directory.
    The config directory can be overridden via RATCATCHER_CONFIG_DIR env var.
    """
    if config_path is not None:
        config_file = Path(config_path)
    else:
        config_dir = Path(
            os.environ.get("RATCATCHER_CONFIG_DIR", str(DEFAULT_CONFIG_DIR))
        )
        config_file = config_dir / "default.yaml"

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {config_file}: {e}") from e

    if raw is None:
        raw = {}

    cameras = []
    for cam_raw in raw.get("cameras", []):
        cameras.append(_build_camera_config(cam_raw))

    return Config(
        system=_build_section(SystemConfig, raw.get("system")),
        cameras=cameras,
        motion=_build_section(MotionConfig, raw.get("motion")),
        detection=_build_section(DetectionConfig, raw.get("detection")),
        classification=_build_section(ClassificationConfig, raw.get("classification")),
        storage=_build_section(StorageConfig, raw.get("storage")),
        audio=_build_section(AudioConfig, raw.get("audio")),
        display=_build_section(DisplayConfig, raw.get("display")),
        alerts=_build_section(AlertConfig, raw.get("alerts")),
        syslog=_build_section(SyslogConfig, raw.get("syslog")),
        monitoring=_build_section(MonitoringConfig, raw.get("monitoring")),
    )
