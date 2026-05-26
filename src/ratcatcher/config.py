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
    alerts: AlertConfig = field(default_factory=AlertConfig)
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
        alerts=_build_section(AlertConfig, raw.get("alerts")),
        monitoring=_build_section(MonitoringConfig, raw.get("monitoring")),
    )
