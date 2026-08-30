"""Tests for ratcatcher.config -- configuration loading and validation."""

import dataclasses
import tempfile
from pathlib import Path

import pytest

from ratcatcher.config import (
    CameraConfig,
    ClassificationConfig,
    Config,
    DetectionConfig,
    MonitoringConfig,
    MotionConfig,
    StorageConfig,
    SystemConfig,
    load_config,
)


CONFIG_DIR = Path(__file__).parent.parent / "config"
DEFAULT_YAML = CONFIG_DIR / "default.yaml"


# -------------------------------------------------------------------
# 1. Load default config and verify all sections present
# -------------------------------------------------------------------

class TestLoadDefaultConfig:
    def test_load_default_config(self) -> None:
        cfg = load_config(DEFAULT_YAML)

        assert isinstance(cfg, Config)
        assert isinstance(cfg.system, SystemConfig)
        assert isinstance(cfg.cameras, list)
        assert isinstance(cfg.motion, MotionConfig)
        assert isinstance(cfg.detection, DetectionConfig)
        assert isinstance(cfg.classification, ClassificationConfig)
        assert isinstance(cfg.storage, StorageConfig)

        # Top-level system values from default.yaml
        assert cfg.system.data_dir == "data"
        assert cfg.system.log_level == "INFO"
        assert cfg.system.platform == "auto"


# -------------------------------------------------------------------
# 2. Config is frozen -- attribute assignment should fail
# -------------------------------------------------------------------

class TestConfigIsFrozen:
    def test_config_is_frozen(self) -> None:
        cfg = load_config(DEFAULT_YAML)
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.system = SystemConfig()  # type: ignore[misc]

    def test_system_config_is_frozen(self) -> None:
        cfg = load_config(DEFAULT_YAML)
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.system.log_level = "DEBUG"  # type: ignore[misc]

    def test_motion_config_is_frozen(self) -> None:
        cfg = load_config(DEFAULT_YAML)
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.motion.enabled = False  # type: ignore[misc]


# -------------------------------------------------------------------
# 3. Camera list parsed correctly (2 cameras in default.yaml)
# -------------------------------------------------------------------

class TestCameraConfigFromYaml:
    def test_camera_config_from_yaml(self) -> None:
        cfg = load_config(DEFAULT_YAML)

        assert len(cfg.cameras) == 2

        cam0 = cfg.cameras[0]
        assert isinstance(cam0, CameraConfig)
        assert cam0.id == 0
        assert cam0.enabled is True
        assert cam0.source_type == "auto"
        assert cam0.resolution == (1920, 1080)
        # Full sensor readout, so the detector gets native pixels; 14 fps
        # is the sensor maximum at that size.
        assert cam0.capture_resolution == (4056, 3040)
        assert cam0.fps == 14
        assert cam0.file_path is None
        assert cam0.device_index == 0
        assert cam0.roi == []

        cam1 = cfg.cameras[1]
        assert cam1.id == 1
        assert cam1.device_index == 1
        assert cam1.capture_resolution == (4056, 3040)


class TestCaptureResolutionIsOptional:
    def test_absent_capture_resolution_stays_none(self, tmp_path) -> None:
        """An upgrade whose conffile predates this key must not change."""
        path = tmp_path / "old.yaml"
        path.write_text(
            "cameras:\n"
            "  - id: 0\n"
            "    resolution: [1920, 1080]\n"
            "    fps: 30\n"
        )
        cam = load_config(path).cameras[0]
        assert cam.capture_resolution is None
        assert cam.resolution == (1920, 1080)
        assert cam.fps == 30


# -------------------------------------------------------------------
# 4. MotionConfig defaults match expected values
# -------------------------------------------------------------------

class TestMotionConfigDefaults:
    def test_motion_config_defaults(self) -> None:
        cfg = load_config(DEFAULT_YAML)
        m = cfg.motion

        assert m.enabled is True
        assert m.history == 500
        assert m.var_threshold == 16
        assert m.detect_shadows is False
        assert m.process_width == 640
        assert m.process_height == 480
        assert m.erode_kernel == 3
        assert m.dilate_kernel == 7
        assert m.min_area_pct == pytest.approx(0.0002)
        assert m.cooldown_seconds == pytest.approx(2.0)
        assert m.learning_rate == pytest.approx(-1.0)


# -------------------------------------------------------------------
# 5. DetectionConfig defaults
# -------------------------------------------------------------------

class TestDetectionConfigDefaults:
    def test_detection_config_defaults(self) -> None:
        cfg = load_config(DEFAULT_YAML)
        d = cfg.detection

        assert d.enabled is True
        assert d.backend == "auto"
        # The custom 5-class model, not stock COCO yolov8n: COCO has no
        # squirrel or rat class, so shipping it as the default would leave
        # the detector unable to report a pest at all.
        assert d.model_path == "ratcatcher_best.onnx"
        assert d.confidence_threshold == pytest.approx(0.45)
        assert d.nms_threshold == pytest.approx(0.45)
        assert d.input_size == (640, 640)


# -------------------------------------------------------------------
# 6. ClassificationConfig defaults
# -------------------------------------------------------------------

class TestClassificationConfigDefaults:
    def test_classification_config_defaults(self) -> None:
        cfg = load_config(DEFAULT_YAML)
        c = cfg.classification

        assert c.enabled is True
        assert c.model_path == "mobilenet_v2_inat_bird_quant.tflite"
        assert c.species_config == "species.yaml"
        assert c.input_size == (224, 224)
        assert c.top_k == 5
        assert c.min_confidence == pytest.approx(0.70)
        assert c.use_xnnpack is True


# -------------------------------------------------------------------
# 7. StorageConfig defaults
# -------------------------------------------------------------------

class TestStorageConfigDefaults:
    def test_storage_config_defaults(self) -> None:
        cfg = load_config(DEFAULT_YAML)
        s = cfg.storage

        assert s.db_path == "detections.db"
        assert s.clip_dir == "clips"
        assert s.thumbnail_dir == "thumbnails"
        assert s.clip_pre_seconds == 5
        assert s.clip_post_seconds == 10
        assert s.retention_days == 30
        assert s.max_disk_gb == pytest.approx(10.0)
        assert s.ffmpeg_preset == "fast"


# -------------------------------------------------------------------
# 8. FileNotFoundError for nonexistent path
# -------------------------------------------------------------------

class TestLoadMissingFileRaises:
    def test_load_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path/that/does/not/exist.yaml")


# -------------------------------------------------------------------
# 9. ValueError for malformed YAML
# -------------------------------------------------------------------

class TestLoadInvalidYamlRaises:
    def test_load_invalid_yaml_raises(self, tmp_path: Path) -> None:
        bad_yaml = tmp_path / "broken.yaml"
        bad_yaml.write_text("{{invalid yaml content::: [[[", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid YAML"):
            load_config(bad_yaml)


# -------------------------------------------------------------------
# 10. Data-path derived properties
# -------------------------------------------------------------------

class TestConfigDataPaths:
    def test_config_data_paths(self) -> None:
        cfg = load_config(DEFAULT_YAML)

        assert cfg.data_path == Path("data")
        assert cfg.db_full_path == Path("data") / "detections.db"
        assert cfg.clip_full_path == Path("data") / "clips"
        assert cfg.thumbnail_full_path == Path("data") / "thumbnails"


# -------------------------------------------------------------------
# 11. Minimal YAML (only system section) -- other sections get defaults
# -------------------------------------------------------------------

class TestLoadMinimalYaml:
    def test_load_minimal_yaml(self, tmp_path: Path) -> None:
        minimal = tmp_path / "minimal.yaml"
        minimal.write_text(
            'system:\n  data_dir: "custom_data"\n  log_level: "DEBUG"\n',
            encoding="utf-8",
        )
        cfg = load_config(minimal)

        # System section should reflect the YAML values
        assert cfg.system.data_dir == "custom_data"
        assert cfg.system.log_level == "DEBUG"

        # No cameras defined
        assert cfg.cameras == []

        # Other sections should be populated with their dataclass defaults
        default_motion = MotionConfig()
        assert cfg.motion.enabled == default_motion.enabled
        assert cfg.motion.history == default_motion.history
        assert cfg.motion.var_threshold == default_motion.var_threshold

        default_detection = DetectionConfig()
        assert cfg.detection.enabled == default_detection.enabled
        assert cfg.detection.model_path == default_detection.model_path

        default_classification = ClassificationConfig()
        assert cfg.classification.enabled == default_classification.enabled
        assert cfg.classification.model_path == default_classification.model_path

        default_storage = StorageConfig()
        assert cfg.storage.db_path == default_storage.db_path
        assert cfg.storage.clip_dir == default_storage.clip_dir

        # Derived paths should use the custom data_dir
        assert cfg.data_path == Path("custom_data")
        assert cfg.db_full_path == Path("custom_data") / "detections.db"
