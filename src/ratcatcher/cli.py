"""RatCatcher AI command-line interface."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ratcatcher",
        description="Wildlife detection system for bird feeders",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_get_version()}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    run_parser = subparsers.add_parser("run", help="Start the detection pipeline")
    run_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file",
    )
    run_parser.add_argument(
        "--source",
        type=str,
        choices=["auto", "picamera", "webcam", "file"],
        default=None,
        help="Camera source type override",
    )
    run_parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input file or directory (for file source)",
    )
    run_parser.add_argument(
        "--cameras",
        type=int,
        nargs="+",
        default=None,
        help="Camera IDs to use (default: all enabled)",
    )
    run_parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "motion", "detect"],
        default="full",
        help="Pipeline mode: full (detect+classify), motion (motion only), detect (no species ID)",
    )
    run_parser.add_argument(
        "--no-classify",
        action="store_true",
        help="Skip species classification (detection only)",
    )
    run_parser.add_argument(
        "--backend",
        type=str,
        choices=["auto", "hailo", "ncnn", "opencv_dnn"],
        default=None,
        help="Detection backend override",
    )

    stats_parser = subparsers.add_parser("stats", help="Show detection statistics")
    stats_parser.add_argument(
        "--last",
        type=str,
        default=None,
        help="Time window (e.g., '24h', '7d', '1w')",
    )
    stats_parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Path to detections database",
    )

    subparsers.add_parser("health", help="Show system health status")

    test_cam_parser = subparsers.add_parser("test-camera", help="Capture a test frame")
    test_cam_parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera ID to test",
    )

    return parser


def _get_version() -> str:
    from ratcatcher import __version__
    return __version__


def _parse_time_window(window: str) -> str:
    """Convert a time window string like '24h' or '7d' to an ISO timestamp."""
    now = datetime.now()
    value = int(window[:-1])
    unit = window[-1].lower()
    if unit == "h":
        delta = timedelta(hours=value)
    elif unit == "d":
        delta = timedelta(days=value)
    elif unit == "w":
        delta = timedelta(weeks=value)
    else:
        raise ValueError(f"Unknown time unit: {unit}. Use h, d, or w.")
    return (now - delta).isoformat()


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )


def _cmd_run(args: argparse.Namespace) -> int:
    """Start the detection pipeline."""
    from ratcatcher.config import load_config

    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    _setup_logging(config.system.log_level)

    if args.source is not None:
        cameras = []
        for cam in config.cameras:
            from dataclasses import replace
            updated = replace(cam, source_type=args.source)
            if args.source == "file" and args.input is not None:
                updated = replace(updated, file_path=args.input)
            cameras.append(updated)
        from dataclasses import replace as dc_replace
        config = dc_replace(config, cameras=cameras)

    if args.backend is not None:
        from dataclasses import replace as dc_replace2
        config = dc_replace2(
            config,
            detection=dc_replace2(config.detection, backend=args.backend),
        )

    skip_classify = args.no_classify or args.mode == "motion"
    skip_detect = args.mode == "motion"

    if skip_detect:
        from dataclasses import replace as dc_replace3
        config = dc_replace3(
            config,
            detection=dc_replace3(config.detection, enabled=False),
        )

    print(f"RatCatcher AI v{_get_version()}")
    print(f"Mode: {args.mode}")

    from ratcatcher.pipeline.engine import PipelineEngine

    engine = PipelineEngine(config)
    try:
        engine.start(
            camera_ids=args.cameras,
            skip_classification=skip_classify,
        )
        engine.run_until_stopped()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        engine.stop()

    stats = engine.stats
    print(f"\nSession stats: {stats}")
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    """Show detection statistics from the database."""
    from ratcatcher.config import load_config
    from ratcatcher.storage.database import DetectionDatabase

    if args.db is not None:
        db_path = Path(args.db)
    else:
        try:
            config = load_config()
            db_path = config.db_full_path
        except FileNotFoundError:
            db_path = Path("data/detections.db")

    if not db_path.exists():
        print(f"Database not found: {db_path}", file=sys.stderr)
        return 1

    since = None
    if args.last is not None:
        try:
            since = _parse_time_window(args.last)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    with DetectionDatabase(db_path) as db:
        total = db.get_detection_count(since=since)
        counts = db.get_species_counts(since=since)

    window_label = f"last {args.last}" if args.last else "all time"
    print(f"Detection Statistics ({window_label})")
    print(f"{'=' * 50}")
    print(f"Total detections: {total}")

    if counts:
        print(f"\nBy species:")
        for species_name, count in counts.items():
            print(f"  {species_name}: {count}")
    else:
        print("No detections recorded.")

    return 0


def _cmd_health(args: argparse.Namespace) -> int:
    """Show system health status."""
    import platform
    import os

    print("System Health")
    print(f"{'=' * 50}")
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python: {platform.python_version()}")

    try:
        import cv2
        print(f"OpenCV: {cv2.__version__}")
    except ImportError:
        print("OpenCV: NOT INSTALLED")

    try:
        import numpy as np
        print(f"NumPy: {np.__version__}")
    except ImportError:
        print("NumPy: NOT INSTALLED")

    hailo_available = False
    try:
        import hailo_platform  # type: ignore[import-untyped]
        hailo_available = True
        print("Hailo: available")
    except ImportError:
        print("Hailo: not available")

    tflite_available = False
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore[import-untyped]
        tflite_available = True
        print("TFLite Runtime: available")
    except ImportError:
        try:
            import tensorflow.lite  # type: ignore[import-untyped]
            tflite_available = True
            print("TFLite (via TensorFlow): available")
        except ImportError:
            print("TFLite: not available")

    import shutil
    print(f"FFmpeg: {'available' if shutil.which('ffmpeg') else 'not available'}")

    if platform.system() == "Linux" and platform.machine() == "aarch64":
        try:
            with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                temp_mc = int(f.read().strip())
                print(f"CPU Temperature: {temp_mc / 1000:.1f} C")
        except (FileNotFoundError, ValueError, PermissionError):
            print("CPU Temperature: unavailable")

    return 0


def _cmd_test_camera(args: argparse.Namespace) -> int:
    """Capture a test frame from a camera."""
    from ratcatcher.config import load_config, CameraConfig
    from ratcatcher.camera.platform_camera import create_camera

    try:
        config = load_config()
        cam_cfgs = [c for c in config.cameras if c.id == args.camera]
        if cam_cfgs:
            cam_cfg = cam_cfgs[0]
        else:
            cam_cfg = CameraConfig(id=args.camera)
    except FileNotFoundError:
        cam_cfg = CameraConfig(id=args.camera)

    try:
        camera = create_camera(cam_cfg)
        camera.start()
    except Exception as exc:
        print(f"ERROR: Failed to open camera {args.camera}: {exc}", file=sys.stderr)
        return 1

    ok, frame = camera.read()
    camera.stop()

    if not ok or frame is None:
        print(f"ERROR: Failed to capture frame from camera {args.camera}", file=sys.stderr)
        return 1

    h, w = frame.shape[:2]
    print(f"Camera {args.camera}: captured {w}x{h} frame")

    out_path = f"test_camera_{args.camera}.jpg"
    try:
        import cv2
        cv2.imwrite(out_path, frame)
        print(f"Saved to {out_path}")
    except Exception as exc:
        print(f"ERROR: Failed to save frame: {exc}", file=sys.stderr)
        return 1

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    commands = {
        "run": _cmd_run,
        "stats": _cmd_stats,
        "health": _cmd_health,
        "test-camera": _cmd_test_camera,
    }

    handler = commands.get(args.command)
    if handler is not None:
        return handler(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
