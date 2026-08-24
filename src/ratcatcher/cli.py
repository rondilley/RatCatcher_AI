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

    audio_group = run_parser.add_mutually_exclusive_group()
    audio_group.add_argument(
        "--audio",
        dest="audio",
        action="store_true",
        default=None,
        help="Enable bird song detection from the I2S microphones",
    )
    audio_group.add_argument(
        "--no-audio",
        dest="audio",
        action="store_false",
        help="Disable bird song detection even if enabled in the config",
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

    test_mic_parser = subparsers.add_parser(
        "test-mic", help="Record from the I2S microphones and report levels"
    )
    test_mic_parser.add_argument(
        "--seconds",
        type=float,
        default=5.0,
        help="Recording duration in seconds (default: 5)",
    )
    test_mic_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write the recording to this WAV file",
    )
    test_mic_parser.add_argument(
        "--identify",
        action="store_true",
        help="Run BirdNET on the recording and report any species heard",
    )

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

    if args.audio is not None:
        from dataclasses import replace as dc_replace4
        config = dc_replace4(
            config,
            audio=dc_replace4(config.audio, enabled=args.audio),
        )

    from ratcatcher.pipeline.engine import PipelineEngine

    engine = PipelineEngine(config)

    # The audio engine is deliberately independent: it owns its own
    # threads and its own database handle, so a failure to open the
    # microphones must not stop the cameras from running.
    audio_engine = None
    if config.audio.enabled:
        from ratcatcher.pipeline.audio_engine import AudioEngine

        audio_engine = AudioEngine(config)
        try:
            audio_engine.start()
            print("Audio: bird song detection active")
        except (RuntimeError, OSError) as exc:
            print(f"WARNING: audio disabled -- {exc}", file=sys.stderr)
            audio_engine = None
    else:
        print("Audio: disabled")

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
    finally:
        if audio_engine is not None:
            audio_engine.stop()

    stats = engine.stats
    print(f"\nSession stats: {stats}")
    if audio_engine is not None:
        print(f"Audio stats:   {audio_engine.stats}")
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


def _cmd_test_mic(args: argparse.Namespace) -> int:
    """Record from the I2S microphones and report per-channel levels.

    The audio counterpart of ``test-camera``. Its main job is answering
    the first question after wiring the SPH0645s: is each microphone
    actually producing signal, and are they on the channels expected?
    """
    import numpy as np

    from ratcatcher.audio.capture import ArecordSource
    from ratcatcher.audio.preprocess import (
        DCBlocker,
        peak_dbfs,
        rms_dbfs,
        split_channels,
    )
    from ratcatcher.config import load_config

    try:
        config = load_config()
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    _setup_logging(config.system.log_level)
    audio_config = config.audio

    devices = ArecordSource.list_capture_devices()
    if not devices:
        print("ERROR: no ALSA capture devices found.", file=sys.stderr)
        print("", file=sys.stderr)
        print("The I2S microphones are not enabled yet. Run:", file=sys.stderr)
        print("  sudo ./scripts/enable_i2s_mics.sh", file=sys.stderr)
        print("then reboot and try again.", file=sys.stderr)
        return 1

    print("Capture devices:")
    for device in devices:
        print(f"  {device}")
    print()

    source = ArecordSource(
        device=audio_config.device,
        sample_rate=audio_config.sample_rate,
        channels=audio_config.channels,
    )

    frames = int(audio_config.sample_rate * args.seconds)
    print(
        f"Recording {args.seconds:.1f} s from '{audio_config.device}' "
        f"({audio_config.channels} channel(s) @ {audio_config.sample_rate} Hz)..."
    )

    try:
        source.start()
        ok, block = source.read(frames)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    finally:
        source.stop()

    if not ok or block is None:
        print("ERROR: capture failed or returned no audio.", file=sys.stderr)
        return 1

    raw_offset = float(np.mean(block))
    filtered = DCBlocker(
        sample_rate=audio_config.sample_rate,
        cutoff_hz=audio_config.highpass_hz,
        channels=block.shape[1],
    ).process(block)

    print()
    print(f"Captured {block.shape[0] / audio_config.sample_rate:.2f} s")
    print(f"DC offset before filtering: {raw_offset:+.5f} (SPH0645 always has one)")
    print()
    print(f"  {'channel':<12} {'RMS':>10} {'peak':>10}   status")

    channels = split_channels(filtered)
    problems = 0
    for index, signal in enumerate(channels):
        rms = rms_dbfs(signal)
        peak = peak_dbfs(signal)
        side = {0: "left", 1: "right"}.get(index, f"ch{index}")

        # A live SPH0645 always shows some self-noise. A channel pinned at
        # digital silence means SEL, DOUT or power is not connected.
        if peak == float("-inf") or rms < -90.0:
            status = "SILENT -- check wiring (SEL, DOUT, power)"
            problems += 1
        elif peak > -1.0:
            status = "CLIPPING -- sound source is too loud"
            problems += 1
        else:
            status = "OK"

        rms_text = "-inf" if rms == float("-inf") else f"{rms:.1f}"
        peak_text = "-inf" if peak == float("-inf") else f"{peak:.1f}"
        print(f"  {index} ({side:<5}) {rms_text:>10} {peak_text:>10}   {status}")

    if len(channels) >= 2:
        difference = abs(rms_dbfs(channels[0]) - rms_dbfs(channels[1]))
        if difference > 20.0:
            print()
            print(
                f"WARNING: channels differ by {difference:.0f} dB. If both mics "
                f"hear the same scene, check that one SEL is tied to GND and "
                f"the other to 3.3V."
            )

    if args.output:
        from ratcatcher.audio.clip_writer import write_wav

        try:
            written = write_wav(args.output, filtered, audio_config.sample_rate)
            print(f"\nWrote {written}")
        except (OSError, ValueError) as exc:
            print(f"ERROR: could not write {args.output}: {exc}", file=sys.stderr)
            return 1

    if args.identify:
        from ratcatcher.audio.birdnet import BirdNetClassifier

        print()
        try:
            classifier = BirdNetClassifier(
                model_path=Path("models") / audio_config.model_path,
                labels_path=Path("models") / audio_config.labels_path,
                min_confidence=audio_config.min_confidence,
                top_k=audio_config.top_k,
                num_threads=audio_config.num_threads,
            )
        except (ImportError, FileNotFoundError, ValueError, RuntimeError) as exc:
            print(f"ERROR: BirdNET unavailable -- {exc}", file=sys.stderr)
            return 1

        print("BirdNET identification:")
        found = False
        for index, signal in enumerate(channels):
            for start in range(0, max(1, signal.size), classifier.window_samples):
                window = signal[start : start + classifier.window_samples]
                if window.size < classifier.window_samples // 2:
                    break
                for detection in classifier.identify(window, channel=index):
                    offset = start / audio_config.sample_rate
                    print(f"  ch{index} t={offset:5.1f}s  {detection}")
                    found = True
        if not found:
            print("  (nothing identified above the confidence threshold)")

    return 0 if problems == 0 else 1


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
        "test-mic": _cmd_test_mic,
    }

    handler = commands.get(args.command)
    if handler is not None:
        return handler(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
