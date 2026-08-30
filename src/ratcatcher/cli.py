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

    display_group = run_parser.add_mutually_exclusive_group()
    display_group.add_argument(
        "--display",
        dest="display",
        action="store_true",
        default=None,
        help="Enable the CrowPanel e-paper status panel",
    )
    display_group.add_argument(
        "--no-display",
        dest="display",
        action="store_false",
        help="Disable the status panel even if enabled in the config",
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

    display_parser = subparsers.add_parser(
        "display", help="Drive or test the CrowPanel e-paper status panel"
    )
    display_parser.add_argument(
        "--port",
        type=str,
        default=None,
        help="Serial device path (default: the configured port, or auto-detect)",
    )
    display_parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Path to detections database",
    )
    display_parser.add_argument(
        "--list-ports",
        action="store_true",
        help="List every serial port worth probing, then exit",
    )
    display_parser.add_argument(
        "--preview",
        action="store_true",
        help="Print the screen as text and exit. Opens no serial port.",
    )
    display_parser.add_argument(
        "--once",
        action="store_true",
        help="Send one frame to the panel and exit",
    )

    test_cam_parser = subparsers.add_parser("test-camera", help="Capture a test frame")
    test_cam_parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera ID to test",
    )

    focus_parser = subparsers.add_parser(
        "focus",
        help="Live focus and exposure meter on the e-paper panel",
    )
    focus_parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Measure only this camera (default: every enabled camera)",
    )
    focus_parser.add_argument(
        "--port",
        type=str,
        default=None,
        help="Serial device path (default: the configured port, or auto-detect)",
    )
    focus_parser.add_argument(
        "--preview",
        action="store_true",
        help="Print readings to the terminal instead. Opens no serial port.",
    )
    # Defaults are resolved in the handler rather than here, so that
    # building the parser for --help does not have to import OpenCV.
    focus_parser.add_argument(
        "--ceiling",
        type=float,
        default=None,
        help="Blur ratio that reads as 100%%. Raise it if a focused lens never reaches 100.",
    )
    focus_parser.add_argument(
        "--continuous",
        action="store_true",
        help=(
            "Read on a timer instead of waiting for a button. Wears the "
            "panel considerably faster."
        ),
    )
    focus_parser.add_argument(
        "--interval",
        type=float,
        default=None,
        help="Seconds between readings, with --continuous.",
    )
    focus_parser.add_argument(
        "--web",
        action="store_true",
        help=(
            "Also serve live camera views over HTTP, for setting a lens by "
            "eye from a phone. Binds to every interface, no authentication."
        ),
    )
    focus_parser.add_argument(
        "--web-port",
        type=int,
        default=None,
        help="Port for --web (default: 8080).",
    )
    focus_parser.add_argument(
        "--web-only",
        action="store_true",
        help="Serve the web view and do not touch the e-paper panel.",
    )
    focus_parser.add_argument(
        "--web-refresh",
        type=float,
        default=None,
        help=(
            "Seconds between web view refreshes (default: 4). A lens is "
            "turned and then looked at, so this is a viewer, not a video feed."
        ),
    )

    return parser


def _get_version() -> str:
    from ratcatcher import __version__
    return __version__


def _parse_time_window(window: str) -> float:
    """Convert a time window string like '24h' or '7d' to a number of hours."""
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
    return delta.total_seconds() / 3600.0


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

    # Before anything can detect, so the first detection has somewhere
    # to go. A syslog socket that cannot be opened is reported and then
    # ignored; the lines still reach journald either way.
    from ratcatcher.monitoring.events import configure_event_log

    configure_event_log(config.syslog)

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

    if args.display is not None:
        from dataclasses import replace as dc_replace5
        config = dc_replace5(
            config,
            display=dc_replace5(config.display, enabled=args.display),
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

    # The status panel is independent for the same reason the audio path
    # is: it must never be able to stop the cameras. It reports on the
    # system, so a panel fault that took the system down would be the
    # worst possible failure mode.
    display_engine = None
    if config.display.enabled:
        from ratcatcher.display.engine import DisplayEngine
        from ratcatcher.display.panel import NullPanel

        display_engine = DisplayEngine(config)
        try:
            display_engine.start()
            display_engine.set_state(
                "RUN",
                cameras=sum(1 for camera in config.cameras if camera.enabled),
                audio_active=audio_engine is not None,
            )
            if isinstance(display_engine.panel, NullPanel):
                print("Display: no panel found, continuing without one")
            else:
                print(f"Display: {display_engine.panel.description}")
        except (RuntimeError, OSError) as exc:
            print(f"WARNING: status panel disabled -- {exc}", file=sys.stderr)
            display_engine = None
    else:
        print("Display: disabled")

    # Independent for the same reason the panel is. It reports on the
    # system, so a reporting fault that stopped the cameras would be the
    # worst possible failure mode.
    status_reporter = None
    if config.syslog.enabled:
        from ratcatcher.monitoring.reporter import StatusReporter

        status_reporter = StatusReporter(config)
        try:
            status_reporter.set_state(
                cameras=sum(1 for camera in config.cameras if camera.enabled),
                audio_active=audio_engine is not None,
            )
            status_reporter.start()
            print(f"Syslog: reporting to {config.syslog.address}")
        except (RuntimeError, OSError) as exc:
            print(f"WARNING: syslog status disabled -- {exc}", file=sys.stderr)
            status_reporter = None
    else:
        print("Syslog: disabled")

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
        if status_reporter is not None:
            status_reporter.stop()
        if display_engine is not None:
            display_engine.stop()
        if audio_engine is not None:
            audio_engine.stop()

    stats = engine.stats
    print(f"\nSession stats: {stats}")
    if audio_engine is not None:
        print(f"Audio stats:   {audio_engine.stats}")
    if display_engine is not None:
        print(f"Panel stats:   {display_engine.stats}")
    if status_reporter is not None:
        print(f"Syslog stats:  {status_reporter.stats}")
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    """Show detection statistics from the database."""
    from ratcatcher.config import load_config
    from ratcatcher.monitoring.stats import format_stats, get_stats
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

    hours = None
    if args.last is not None:
        try:
            hours = _parse_time_window(args.last)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    label = f"last {args.last}" if args.last else "all time"

    with DetectionDatabase(db_path) as db:
        stats = get_stats(db, hours=hours, label=label)

    print(format_stats(stats))
    if stats.total_detections == 0:
        print("\nNo detections recorded.")

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
            # Same order as the classifiers: tflite_runtime, then
            # ai_edge_litert, then tensorflow. ai-edge-litert is the only
            # one of the three with wheels for current Python versions.
            from ai_edge_litert.interpreter import Interpreter  # type: ignore[import-untyped]  # noqa: F401
            tflite_available = True
            print("TFLite (via ai-edge-litert): available")
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


def _cmd_display(args: argparse.Namespace) -> int:
    """Drive or test the e-paper status panel.

    With no options it runs the panel until interrupted. ``--preview``
    answers the first question when the screen looks wrong: is the
    number wrong, or is the panel wrong? It builds exactly the frame
    that would be sent and prints it, touching no serial port.
    """
    import time
    from dataclasses import replace as dc_replace

    from ratcatcher.config import load_config
    from ratcatcher.display.factory import list_candidate_ports
    from ratcatcher.display.render import render_box
    from ratcatcher.storage.database import DetectionDatabase

    if args.list_ports:
        ports = list_candidate_ports()
        if not ports:
            print("No USB serial ports found.")
            print("")
            print("If the panel is plugged in, this account may not be able")
            print("to see it. Add it to the 'dialout' group and log in again:")
            print("  sudo usermod -aG dialout $USER")
            return 1
        print("Serial ports that could carry a panel:")
        for port, description in ports:
            print(f"  {port}  {description}")
        return 0

    try:
        config = load_config()
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    _setup_logging(config.system.log_level)

    if args.port is not None:
        config = dc_replace(
            config,
            display=dc_replace(config.display, port=args.port, source_type="serial"),
        )

    db_path = Path(args.db) if args.db is not None else config.db_full_path
    if not db_path.exists() and args.db is not None:
        print(f"Database not found: {db_path}", file=sys.stderr)
        return 1

    with DetectionDatabase(db_path) as db:
        if args.preview:
            from ratcatcher.display.status import build_status_frame

            frame = build_status_frame(db, config)
            print(render_box(frame))
            return 0

        from ratcatcher.display.engine import DisplayEngine
        from ratcatcher.display.panel import NullPanel

        engine = DisplayEngine(config)

        # A single frame needs no background thread, so --once opens the
        # panel, writes, and closes without ever starting one.
        try:
            if args.once:
                engine.open(database=db)
            else:
                engine.start(database=db)
        except RuntimeError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

        panel = engine.panel
        if isinstance(panel, NullPanel):
            print("ERROR: no panel found.", file=sys.stderr)
            print("", file=sys.stderr)
            print("Check the port with: ratcatcher display --list-ports", file=sys.stderr)
            print(
                "Flash the firmware with: ./scripts/build_panel_firmware.sh --upload",
                file=sys.stderr,
            )
            engine.stop()
            return 1

        print(f"Panel: {panel.description}")

        if args.once:
            sent = engine.refresh_now(full=True)
            print(render_box(engine.last_frame) if engine.last_frame else "")
            # Leave the reading on the screen. The closing STOP frame is
            # for a system shutting down, not for a one-shot reading.
            engine.close(announce_stop=False)
            print("Frame sent." if sent else "ERROR: the frame was not accepted.")
            return 0 if sent else 1

        print("Driving the panel. Press Ctrl-C to stop.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("")
        finally:
            engine.stop()

        print(f"Panel stats: {engine.stats}")
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


def _cmd_focus(args: argparse.Namespace) -> int:
    """Live focus meter for setting the lenses by hand.

    The lenses have no software focus control, so this reports numbers
    and a person turns the ring. ``--preview`` prints to the terminal
    for bench work; the panel is what makes it usable in the field.
    """
    import time
    from dataclasses import replace as dc_replace

    from ratcatcher.camera.focus import (
        DEFAULT_CEILING,
        analyse_frame,
        exposure_sample,
        exposure_settled,
        readings_settled,
    )
    from ratcatcher.camera.platform_camera import create_camera
    from ratcatcher.config import CameraConfig, load_config
    from ratcatcher.display.render import render_screen_box
    from ratcatcher.pipeline.focus_engine import (
        DEFAULT_INTERVAL,
        FocusEngine,
        _metadata_of,
    )

    ceiling = args.ceiling if args.ceiling is not None else DEFAULT_CEILING
    interval = args.interval if args.interval is not None else DEFAULT_INTERVAL

    if ceiling <= 1.0:
        print("ERROR: --ceiling must be greater than 1.0", file=sys.stderr)
        return 1

    try:
        config = load_config()
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    _setup_logging(config.system.log_level)

    if _service_is_active():
        print("ERROR: the ratcatcher service is running and holds the cameras.")
        print("")
        print("  sudo systemctl stop ratcatcher")
        print("  ratcatcher focus")
        print("  sudo systemctl start ratcatcher")
        return 1

    cam_cfgs = [c for c in config.cameras if c.enabled]
    if args.camera is not None:
        cam_cfgs = [c for c in config.cameras if c.id == args.camera] or [
            CameraConfig(id=args.camera)
        ]
    if not cam_cfgs:
        print("ERROR: no cameras are enabled in the configuration.", file=sys.stderr)
        return 1

    cameras = []
    try:
        for cam_cfg in cam_cfgs:
            camera = create_camera(cam_cfg)
            camera.start()
            cameras.append(camera)
    except Exception as exc:
        for camera in cameras:
            camera.stop()
        print(f"ERROR: cannot open camera: {exc}", file=sys.stderr)
        return 1

    # The first frame of a picamera source arrives from a background
    # thread, so a reading taken immediately would measure nothing. This
    # used to be a flat one-second sleep, which was enough at 1920x1080
    # and silently stopped being enough when the cameras moved to the
    # full 4056x3040 sensor: every reading came back "no frames from any
    # camera", which reads as a hardware fault rather than a tool that
    # gave up too early. Wait for the frames themselves instead.
    deadline = time.monotonic() + 15.0
    pending = list(cameras)
    while pending and time.monotonic() < deadline:
        pending = [c for c in pending if not c.read()[0]]
        if pending:
            time.sleep(0.1)
    if pending:
        print(
            f"WARNING: {len(pending)} of {len(cameras)} camera(s) produced no "
            "frame within 15s; readings for those will be unavailable",
            file=sys.stderr,
        )

    # A frame is not enough: it has to be exposed correctly. Camera 1
    # here delivers its first frame at 193 us and luma 121 and measures
    # 57%; a second later auto exposure has settled on 303 us and the
    # same lens measures 41%. Reading before that reports a lens as
    # acceptable when it needs turning, which is the one mistake this
    # tool must not make.
    # Exposure settling is necessary and not sufficient: the ISP's
    # temporal denoise keeps converging for seconds after every control
    # has frozen, and the blur ratio counts the early frames' sensor
    # noise as detail. So wait for the reading itself to stop moving.
    # This is what made the tool feel temperamental -- the same lens
    # position reported a different number depending on how long you
    # happened to wait before pressing the button.
    settling = {id(c): [] for c in cameras}
    readings = {id(c): [] for c in cameras}
    deadline = time.monotonic() + 25.0
    while time.monotonic() < deadline:
        for camera in cameras:
            metadata = _metadata_of(camera)
            settling[id(camera)].append(exposure_sample(metadata))
            ok, frame = camera.read()
            if ok and frame is not None:
                readings[id(camera)].append(
                    analyse_frame(frame, 0, metadata=metadata, ceiling=ceiling).focus_pct
                )
        if all(
            exposure_settled(settling[k]) and readings_settled(readings[k])
            for k in settling
        ):
            break
        time.sleep(0.4)
    else:
        stalled = sum(
            1
            for k in settling
            if not (exposure_settled(settling[k]) and readings_settled(readings[k]))
        )
        if stalled:
            print(
                f"WARNING: reading still drifting on {stalled} camera(s) after "
                "25s; the value shown may not describe the lens",
                file=sys.stderr,
            )

    # Serving the pixels is independent of the panel, and of the score:
    # the blur ratio hill-climbs well but is unreliable on a badly
    # defocused lens, where a soft frame carries so little real detail
    # that the measurement is mostly sensor noise. The web view exists so
    # the eye can decide instead.
    web = None
    if args.web or args.web_only:
        from ratcatcher.web.focus_server import (
            DEFAULT_PORT,
            DEFAULT_REFRESH_MS,
            FocusWebServer,
        )

        port = args.web_port if args.web_port is not None else DEFAULT_PORT
        refresh_ms = (
            int(args.web_refresh * 1000)
            if args.web_refresh is not None
            else DEFAULT_REFRESH_MS
        )
        try:
            web = FocusWebServer(
                cameras, port=port, ceiling=ceiling, refresh_ms=refresh_ms
            )
            web.start()
        except OSError as exc:
            for camera in cameras:
                camera.stop()
            print(f"ERROR: cannot serve on port {port}: {exc}", file=sys.stderr)
            return 1
        print("")
        print(f"  Focus viewer: {web.urls[0]}")
        print("  Open that on a phone on the same network.")
        print("  1:1 crop is the view to judge focus on; the 3x3 grid moves it.")
        print(f"  Refreshing every {refresh_ms / 1000:.0f}s; 'refresh now' after each adjustment.")
        print("")

    if args.web_only:
        print("Serving until Ctrl-C.")
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        finally:
            if web is not None:
                web.stop()
            for camera in cameras:
                camera.stop()
        return 0

    panel = None
    engine = None
    try:
        if args.preview:
            engine = FocusEngine(cameras, _PrintPanel(), ceiling=ceiling, interval=interval)
            readings = engine.read_once()
            frame = engine.build_screen(
                readings, clock=datetime.now().strftime("%H:%M")
            )
            print(render_screen_box(frame))
            return 0

        from ratcatcher.display.factory import create_panel
        from ratcatcher.display.panel import NullPanel

        if args.port is not None:
            config = dc_replace(
                config,
                display=dc_replace(config.display, port=args.port, source_type="serial"),
            )

        panel = create_panel(config.display)
        panel.open()

        if isinstance(panel, NullPanel):
            print("ERROR: no panel found.", file=sys.stderr)
            print("", file=sys.stderr)
            print("Check the port with: ratcatcher display --list-ports", file=sys.stderr)
            return 1

        _warn_if_firmware_stale(panel)

        print(f"Focus meter running on {panel.description}.")
        if args.continuous:
            print(f"Reading every {interval:.1f}s.")
        else:
            print("Adjust a lens, then press a panel button to read it again.")
            print("The sample counter on the panel proves the press landed.")
        print("Panel: EXIT quits, any other button takes a reading.")
        print("Ctrl-C also quits.")

        engine = FocusEngine(
            cameras,
            panel,
            ceiling=ceiling,
            interval=interval,
            continuous=args.continuous,
        )
        try:
            engine.run()
        except KeyboardInterrupt:
            engine.stop()

        for camera_id, peak in sorted(engine.peaks.items()):
            print(f"camera {camera_id}: best reading {peak:.0f}%")
        if web is not None:
            web.stop()
        # Panel refreshes are a consumable. Reporting them keeps the cost
        # of a long session visible rather than silent.
        print(
            f"panel: {engine.refreshes} refreshes, "
            f"{engine.skipped} frames skipped as unchanged"
        )
        return 0
    finally:
        # E-paper holds its last image with no power, so a panel left
        # showing live-looking numbers is how a stale reading gets
        # trusted. Say the tool has gone before releasing the port.
        if engine is not None and panel is not None:
            try:
                engine.draw_exit_screen()
            except OSError as exc:
                print(f"WARNING: could not draw the exit screen: {exc}", file=sys.stderr)
        for camera in cameras:
            camera.stop()
        if panel is not None:
            panel.close()


def _service_is_active() -> bool:
    """True when the systemd unit is running and holding the devices.

    Asked up front because the failure it prevents is unhelpful:
    libcamera reports "Pipeline handler in use by another process" from
    inside a stack trace that names neither the service nor the fix.
    """
    import subprocess

    try:
        result = subprocess.run(
            ["systemctl", "is-active", "ratcatcher"],
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        # No systemd, or systemctl is not on PATH. A development machine
        # has no service to conflict with, so carry on.
        return False
    return result.stdout.strip() == "active"


def _warn_if_firmware_stale(panel: object) -> None:
    """Say so when the panel is too old to draw the focus screen.

    Older firmware ignores an unknown frame type without complaint, and
    an e-paper panel holds its last image with no power, so the symptom
    is a screen that simply never changes.
    """
    import time

    from ratcatcher.display.protocol import supports_screen

    ping = getattr(panel, "ping", None)
    if ping is None:
        return
    ping()

    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        panel.poll()
        if getattr(panel, "firmware", None) is not None:
            break
        time.sleep(0.1)

    firmware = getattr(panel, "firmware", None)
    if not supports_screen(firmware):
        print("")
        print(f"WARNING: panel firmware {firmware} cannot draw the focus screen.")
        print("The panel will keep showing its previous image.")
        print("Reflash with: ./scripts/build_panel_firmware.sh --upload")
        print("")


class _PrintPanel:
    """Accepts screens and discards them, for --preview.

    The preview renders the frame itself, so this only has to satisfy
    the engine's calls without opening anything.
    """

    @property
    def description(self) -> str:
        return "preview (no panel)"

    def open(self) -> None:
        return None

    def send(self, frame: object) -> bool:
        return True

    def send_screen(self, frame: object) -> bool:
        return True

    def poll(self) -> list[dict[str, object]]:
        return []

    def close(self) -> None:
        return None


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
        "display": _cmd_display,
        "focus": _cmd_focus,
        "test-camera": _cmd_test_camera,
        "test-mic": _cmd_test_mic,
    }

    handler = commands.get(args.command)
    if handler is not None:
        return handler(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
