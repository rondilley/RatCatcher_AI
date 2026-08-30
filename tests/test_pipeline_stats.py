"""Tests for the periodic pipeline counters line.

The real PipelineEngine and a real thread; no cameras are started,
because _stats_loop reads only the counter dict and the stop event.
"""

from __future__ import annotations

import logging
import threading
import time

from ratcatcher.config import Config, MonitoringConfig
from ratcatcher.pipeline.engine import PipelineEngine


def _engine(interval: float) -> PipelineEngine:
    config = Config(
        monitoring=MonitoringConfig(pipeline_stats_interval_seconds=interval)
    )
    return PipelineEngine(config)


def _run_loop(engine: PipelineEngine) -> threading.Thread:
    engine._stop_event.clear()
    t = threading.Thread(target=engine._stats_loop, daemon=True, name="stats")
    t.start()
    return t


def _pipeline_lines(caplog) -> list[str]:
    return [
        r.getMessage()
        for r in caplog.records
        if r.getMessage().startswith("pipeline ")
    ]


class TestPipelineStatsLine:
    def test_it_reports_the_counters_it_was_given(self, caplog) -> None:
        engine = _engine(0.05)
        with engine._stats_lock:
            engine._stats["frames_captured"] = 120
            engine._stats["motion_events"] = 7
            engine._stats["crop_windows"] = 13
            engine._stats["detections"] = 2
            engine._stats["stored"] = 1

        with caplog.at_level(logging.INFO, logger="ratcatcher.pipeline.engine"):
            t = _run_loop(engine)
            deadline = time.monotonic() + 5.0
            while not _pipeline_lines(caplog) and time.monotonic() < deadline:
                time.sleep(0.02)
            engine._stop_event.set()
            t.join(timeout=5.0)

        lines = _pipeline_lines(caplog)
        assert lines, "no pipeline stats line was emitted"
        line = lines[0]
        assert "frames=120" in line
        assert "motion=7" in line
        assert "crops=13" in line
        assert "detections=2" in line
        assert "stored=1" in line

    def test_the_frame_rate_is_measured_over_the_interval(self, caplog) -> None:
        """fps is a delta, so a counter that does not move reports 0."""
        engine = _engine(0.05)
        with engine._stats_lock:
            engine._stats["frames_captured"] = 5000

        with caplog.at_level(logging.INFO, logger="ratcatcher.pipeline.engine"):
            t = _run_loop(engine)
            deadline = time.monotonic() + 5.0
            while not _pipeline_lines(caplog) and time.monotonic() < deadline:
                time.sleep(0.02)
            engine._stop_event.set()
            t.join(timeout=5.0)

        lines = _pipeline_lines(caplog)
        assert lines
        # A large total with no movement during the interval is 0 fps,
        # not 5000 divided by the interval.
        assert "fps=0.0" in lines[0]
        assert "frames=5000" in lines[0]

    def test_zero_interval_disables_the_line(self, caplog) -> None:
        engine = _engine(0.0)
        with caplog.at_level(logging.INFO, logger="ratcatcher.pipeline.engine"):
            t = _run_loop(engine)
            t.join(timeout=5.0)
            assert not t.is_alive(), "the loop should return immediately"
            time.sleep(0.2)
        assert not _pipeline_lines(caplog)

    def test_it_stops_without_waiting_out_the_interval(self) -> None:
        """stop() must not block for a full reporting period."""
        engine = _engine(30.0)
        t = _run_loop(engine)
        time.sleep(0.05)

        started = time.monotonic()
        engine._stop_event.set()
        t.join(timeout=5.0)
        elapsed = time.monotonic() - started

        assert not t.is_alive()
        assert elapsed < 1.0, f"shutdown waited {elapsed:.1f}s on the interval"
