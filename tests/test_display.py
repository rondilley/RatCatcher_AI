"""Tests for the CrowPanel e-paper status panel.

No test doubles, following the rest of the suite. The database is a real
SQLite file, the serial tests run over a real pseudo-terminal with a
real pyserial port on one end, and ``PanelEmulator`` speaks the same
wire protocol as the firmware. It stands in for the hardware the way
``WavFileSource`` stands in for the microphones.
"""

from __future__ import annotations

import json
import os
import pty
import threading
import time
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from ratcatcher.config import Config, DisplayConfig, SystemConfig
from ratcatcher.display.engine import DisplayEngine
from ratcatcher.display.factory import find_panel_port
from ratcatcher.display.panel import FilePanel, NullPanel, SerialPanel
from ratcatcher.display.protocol import (
    MAX_LINE_BYTES,
    PANEL_MODEL,
    PROTOCOL_VERSION,
    LastSighting,
    StatusFrame,
    SystemStatus,
    decode_line,
    encode_frame,
)
from ratcatcher.display.render import WIDTH, render_text
from ratcatcher.display.status import build_status_frame, format_uptime, window_start
from ratcatcher.monitoring.stats import categorize, get_category_counts
from ratcatcher.storage.database import DetectionDatabase


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db(tmp_path: Path) -> DetectionDatabase:
    database = DetectionDatabase(tmp_path / "detections.db")
    yield database
    database.close()


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(
        system=SystemConfig(data_dir=str(tmp_path)),
        display=DisplayConfig(enabled=True, source_type="null"),
    )


def add_video(
    db: DetectionDatabase,
    class_name: str,
    when: datetime,
    count: int = 1,
    common_name: str | None = None,
) -> None:
    for _ in range(count):
        db.insert_detection(
            timestamp=when.isoformat(),
            camera_id=0,
            stage="classification" if common_name else "detection",
            class_name=class_name,
            common_name=common_name,
            confidence=0.9,
        )


def add_audio(
    db: DetectionDatabase,
    species: str,
    when: datetime,
    count: int = 1,
    common_name: str | None = None,
) -> None:
    for _ in range(count):
        db.insert_audio_detection(
            timestamp=when.isoformat(),
            channel=1,
            species=species,
            common_name=common_name or species,
            confidence=0.5,
        )


def sample_frame(**overrides) -> StatusFrame:
    defaults = dict(
        clock="14:32",
        state="RUN",
        window="today",
        counts={"bird": (18, 6), "rodent": (3, 0), "other": (1, 0)},
        system=SystemStatus(
            cameras=2, npu=True, audio=True, temp_c=47.4, disk_pct=61.2, uptime="3d"
        ),
        last=LastSighting(name="Dark-eyed Junco", sense="ear", clock="14:29"),
    )
    defaults.update(overrides)
    return StatusFrame(**defaults)


# ---------------------------------------------------------------------------
# Counting
# ---------------------------------------------------------------------------


def test_counts_split_video_from_audio(db: DetectionDatabase) -> None:
    now = datetime.now()
    add_video(db, "bird", now, count=18, common_name="Dark-eyed Junco")
    add_video(db, "squirrel", now, count=3)
    add_video(db, "cat", now, count=1)
    add_audio(db, "Junco hyemalis", now, count=6)

    counts = get_category_counts(db)

    assert counts.bird.seen == 18
    assert counts.bird.heard == 6
    assert counts.bird.total == 24
    assert counts.rodent.seen == 3
    assert counts.rodent.heard == 0
    assert counts.other.seen == 1
    assert counts.total == 28


def test_motion_only_rows_are_not_counted(db: DetectionDatabase) -> None:
    """A motion event records that something moved, not what it was."""
    now = datetime.now()
    db.insert_detection(
        timestamp=now.isoformat(), camera_id=0, stage="motion", class_name=None
    )
    add_video(db, "bird", now)

    counts = get_category_counts(db)

    assert counts.total == 1


def test_rat_and_squirrel_are_both_rodents(db: DetectionDatabase) -> None:
    now = datetime.now()
    add_video(db, "squirrel", now, count=2)
    add_video(db, "rat", now, count=5)

    counts = get_category_counts(db)

    assert counts.rodent.seen == 7
    assert counts.bird.total == 0


def test_birdnet_noise_labels_are_discarded(db: DetectionDatabase) -> None:
    """A passing engine is not a visitor to the feeder."""
    now = datetime.now()
    add_audio(db, "Engine", now, count=4)
    add_audio(db, "Siren", now, count=2)
    add_audio(db, "Junco hyemalis", now, count=1)

    counts = get_category_counts(db)

    assert counts.bird.heard == 1
    assert counts.total == 1


def test_birdnet_rodent_genus_is_counted_as_a_rodent(db: DetectionDatabase) -> None:
    """The unified view labels every audio row 'bird'. The genus corrects it."""
    now = datetime.now()
    add_audio(db, "Sciurus carolinensis", now, count=3)

    counts = get_category_counts(db)

    assert counts.rodent.heard == 3
    assert counts.bird.heard == 0


def test_categorize_handles_an_unknown_detector_class() -> None:
    assert categorize("video", "porcupine", None) == "other"
    assert categorize("video", None, None) is None
    assert categorize("audio", None, None) == "bird"


def test_today_window_excludes_yesterday(db: DetectionDatabase, config: Config) -> None:
    now = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0)
    add_video(db, "bird", now, count=4)
    add_video(db, "bird", now - timedelta(days=1), count=99)

    since, label = window_start("today", now=now)
    counts = get_category_counts(db, since=since)

    assert label == "today"
    assert counts.bird.seen == 4


def test_all_window_counts_everything(db: DetectionDatabase) -> None:
    now = datetime.now()
    add_video(db, "bird", now, count=2)
    add_video(db, "bird", now - timedelta(days=30), count=3)

    since, label = window_start("all", now=now)

    assert since is None
    assert label == "all"
    assert get_category_counts(db, since=since).bird.seen == 5


def test_unknown_window_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown display window"):
        window_start("fortnight")


# ---------------------------------------------------------------------------
# Frame building
# ---------------------------------------------------------------------------


def test_frame_reports_the_last_sighting_from_either_sense(
    db: DetectionDatabase, config: Config
) -> None:
    now = datetime.now()
    add_video(db, "bird", now - timedelta(hours=2), common_name="House Finch")
    add_audio(db, "Junco hyemalis", now, common_name="Dark-eyed Junco")

    frame = build_status_frame(db, config, now=now)

    assert frame.last is not None
    assert frame.last.name == "Dark-eyed Junco"
    assert frame.last.sense == "ear"


def test_frame_has_no_last_sighting_on_an_empty_database(
    db: DetectionDatabase, config: Config
) -> None:
    frame = build_status_frame(db, config)

    assert frame.last is None
    assert frame.counts["bird"] == (0, 0)


def test_frame_state_defaults_to_run(db: DetectionDatabase, config: Config) -> None:
    frame = build_status_frame(db, config, state="RUN")

    assert frame.state == "RUN"
    assert frame.window == "today"


def test_format_uptime_picks_the_readable_unit() -> None:
    assert format_uptime(None) == "?"
    assert format_uptime(90) == "1m"
    assert format_uptime(7200) == "2h"
    assert format_uptime(3 * 86400 + 60) == "3d"


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


def test_encoded_frame_is_one_line_within_the_panel_buffer() -> None:
    line = encode_frame(sample_frame())

    assert line.endswith(b"\n")
    assert line.count(b"\n") == 1
    assert len(line) <= MAX_LINE_BYTES + 1

    payload = json.loads(line)
    assert payload["v"] == PROTOCOL_VERSION
    assert payload["t"] == "status"
    assert payload["n"]["bird"] == [18, 6]
    assert payload["sys"]["temp"] == 47


def test_a_very_long_name_is_shortened_rather_than_cut_by_the_panel() -> None:
    frame = sample_frame(
        last=LastSighting(name="A" * 400, sense="eye", clock="09:15")
    )

    line = encode_frame(frame)

    assert len(line) <= MAX_LINE_BYTES + 1
    payload = json.loads(line)
    # The frame still parses, and the fields the panel must show exactly
    # are all present.
    assert payload["n"]["bird"] == [18, 6]
    assert payload["sys"]["cam"] == 2


def test_content_key_ignores_the_sequence_number_and_refresh_mode() -> None:
    first = sample_frame(seq=1, full_refresh=False)
    second = sample_frame(seq=99, full_refresh=True)

    assert first.content_key() == second.content_key()


def test_content_key_changes_when_a_count_changes() -> None:
    first = sample_frame()
    second = sample_frame(counts={"bird": (19, 6), "rodent": (3, 0), "other": (1, 0)})

    assert first.content_key() != second.content_key()


def test_decode_ignores_boot_loader_noise() -> None:
    """The ESP32 boot loader prints its own banner on this same port."""
    assert decode_line(b"rst:0x1 (POWERON_RESET),boot:0x8") is None
    assert decode_line(b"") is None
    assert decode_line(b"{not json}") is None
    assert decode_line(b'{"no":"type"}') is None
    assert decode_line(b'\xff\xfe garbage') is None

    message = decode_line(b'{"v":1,"t":"ack","seq":7}')
    assert message is not None
    assert message["t"] == "ack"


def test_null_values_survive_the_round_trip() -> None:
    frame = sample_frame(
        system=SystemStatus(cameras=0, npu=False, audio=False, uptime="?")
    )

    payload = json.loads(encode_frame(frame))

    assert payload["sys"]["temp"] is None
    assert payload["sys"]["disk"] is None
    assert payload["sys"]["npu"] == 0


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_render_shows_both_senses_and_the_total() -> None:
    text = render_text(sample_frame())
    lines = text.split("\n")

    assert all(len(line) == WIDTH for line in lines)
    assert "RatCatcher today" in lines[0]
    assert "RUN" in lines[0]

    birds = next(line for line in lines if "BIRDS" in line)
    assert "18" in birds and "6" in birds and "24" in birds

    assert "Dark-eyed Junco" in text
    assert "CAM 2" in text
    assert "NPU ok" in text


def test_render_says_so_when_nothing_has_been_seen() -> None:
    frame = sample_frame(
        counts={"bird": (0, 0), "rodent": (0, 0), "other": (0, 0)}, last=None
    )

    text = render_text(frame)

    assert "nothing yet" in text


def test_render_marks_missing_hardware() -> None:
    frame = sample_frame(
        system=SystemStatus(cameras=0, npu=False, audio=False, uptime="5m")
    )

    text = render_text(frame)

    assert "NPU --" in text
    assert "MIC --" in text


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def test_file_panel_writes_replayable_frames(tmp_path: Path) -> None:
    path = tmp_path / "panel.jsonl"
    panel = FilePanel(path)
    panel.open()
    panel.send(sample_frame(seq=0))
    panel.send(sample_frame(seq=1))
    panel.close()

    lines = path.read_bytes().splitlines()

    assert len(lines) == 2
    assert json.loads(lines[1])["seq"] == 1


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


def test_engine_skips_a_frame_that_would_change_nothing(
    db: DetectionDatabase, config: Config
) -> None:
    panel = NullPanel()
    engine = DisplayEngine(config, panel=panel)
    engine.open(database=db)
    try:
        assert engine.refresh_now() is True
        # Nothing has changed, so the panel is left alone. Every refresh
        # costs power and a part of the panel's life.
        assert engine.refresh_now_if_changed() is False
        assert len(panel.frames) == 1
        assert engine.stats["frames_skipped"] == 1

        add_video(db, "bird", datetime.now())
        assert engine.refresh_now_if_changed() is True
        assert len(panel.frames) == 2
    finally:
        engine.close()


def test_engine_asks_for_a_full_refresh_on_the_first_frame(
    db: DetectionDatabase, config: Config
) -> None:
    """The panel holds whatever image the last run left on it."""
    panel = NullPanel()
    engine = DisplayEngine(config, panel=panel)
    engine.open(database=db)
    try:
        engine.refresh_now()
        assert panel.frames[0].full_refresh is True
    finally:
        engine.close()


def test_engine_reports_stop_when_it_shuts_down(
    db: DetectionDatabase, config: Config, tmp_path: Path
) -> None:
    """A panel still reading RUN for a system that has exited lies."""
    path = tmp_path / "panel.jsonl"
    panel = FilePanel(path)
    engine = DisplayEngine(config, panel=panel)
    engine.open(database=db)
    engine.refresh_now()
    engine.close()

    frames = [json.loads(line) for line in path.read_bytes().splitlines()]

    assert frames[0]["st"] == "RUN"
    assert frames[-1]["st"] == "STOP"


def test_a_single_reading_is_left_on_the_panel(
    db: DetectionDatabase, config: Config, tmp_path: Path
) -> None:
    """STOP is for a system shutting down, not for a one-shot reading.

    Announcing STOP here would overwrite the reading that was asked for,
    and would cost a second full refresh of the panel to do it.
    """
    path = tmp_path / "panel.jsonl"
    panel = FilePanel(path)
    engine = DisplayEngine(config, panel=panel)
    engine.open(database=db)
    engine.refresh_now(full=True)
    engine.close(announce_stop=False)

    frames = [json.loads(line) for line in path.read_bytes().splitlines()]

    assert len(frames) == 1
    assert frames[0]["st"] == "RUN"


def test_engine_survives_a_panel_that_refuses_every_write(
    db: DetectionDatabase, config: Config
) -> None:
    class DeadPanel:
        description = "dead"

        def open(self):
            return None

        def send(self, frame):
            return False

        def poll(self):
            return []

        def close(self):
            return None

    engine = DisplayEngine(config, panel=DeadPanel())
    engine.start(database=db)
    try:
        assert engine.refresh_now() is False
        assert engine.stats["write_failures"] >= 1
        # The thread is still there. A panel that cannot be written to
        # must not end the process that reports on the whole system.
        assert engine.is_running
    finally:
        engine.stop()


# ---------------------------------------------------------------------------
# Serial link, over a real pseudo-terminal
# ---------------------------------------------------------------------------


class PanelEmulator:
    """Speaks the firmware's side of the wire protocol over a pty.

    Answers a ping with a hello and a status frame with an ack, exactly
    as ratcatcher_panel.ino does. It stands in for the hardware, not for
    any code under test.
    """

    def __init__(self) -> None:
        self._master, self._slave = pty.openpty()
        self.port = os.ttyname(self._slave)
        self.received: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        if self._stop.is_set():
            return  # a test may close the panel before the fixture does
        self._stop.set()
        self._thread.join(timeout=2.0)
        for handle in (self._master, self._slave):
            try:
                os.close(handle)
            except OSError:
                pass

    def _run(self) -> None:
        buffer = bytearray()
        while not self._stop.is_set():
            try:
                chunk = os.read(self._master, 1024)
            except OSError:
                return
            if not chunk:
                continue
            buffer.extend(chunk)
            while b"\n" in buffer:
                raw, _, rest = bytes(buffer).partition(b"\n")
                buffer = bytearray(rest)
                message = decode_line(raw)
                if message is None:
                    continue
                self.received.append(message)
                self._answer(message)

    def _answer(self, message: dict) -> None:
        kind = message.get("t")
        if kind == "ping":
            reply = {
                "v": PROTOCOL_VERSION,
                "t": "hello",
                "panel": PANEL_MODEL,
                "fw": "1.0.0",
            }
        elif kind == "status":
            reply = {"v": PROTOCOL_VERSION, "t": "ack", "seq": message.get("seq", 0)}
        else:
            return
        os.write(self._master, (json.dumps(reply) + "\n").encode())


@pytest.fixture
def emulator():
    panel = PanelEmulator()
    panel.start()
    yield panel
    panel.stop()


def _wait_for(predicate, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_serial_panel_sends_a_frame_and_reads_the_acknowledgement(emulator) -> None:
    panel = SerialPanel(emulator.port)
    panel.open()
    try:
        # Opening the port sends a ping, which the panel answers.
        assert _wait_for(lambda: any(m["t"] == "hello" for m in panel.poll()) or panel.confirmed)
        assert panel.confirmed is True

        assert panel.send(sample_frame(seq=5)) is True
        assert _wait_for(
            lambda: any(m.get("t") == "status" for m in emulator.received)
        )

        sent = next(m for m in emulator.received if m.get("t") == "status")
        assert sent["seq"] == 5
        assert sent["n"]["bird"] == [18, 6]
        assert sent["last"]["name"] == "Dark-eyed Junco"

        acks: list[dict] = []
        assert _wait_for(lambda: acks.extend(panel.poll()) or any(
            m.get("t") == "ack" for m in acks
        ))
    finally:
        panel.close()


def test_serial_panel_reports_the_link_down_when_the_port_disappears(
    emulator,
) -> None:
    panel = SerialPanel(emulator.port, reconnect_seconds=60.0)
    panel.open()
    emulator.stop()

    # The port is gone. Writes must fail cleanly and never raise, so a
    # panel that is unplugged cannot take the pipeline down with it.
    for _ in range(5):
        panel.send(sample_frame())
    assert panel.connected is False
    panel.close()


def test_port_detection_needs_the_handshake_not_just_a_serial_port(
    emulator,
) -> None:
    """A plain CH340 descriptor cannot tell a panel from any other board."""
    silent_master, silent_slave = pty.openpty()
    silent_port = os.ttyname(silent_slave)
    try:
        assert (
            find_panel_port(probe_seconds=0.5, candidates=[silent_port]) is None
        )
        assert (
            find_panel_port(
                probe_seconds=2.0, candidates=[silent_port, emulator.port]
            )
            == emulator.port
        )
    finally:
        os.close(silent_master)
        os.close(silent_slave)


def test_engine_drives_a_real_serial_panel(
    db: DetectionDatabase, config: Config, emulator
) -> None:
    now = datetime.now()
    add_video(db, "bird", now, count=7, common_name="House Finch")
    add_audio(db, "Junco hyemalis", now, count=2, common_name="Dark-eyed Junco")

    serial_config = replace(
        config.display, source_type="serial", port=emulator.port
    )
    engine = DisplayEngine(replace(config, display=serial_config))
    engine.start(database=db)
    try:
        assert engine.refresh_now(full=True) is True
        assert _wait_for(
            lambda: any(m.get("t") == "status" for m in emulator.received)
        )
        frame = next(m for m in emulator.received if m.get("t") == "status")
        assert frame["n"]["bird"] == [7, 2]
        assert frame["full"] == 1
    finally:
        engine.stop()
