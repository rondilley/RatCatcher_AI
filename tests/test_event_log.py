"""Tests for syslog detection and status reporting.

No test doubles, following the rest of the suite. The transport tests
send over a real UDP socket bound to the loopback interface and assert
on the datagram that actually arrives, so what is verified is the bytes
a loghost would receive rather than a call that was made. The reporter
runs against a real SQLite database with real rows in it.
"""

from __future__ import annotations

import logging
import socket
import threading
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from ratcatcher.config import Config, SyslogConfig, SystemConfig
from ratcatcher.monitoring.events import (
    EVENT_LOGGER_NAME,
    close_event_log,
    configure_event_log,
    format_fields,
    log_audio_detection,
    log_status,
    log_video_detection,
)
from ratcatcher.monitoring.reporter import StatusReporter
from ratcatcher.monitoring.stats import get_category_counts
from ratcatcher.storage.database import DetectionDatabase


# -- fixtures --------------------------------------------------------------


@pytest.fixture
def db(tmp_path: Path) -> DetectionDatabase:
    database = DetectionDatabase(tmp_path / "detections.db")
    yield database
    database.close()


@pytest.fixture
def syslog_server():
    """A real UDP socket standing in for a loghost.

    Yields (host, port, receive), where receive returns the next
    datagram as text or raises TimeoutError.

    The trailing NUL is stripped here rather than suppressed in the
    handler: it is part of the traditional syslog wire format that real
    daemons expect, not something this project adds.
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1", 0))
    host, port = server.getsockname()

    def receive(timeout: float = 5.0) -> str:
        server.settimeout(timeout)
        datagram = server.recv(8192).decode("utf-8", errors="replace")
        return datagram.rstrip("\x00")

    yield host, port, receive
    server.close()


@pytest.fixture(autouse=True)
def clean_event_log():
    """Leave the shared event logger as it was found.

    The handler lives on a module-level logger, so a test that attaches
    one and does not remove it would send every later test's output to a
    closed socket.
    """
    yield
    close_event_log()
    configure_event_log(SyslogConfig(enabled=False))


@pytest.fixture
def captured() -> list[logging.LogRecord]:
    """Collect records from the event logger with a real handler."""
    records: list[logging.LogRecord] = []

    class Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = Collector()
    logger = logging.getLogger(EVENT_LOGGER_NAME)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    yield records
    logger.removeHandler(handler)


def add_video(
    db: DetectionDatabase,
    class_name: str,
    when: datetime,
    count: int = 1,
) -> None:
    for _ in range(count):
        db.insert_detection(
            timestamp=when.isoformat(),
            camera_id=0,
            stage="detection",
            class_name=class_name,
            confidence=0.9,
        )


def add_audio(
    db: DetectionDatabase,
    species: str,
    when: datetime,
    count: int = 1,
) -> None:
    for _ in range(count):
        db.insert_audio_detection(
            timestamp=when.isoformat(),
            channel=1,
            species=species,
            common_name=species,
            confidence=0.5,
        )


def parse_logfmt(text: str) -> dict[str, str]:
    """Parse a logfmt fragment back into a dict.

    Deliberately a separate implementation from the encoder: a test that
    round-trips through the encoder's own inverse proves only that the
    two agree with each other.
    """
    fields: dict[str, str] = {}
    rest = text
    while "=" in rest:
        key, _, rest = rest.partition("=")
        key = key.strip()
        if rest.startswith('"'):
            value_chars = []
            index = 1
            while index < len(rest):
                char = rest[index]
                if char == "\\":
                    index += 1
                    value_chars.append(rest[index])
                elif char == '"':
                    break
                else:
                    value_chars.append(char)
                index += 1
            fields[key] = "".join(value_chars)
            rest = rest[index + 1:]
        else:
            value, _, rest = rest.partition(" ")
            fields[key] = value
    return fields


# -- the encoder -----------------------------------------------------------


def test_format_fields_writes_plain_values_unquoted():
    assert format_fields(modality="video", camera=0) == "modality=video camera=0"


def test_format_fields_quotes_values_containing_spaces():
    text = format_fields(common="Western Gray Squirrel")
    assert text == 'common="Western Gray Squirrel"'
    assert parse_logfmt(text)["common"] == "Western Gray Squirrel"


def test_format_fields_omits_none_rather_than_writing_an_empty_token():
    text = format_fields(species=None, camera=1)
    assert "species" not in text
    assert text == "camera=1"


def test_format_fields_renders_booleans_as_words():
    assert format_fields(pest=True, npu=False) == "pest=yes npu=no"


def test_format_fields_trims_float_noise():
    assert format_fields(confidence=0.8700000000000001) == "confidence=0.87"
    assert format_fields(temp_c=60.0) == "temp_c=60"


def test_format_fields_escapes_quotes_and_backslashes():
    text = format_fields(common='a "quoted" name')
    assert parse_logfmt(text)["common"] == 'a "quoted" name'


def test_format_fields_quotes_an_empty_string():
    assert format_fields(common="") == 'common=""'


def test_format_fields_preserves_argument_order():
    text = format_fields(modality="audio", channel=1, species="Corvus")
    assert text == "modality=audio channel=1 species=Corvus"


# -- the transport ---------------------------------------------------------


def test_detection_reaches_a_real_syslog_socket(syslog_server):
    host, port, receive = syslog_server

    assert configure_event_log(
        SyslogConfig(enabled=True, address=f"{host}:{port}", ident="ratcatcher")
    )

    log_video_detection(
        camera=0,
        class_name="squirrel",
        common_name="Western Gray Squirrel",
        confidence=0.87,
        pest=True,
        clip_path="/opt/ratcatcher/data/clips/a.mp4",
    )

    datagram = receive()

    assert "ratcatcher[" in datagram
    assert "detection " in datagram

    fields = parse_logfmt(datagram.split("detection ", 1)[1])
    assert fields["modality"] == "video"
    assert fields["camera"] == "0"
    assert fields["class"] == "squirrel"
    assert fields["common"] == "Western Gray Squirrel"
    assert fields["confidence"] == "0.87"
    assert fields["pest"] == "yes"
    assert fields["clip"] == "/opt/ratcatcher/data/clips/a.mp4"


def test_audio_detection_carries_the_channel_and_acoustics(syslog_server):
    host, port, receive = syslog_server
    configure_event_log(SyslogConfig(enabled=True, address=f"{host}:{port}"))

    log_audio_detection(
        channel=1,
        species="Zonotrichia leucophrys",
        common_name="White-crowned Sparrow",
        confidence=0.62,
        rms_dbfs=-41.25,
        flatness=0.31,
        peak_hz=3800.0,
    )

    fields = parse_logfmt(receive().split("detection ", 1)[1])
    assert fields["modality"] == "audio"
    assert fields["channel"] == "1"
    assert fields["common"] == "White-crowned Sparrow"
    assert fields["rms_dbfs"] == "-41.25"
    assert fields["peak_hz"] == "3800"


def test_status_line_carries_every_count(syslog_server, db):
    host, port, receive = syslog_server
    configure_event_log(SyslogConfig(enabled=True, address=f"{host}:{port}"))

    now = datetime.now()
    add_video(db, "bird", now, count=3)
    add_audio(db, "Corvus corax", now, count=2)

    log_status(
        window="today",
        counts=get_category_counts(db),
        cameras=2,
        npu=True,
        audio=True,
        temp_c=60.9,
        disk_pct=41.0,
        uptime="3h",
    )

    fields = parse_logfmt(receive().split("status ", 1)[1])
    assert fields["window"] == "today"
    assert fields["bird_seen"] == "3"
    assert fields["bird_heard"] == "2"
    assert fields["cameras"] == "2"
    assert fields["npu"] == "yes"
    assert fields["temp_c"] == "60.9"


def test_a_bad_socket_path_is_reported_and_survived(tmp_path, caplog):
    missing = tmp_path / "no-such-syslog-socket"

    with caplog.at_level(logging.WARNING):
        attached = configure_event_log(
            SyslogConfig(enabled=True, address=str(missing))
        )

    assert attached is False
    assert any("Syslog disabled" in record.message for record in caplog.records)

    # And the pipeline keeps reporting, to journald, with nothing raised.
    log_video_detection(camera=0, class_name="bird")


def test_a_bare_hostname_is_refused_rather_than_given_a_default_port(caplog):
    with caplog.at_level(logging.WARNING):
        assert configure_event_log(
            SyslogConfig(enabled=True, address="loghost")
        ) is False


def test_an_unknown_facility_is_refused(caplog):
    with caplog.at_level(logging.WARNING):
        assert configure_event_log(
            SyslogConfig(enabled=True, address="/dev/log", facility="nonsense")
        ) is False


def test_disabled_syslog_attaches_no_handler_but_still_logs(captured):
    assert configure_event_log(SyslogConfig(enabled=False)) is False

    log_video_detection(camera=0, class_name="bird")

    assert len(captured) == 1
    assert "modality=video" in captured[0].getMessage()


def test_configuring_twice_does_not_duplicate_the_handler(syslog_server):
    host, port, receive = syslog_server
    address = f"{host}:{port}"

    configure_event_log(SyslogConfig(enabled=True, address=address))
    configure_event_log(SyslogConfig(enabled=True, address=address))

    log_video_detection(camera=0, class_name="bird")

    receive()
    with pytest.raises(TimeoutError):
        receive(timeout=0.5)


def test_detections_flag_silences_detections_but_not_status(captured, db):
    configure_event_log(SyslogConfig(enabled=False, detections=False))

    log_video_detection(camera=0, class_name="bird")
    log_audio_detection(channel=0, species="Corvus corax")
    assert captured == []

    log_status(
        window="today",
        counts=get_category_counts(db),
        cameras=2,
        npu=True,
        audio=False,
    )
    assert len(captured) == 1


# -- the reporter ----------------------------------------------------------


@pytest.fixture
def config(tmp_path: Path) -> Config:
    return Config(
        system=SystemConfig(data_dir=str(tmp_path)),
        syslog=SyslogConfig(enabled=True, status_interval_seconds=0.05),
    )


def test_reporter_counts_agree_with_the_panel(config, db, captured):
    """The syslog counts and the e-paper counts must not drift apart."""
    now = datetime.now()
    add_video(db, "bird", now, count=4)
    add_video(db, "squirrel", now, count=2)
    add_audio(db, "Corvus corax", now, count=7)

    reporter = StatusReporter(config)
    reporter.open(db)
    try:
        assert reporter.report_now()
    finally:
        reporter.close()

    expected = get_category_counts(db, since=_midnight(now))
    fields = parse_logfmt(captured[-1].getMessage().split("status ", 1)[1])

    assert fields["bird_seen"] == str(expected.bird.seen)
    assert fields["bird_heard"] == str(expected.bird.heard)
    assert fields["rodent_seen"] == str(expected.rodent.seen)
    assert fields["total"] == str(expected.total)


def test_reporter_window_excludes_older_records(config, db, captured):
    now = datetime.now()
    add_video(db, "bird", now, count=2)
    add_video(db, "bird", now - timedelta(days=3), count=9)

    reporter = StatusReporter(config)
    reporter.open(db)
    try:
        reporter.report_now()
    finally:
        reporter.close()

    fields = parse_logfmt(captured[-1].getMessage().split("status ", 1)[1])
    assert fields["bird_seen"] == "2"


def test_reporter_reports_what_the_pipeline_opened(config, db, captured):
    reporter = StatusReporter(config)
    reporter.open(db)
    reporter.set_state(cameras=1, audio_active=False)
    try:
        reporter.report_now()
    finally:
        reporter.close()

    fields = parse_logfmt(captured[-1].getMessage().split("status ", 1)[1])
    assert fields["cameras"] == "1"
    assert fields["audio"] == "no"


def test_reporter_thread_reports_immediately_and_then_repeats(config, db, captured):
    reporter = StatusReporter(config)
    reporter.start(db)
    try:
        deadline = threading.Event()
        deadline.wait(0.4)
    finally:
        reporter.stop()

    assert len(captured) >= 2
    assert reporter.stats["reports_sent"] >= 2
    assert reporter.stats["report_failures"] == 0


def test_reporter_does_not_close_a_database_it_was_given(config, db):
    reporter = StatusReporter(config)
    reporter.open(db)
    reporter.close()

    # Still usable: closing a handle the caller owns would take the
    # pipeline's database out from under it.
    assert get_category_counts(db).total == 0


# -- the pipeline call site ------------------------------------------------


def run_storage_loop(config: Config, db: DetectionDatabase, events) -> None:
    """Drive the real storage loop over real events, then stop it.

    A real PipelineEngine with a real database and a real queue. Only
    the camera and the NPU are left out, because the storage loop
    reaches neither.
    """
    from ratcatcher.pipeline import engine as engine_module

    engine = engine_module.PipelineEngine(config)
    engine._db = db
    config.thumbnail_full_path.mkdir(parents=True, exist_ok=True)

    for event in events:
        engine._storage_queue.put(event)
    engine._storage_queue.put(engine_module._SENTINEL)

    engine._storage_loop()


def detection_event(**overrides):
    from ratcatcher.pipeline.event import DetectionEvent

    defaults = dict(
        timestamp=datetime.now(),
        camera_id=0,
        frame=np.zeros((16, 16, 3), dtype=np.uint8),
        frame_width=16,
        frame_height=16,
    )
    defaults.update(overrides)
    return DetectionEvent(**defaults)


def test_storage_loop_reports_an_identified_animal(config, db, captured):
    run_storage_loop(
        config,
        db,
        [
            detection_event(
                stage="classification",
                class_name="bird",
                confidence=0.91,
                species="Junco hyemalis",
                common_name="Dark-eyed Junco",
                species_confidence=0.78,
            )
        ],
    )

    lines = [r.getMessage() for r in captured if r.getMessage().startswith("detection")]
    assert len(lines) == 1

    fields = parse_logfmt(lines[0].split("detection ", 1)[1])
    assert fields["modality"] == "video"
    assert fields["class"] == "bird"
    assert fields["common"] == "Dark-eyed Junco"
    assert fields["species_confidence"] == "0.78"
    assert fields["pest"] == "no"


def test_storage_loop_marks_a_pest(config, db, captured):
    run_storage_loop(
        config, db, [detection_event(class_name="rat", confidence=0.7)]
    )

    fields = parse_logfmt(captured[-1].getMessage().split("detection ", 1)[1])
    assert fields["pest"] == "yes"


def test_storage_loop_stays_quiet_for_motion_with_nothing_identified(
    config, db, captured
):
    """Motion alone names no animal, and would bury the real sightings."""
    run_storage_loop(config, db, [detection_event(stage="motion")])

    assert [r for r in captured if r.getMessage().startswith("detection")] == []


def test_storage_loop_reports_nothing_when_the_insert_fails(config, db, captured):
    """A detection that was not stored must not be reported as stored."""
    db.close()

    run_storage_loop(config, db, [detection_event(class_name="cat", confidence=0.6)])

    assert [r for r in captured if r.getMessage().startswith("detection")] == []


def run_audio_store(config: Config, db: DetectionDatabase, detections) -> None:
    """Drive the real audio store over a real window, then read it back.

    A real AudioEngine with a real database and a real WAV clip. Only
    the microphone and BirdNET are left out, because the store reaches
    neither -- it is handed the windows they would have produced.
    """
    from ratcatcher.audio.activity import ActivityResult
    from ratcatcher.pipeline.audio_engine import AudioEngine

    engine = AudioEngine(config)
    engine._database = db

    activity = ActivityResult(
        triggered=True,
        band_rms_dbfs=-42.5,
        noise_floor_dbfs=-58.0,
        snr_db=15.5,
        spectral_flatness=0.21,
        peak_frequency_hz=4200.0,
    )
    signal = np.zeros(config.audio.sample_rate * 3, dtype=np.float32)

    engine._store(datetime.now(), 0, signal, activity, detections)


def song_detection(**overrides):
    from ratcatcher.audio.birdnet import SongDetection

    defaults = dict(
        scientific_name="Junco hyemalis",
        common_name="Dark-eyed Junco",
        confidence=0.66,
        channel=0,
    )
    defaults.update(overrides)
    return SongDetection(**defaults)


def test_audio_store_reports_an_identified_species(config, db, captured):
    run_audio_store(config, db, [song_detection()])

    lines = [r.getMessage() for r in captured if r.getMessage().startswith("detection")]
    assert len(lines) == 1

    fields = parse_logfmt(lines[0].split("detection ", 1)[1])
    assert fields["modality"] == "audio"
    assert fields["species"] == "Junco hyemalis"
    assert fields["common"] == "Dark-eyed Junco"


def test_audio_store_stays_quiet_for_a_window_that_matched_no_species(
    config, db, captured
):
    """A gated window names no bird, and at gate rates would bury the songs."""
    run_audio_store(config, db, [])

    assert [r for r in captured if r.getMessage().startswith("detection")] == []


def test_an_unmatched_window_is_still_stored(config, db):
    """Only the syslog line is suppressed. The database keeps the row."""
    run_audio_store(db=db, config=config, detections=[])

    rows = db.get_audio_detections(limit=10)
    assert len(rows) == 1
    assert rows[0]["species"] is None


def _midnight(moment: datetime) -> str:
    return moment.replace(
        hour=0, minute=0, second=0, microsecond=0
    ).isoformat()
