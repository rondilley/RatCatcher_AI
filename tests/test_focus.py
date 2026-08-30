"""Tests for the focus meter.

No test doubles, following the rest of the suite. The metric is measured
against real images -- synthetic ones built with numpy and blurred with
the same OpenCV call the production path uses -- rather than against a
mocked score, so what is verified is the number a lens would actually
produce. The engine runs against a real FileSource reading real JPEGs
and a real NullPanel, and the wire tests encode and decode real bytes.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from ratcatcher.camera.focus import (
    DEFAULT_CEILING,
    STATE_BRIGHT,
    STATE_LOW_DETAIL,
    STATE_NO_IMAGE,
    STATE_OK,
    analyse_frame,
    blur_ratio,
    centre_crop,
    focus_percent,
)
from ratcatcher.display.protocol import (
    MAX_LINE_BYTES,
    MIN_SCREEN_FIRMWARE,
    ScreenFrame,
    ScreenLine,
    decode_line,
    encode_screen,
    supports_screen,
)
from ratcatcher.display.render import WIDTH as RENDER_WIDTH
from ratcatcher.display.render import render_screen_text
from ratcatcher.pipeline.focus_engine import FocusEngine


# ---------------------------------------------------------------------------
# Test images
# ---------------------------------------------------------------------------


def _textured(size: int = 480, seed: int = 7) -> np.ndarray:
    """A sharp, detailed BGR frame.

    Random noise softened just enough to look like optical detail rather
    than single-pixel salt, which is what a real sensor delivers.
    """
    rng = np.random.default_rng(seed)
    mono = rng.integers(0, 256, size=(size, size), dtype=np.uint8)
    mono = cv2.GaussianBlur(mono, (0, 0), 0.6)
    return cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)


def _flat(value: int = 66, size: int = 480) -> np.ndarray:
    """A featureless frame: what a covered lens returns."""
    return np.full((size, size, 3), value, np.uint8)


# ---------------------------------------------------------------------------
# The metric
# ---------------------------------------------------------------------------


def test_focus_falls_monotonically_as_blur_increases():
    """The whole tool rests on this: more blur must never read better.

    The metric this replaced (Laplacian variance over luma variance)
    failed exactly here, rising from 0.072 to 0.217 between sigma 4 and
    sigma 8 -- reporting a worse image as a better one.
    """
    sharp = _textured()
    scores = [
        analyse_frame(
            sharp if sigma == 0 else cv2.GaussianBlur(sharp, (0, 0), sigma), 0
        ).focus_pct
        for sigma in (0, 1, 2, 4, 8)
    ]

    assert scores == sorted(scores, reverse=True), scores
    # The range has to be wide enough to steer by, not merely ordered.
    assert scores[0] - scores[-1] > 40


def test_fully_blurred_frame_sits_at_the_floor():
    """A ratio of 1.0 means "already as soft as it can look" -> 0%."""
    assert blur_ratio(np.zeros((200, 200), np.uint8)) == pytest.approx(1.0)
    assert focus_percent(1.0) == pytest.approx(0.0)


def test_focus_percent_is_clamped_to_the_scale():
    assert focus_percent(0.2) == pytest.approx(0.0)
    assert focus_percent(DEFAULT_CEILING) == pytest.approx(100.0)
    assert focus_percent(DEFAULT_CEILING * 100) == pytest.approx(100.0)


def test_focus_percent_rejects_a_ceiling_that_cannot_be_a_scale():
    with pytest.raises(ValueError):
        focus_percent(5.0, ceiling=1.0)


def test_centre_crop_takes_the_middle_without_resizing():
    """Cropping, not scaling: downscaling would low-pass the detail."""
    frame = np.zeros((900, 600), np.uint8)
    crop = centre_crop(frame, 1.0 / 3.0)
    assert crop.shape == (300, 200)


def test_centre_crop_falls_back_rather_than_returning_nothing():
    """An empty slice would make every downstream statistic NaN."""
    frame = np.zeros((10, 10), np.uint8)
    assert centre_crop(frame, 0.0).shape == (10, 10)


# ---------------------------------------------------------------------------
# States: the reading that must not be a percentage
# ---------------------------------------------------------------------------


def test_blocked_camera_reports_no_image_rather_than_a_low_score():
    """The camera 1 regression.

    A covered lens measured 5% focus, which reads as "keep turning" when
    the truth is that no light is reaching the sensor. It must report
    the fault instead, and must not offer a percentage at all.
    """
    reading = analyse_frame(
        _flat(66), 1, metadata={"Lux": 7.3, "ExposureTime": 66656, "AnalogueGain": 7.9}
    )

    assert reading.state == STATE_NO_IMAGE
    assert not reading.measurable
    assert reading.lux == pytest.approx(7.3)


def test_featureless_but_lit_scene_is_low_detail_not_a_fault():
    """A blank wall in daylight is a framing problem, not a blocked lens."""
    reading = analyse_frame(_flat(128), 0, metadata={"Lux": 8000.0})

    assert reading.state == STATE_LOW_DETAIL
    assert not reading.measurable


def test_blown_out_frame_names_the_exposure_not_the_missing_detail():
    """Clipping explains an absent texture, so it is the better report."""
    reading = analyse_frame(_flat(255), 0, metadata={"Lux": 90000.0})

    assert reading.state == STATE_BRIGHT
    assert reading.clip_high_pct == pytest.approx(100.0)


def test_detailed_daylight_frame_is_measurable():
    reading = analyse_frame(_textured(), 0, metadata={"Lux": 19242.0})

    assert reading.state == STATE_OK
    assert reading.measurable
    assert reading.focus_pct > 50


def test_reading_survives_absent_metadata():
    """A file source has no exposure to report; the tool still works."""
    reading = analyse_frame(_textured(), 0)

    assert reading.measurable
    assert reading.lux is None
    assert reading.exposure_us is None


# ---------------------------------------------------------------------------
# Peak-hold
# ---------------------------------------------------------------------------


def test_peak_holds_the_best_reading_and_ignores_a_worse_one():
    sharp = _textured()
    soft = cv2.GaussianBlur(sharp, (0, 0), 4)

    best = analyse_frame(sharp, 0).focus_pct
    after = analyse_frame(soft, 0, peak_pct=best)

    assert after.focus_pct < best
    assert after.peak_pct == pytest.approx(best)


def test_an_unmeasurable_frame_does_not_reset_the_peak():
    """A hand passing the lens must not wipe the target you were chasing."""
    best = analyse_frame(_textured(), 0).focus_pct
    blocked = analyse_frame(_flat(66), 0, metadata={"Lux": 5.0}, peak_pct=best)

    assert blocked.peak_pct == pytest.approx(best)


# ---------------------------------------------------------------------------
# The wire
# ---------------------------------------------------------------------------


def test_screen_frame_round_trips():
    frame = ScreenFrame(
        title="FOCUS  09:14",
        lines=(
            ScreenLine(text="CAM0  63%", bar=63),
            ScreenLine(text="PEAK 71  LUX 19k  OK", rule=True),
        ),
        seq=7,
    )

    message = decode_line(encode_screen(frame))

    assert message["t"] == "screen"
    assert message["title"] == "FOCUS  09:14"
    assert message["seq"] == 7
    assert message["l"][0] == {"t": "CAM0  63%", "b": 63}
    assert message["l"][1] == {"t": "PEAK 71  LUX 19k  OK", "r": 1}


def test_screen_frame_stays_inside_the_firmware_buffer():
    """The panel reads each line into a fixed buffer.

    A frame longer than the limit would be parsed in half, so the
    encoder must bound it at the source however much it is given.
    """
    frame = ScreenFrame(
        title="X" * 200,
        lines=tuple(ScreenLine(text="Y" * 200, bar=50) for _ in range(20)),
    )

    line = encode_screen(frame)

    assert len(line) <= MAX_LINE_BYTES + 1  # +1 for the newline
    assert decode_line(line) is not None


def test_bar_is_clamped_to_the_scale():
    frame = ScreenFrame(
        title="t", lines=(ScreenLine(text="a", bar=430), ScreenLine(text="b", bar=-20))
    )

    message = decode_line(encode_screen(frame))

    assert message["l"][0]["b"] == 100
    assert message["l"][1]["b"] == 0


def test_content_key_ignores_the_sequence_number():
    """Otherwise every frame looks new and drives a needless refresh."""
    first = ScreenFrame(title="t", lines=(ScreenLine(text="a"),), seq=1)
    second = ScreenFrame(title="t", lines=(ScreenLine(text="a"),), seq=99)

    assert first.content_key() == second.content_key()


# ---------------------------------------------------------------------------
# Firmware capability
# ---------------------------------------------------------------------------


def test_firmware_before_the_screen_frame_is_reported_as_unable():
    assert not supports_screen("1.0.0")


def test_firmware_from_the_screen_frame_onwards_is_able():
    assert supports_screen(".".join(str(part) for part in MIN_SCREEN_FIRMWARE))
    assert supports_screen("2.0.0")


def test_an_unknown_version_is_given_the_benefit_of_the_doubt():
    """A panel that never answered may still draw perfectly well."""
    assert supports_screen(None)
    assert supports_screen("")
    assert supports_screen("experimental")


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def test_rendered_screen_shows_the_bar_and_the_rules():
    frame = ScreenFrame(
        title="FOCUS  09:14",
        lines=(
            ScreenLine(text="CAM0  63%", bar=63),
            ScreenLine(text="PEAK 71  OK", rule=True),
        ),
    )

    text = render_screen_text(frame)
    lines = text.split("\n")

    assert lines[0].startswith("FOCUS  09:14")
    assert "[" in lines[2] and "#" in lines[2]
    assert set(lines[4].strip()) == {"-"}


def test_every_rendered_row_fits_the_panel_width():
    """Over-long text is truncated, never wrapped onto the next row.

    A wrapped row would push the last reading off a 122-pixel screen.
    """
    frame = ScreenFrame(title="T" * 80, lines=(ScreenLine(text="L" * 80, bar=100),))

    for line in render_screen_text(frame).split("\n"):
        assert len(line) == RENDER_WIDTH


# ---------------------------------------------------------------------------
# The engine, against real cameras and a real panel object
# ---------------------------------------------------------------------------


@pytest.fixture()
def image_dir(tmp_path):
    """A directory of real JPEGs: one sharp, for FileSource to read."""
    cv2.imwrite(str(tmp_path / "frame_000.jpg"), _textured())
    return tmp_path


def _engine(image_dir, panel, **options):
    from ratcatcher.camera.capture import FileSource

    source = FileSource(str(image_dir))
    source.start()
    return FocusEngine([source], panel, interval=0.0, **options), source


def test_engine_measures_a_camera_and_builds_a_screen(image_dir):
    from ratcatcher.display.panel import NullPanel

    panel = NullPanel()
    engine, source = _engine(image_dir, panel)
    try:
        readings = engine.read_once()
        assert len(readings) == 1
        assert readings[0].measurable

        frame = engine.build_screen(readings, clock="09:14")
        assert frame.title == "FOCUS  09:14"
        # Two rows for the camera, then the key legend.
        assert len(frame.lines) == 3
        assert frame.lines[0].text.startswith("CAM0")
        assert frame.lines[0].bar is not None
    finally:
        source.stop()


def test_engine_holds_the_peak_across_readings(image_dir):
    from ratcatcher.display.panel import NullPanel

    engine, source = _engine(image_dir, NullPanel())
    try:
        engine.read_once()
        first = engine.peaks[0]
        engine.read_once()

        assert engine.peaks[0] == pytest.approx(first)
        assert first > 0
    finally:
        source.stop()


def test_ok_button_resets_the_peak(image_dir):
    """The peak has to be clearable when the scene changes."""
    from ratcatcher.display.panel import NullPanel

    engine, source = _engine(image_dir, NullPanel())
    try:
        engine.read_once()
        assert engine.peaks

        engine.reset_peaks()
        assert engine.peaks == {}
    finally:
        source.stop()


def test_first_frame_of_a_session_is_a_full_refresh(image_dir):
    """E-paper keeps whatever was on it, including another tool's screen."""
    from ratcatcher.display.panel import NullPanel

    panel = NullPanel()
    engine, source = _engine(image_dir, panel)
    try:
        engine.send_if_changed(engine.build_screen(engine.read_once(), clock="09:14"))

        assert panel.screens[0].full_refresh
    finally:
        source.stop()


def test_an_unchanged_reading_is_not_redrawn(image_dir):
    """The regression that mottled the panel.

    Sending every reading unconditionally at 2 Hz was about 1140
    refreshes in a ten-minute session, nearly all of them redrawing a
    number that had not moved. E-paper ghosts under that.
    """
    from ratcatcher.display.panel import NullPanel

    panel = NullPanel()
    engine, source = _engine(image_dir, panel)
    try:
        frame = engine.build_screen(engine.read_once(), clock="09:14")
        assert engine.send_if_changed(frame) is True

        for _ in range(20):
            assert engine.send_if_changed(frame) is False

        assert len(panel.screens) == 1
        assert engine.refreshes == 1
        assert engine.skipped == 20
    finally:
        source.stop()


def test_a_reading_that_moves_is_redrawn(image_dir):
    """Deduplication must not cost responsiveness while a ring turns."""
    from ratcatcher.display.panel import NullPanel

    panel = NullPanel()
    engine, source = _engine(image_dir, panel)
    try:
        engine.send_if_changed(engine.build_screen(engine.read_once(), clock="09:14"))
        engine.send_if_changed(engine.build_screen(engine.read_once(), clock="09:15"))

        assert len(panel.screens) == 2
        assert engine.refreshes == 2
    finally:
        source.stop()


def test_heartbeat_redraws_a_screen_that_has_not_changed(image_dir):
    """Silence past the firmware's stale timeout blanks the panel.

    The firmware declares the host dead after seven minutes and draws a
    splash over whatever was there, so deduplication cannot be allowed
    to go quiet indefinitely.
    """
    from ratcatcher.display.panel import NullPanel

    panel = NullPanel()
    engine, source = _engine(image_dir, panel, heartbeat_seconds=0.0)
    try:
        frame = engine.build_screen(engine.read_once(), clock="09:14")
        engine.send_if_changed(frame)
        engine.send_if_changed(frame)

        assert len(panel.screens) == 2
        assert engine.skipped == 0
    finally:
        source.stop()


def test_home_button_forces_a_redraw_of_an_unchanged_screen(image_dir):
    """HOME clears ghosting, so it must survive deduplication."""
    from ratcatcher.display.panel import NullPanel

    panel = NullPanel()
    engine, source = _engine(image_dir, panel)
    try:
        frame = engine.build_screen(engine.read_once(), clock="09:14")
        engine.send_if_changed(frame)
        assert engine.send_if_changed(frame) is False

        engine.force_full_refresh()
        assert engine.send_if_changed(frame) is True
        assert panel.screens[-1].full_refresh
    finally:
        source.stop()


# ---------------------------------------------------------------------------
# Refresh discipline: what keeps a still scene from redrawing the panel
# ---------------------------------------------------------------------------


def _reading(camera=0, pct=50.0, peak=50.0, lux=19242.0, exposure=331, state=STATE_OK):
    """A FocusReading with plausible field values, built directly."""
    from ratcatcher.camera.focus import FocusReading

    return FocusReading(
        camera=camera,
        focus_pct=pct,
        peak_pct=peak,
        blur_ratio=5.0,
        luma_mean=110.0,
        luma_std=57.0,
        clip_high_pct=0.0,
        clip_low_pct=0.0,
        state=state,
        lux=lux,
        exposure_us=exposure,
        analogue_gain=1.0,
    )


def _detail_of(frame):
    """The second row, which carries peak, light and state."""
    return frame.lines[1].text


def test_telemetry_is_shown_to_one_significant_figure():
    """Auto-exposure nudges lux and exposure on every single frame.

    At full precision those two fields alone differ every time, which
    defeats deduplication and redraws the panel once a second however
    still the scene is.
    """
    from ratcatcher.display.panel import NullPanel

    engine = FocusEngine([], NullPanel())
    detail = _detail_of(engine.build_screen([_reading()], clock="09:14"))

    assert "LUX 20k" in detail
    assert "EXP 300u" in detail


def test_small_light_changes_do_not_change_the_screen():
    from ratcatcher.display.panel import NullPanel

    engine = FocusEngine([], NullPanel())
    first = engine.build_screen([_reading(lux=19242.0, exposure=331)], clock="09:14")
    second = engine.build_screen([_reading(lux=20105.0, exposure=347)], clock="09:14")

    assert first.content_key() == second.content_key()


def test_a_reading_inside_the_deadband_holds_its_displayed_value():
    """The metric repeats to about +/-1%; that flicker must not redraw."""
    from ratcatcher.display.panel import NullPanel

    engine = FocusEngine([], NullPanel())
    first = engine.build_screen([_reading(pct=50.0, peak=50.0)], clock="09:14")
    second = engine.build_screen([_reading(pct=51.0, peak=50.0)], clock="09:14")

    assert first.content_key() == second.content_key()
    assert first.lines[0].bar == second.lines[0].bar


def test_a_real_turn_of_the_ring_moves_the_display():
    """Hysteresis must not cost responsiveness where it matters."""
    from ratcatcher.display.panel import NullPanel

    engine = FocusEngine([], NullPanel())
    first = engine.build_screen([_reading(pct=50.0, peak=50.0)], clock="09:14")
    second = engine.build_screen([_reading(pct=62.0, peak=62.0)], clock="09:14")

    assert first.content_key() != second.content_key()
    assert second.lines[0].bar > first.lines[0].bar


def test_losing_the_image_clears_the_held_value():
    """A blocked lens must not keep showing the number it had before."""
    from ratcatcher.display.panel import NullPanel

    engine = FocusEngine([], NullPanel())
    engine.build_screen([_reading(pct=50.0)], clock="09:14")
    blocked = engine.build_screen(
        [_reading(pct=4.0, lux=7.3, exposure=66656, state=STATE_NO_IMAGE)],
        clock="09:14",
    )

    assert blocked.lines[0].bar == 0
    assert "--" in blocked.lines[0].text
    assert STATE_NO_IMAGE in _detail_of(blocked)


# ---------------------------------------------------------------------------
# Press to sample
# ---------------------------------------------------------------------------


def test_a_press_always_redraws_even_when_nothing_changed(image_dir):
    """The press is the feedback.

    Deduplication would otherwise swallow a press taken against an
    unchanged scene, and a button that silently does nothing is
    indistinguishable from a broken one.
    """
    from ratcatcher.display.panel import NullPanel

    panel = NullPanel()
    engine, source = _engine(image_dir, panel)
    try:
        engine.sample_and_draw()
        before = len(panel.screens)

        engine.sample_and_draw(trigger="ok")
        engine.sample_and_draw(trigger="ok")

        assert len(panel.screens) == before + 2
    finally:
        source.stop()


def test_the_footer_counts_samples_so_a_press_is_visible(image_dir):
    """Identical readings still have to look like something happened."""
    from ratcatcher.display.panel import NullPanel

    panel = NullPanel()
    engine, source = _engine(image_dir, panel)
    try:
        engine.sample_and_draw(trigger="ok")
        engine.sample_and_draw(trigger="ok")

        assert "sample 1" in panel.screens[0].lines[-1].text
        assert "sample 2" in panel.screens[1].lines[-1].text
        assert "key ok" in panel.screens[1].lines[-1].text
    finally:
        source.stop()


def test_the_release_edge_does_not_take_a_second_reading(image_dir):
    """Firmware 1.2.0 reports both edges; only the press should act."""
    from ratcatcher.display.panel import NullPanel

    panel = NullPanel()
    engine, source = _engine(image_dir, panel)
    try:
        panel.press("ok", down=True)
        assert engine._handle_buttons() == "ok"

        panel.press("ok", down=False)
        assert engine._handle_buttons() is None
    finally:
        source.stop()


def test_exit_stops_the_session(image_dir):
    from ratcatcher.display.panel import NullPanel

    panel = NullPanel()
    engine, source = _engine(image_dir, panel)
    try:
        panel.press("exit")
        engine._handle_buttons()

        assert engine._stop.is_set()
    finally:
        source.stop()


def test_every_non_exit_button_asks_for_a_reading(image_dir):
    """The physical mapping is unconfirmed, so none of them may be dead."""
    from ratcatcher.display.panel import NullPanel

    for button in ("ok", "home", "prev", "next"):
        panel = NullPanel()
        engine, source = _engine(image_dir, panel)
        try:
            panel.press(button)
            assert engine._handle_buttons() == button
        finally:
            source.stop()


def test_exit_screen_says_the_tool_is_gone(image_dir):
    """E-paper keeps its last image, so a dead tool must not look live."""
    from ratcatcher.display.panel import NullPanel

    panel = NullPanel()
    engine, source = _engine(image_dir, panel)
    try:
        engine.sample_and_draw()
        engine.draw_exit_screen()

        final = panel.screens[-1]
        assert "EXITED" in final.title
        assert final.full_refresh
        assert any("not live" in line.text for line in final.lines)
    finally:
        source.stop()


# -- Exposure settling ---------------------------------------------------------
#
# A frame captured before auto-exposure has settled is measured at the wrong
# brightness and the metric reads it as a different lens. Camera 1 on this Pi
# opens at 193 us / luma 121 and scores 57%; a second later it has settled on
# 303 us / luma 94 and the same lens scores 41%. The tool used to report the
# first of those, which is the one mistake it must not make.

from ratcatcher.camera.focus import (  # noqa: E402
    SETTLE_SAMPLES,
    exposure_sample,
    exposure_settled,
)


def test_exposure_sample_reads_the_three_controls():
    assert exposure_sample(
        {"ExposureTime": 303, "AnalogueGain": 1.0, "DigitalGain": 1.08}
    ) == (303.0, 1.0, 1.08)


def test_exposure_sample_is_none_without_metadata():
    assert exposure_sample(None) is None
    assert exposure_sample({}) is None


def test_exposure_sample_is_none_when_a_control_is_missing():
    """A partial sample must not be treated as agreement."""
    assert exposure_sample({"ExposureTime": 303, "AnalogueGain": 1.0}) is None


def test_not_settled_before_enough_samples():
    stable = (303.0, 1.0, 1.08)
    assert not exposure_settled([stable] * (SETTLE_SAMPLES - 1))


def test_settled_when_samples_agree():
    assert exposure_settled([(303.0, 1.0, 1.08)] * SETTLE_SAMPLES)


def test_not_settled_while_exposure_is_still_moving():
    """The real camera-1 sequence: 193 us jumping to 303 us."""
    assert not exposure_settled(
        [(193.0, 1.0, 2.57), (303.0, 1.0, 1.08), (303.0, 1.0, 1.08)]
    )


def test_settled_once_the_jump_has_passed():
    assert exposure_settled(
        [(193.0, 1.0, 2.57)] + [(303.0, 1.0, 1.08)] * SETTLE_SAMPLES
    )


def test_small_drift_still_counts_as_settled():
    """Lux and gain jitter by a fraction of a percent forever."""
    assert exposure_settled(
        [(303.0, 1.0, 1.0796), (303.0, 1.0, 1.0797), (303.0, 1.0, 1.0796)]
    )


def test_gain_change_alone_blocks_settling():
    """Exposure can hold steady while gain is still being traded against it."""
    assert not exposure_settled(
        [(303.0, 1.0, 2.57), (303.0, 1.0, 1.08), (303.0, 1.0, 1.08)]
    )


def test_none_samples_are_not_counted_as_agreement():
    assert not exposure_settled([None] * (SETTLE_SAMPLES + 2))


# -- Reading settling ----------------------------------------------------------
#
# Exposure settling is necessary and not sufficient. On camera 1 every control
# freezes within 0.7 s while the focus reading goes on falling from 56% to 32%
# over the next three seconds: the ISP's temporal denoise is converging, and
# the blur ratio counts the early frames' sensor noise as detail.

from ratcatcher.camera.focus import (  # noqa: E402
    READING_TOLERANCE_PCT,
    readings_settled,
)


def test_not_settled_before_enough_readings():
    assert not readings_settled([32.0] * (SETTLE_SAMPLES - 1))


def test_settled_when_readings_agree():
    assert readings_settled([32.0] * SETTLE_SAMPLES)


def test_not_settled_during_the_denoise_decay():
    """The real camera-1 sequence, sampled while it was still falling."""
    assert not readings_settled([56.1, 46.9, 40.4])
    assert not readings_settled([46.9, 40.4, 36.3])
    assert not readings_settled([40.4, 36.3, 34.2])


def test_settled_once_the_decay_has_flattened():
    """The tail of that same sequence, where the true value lives."""
    assert readings_settled([32.7, 32.4, 32.9])


def test_jitter_inside_the_tolerance_still_settles():
    assert readings_settled([32.0, 32.0 + READING_TOLERANCE_PCT * 0.9, 32.0])


def test_jitter_outside_the_tolerance_does_not():
    assert not readings_settled([32.0, 32.0 + READING_TOLERANCE_PCT * 1.5, 32.0])


def test_a_sharp_camera_settles_immediately():
    """Camera 0 drifts about two points, which is why this hid for so long."""
    assert readings_settled([67.3, 66.5, 67.0])


def test_button_triggered_reading_uses_a_full_refresh(image_dir, tmp_path):
    """Partial updates on this panel resolve over two writes.

    That is what made every other press show noise: the first press left
    a half-formed image and the second completed it. Checked against the
    encoded bytes a real FilePanel wrote, not a recorded call, so this
    asserts what would actually go down the wire.
    """
    import json

    from ratcatcher.display.panel import FilePanel

    path = tmp_path / "panel.ndjson"
    panel = FilePanel(path)
    panel.open()
    engine, source = _engine(image_dir, panel)
    try:
        for _ in range(3):
            engine.sample_and_draw(trigger="ok")
    finally:
        source.stop()
        panel.close()

    frames = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    assert frames, "nothing was written to the panel"
    assert all(f.get("full") for f in frames), (
        "a button-triggered reading must ask for a full refresh; got "
        f"{[f.get('full') for f in frames]}"
    )
