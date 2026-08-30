"""Tests for the focus web viewer.

A real ThreadingHTTPServer on a real port, served from a real FileSource,
answered with real HTTP requests and decoded with real cv2. The one thing
worth guarding above all is that the 1:1 crop is genuinely 1:1 -- a view
that quietly rescales would be a low-pass filter over exactly the detail
the tool exists to show, and would look plausible at every lens position.
"""

from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pytest

from ratcatcher.camera.capture import FileSource
from ratcatcher.web.focus_server import (
    CROP_EDGE,
    FIT_WIDTH,
    FocusWebServer,
    crop_window,
    render_page,
)


def _scene(width=1280, height=960):
    """A frame with detail everywhere and identifiable corners.

    Structured rather than random noise: these are written to disk many
    times over, and incompressible noise filled two gigabytes of the
    Pi's root filesystem the first time this suite ran.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Fine checkerboard: high spatial frequency, but compresses.
    img[::4, :] = 200
    img[:, ::4] = 160
    cv2.circle(img, (width // 2, height // 2), 180, (90, 180, 240), 9)
    # Corner markers, so a returned window can be located unambiguously.
    img[0:80, 0:80] = 255
    img[0:80, width - 80 :] = 0
    img[height - 80 :, 0:80] = 0
    img[height - 80 :, width - 80 :] = 255
    return img


@pytest.fixture
def server(tmp_path):
    # FileSource consumes one image per read and stops when they run out,
    # so a live view needs a supply of them. Identical frames, so a crop
    # taken on one request is comparable with one taken on the next.
    # Eight is comfortably more than any test consumes; each request
    # takes one. Kept small on purpose -- this runs on a Pi with a few
    # gigabytes free.
    scene = _scene()
    for index in range(8):
        cv2.imwrite(
            str(tmp_path / f"frame_{index:03d}.jpg"),
            scene,
            [cv2.IMWRITE_JPEG_QUALITY, 85],
        )
    source = FileSource(str(tmp_path))
    source.start()
    srv = FocusWebServer([source], port=0)
    srv.start()
    yield srv
    srv.stop()
    source.stop()


def _get(server, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{server.port}{path}", timeout=10) as r:
        return r.status, r.read(), r.headers.get("Content-Type")


def _image(server, path):
    status, body, ctype = _get(server, path)
    assert status == 200
    assert ctype == "image/jpeg"
    return cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)


# -- geometry, no server needed ------------------------------------------------


def test_grid_positions_are_inside_the_frame():
    for region in ("tl", "tc", "tr", "ml", "mc", "mr", "bl", "bc", "br"):
        x, y, w, h = crop_window(4056, 3040, region)
        assert 0 <= x and 0 <= y
        assert x + w <= 4056 and y + h <= 3040
        assert (w, h) == (CROP_EDGE, CROP_EDGE)


def test_grid_corners_are_actually_at_the_corners():
    assert crop_window(4056, 3040, "tl")[:2] == (0, 0)
    assert crop_window(4056, 3040, "br")[:2] == (4056 - CROP_EDGE, 3040 - CROP_EDGE)


def test_crop_shrinks_to_fit_a_small_frame():
    x, y, w, h = crop_window(200, 150, "mc")
    assert (w, h) == (150, 150)
    assert x + w <= 200 and y + h <= 150


# -- the server ----------------------------------------------------------------


def test_index_is_served(server):
    status, body, ctype = _get(server, "/")
    assert status == 200
    assert "text/html" in ctype
    assert b"1:1 crop" in body


def test_crop_is_returned_at_native_resolution(server):
    """The whole point: no rescaling of the view used to judge focus."""
    img = _image(server, "/view?cam=0&mode=crop&region=mc")
    assert img.shape[0] == CROP_EDGE and img.shape[1] == CROP_EDGE


def test_crop_region_actually_moves_the_window(server):
    """A grid that does not move the window would be worse than none."""
    top_left = _image(server, "/view?cam=0&mode=crop&region=tl")
    bottom_right = _image(server, "/view?cam=0&mode=crop&region=br")
    # The scene's corner markers differ, so the two windows must too.
    assert top_left[:40, :40].mean() > 200      # white marker
    assert bottom_right[-40:, -40:].mean() > 200
    assert not np.array_equal(top_left, bottom_right)


def test_crop_edge_is_configurable_and_bounded(server):
    small = _image(server, "/view?cam=0&mode=crop&edge=256")
    assert small.shape[:2] == (256, 256)
    # Absurd values are clamped rather than serving the whole sensor.
    huge = _image(server, "/view?cam=0&mode=crop&edge=99999")
    assert max(huge.shape[:2]) <= 2000


def test_fit_view_is_downscaled_for_aiming(server):
    img = _image(server, "/view?cam=0&mode=fit")
    assert img.shape[1] == FIT_WIDTH
    assert img.shape[1] < 1280


def test_telemetry_reports_the_reading(server):
    status, body, ctype = _get(server, "/telemetry")
    assert status == 200
    assert "application/json" in ctype
    rows = json.loads(body)
    assert len(rows) == 1
    row = rows[0]
    assert row["camera"] == 0
    assert row["size"] == [1280, 960]
    for key in ("state", "blur_ratio", "luma_mean", "clip_high_pct"):
        assert key in row


def test_unknown_region_falls_back_rather_than_failing(server):
    img = _image(server, "/view?cam=0&mode=crop&region=nonsense")
    assert img.shape[:2] == (CROP_EDGE, CROP_EDGE)


def test_bad_camera_index_is_reported_not_crashed(server):
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(server, "/view?cam=9")
    assert exc.value.code == 503


def test_bad_parameter_is_a_client_error(server):
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(server, "/view?cam=notanumber")
    assert exc.value.code == 400


def test_unknown_path_is_404(server):
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(server, "/nope")
    assert exc.value.code == 404


def test_images_are_not_cached(server):
    """A phone caching the view would show a stale lens position."""
    url = f"http://127.0.0.1:{server.port}/view?cam=0"
    with urllib.request.urlopen(url, timeout=10) as r:
        assert "no-store" in r.headers.get("Cache-Control", "")


def test_page_renders_one_panel_per_camera():
    assert render_page(1).count('class="cam"') == 1
    assert render_page(2).count('class="cam"') == 2


# -- Refresh cadence -----------------------------------------------------------
#
# A lens is turned and then looked at, so this is a viewer rather than a video
# feed. Every refresh costs a full-resolution crop and a JPEG encode per
# camera, on a board already running two of them.


def test_default_refresh_is_seconds_not_milliseconds():
    from ratcatcher.web.focus_server import DEFAULT_REFRESH_MS

    assert 3000 <= DEFAULT_REFRESH_MS <= 5000


def test_page_carries_the_configured_interval(tmp_path):
    from ratcatcher.web.focus_server import render_page

    assert "every = 4000" in render_page(2, refresh_ms=4000)
    assert "every = 9000" in render_page(2, refresh_ms=9000)


def test_server_serves_its_configured_interval(tmp_path):
    scene = _scene()
    for index in range(4):
        cv2.imwrite(str(tmp_path / f"frame_{index:03d}.jpg"), scene)
    source = FileSource(str(tmp_path))
    source.start()
    srv = FocusWebServer([source], port=0, refresh_ms=6000)
    srv.start()
    try:
        status, body, _ = _get(srv, "/")
        assert status == 200
        assert b"every = 6000" in body
    finally:
        srv.stop()
        source.stop()


def test_absurdly_fast_refresh_is_floored():
    """Guards against a typo turning the viewer into a CPU-bound video feed."""
    from ratcatcher.web.focus_server import FocusWebServer as S

    srv = S([], port=0, refresh_ms=1)
    try:
        assert srv._refresh_ms >= 250
    finally:
        srv.stop()


def test_stopping_a_server_that_never_started_returns():
    """An aborted start-up must still be able to release the port.

    shutdown() waits for serve_forever() to acknowledge it, so calling it
    on a server that was only constructed hangs forever.
    """
    srv = FocusWebServer([], port=0)
    srv.stop()          # must not block


# -- Connection hygiene --------------------------------------------------------
#
# A phone that sleeps or roams mid-response used to leave its socket
# ESTABLISHED here with a thread parked in wfile.write() forever. Those
# corpses occupied the browser's six-connections-per-host budget, so the next
# page load waited on a socket the server still believed was live -- which
# looks exactly like the server having stopped accepting connections, while it
# is idle and answering instantly on loopback.

import socket  # noqa: E402
import threading  # noqa: E402


def test_response_closes_the_connection(server):
    with urllib.request.urlopen(
        f"http://127.0.0.1:{server.port}/view?cam=0", timeout=10
    ) as r:
        assert r.headers.get("Connection", "").lower() == "close"


def test_a_client_that_vanishes_does_not_leak_a_thread(server):
    """Connect, ask, then walk away without reading the answer."""
    before = threading.active_count()

    for _ in range(12):
        sock = socket.create_connection(("127.0.0.1", server.port), timeout=5)
        sock.sendall(b"GET /view?cam=0 HTTP/1.0\r\n\r\n")
        # Read nothing at all, then abort the connection outright: RST
        # rather than a polite FIN, which is what a phone dropping off
        # wifi actually looks like.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, b"\x01\x00\x00\x00\x00\x00\x00\x00")
        sock.close()

    # The server must still answer normally afterwards.
    status, _, _ = _get(server, "/telemetry")
    assert status == 200

    # And it must not be accumulating a thread per abandoned client.
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline and threading.active_count() > before + 4:
        time.sleep(0.25)
    assert threading.active_count() <= before + 4, (
        f"threads grew from {before} to {threading.active_count()} after 12 "
        "abandoned connections"
    )


def test_many_sequential_requests_keep_working(tmp_path):
    """The symptom was the server appearing to stop accepting connections.

    Its own small frames: FileSource consumes one per request, and forty
    of the full-size fixture would be a waste of a Pi's root filesystem.
    """
    scene = _scene(320, 240)
    for index in range(50):
        cv2.imwrite(str(tmp_path / f"f{index:03d}.jpg"), scene)
    source = FileSource(str(tmp_path))
    source.start()
    srv = FocusWebServer([source], port=0)
    srv.start()
    try:
        for _ in range(40):
            status, _, _ = _get(srv, "/view?cam=0")
            assert status == 200
    finally:
        srv.stop()
        source.stop()
