"""A phone-sized viewer for setting a lens by eye.

The blur ratio in ``camera/focus.py`` answers "is this sharper than the
last reading" well enough to hill-climb with, and it is unreliable in
exactly the place where the help is needed most. A badly defocused frame
holds almost no real detail, so what the metric measures is mostly sensor
noise -- which is why camera 1, at roughly 35%, produced numbers that
moved without the lens moving. A number that cannot be trusted while it
is being used to make a decision is worse than no number.

So this serves the pixels instead and lets the eye decide. The score
stays on the page as telemetry, next to the exposure and light readings
that say whether the frame is worth judging at all, but it is no longer
what the tool is for.

Two things make it usable on a phone at the end of a garden:

* **The crop is 1:1.** Focus lives in the highest spatial frequencies,
  and any downscale is a low-pass filter -- a 4056-wide frame fitted to a
  phone screen has had exactly the detail being judged averaged away, and
  looks acceptable at every lens position. The default view is therefore
  a native-resolution window, not the whole frame.

* **The window can be moved.** Focus is not uniform: measured across this
  installation the centre and the corners differ by tens of points, so a
  lens set on the middle of the frame can leave the feeders soft. A
  three-by-three grid selector is enough to check that without a
  pan-and-zoom gesture on a wet phone.

Only the standard library and OpenCV, both already dependencies. The
server binds to every interface so a phone on the same network can reach
it, and it serves camera images with no authentication -- it is a bring-up
tool for a private network, not something to leave running.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Sequence
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from ratcatcher.camera.capture import CameraSource
from ratcatcher.camera.focus import DEFAULT_CEILING, analyse_frame

logger = logging.getLogger(__name__)

DEFAULT_PORT = 8080

# Edge of the native-resolution window, in sensor pixels. Chosen to be
# legible on a phone held at arm's length without scrolling, and small
# enough that the JPEG encode stays cheap enough to poll a few times a
# second.
CROP_EDGE = 720

# JPEG quality for the crop. High, deliberately: this is the one image
# where compression artefacts would be mistaken for the thing being
# measured.
CROP_QUALITY = 92

# The fitted whole-frame view exists for aiming, not for focus, so it can
# be small and cheap.
FIT_WIDTH = 960
FIT_QUALITY = 75

# How often the page fetches a new view, in milliseconds. A lens is
# turned and then looked at, so this is a viewer rather than a video
# feed: a few seconds is plenty, and every refresh is a full-resolution
# crop plus a JPEG encode per camera. The tighter interval this started
# with bought nothing and cost CPU on a board that is also running two
# cameras.
DEFAULT_REFRESH_MS = 4000

_GRID = ("tl", "tc", "tr", "ml", "mc", "mr", "bl", "bc", "br")


def crop_window(
    width: int, height: int, region: str, edge: int = CROP_EDGE
) -> tuple[int, int, int, int]:
    """The native-resolution window for a grid position, clamped to frame."""
    edge = max(1, min(edge, width, height))
    row, col = (region[0] if len(region) == 2 else "m"), (
        region[1] if len(region) == 2 else "c"
    )

    if col == "l":
        x = 0
    elif col == "r":
        x = width - edge
    else:
        x = (width - edge) // 2

    if row == "t":
        y = 0
    elif row == "b":
        y = height - edge
    else:
        y = (height - edge) // 2

    return max(0, x), max(0, y), edge, edge


def _metadata_of(camera: CameraSource) -> dict[str, Any] | None:
    getter = getattr(camera, "capture_metadata", None)
    if getter is None:
        return None
    try:
        return getter()
    except Exception:  # pragma: no cover -- a camera that cannot answer
        return None


class FocusWebServer:
    """Serves live camera windows and their telemetry over HTTP."""

    def __init__(
        self,
        cameras: Sequence[CameraSource],
        *,
        port: int = DEFAULT_PORT,
        ceiling: float = DEFAULT_CEILING,
        host: str = "0.0.0.0",
        refresh_ms: int = DEFAULT_REFRESH_MS,
    ) -> None:
        self._cameras = list(cameras)
        self._ceiling = ceiling
        self._refresh_ms = max(250, int(refresh_ms))
        handler = self._make_handler()

        class _Server(ThreadingHTTPServer):
            # Default is 5. A browser opens several parallel connections
            # for the page plus one image per camera, and a phone
            # reconnecting on a weak link adds more; at 5 the surplus are
            # refused rather than queued, which looks exactly like the
            # server being down.
            request_queue_size = 64
            allow_reuse_address = True

        self._server = _Server((host, port), handler)
        self._server.daemon_threads = True
        self._thread: threading.Thread | None = None

    @property
    def port(self) -> int:
        return self._server.server_address[1]

    @property
    def urls(self) -> list[str]:
        """Every address a phone might reach this on."""
        out = [f"http://{lan_address()}:{self.port}/"]
        return out

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="focus-web",
            daemon=True,
        )
        self._thread.start()
        logger.info("Focus viewer on %s", ", ".join(self.urls))

    def stop(self) -> None:
        # shutdown() blocks until serve_forever() acknowledges it, so on a
        # server that was never started it waits for a loop that will
        # never run. Closing a server that was only constructed has to
        # work: it is what an aborted start-up does.
        if self._thread is not None:
            self._server.shutdown()
            self._thread.join(timeout=5.0)
            self._thread = None
        self._server.server_close()

    # -- internals ---------------------------------------------------------

    def _frame(self, index: int) -> np.ndarray | None:
        if not 0 <= index < len(self._cameras):
            return None
        ok, frame = self._cameras[index].read()
        return frame if ok else None

    def _telemetry(self) -> list[dict[str, Any]]:
        rows = []
        for index, camera in enumerate(self._cameras):
            frame = self._frame(index)
            if frame is None:
                rows.append({"camera": index, "state": "NO IMAGE"})
                continue
            reading = analyse_frame(
                frame, index, metadata=_metadata_of(camera), ceiling=self._ceiling
            )
            rows.append(
                {
                    "camera": index,
                    "state": reading.state,
                    "focus_pct": round(reading.focus_pct, 1)
                    if reading.measurable
                    else None,
                    "blur_ratio": round(reading.blur_ratio, 3),
                    "luma_mean": round(reading.luma_mean, 1),
                    "luma_std": round(reading.luma_std, 1),
                    "clip_high_pct": round(reading.clip_high_pct, 2),
                    "lux": round(reading.lux) if reading.lux is not None else None,
                    "exposure_us": reading.exposure_us,
                    "analogue_gain": reading.analogue_gain,
                    "size": [frame.shape[1], frame.shape[0]],
                }
            )
        return rows

    def _encode(self, index: int, mode: str, region: str, edge: int) -> bytes | None:
        frame = self._frame(index)
        if frame is None:
            return None

        if mode == "fit":
            height, width = frame.shape[:2]
            scale = FIT_WIDTH / float(width)
            view = cv2.resize(
                frame,
                (FIT_WIDTH, max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
            params = [cv2.IMWRITE_JPEG_QUALITY, FIT_QUALITY]
        else:
            height, width = frame.shape[:2]
            x, y, w, h = crop_window(width, height, region, edge)
            view = frame[y : y + h, x : x + w]
            params = [cv2.IMWRITE_JPEG_QUALITY, CROP_QUALITY]

        ok, buf = cv2.imencode(".jpg", view, params)
        return buf.tobytes() if ok else None

    def _make_handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            # One response per connection, deliberately.
            #
            # With keep-alive, a phone that sleeps or roams between
            # access points leaves its socket ESTABLISHED on this side
            # with a thread parked in wfile.write() forever. Those
            # corpses then occupy the browser's six-connections-per-host
            # budget, so the next page load waits on a socket the server
            # still believes is live -- which presents as "the server
            # stopped accepting connections" while it is in fact idle and
            # answering instantly on loopback.
            #
            # At one request per camera every few seconds a fresh TCP
            # handshake costs nothing worth measuring, and closing after
            # each response makes the whole class of half-dead mobile
            # connections impossible rather than merely rarer.
            protocol_version = "HTTP/1.0"

            # Belt and braces: a client that stops reading mid-response
            # must not park a thread indefinitely. socketserver applies
            # this to the connection, so it bounds reads and writes both.
            timeout = 15

            def log_message(self, *args: Any) -> None:
                # One line per image per camera would bury everything
                # else the session prints.
                pass

            def handle_one_request(self) -> None:
                # A timed-out or reset client is ordinary on wifi, not an
                # error worth a traceback on the console.
                try:
                    super().handle_one_request()
                except (TimeoutError, ConnectionError, OSError) as exc:
                    logger.debug("Focus viewer client went away: %s", exc)
                    self.close_connection = True

            def _send(self, code, body: bytes, content_type: str, cache=False):
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                if not cache:
                    self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                query = parse_qs(parsed.query)

                if parsed.path in ("/", "/index.html"):
                    body = render_page(
                        len(server._cameras), refresh_ms=server._refresh_ms
                    ).encode("utf-8")
                    self._send(200, body, "text/html; charset=utf-8")
                    return

                if parsed.path == "/telemetry":
                    body = json.dumps(server._telemetry()).encode("utf-8")
                    self._send(200, body, "application/json")
                    return

                if parsed.path == "/view":
                    try:
                        index = int(query.get("cam", ["0"])[0])
                        edge = int(query.get("edge", [str(CROP_EDGE)])[0])
                    except ValueError:
                        self._send(400, b"bad parameter", "text/plain")
                        return
                    mode = query.get("mode", ["crop"])[0]
                    region = query.get("region", ["mc"])[0]
                    if region not in _GRID:
                        region = "mc"
                    edge = max(64, min(edge, 2000))

                    jpeg = server._encode(index, mode, region, edge)
                    if jpeg is None:
                        self._send(503, b"no frame", "text/plain")
                        return
                    self._send(200, jpeg, "image/jpeg")
                    return

                self._send(404, b"not found", "text/plain")

        return Handler


def lan_address() -> str:
    """The address a phone on the same network should use.

    Asks the routing table which interface would reach the outside world
    rather than resolving the hostname, which on a Pi commonly answers
    127.0.1.1 and would send the user to their own phone.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 53))
        return str(sock.getsockname()[0])
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def render_page(cameras: int, refresh_ms: int = DEFAULT_REFRESH_MS) -> str:
    """The whole interface: one file, no assets, no framework."""
    panels = "\n".join(
        f"""
    <section class="cam" data-cam="{index}">
      <h2>Camera {index} <span class="tel" id="tel{index}">...</span></h2>
      <img id="img{index}" alt="camera {index}">
    </section>"""
        for index in range(cameras)
    )
    return f"""<!doctype html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RatCatcher focus</title>
<style>
 body{{margin:0;background:#111;color:#eee;font:14px/1.4 system-ui,sans-serif}}
 header{{padding:8px 10px;background:#000;position:sticky;top:0;z-index:2}}
 h2{{font-size:14px;margin:10px 0 4px;font-weight:600}}
 .tel{{font-weight:400;color:#9bd;font-variant-numeric:tabular-nums}}
 img{{width:100%;display:block;background:#000;image-rendering:pixelated}}
 section{{padding:0 6px 10px}}
 button{{background:#222;color:#eee;border:1px solid #444;border-radius:6px;
   padding:7px 9px;font-size:14px;margin:2px}}
 button.on{{background:#2a5d8f;border-color:#4a9}}
 .grid{{display:inline-grid;grid-template-columns:repeat(3,32px);gap:2px;
   vertical-align:middle;margin-left:6px}}
 .grid button{{padding:0;height:28px;margin:0}}
 .warn{{color:#fb6}}
</style></head><body>
<header>
  <button id="mode-crop" class="on">1:1 crop</button>
  <button id="mode-fit">whole frame</button>
  <span class="grid" id="grid"></span>
  <button id="now">refresh now</button>
  <button id="pause">pause</button>
  <span id="rate"></span>
</header>
{panels}
<script>
const N = {cameras};
let mode = "crop", region = "mc", paused = false, every = {refresh_ms};
const rate = document.getElementById("rate");
[2000, 4000, 8000].forEach(ms => {{
  const b = document.createElement("button");
  b.textContent = (ms / 1000) + "s";
  b.dataset.ms = ms;
  if (ms === every) b.className = "on";
  b.onclick = () => {{
    every = ms;
    [...rate.children].forEach(c => c.className = +c.dataset.ms === ms ? "on" : "");
  }};
  rate.appendChild(b);
}});
const grid = document.getElementById("grid");
["tl","tc","tr","ml","mc","mr","bl","bc","br"].forEach(r => {{
  const b = document.createElement("button");
  b.textContent = "\\u00b7"; b.dataset.r = r;
  if (r === region) b.className = "on";
  b.onclick = () => {{
    region = r;
    [...grid.children].forEach(c => c.className = c.dataset.r === r ? "on" : "");
  }};
  grid.appendChild(b);
}});
function setMode(m) {{
  mode = m;
  document.getElementById("mode-crop").className = m === "crop" ? "on" : "";
  document.getElementById("mode-fit").className = m === "fit" ? "on" : "";
}}
document.getElementById("mode-crop").onclick = () => setMode("crop");
document.getElementById("mode-fit").onclick = () => setMode("fit");
document.getElementById("pause").onclick = e => {{
  paused = !paused;
  e.target.className = paused ? "on" : "";
  e.target.textContent = paused ? "resume" : "pause";
}};
function draw() {{
    const t = Date.now();
    for (let i = 0; i < N; i++) {{
      document.getElementById("img" + i).src =
        `/view?cam=${{i}}&mode=${{mode}}&region=${{region}}&t=${{t}}`;
    }}
    fetch("/telemetry").then(r => r.json()).then(rows => {{
      rows.forEach(row => {{
        const el = document.getElementById("tel" + row.camera);
        if (!el) return;
        if (row.state === "NO IMAGE") {{ el.textContent = "no image"; return; }}
        const f = row.focus_pct === null ? "--" : row.focus_pct.toFixed(0) + "%";
        let s = `${{f}}  lux ${{row.lux}}  exp ${{row.exposure_us}}us  ${{row.state}}`;
        if (row.clip_high_pct > 5) s += "  CLIPPING";
        el.textContent = s;
        el.className = row.clip_high_pct > 5 ? "tel warn" : "tel";
      }});
    }}).catch(() => {{}});
}}
document.getElementById("now").onclick = draw;
function tick() {{
  if (!paused) draw();
  setTimeout(tick, every);
}}
draw();
tick();
</script>
</body></html>"""
