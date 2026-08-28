"""Text preview of the panel screen.

The panel itself draws the screen: the host sends numbers, not pixels.
This module reproduces the same layout as characters so that the layout
can be seen, and tested, with no hardware attached.

It is a preview, not a simulator. The panel mixes an 8-pixel and a
6-pixel font on a 248 by 122 frame, which no fixed character grid can
reproduce exactly. Column order, field widths and rounding are faithful;
pixel positions are not.

The pixel geometry the firmware uses is recorded in ``PIXEL_LAYOUT``
below. ``firmware/ratcatcher_panel/ratcatcher_panel.ino`` holds the same
numbers. They are two statements of one design and must be changed
together.
"""

from __future__ import annotations

from ratcatcher.display.protocol import StatusFrame

# Character width of the preview.
WIDTH = 41

# Right-hand edge of each number column, in preview characters.
_COL_EYE = 21
_COL_EAR = 29
_COL_ALL = 37

_ROW_LABELS = (("bird", "BIRDS"), ("rodent", "RODENTS"), ("other", "OTHER"))

# The firmware geometry, in pixels on the 248 by 122 landscape frame.
# Kept here so that the two layouts can be compared side by side.
PIXEL_LAYOUT = {
    "frame": (248, 122),
    "font_header": 16,
    "font_row": 16,
    "font_small": 12,
    "header_y": 1,
    "rule_top_y": 19,
    "colhead_y": 21,
    "row_y": (35, 53, 71),
    "rule_bottom_y": 89,
    "footer_y": (92, 107),
    "col_right_px": (130, 180, 238),
    "label_x": 2,
}


def render_text(frame: StatusFrame) -> str:
    """Render one status frame as the panel will show it."""
    lines = [
        _header(frame),
        "-" * WIDTH,
        _column_heads(),
    ]
    for key, label in _ROW_LABELS:
        lines.append(_count_row(label, frame.counts.get(key, (0, 0))))
    lines.append("-" * WIDTH)
    lines.append(_last_line(frame))
    lines.append(_system_line(frame))
    return "\n".join(line[:WIDTH].ljust(WIDTH) for line in lines)


def render_box(frame: StatusFrame) -> str:
    """Render the frame inside a border, for the command line."""
    body = render_text(frame).split("\n")
    rule = "+" + "-" * (WIDTH + 2) + "+"
    return "\n".join([rule] + [f"| {line} |" for line in body] + [rule])


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _header(frame: StatusFrame) -> str:
    left = f"RatCatcher {frame.window}"
    right = f"{frame.clock}  {frame.state}"
    pad = max(1, WIDTH - len(left) - len(right))
    return left + " " * pad + right


def _column_heads() -> str:
    line = [" "] * WIDTH
    _place(line, "EYE", _COL_EYE)
    _place(line, "EAR", _COL_EAR)
    _place(line, "ALL", _COL_ALL)
    return "".join(line)


def _count_row(label: str, counts: tuple[int, int]) -> str:
    seen, heard = counts
    line = [" "] * WIDTH
    for index, char in enumerate(f" {label}"):
        if index < WIDTH:
            line[index] = char
    _place(line, str(seen), _COL_EYE)
    _place(line, str(heard), _COL_EAR)
    _place(line, str(seen + heard), _COL_ALL)
    return "".join(line)


def _last_line(frame: StatusFrame) -> str:
    if frame.last is None:
        return "Last  (nothing yet)"
    tail = f"{frame.last.sense}  {frame.last.clock}"
    head = f"Last  {frame.last.name}"
    pad = max(1, WIDTH - len(head) - len(tail))
    return head + " " * pad + tail


def _system_line(frame: StatusFrame) -> str:
    system = frame.system
    parts = [
        f"CAM {system.cameras}",
        f"NPU {'ok' if system.npu else '--'}",
        # CAM and MIC name the devices. EYE and EAR above name the
        # senses. Keeping the two vocabularies apart stops the footer
        # from reading as a fourth count column.
        f"MIC {'on' if system.audio else '--'}",
    ]
    if system.temp_c is not None:
        parts.append(f"{system.temp_c:.0f}C")
    if system.disk_pct is not None:
        parts.append(f"disk {system.disk_pct:.0f}%")
    parts.append(f"up {system.uptime}")
    return "  ".join(parts)


def _place(line: list[str], text: str, right_edge: int) -> None:
    """Write text right-aligned so its last character sits at right_edge."""
    start = right_edge - len(text)
    for index, char in enumerate(text):
        position = start + index
        if 0 <= position < len(line):
            line[position] = char
