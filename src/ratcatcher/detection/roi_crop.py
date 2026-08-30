"""Native-resolution detection windows cut around motion.

The detector takes a fixed input (640x640 for the shipped models) and
both backends reach it by stretching whatever frame they are handed.
That is fine when the animal fills the frame and ruinous when it does
not: squeezing 1920x1080 into 640x640 divides object height by 1.7 and
width by 3, which turns a 29 px finch into roughly 17x10 px.  Measured
against the val set, detection recall collapses below about 100 px of
object height and reaches zero at 50 px, so at that framing every target
species is invisible.

Cutting a 640x640 window straight out of a full-resolution frame skips
the stretch entirely.  With the camera reading out its full 4056x3040
sensor the same finch arrives at the network about 62 px tall, which is
back inside the range the detector can work in.

The functions here only decide *where* to cut.  They are pure and take no
frames, which is what lets the geometry be tested without a camera.
"""

from __future__ import annotations

from dataclasses import dataclass

from ratcatcher.motion.detector import MotionRegion


@dataclass(frozen=True)
class CropWindow:
    """A rectangle to cut from the capture frame, in capture pixels."""

    x: int
    y: int
    width: int
    height: int

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)


def _rect_of(region: MotionRegion, scale: float) -> tuple[int, int, int, int]:
    """A motion region in capture coordinates."""
    return (
        int(round(region.x * scale)),
        int(round(region.y * scale)),
        int(round(region.width * scale)),
        int(round(region.height * scale)),
    )


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw <= bx or bx + bw <= ax or ay + ah <= by or by + bh <= ay)


def _union(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = min(ax, bx), min(ay, by)
    x1, y1 = max(ax + aw, bx + bw), max(ay + ah, by + bh)
    return (x0, y0, x1 - x0, y1 - y0)


def plan_windows(
    regions: list[MotionRegion],
    frame_width: int,
    frame_height: int,
    window: int,
    max_windows: int = 2,
    region_scale: float = 1.0,
) -> list[CropWindow]:
    """Choose detection windows covering *regions*.

    Parameters
    ----------
    regions:
        Motion regions, in the coordinate space their detector reported.
    frame_width, frame_height:
        Size of the capture frame the windows will be cut from.
    window:
        Desired edge length, in capture pixels.  Normally the detector's
        input size, so that no rescaling happens at all.
    max_windows:
        Cap on the number returned.  Each window costs a full inference,
        so the largest motion regions win and the rest are dropped.
    region_scale:
        Multiplier taking *regions* into capture coordinates, for when
        motion ran on a smaller frame than the one being cut.

    Returns
    -------
    Windows clamped inside the frame, largest motion first.  A region
    bigger than *window* yields a window bigger than *window*; the caller
    is expected to let the detector rescale that one, which is still a
    tighter crop than the whole frame.
    """
    if not regions or window <= 0:
        return []

    edge = min(window, frame_width, frame_height)

    # Merge overlapping regions first.  Two contours on one animal --
    # common when a limb is separated from a body by the morphological
    # cleanup -- would otherwise spend two of the available windows on
    # the same subject.
    rects = [_rect_of(r, region_scale) for r in regions]
    merged: list[tuple[int, int, int, int]] = []
    for rect in sorted(rects, key=lambda r: r[2] * r[3], reverse=True):
        for i, seen in enumerate(merged):
            if _overlaps(rect, seen):
                merged[i] = _union(rect, seen)
                break
        else:
            merged.append(rect)

    merged.sort(key=lambda r: r[2] * r[3], reverse=True)

    windows: list[CropWindow] = []
    for rx, ry, rw, rh in merged[:max_windows]:
        # Grow to at least the window size, centred on the region, so a
        # small animal keeps context around it rather than being cropped
        # to its own bounding box.
        w = max(edge, min(rw, frame_width))
        h = max(edge, min(rh, frame_height))

        cx = rx + rw / 2
        cy = ry + rh / 2
        x = int(round(cx - w / 2))
        y = int(round(cy - h / 2))

        # Clamp inside the frame without shrinking the window: slide it
        # back in, so an animal at the frame edge still gets a full-size
        # window rather than a sliver.
        x = max(0, min(x, frame_width - w))
        y = max(0, min(y, frame_height - h))

        windows.append(CropWindow(x=x, y=y, width=w, height=h))

    return windows
