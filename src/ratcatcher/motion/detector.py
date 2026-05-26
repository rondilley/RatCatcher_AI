"""Motion detection via OpenCV MOG2 background subtraction.

This module is the first stage of the RatCatcher AI pipeline: it finds
regions of a video frame where motion is occurring, filters them by
minimum area, applies a per-grid-cell cooldown to suppress duplicate
detections, and returns bounding boxes in the original frame coordinate
system.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from ratcatcher.config import MotionConfig


@dataclass
class MotionRegion:
    """A rectangular region where motion was detected.

    Coordinates are expressed in the original (full-resolution) frame
    space, not in the down-scaled processing space.
    """

    x: int       # top-left x in original frame coords
    y: int       # top-left y in original frame coords
    width: int   # width in original frame coords
    height: int  # height in original frame coords
    area: float  # contour area normalised to 0-1 of frame area


class MotionDetector:
    """Detect motion in video frames using MOG2 background subtraction.

    Parameters
    ----------
    config:
        A ``MotionConfig`` instance that controls background subtractor
        parameters, morphological kernel sizes, minimum contour area,
        cooldown duration, and the processing resolution.
    """

    # Number of cells on each axis for the cooldown grid.
    _GRID_COLS = 8
    _GRID_ROWS = 6

    def __init__(self, config: MotionConfig) -> None:
        self._config = config

        self._bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=config.history,
            varThreshold=config.var_threshold,
            detectShadows=config.detect_shadows,
        )

        self._erode_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (config.erode_kernel, config.erode_kernel),
        )
        self._dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (config.dilate_kernel, config.dilate_kernel),
        )

        # Per-cell cooldown timestamps.  Each cell stores the last time
        # a detection was emitted whose centre fell inside that cell.
        self._cooldown_grid: dict[tuple[int, int], float] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> list[MotionRegion]:
        """Run motion detection on a single video frame.

        Parameters
        ----------
        frame:
            BGR image as a NumPy array (any resolution).

        Returns
        -------
        A (possibly empty) list of ``MotionRegion`` objects for regions
        that passed both the area and cooldown filters.
        """
        orig_h, orig_w = frame.shape[:2]
        proc_w = self._config.process_width
        proc_h = self._config.process_height

        # --- 1. Resize to processing resolution ---
        small = cv2.resize(
            frame, (proc_w, proc_h), interpolation=cv2.INTER_LINEAR
        )

        # --- 2. Background subtraction ---
        fg_mask = self._bg_sub.apply(
            small, learningRate=self._config.learning_rate
        )

        # --- 3. Morphological cleanup ---
        fg_mask = cv2.erode(fg_mask, self._erode_kernel, iterations=1)
        fg_mask = cv2.dilate(fg_mask, self._dilate_kernel, iterations=1)

        # --- 4. Find contours ---
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # --- 5. Filter by area, apply cooldown, scale back ---
        proc_area = proc_w * proc_h
        min_area_abs = self._config.min_area_pct * proc_area

        scale_x = orig_w / proc_w
        scale_y = orig_h / proc_h
        orig_area = orig_w * orig_h

        now = time.monotonic()
        results: list[MotionRegion] = []

        for cnt in contours:
            cnt_area = cv2.contourArea(cnt)
            if cnt_area < min_area_abs:
                continue

            bx, by, bw, bh = cv2.boundingRect(cnt)

            # Centre of the bounding box in processing coords, used for
            # the grid-based cooldown check.
            cx = bx + bw / 2
            cy = by + bh / 2
            cell = self._grid_cell(cx, cy, proc_w, proc_h)

            last_fire = self._cooldown_grid.get(cell, 0.0)
            if (now - last_fire) < self._config.cooldown_seconds:
                continue

            self._cooldown_grid[cell] = now

            # Scale bounding box back to original resolution.
            ox = int(round(bx * scale_x))
            oy = int(round(by * scale_y))
            ow = int(round(bw * scale_x))
            oh = int(round(bh * scale_y))

            # Clamp to frame bounds.
            ox = max(0, min(ox, orig_w - 1))
            oy = max(0, min(oy, orig_h - 1))
            ow = min(ow, orig_w - ox)
            oh = min(oh, orig_h - oy)

            normalised_area = cnt_area / proc_area

            results.append(
                MotionRegion(
                    x=ox,
                    y=oy,
                    width=ow,
                    height=oh,
                    area=normalised_area,
                )
            )

        return results

    def reset(self) -> None:
        """Clear the background model and cooldown state.

        After calling this the next few frames will be treated as a
        fresh learning period by MOG2.
        """
        self._bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=self._config.history,
            varThreshold=self._config.var_threshold,
            detectShadows=self._config.detect_shadows,
        )
        self._cooldown_grid.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _grid_cell(
        self, cx: float, cy: float, frame_w: int, frame_h: int
    ) -> tuple[int, int]:
        """Map a point to a cooldown-grid cell index.

        The frame is divided into ``_GRID_COLS x _GRID_ROWS`` cells.
        Two detections whose centres land in the same cell are considered
        "the same region" for cooldown purposes.
        """
        col = int(cx / frame_w * self._GRID_COLS)
        row = int(cy / frame_h * self._GRID_ROWS)
        col = max(0, min(col, self._GRID_COLS - 1))
        row = max(0, min(row, self._GRID_ROWS - 1))
        return (col, row)
