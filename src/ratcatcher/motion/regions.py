"""ROI (Region of Interest) masking for motion detection.

Provides polygon-based masking so motion detection can be restricted
to specific areas of the frame (e.g., the feeder zone only).
"""

from __future__ import annotations

import cv2
import numpy as np


class ROIMask:
    """Binary mask built from one or more ROI polygons.

    Polygon vertices are specified in normalized coordinates (0.0 to 1.0),
    so the same ROI definition works across different frame resolutions.
    The mask is rasterized once at construction time for the given
    frame dimensions, then reused on every call to ``apply``.

    Parameters
    ----------
    polygons:
        A list of polygons, where each polygon is a list of ``[x, y]``
        pairs with values in the 0-1 range.  An empty list means the
        entire frame is active.
    frame_width:
        Width of the frames that will be passed to ``apply``.
    frame_height:
        Height of the frames that will be passed to ``apply``.
    """

    def __init__(
        self,
        polygons: list[list[list[float]]],
        frame_width: int,
        frame_height: int,
    ) -> None:
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._polygons_norm = polygons

        if not polygons:
            # No ROI defined -- the whole frame is active.
            self._mask = np.full(
                (frame_height, frame_width), 255, dtype=np.uint8
            )
        else:
            self._mask = np.zeros(
                (frame_height, frame_width), dtype=np.uint8
            )
            for poly_norm in polygons:
                pts = np.array(
                    [
                        [
                            int(round(pt[0] * frame_width)),
                            int(round(pt[1] * frame_height)),
                        ]
                        for pt in poly_norm
                    ],
                    dtype=np.int32,
                )
                cv2.fillPoly(self._mask, [pts], 255)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def mask(self) -> np.ndarray:
        """Return the rasterized binary mask (read-only view)."""
        result: np.ndarray = self._mask
        return result

    def apply(self, foreground_mask: np.ndarray) -> np.ndarray:
        """Bitwise-AND a foreground mask with this ROI mask.

        If the foreground mask has a different size than the ROI mask,
        the ROI mask is resized to match on-the-fly.

        Parameters
        ----------
        foreground_mask:
            Single-channel uint8 mask (e.g., from background subtraction).

        Returns
        -------
        The masked result, same shape and dtype as *foreground_mask*.
        """
        if foreground_mask.shape[:2] != self._mask.shape[:2]:
            resized = cv2.resize(
                self._mask,
                (foreground_mask.shape[1], foreground_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            result: np.ndarray = cv2.bitwise_and(foreground_mask, resized)
            return result

        result = cv2.bitwise_and(foreground_mask, self._mask)
        return result

    def contains_point(self, x: float, y: float) -> bool:
        """Check whether a *normalized* point lies inside any ROI polygon.

        Parameters
        ----------
        x:
            Horizontal coordinate in the 0-1 range.
        y:
            Vertical coordinate in the 0-1 range.

        Returns
        -------
        ``True`` if the point is inside at least one polygon (or if no
        polygons were defined, meaning the full frame is active).
        """
        if not self._polygons_norm:
            return True

        for poly_norm in self._polygons_norm:
            # cv2.pointPolygonTest expects pixel-space coordinates on a
            # contour expressed as an array of shape (N, 1, 2) float32.
            pts = np.array(
                [
                    [pt[0] * self._frame_width, pt[1] * self._frame_height]
                    for pt in poly_norm
                ],
                dtype=np.float32,
            ).reshape(-1, 1, 2)

            px = x * self._frame_width
            py = y * self._frame_height

            # measureDist=False returns +1 inside, 0 on edge, -1 outside.
            if cv2.pointPolygonTest(pts, (px, py), False) >= 0:
                return True

        return False
