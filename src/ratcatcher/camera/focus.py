"""Focus and exposure measurement for setting a lens by hand.

The Arducam UC-517 lenses have no software focus control -- libcamera
exposes neither ``LensPosition`` nor any ``Af`` control for them -- so
focus is a ring turned by hand at the enclosure. This module turns a
frame into the numbers that make that possible without a monitor.

Why the blur ratio and not the usual Laplacian variance
-------------------------------------------------------
Sharpness has to be reported as a percentage, and a percentage needs a
fixed zero. Three metrics were measured against the same frames under
known Gaussian blur:

- **Laplacian variance** spans four decades (0.88 to 10144 across a real
  frame and a random-texture target) and depends entirely on how much
  detail the scene happens to contain. A blank wall in perfect focus
  scores near zero. There is no value that means "focused".
- **Laplacian variance normalised by luma variance** is worse: it is not
  monotonic. Measured 0.0718 at sigma 4 and 0.2171 at sigma 8, so it
  reads *better* as the image gets blurrier. Unusable.
- **The blur ratio** used here re-blurs the frame and asks how much
  detail that destroys. A sharp frame loses a great deal, so the ratio
  is large; a frame that is already blurred loses almost nothing, so the
  ratio approaches 1.0.

That 1.0 is the point of the whole exercise: it is a floor with a
physical meaning -- "blurring this further changes nothing, it is
already as defocused as it can look" -- and it anchors the 0% end of the
scale to something real rather than to a scene-dependent constant.

The ceiling is the honest weakness
----------------------------------
``DEFAULT_CEILING`` maps the top of the scale and cannot be calibrated
until a genuinely in-focus frame exists on this hardware. Until then the
absolute percentage is approximate. This does not make the tool useless,
because setting a lens by hand needs a *comparison*, not an absolute:
the caller peak-holds the best reading of the session, and "better or
worse than a moment ago" is exact whatever the ceiling is.

Why a percentage alone would mislead
------------------------------------
Camera 1 on this system reads 5% while returning no image at all: a
uniform field at 7.3 lux with auto-exposure pinned at maximum. Five
percent invites someone to keep turning a ring that was never the
problem. Detail is the tell -- standard deviation 10 against camera 0's
57 -- so a frame with no detail to measure is reported as such instead
of as a low score. ``NO IMAGE`` and ``LOW DETAIL`` are separated by
brightness, because a blocked lens and a blank wall look identical in
pixels alone and only the first is a fault.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np

# Fraction of the frame, per axis, used to judge focus. The feeder sits
# in the middle of the frame, and the corners of a cheap lens are soft
# even when the centre is sharp.
CENTRE_FRACTION = 1.0 / 3.0

# Blur ratio that maps to 100%. See the module docstring: this is the
# uncalibrated constant, and the value to revisit once a sharp frame
# exists. A random-texture target reaches 224, a defocused real frame
# 3.9, so 60 sits well inside the achievable range for a real scene.
DEFAULT_CEILING = 60.0

# Sigma of the reference blur. Large enough to destroy real detail,
# small enough not to erase the coarse structure that survives in an
# already-blurred frame.
_BLUR_SIGMA = 1.5

# Laplacian energy below this is noise, not structure. Guards the
# division and makes a featureless frame return the 1.0 floor rather
# than a ratio built from rounding error.
_MIN_ENERGY = 1e-6

# Pixel values counted as clipped. 8-bit, so 255 is full scale; a little
# margin catches highlights already flattened by the ISP.
_CLIP_HIGH = 250
_CLIP_LOW = 5

# Standard deviation below which there is not enough detail to judge
# focus at all. Camera 1 measured 6 to 10 while blocked; camera 0
# measured 57 while merely out of focus.
_MIN_TEXTURE_STD = 12.0

# Scene illumination below which a featureless frame is a blocked lens
# rather than a blank subject. Camera 1 reported 7.3 lux against camera
# 0's 19242 in the same daylight.
_DARK_LUX = 50.0

# Mean luma standing in for lux when no camera metadata is available.
_DARK_LUMA = 30.0

# Fraction of the frame clipped white before exposure is the problem
# rather than the focus.
_BRIGHT_CLIP_PCT = 20.0

STATE_OK = "OK"
STATE_NO_IMAGE = "NO IMAGE"
STATE_LOW_DETAIL = "LOW DETAIL"
STATE_DARK = "DARK"
STATE_BRIGHT = "BRIGHT"

# States in which the focus percentage means nothing and must not be
# shown as though it did.
_UNMEASURABLE = frozenset({STATE_NO_IMAGE, STATE_LOW_DETAIL})


@dataclass(frozen=True)
class FocusReading:
    """Everything one frame can say about focus and exposure."""

    camera: int
    focus_pct: float
    peak_pct: float
    blur_ratio: float
    luma_mean: float
    luma_std: float
    clip_high_pct: float
    clip_low_pct: float
    state: str
    lux: float | None = None
    exposure_us: int | None = None
    analogue_gain: float | None = None

    @property
    def measurable(self) -> bool:
        """True when the focus percentage describes the lens.

        False when the frame carries too little detail to judge, in
        which case the state is the reading and the percentage is not.
        """
        return self.state not in _UNMEASURABLE


def centre_crop(gray: np.ndarray, fraction: float = CENTRE_FRACTION) -> np.ndarray:
    """The middle of the frame at full resolution.

    Cropping rather than resizing is deliberate. Downscaling is a
    low-pass filter, and it would attenuate exactly the high-frequency
    detail this module exists to measure, flattening the difference
    between a sharp lens and a soft one.
    """
    height, width = gray.shape[:2]
    half = max(0.0, min(1.0, fraction)) / 2.0
    y0, y1 = int(height * (0.5 - half)), int(height * (0.5 + half))
    x0, x1 = int(width * (0.5 - half)), int(width * (0.5 + half))
    # A fraction rounding to an empty slice would make every downstream
    # statistic NaN. Fall back to the whole frame instead.
    if y1 - y0 < 2 or x1 - x0 < 2:
        return gray
    return gray[y0:y1, x0:x1]


def blur_ratio(gray: np.ndarray) -> float:
    """How much detail a reference blur destroys. 1.0 means none left.

    Guarded by the same texture threshold the state machine uses,
    because outside it the measurement inverts. Once an image is nearly
    flat both Laplacian energies are dominated by sensor and rounding
    noise, and their *ratio* can climb as the image gets smoother still:
    measured 1.06 at sigma 4 rising to 1.86 at sigma 12 on a synthetic
    target, and far worse in floating point, where the same sweep ran
    away to 312. Reporting the floor for a frame with nothing in it to
    measure keeps the scale monotonic where it is meaningful and honest
    where it is not.
    """
    if float(gray.std()) < _MIN_TEXTURE_STD:
        return 1.0

    reference = cv2.GaussianBlur(gray, (0, 0), _BLUR_SIGMA)
    blurred_energy = float(cv2.Laplacian(reference, cv2.CV_64F).var())
    if blurred_energy < _MIN_ENERGY:
        return 1.0
    ratio = float(cv2.Laplacian(gray, cv2.CV_64F).var()) / blurred_energy
    # Below the floor the frame is already fully blurred; the excess is
    # noise and reporting it as less than "completely soft" is noise too.
    return max(1.0, ratio)


def focus_percent(ratio: float, ceiling: float = DEFAULT_CEILING) -> float:
    """Map a blur ratio to 0-100.

    Logarithmic because the ratio grows multiplicatively as a lens comes
    into focus: a linear map would leave the whole usable range of a
    hand-turned ring crushed into the bottom few percent of the bar.
    """
    if ceiling <= 1.0:
        raise ValueError(f"ceiling must exceed 1.0, got {ceiling}")
    percent = 100.0 * math.log(max(float(ratio), 1.0)) / math.log(ceiling)
    return max(0.0, min(100.0, percent))


def analyse_frame(
    frame: np.ndarray,
    camera: int,
    *,
    metadata: dict[str, Any] | None = None,
    ceiling: float = DEFAULT_CEILING,
    peak_pct: float = 0.0,
) -> FocusReading:
    """Measure one frame.

    ``frame`` is BGR uint8 as every ``CameraSource`` produces, or already
    grayscale. ``metadata`` is the optional picamera2 control dictionary;
    without it the reading falls back to mean luma to tell a blocked lens
    from a blank subject.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

    # Focus is measured on the centre, where the subject is. Exposure is
    # a property of the whole frame, so clipping is measured across all
    # of it -- a blown-out sky matters even when the feeder is fine.
    centre = centre_crop(gray)
    ratio = blur_ratio(centre)
    percent = focus_percent(ratio, ceiling)

    luma_mean = float(gray.mean())
    # Texture is measured on the same crop as the focus, so the state and
    # the percentage cannot disagree about whether there was anything to
    # measure: the ratio falls back to its floor on exactly the frames
    # this threshold marks unmeasurable.
    luma_std = float(centre.std())
    clip_high = 100.0 * float((gray >= _CLIP_HIGH).mean())
    clip_low = 100.0 * float((gray <= _CLIP_LOW).mean())

    lux = exposure_us = analogue_gain = None
    if metadata:
        lux = _as_float(metadata.get("Lux"))
        exposure_us = _as_int(metadata.get("ExposureTime"))
        analogue_gain = _as_float(metadata.get("AnalogueGain"))

    state = _classify(luma_std, luma_mean, clip_high, lux)

    return FocusReading(
        camera=camera,
        focus_pct=percent,
        peak_pct=max(float(peak_pct), percent if state not in _UNMEASURABLE else 0.0),
        blur_ratio=ratio,
        luma_mean=luma_mean,
        luma_std=luma_std,
        clip_high_pct=clip_high,
        clip_low_pct=clip_low,
        state=state,
        lux=lux,
        exposure_us=exposure_us,
        analogue_gain=analogue_gain,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _classify(
    luma_std: float, luma_mean: float, clip_high: float, lux: float | None
) -> str:
    """Name what the frame shows.

    Order matters. A frame with no detail cannot be judged for focus at
    all, so that test comes first and the exposure tests describe frames
    that do carry detail.
    """
    # Metadata is the better measure of scene light, because auto-exposure
    # hides a dark scene by brightening it: camera 1 reported a mean luma
    # of 66 while sitting at 7.3 lux with the gain pinned at 7.9x.
    dark = lux < _DARK_LUX if lux is not None else luma_mean < _DARK_LUMA

    # Clipping is tested before detail because it explains a lack of it.
    # A frame blown to white has no texture either, and "BRIGHT" points
    # at the cause where "LOW DETAIL" would only restate the symptom.
    # Auto-exposure keeps a normal scene well clear of this: camera 0
    # measured 0.0% clipped against a 19242 lux sky.
    if clip_high >= _BRIGHT_CLIP_PCT:
        return STATE_BRIGHT
    if luma_std < _MIN_TEXTURE_STD:
        # No detail and no light is a covered lens. No detail with light
        # is a real but featureless view, which is a framing problem.
        return STATE_NO_IMAGE if dark else STATE_LOW_DETAIL
    if dark:
        return STATE_DARK
    return STATE_OK


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# -- Exposure settling ---------------------------------------------------------

# How many consecutive agreeing samples count as settled. Three at the
# caller's poll interval is about a second, which is what the IMX477
# needs after its first frame.
SETTLE_SAMPLES = 3

# Fractional change below which two exposure samples are the same value.
SETTLE_TOLERANCE = 0.02


def exposure_sample(metadata: dict[str, Any] | None) -> tuple[float, ...] | None:
    """The exposure state a settling test compares, or None if unknown."""
    if not metadata:
        return None
    values = (
        _as_float(metadata.get("ExposureTime")),
        _as_float(metadata.get("AnalogueGain")),
        _as_float(metadata.get("DigitalGain")),
    )
    if any(v is None for v in values):
        return None
    return values  # type: ignore[return-value]


def exposure_settled(
    samples: Sequence[tuple[float, ...] | None],
    *,
    tolerance: float = SETTLE_TOLERANCE,
    needed: int = SETTLE_SAMPLES,
) -> bool:
    """Has auto-exposure stopped moving?

    A frame captured before it has is measured at the wrong brightness,
    and the focus metric reads it as a different lens. Camera 1 on this
    Pi opens at 193 us and luma 121, scoring 57%; one second later auto
    exposure has settled on 303 us and luma 94, and the same lens scores
    41%. Reporting the first of those is not a small error -- it is the
    difference between a lens that looks acceptable and one that needs
    turning.

    There is no ``AeLocked`` in this pipeline's metadata, so agreement
    across consecutive samples is the available signal.
    """
    usable = [s for s in samples if s is not None]
    if len(usable) < needed:
        return False

    window = usable[-needed:]
    reference = window[0]
    for sample in window[1:]:
        for a, b in zip(reference, sample):
            scale = max(abs(a), abs(b), 1e-6)
            if abs(a - b) > tolerance * scale:
                return False
    return True


# Percentage points within which two readings count as the same value.
READING_TOLERANCE_PCT = 1.5


def readings_settled(
    values: Sequence[float],
    *,
    tolerance: float = READING_TOLERANCE_PCT,
    needed: int = SETTLE_SAMPLES,
) -> bool:
    """Has the focus measurement itself stopped moving?

    Exposure settling is necessary and not sufficient. On camera 1 every
    control freezes within 0.7 s -- exposure 303 us, gains 1.00 and
    1.029, lux 21880, colour temperature 4558, luma mean 102.9, luma
    standard deviation 53.6, all constant -- while the focus reading goes
    on falling from 56% to 32% over the next three seconds. The global
    statistics do not move because what is changing is fine detail: the
    ISP's temporal denoise is converging, and the early frames carry
    sensor noise that the blur ratio counts as detail.

    The size of that error depends on how soft the lens is, which is why
    it hid for so long. A sharp frame has real detail that dwarfs the
    noise, so camera 0 drifts two points; a soft one has little else, so
    camera 1 drifts twenty-four -- and reads as acceptable at exactly the
    moment it is worst.

    No metadata reports this, so the measurement is its own settling
    signal.
    """
    if len(values) < needed:
        return False
    window = values[-needed:]
    return (max(window) - min(window)) <= tolerance
