"""BirdNET bird song identification for RatCatcher AI.

Wraps the BirdNET TFLite model, which classifies a fixed-length window of
raw audio into one of several thousand bird species. It is the same
shape of component as ``classification.SpeciesClassifier`` -- a quantised
TFLite model running on CPU -- but takes a waveform rather than an image
crop.

The model's geometry is read from the interpreter at load time rather
than hardcoded. BirdNET has shipped several revisions with different
window lengths and class counts, and the detection backends in this
project already establish the pattern of adapting to the model file
found on disk instead of trusting a constant.

Model weights are licensed CC BY-NC-SA and are deliberately not vendored
into this repository. ``scripts/download_models.sh`` fetches them.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# TFLite import guard
# ---------------------------------------------------------------------------

_Interpreter: Any = None
_TFLITE_SOURCE: str = "none"

try:
    from tflite_runtime.interpreter import Interpreter as _TFLRInterpreter  # type: ignore[import-untyped]

    _Interpreter = _TFLRInterpreter
    _TFLITE_SOURCE = "tflite_runtime"
except ImportError:
    try:
        # ai-edge-litert is Google's maintained successor to tflite-runtime
        # and is the only one of the three with wheels for Python 3.13 on
        # aarch64, which is what Raspberry Pi OS ships today.
        from ai_edge_litert.interpreter import Interpreter as _LiteRTInterpreter  # type: ignore[import-untyped]

        _Interpreter = _LiteRTInterpreter
        _TFLITE_SOURCE = "ai_edge_litert"
    except ImportError:
        try:
            import tensorflow as _tf  # type: ignore[import-untyped]

            _Interpreter = _tf.lite.Interpreter
            _TFLITE_SOURCE = "tensorflow"
        except ImportError:
            _Interpreter = None
            _TFLITE_SOURCE = "none"


# BirdNET is trained on 48 kHz audio, which is also the native rate of the
# googlevoicehat capture path, so no resampling is needed in practice.
BIRDNET_SAMPLE_RATE = 48000


@dataclass(frozen=True)
class SongDetection:
    """One species identified in one window of audio."""

    scientific_name: str
    common_name: str
    confidence: float
    channel: int

    def __str__(self) -> str:
        return (
            f"{self.common_name} ({self.scientific_name}) "
            f"{self.confidence:.1%} ch{self.channel}"
        )


def parse_labels(path: str | Path) -> list[tuple[str, str]]:
    """Read a BirdNET label file into ``(scientific, common)`` pairs.

    BirdNET labels are written as ``Scientific name_Common Name``. Lines
    without a separator are taken as a scientific name with the common
    name left equal to it, which keeps a malformed file usable rather
    than aborting the whole load.
    """
    label_path = Path(path)
    if not label_path.exists():
        raise FileNotFoundError(f"BirdNET label file not found: {label_path}")

    try:
        raw = label_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError(f"Could not read {label_path}: {exc}") from exc

    labels: list[tuple[str, str]] = []
    for line in raw.splitlines():
        entry = line.strip()
        if not entry:
            continue
        scientific, separator, common = entry.partition("_")
        if separator:
            labels.append((scientific.strip(), common.strip()))
        else:
            labels.append((entry, entry))

    if not labels:
        raise ValueError(f"BirdNET label file is empty: {label_path}")

    return labels


class BirdNetClassifier:
    """Identify bird species from a window of audio.

    Parameters
    ----------
    model_path:
        Path to the BirdNET ``.tflite`` file.
    labels_path:
        Path to the matching label list.
    min_confidence:
        Minimum sigmoid confidence for a species to be reported.
    top_k:
        Maximum number of species returned per window.
    num_threads:
        Interpreter thread count. The Pi 5 has four cores, but the camera
        pipeline needs them too, so the default of 2 leaves headroom.

    Raises
    ------
    ImportError
        If no TFLite interpreter is installed.
    FileNotFoundError
        If the model or label file is missing.
    ValueError
        If the label count does not match the model's output size.
    """

    def __init__(
        self,
        model_path: str | Path,
        labels_path: str | Path,
        min_confidence: float = 0.25,
        top_k: int = 3,
        num_threads: int = 2,
    ) -> None:
        if _Interpreter is None:
            raise ImportError(
                "No TFLite interpreter available. Install one with:\n"
                "  pip install ai-edge-litert\n"
                "tflite-runtime has no wheels for Python 3.13, which is what "
                "current Raspberry Pi OS ships; ai-edge-litert is its "
                "maintained successor and does."
            )

        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(
                f"min_confidence must be in [0, 1], got {min_confidence}"
            )
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        self._model_path = Path(model_path)
        if not self._model_path.exists():
            raise FileNotFoundError(
                f"BirdNET model not found: {self._model_path}\n"
                f"Download it with: ./scripts/download_models.sh"
            )

        self._labels = parse_labels(labels_path)
        self._min_confidence = min_confidence
        self._top_k = top_k

        try:
            self._interpreter = _Interpreter(
                model_path=str(self._model_path),
                num_threads=num_threads,
            )
            self._interpreter.allocate_tensors()
        except (ValueError, RuntimeError, OSError) as exc:
            raise RuntimeError(
                f"Could not load BirdNET model {self._model_path}: {exc}"
            ) from exc

        input_detail = self._interpreter.get_input_details()[0]
        output_detail = self._interpreter.get_output_details()[0]

        self._input_index = input_detail["index"]
        self._output_index = output_detail["index"]
        self._input_dtype = input_detail["dtype"]

        # Read the window length from the model rather than assuming it.
        input_shape = [int(dim) for dim in input_detail["shape"]]
        self._window_samples = int(np.prod(input_shape[1:]))
        self._input_shape = input_shape

        num_classes = int(output_detail["shape"][-1])
        if num_classes != len(self._labels):
            raise ValueError(
                f"BirdNET model outputs {num_classes} classes but the label "
                f"file lists {len(self._labels)}. The model and labels are "
                f"from different releases."
            )

        logger.info(
            "BirdNET loaded via %s: %s (%d classes, %.1f s window @ %d Hz)",
            _TFLITE_SOURCE,
            self._model_path.name,
            num_classes,
            self._window_samples / BIRDNET_SAMPLE_RATE,
            BIRDNET_SAMPLE_RATE,
        )

    # -- inference ---------------------------------------------------------

    def identify(self, window: np.ndarray, channel: int = 0) -> list[SongDetection]:
        """Identify species in one mono window of audio.

        Parameters
        ----------
        window:
            1-D float32 audio at 48 kHz. Windows shorter than the model's
            input are zero-padded; longer windows are truncated. Callers
            should feed exactly ``window_samples`` for best results.
        channel:
            Which microphone this window came from, recorded on each
            resulting detection.

        Returns
        -------
        Up to ``top_k`` detections above ``min_confidence``, most
        confident first. An empty list means nothing was identified.
        """
        if window.ndim != 1:
            raise ValueError(
                f"window must be 1-D mono audio, got shape {window.shape}"
            )

        prepared = self._fit_window(window)

        try:
            self._interpreter.set_tensor(
                self._input_index,
                prepared.reshape(self._input_shape),
            )
            self._interpreter.invoke()
            raw_output = self._interpreter.get_tensor(self._output_index)
        except (ValueError, RuntimeError) as exc:
            logger.error("BirdNET inference failed: %s", exc)
            return []

        scores = _sigmoid(np.asarray(raw_output, dtype=np.float32).flatten())

        # Take the top_k highest scores, then drop those below threshold.
        # argpartition avoids sorting all several thousand classes.
        count = min(self._top_k, scores.size)
        candidates = np.argpartition(-scores, count - 1)[:count]
        candidates = candidates[np.argsort(-scores[candidates])]

        detections: list[SongDetection] = []
        for index in candidates:
            confidence = float(scores[index])
            if confidence < self._min_confidence:
                break
            scientific, common = self._labels[int(index)]
            detections.append(
                SongDetection(
                    scientific_name=scientific,
                    common_name=common,
                    confidence=confidence,
                    channel=channel,
                )
            )

        return detections

    def _fit_window(self, window: np.ndarray) -> np.ndarray:
        """Pad or truncate a window to the model's expected length."""
        prepared = window.astype(np.float32, copy=False)

        if prepared.size < self._window_samples:
            padding = self._window_samples - prepared.size
            prepared = np.pad(prepared, (0, padding), mode="constant")
        elif prepared.size > self._window_samples:
            prepared = prepared[: self._window_samples]

        if self._input_dtype != np.float32:
            prepared = prepared.astype(self._input_dtype)

        return prepared

    # -- properties --------------------------------------------------------

    @property
    def window_samples(self) -> int:
        """Number of samples the model expects per inference."""
        return self._window_samples

    @property
    def window_seconds(self) -> float:
        """Length of the model's input window in seconds."""
        return self._window_samples / BIRDNET_SAMPLE_RATE

    @property
    def num_classes(self) -> int:
        return len(self._labels)

    @property
    def min_confidence(self) -> float:
        return self._min_confidence

    @property
    def backend(self) -> str:
        """Which TFLite implementation is in use."""
        return _TFLITE_SOURCE


def is_available() -> bool:
    """True if a TFLite interpreter is installed."""
    return _Interpreter is not None


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function.

    BirdNET emits logits, and the standard formula overflows on large
    negative inputs. Splitting by sign keeps the exponent negative in
    both branches.
    """
    output = np.empty_like(values, dtype=np.float32)
    positive = values >= 0

    output[positive] = 1.0 / (1.0 + np.exp(-values[positive]))

    exponential = np.exp(values[~positive])
    output[~positive] = exponential / (1.0 + exponential)

    return output
