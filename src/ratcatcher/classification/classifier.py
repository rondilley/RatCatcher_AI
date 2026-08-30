"""TensorFlow Lite species classifier wrapper for RatCatcher AI.

Runs a TFLite image-classification model (e.g. EfficientNet-B3) on a
cropped detection region and maps the softmax output to a species
identity via the project Taxonomy.

The module supports two TFLite back-ends:

* ``tflite_runtime`` -- the lightweight, pip-installable interpreter.
* ``tensorflow.lite``  -- the full TensorFlow package (heavier but
  always includes the interpreter).

If neither is available the ``SpeciesClassifier`` class is still
importable, but its constructor will raise ``ImportError`` with an
actionable message.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ratcatcher.classification.taxonomy import Taxonomy

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

_TFLITE_AVAILABLE: bool = _Interpreter is not None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SpeciesResult:
    """Result of classifying a single detection crop.

    Attributes
    ----------
    species:
        Binomial name of the top prediction (e.g.
        ``"Aphelocoma californica"``).
    common_name:
        Common English name (e.g. ``"California Scrub-Jay"``).
    confidence:
        Softmax probability of the top prediction, 0.0 -- 1.0.
    top_k:
        List of the *k* highest-confidence predictions, each a tuple of
        ``(genus_species, common_name, confidence)``.
    """

    species: str
    common_name: str
    confidence: float
    top_k: list[tuple[str, str, float]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------


class SpeciesClassifier:
    """Classify a detection crop into a bird species using TFLite.

    Parameters
    ----------
    model_path:
        Path to a ``.tflite`` model file.
    taxonomy:
        A loaded ``Taxonomy`` instance for mapping label indices to
        species.
    input_size:
        ``(width, height)`` that the model expects.  Crops are resized
        to this before inference.
    top_k:
        Number of top predictions to include in ``SpeciesResult.top_k``.
    min_confidence:
        Predictions below this threshold cause ``classify`` to return
        ``None``.
    use_xnnpack:
        If ``True``, attempt to load the XNNPACK delegate for faster
        CPU inference.  Falls back silently if the delegate is
        unavailable.

    Raises
    ------
    ImportError
        If no TFLite runtime is installed.
    FileNotFoundError
        If *model_path* does not exist.
    RuntimeError
        If the interpreter fails to load or allocate the model.
    """

    def __init__(
        self,
        model_path: str | Path,
        taxonomy: Taxonomy,
        input_size: tuple[int, int] = (300, 300),
        top_k: int = 5,
        min_confidence: float = 0.70,
        use_xnnpack: bool = True,
    ) -> None:
        if not _TFLITE_AVAILABLE:
            raise ImportError(
                "No TFLite runtime found.  Install one of:\n"
                "  pip install tflite-runtime\n"
                "  pip install tensorflow"
            )

        self._taxonomy = taxonomy
        self._input_size = input_size
        self._top_k = max(1, top_k)

        # The model knows 965 classes and the taxonomy maps 50 of them.
        # Ranking over all 965 and mapping afterwards means the winner is
        # usually a class this system has no name for -- most often the
        # model's own "background" class -- which then reaches the
        # database as a species.  Scoring only the mapped indices asks
        # the question the caller actually has: which of the species we
        # know is this, and is that answer strong enough to keep.
        self._label_indices = np.array(
            sorted(s.label_index for s in taxonomy.get_all_species()),
            dtype=np.int64,
        )
        if self._label_indices.size == 0:
            raise ValueError(
                "Taxonomy maps no label indices; the classifier cannot "
                "name anything"
            )
        self._min_confidence = min_confidence

        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(
                f"TFLite model not found: {model_file}"
            )

        # -- Build interpreter -----------------------------------------------
        try:
            interpreter_kwargs: dict[str, Any] = {
                "model_path": str(model_file),
            }

            if use_xnnpack:
                try:
                    # XNNPACK delegate name differs between runtimes.
                    # On failure we simply skip -- not every build ships it.
                    xnn_delegate = "libXNNPACK.so"
                    interpreter_kwargs["experimental_delegates"] = [
                        _load_delegate(xnn_delegate)
                    ]
                except Exception:
                    logger.debug(
                        "XNNPACK delegate unavailable; using default CPU backend"
                    )

            self._interpreter = _Interpreter(**interpreter_kwargs)
            self._interpreter.allocate_tensors()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load TFLite model from {model_file}: {exc}"
            ) from exc

        # -- Discover input / output tensor details --------------------------
        input_details = self._interpreter.get_input_details()
        output_details = self._interpreter.get_output_details()

        self._input_index: int = int(input_details[0]["index"])
        self._output_index: int = int(output_details[0]["index"])

        self._input_dtype: np.dtype = np.dtype(input_details[0]["dtype"])
        self._output_dtype: np.dtype = np.dtype(output_details[0]["dtype"])

        # Quantization parameters (used only for integer-quantised models).
        input_quant = input_details[0].get("quantization_parameters", {})
        self._input_scale: float = float(
            input_quant.get("scales", [0.0])[0]
            if isinstance(input_quant.get("scales"), (list, np.ndarray))
            else 0.0
        )
        self._input_zero_point: int = int(
            input_quant.get("zero_points", [0])[0]
            if isinstance(input_quant.get("zero_points"), (list, np.ndarray))
            else 0
        )

        output_quant = output_details[0].get("quantization_parameters", {})
        self._output_scale: float = float(
            output_quant.get("scales", [0.0])[0]
            if isinstance(output_quant.get("scales"), (list, np.ndarray))
            else 0.0
        )
        self._output_zero_point: int = int(
            output_quant.get("zero_points", [0])[0]
            if isinstance(output_quant.get("zero_points"), (list, np.ndarray))
            else 0
        )

        logger.info(
            "SpeciesClassifier ready: model=%s  input_dtype=%s  "
            "input_size=%s  classes=%d  (via %s)",
            model_file.name,
            self._input_dtype,
            self._input_size,
            taxonomy.num_classes,
            _TFLITE_SOURCE,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, crop: np.ndarray) -> SpeciesResult | None:
        """Classify a detection crop and return a species prediction.

        Parameters
        ----------
        crop:
            BGR ``uint8`` image (any size).  Will be resized and
            normalised to match the model input format.

        Returns
        -------
        A ``SpeciesResult`` with the top prediction and top-k list, or
        ``None`` if the best confidence is below ``min_confidence``.
        """
        try:
            tensor = self._preprocess(crop)
        except Exception as exc:
            logger.warning("Preprocessing failed: %s", exc)
            return None

        try:
            self._interpreter.set_tensor(self._input_index, tensor)
            self._interpreter.invoke()
            raw_output = self._interpreter.get_tensor(self._output_index)
        except Exception as exc:
            logger.error("TFLite inference failed: %s", exc)
            return None

        scores = self._dequantize_output(raw_output)

        return self._build_result(scores)

    @staticmethod
    def is_available() -> bool:
        """Return ``True`` if a TFLite runtime is installed."""
        return _TFLITE_AVAILABLE

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(self, crop: np.ndarray) -> np.ndarray:
        """Resize and normalise a BGR crop for the model.

        For ``float32`` models the pixel values are scaled to [0, 1].
        For ``uint8`` quantised models the raw bytes are passed through
        (optionally adjusted by scale/zero-point if the model uses
        full-range quantisation).
        """
        # cv2 is imported lazily here so the module can be loaded in
        # environments where OpenCV is available but TFLite is not
        # (the class just raises ImportError on construction).
        import cv2  # noqa: C0415

        # Resize to (width, height) -- cv2.resize takes (w, h).
        resized = cv2.resize(
            crop,
            self._input_size,
            interpolation=cv2.INTER_LINEAR,
        )

        # Convert BGR -> RGB.  Most classification models are trained
        # on RGB input.
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        if np.issubdtype(self._input_dtype, np.floating):
            # Float model: normalise to [0, 1].
            tensor = rgb.astype(np.float32) / 255.0
        else:
            # Integer-quantised model: pass the raw bytes through.
            #
            # scale/zero_point convert between the model's *real* domain
            # and the quantised one, so quantising requires the real
            # value first -- and for an image model the real domain is
            # exactly what the full 0-255 pixel range maps onto.  Writing
            # that out, q = real/scale + zero_point with
            # real = (0 - zp)*scale + (rgb/255)*255*scale, collapses to
            # q = rgb: the identity.  Feeding the raw bytes is correct
            # for any uint8 image input, whatever its scale and offset.
            #
            # Applying the formula to the bytes as if they were already
            # real values does not merely rescale, it destroys the image.
            # This model has scale 1/128 and zero_point 128, so
            # rgb/scale + zp is rgb*128 + 128, which wraps mod 256 to 128
            # for even pixels and 0 for odd ones -- a tensor holding the
            # parity bit of each pixel and nothing else.  The model reads
            # that as "background" with high confidence.
            tensor = rgb.astype(self._input_dtype)

        # Add batch dimension: (H, W, C) -> (1, H, W, C).
        return np.expand_dims(tensor, axis=0)

    def _dequantize_output(self, raw: np.ndarray) -> np.ndarray:
        """Convert raw output tensor to float32 probabilities.

        Applies scale and zero-point to a quantised output, then softmax
        only when the values are logits rather than probabilities.
        """
        output = raw.squeeze()  # remove batch dim -> (num_classes,)

        if np.issubdtype(output.dtype, np.floating):
            scores = output.astype(np.float32)
        else:
            # De-quantise: real = (q - zero_point) * scale
            scores = (
                (output.astype(np.float32) - self._output_zero_point)
                * self._output_scale
            )

        # Probabilities are non-negative and sum to at most 1; anything
        # else is logits and needs softmax.
        #
        # The sum cannot be tested for equality with 1.  A quantised
        # softmax loses mass to rounding: this model has 965 classes at
        # scale 1/256, so the long tail of near-zero probabilities all
        # round to zero and the vector sums to about 0.85.  The previous
        # test required the total to reach 0.99 and so re-applied softmax
        # to values that were already probabilities, on every frame.
        # That is not a small error -- softmax over [0, 1] inputs across
        # 965 classes caps the winner at 1/(1 + 964*e^-1) = 0.0028, far
        # under any usable threshold, so no crop could ever be
        # classified.  Under-summing is expected; over-summing is not.
        total = float(np.sum(scores))
        if float(np.min(scores)) < 0.0 or total > 1.01:
            # Numerically stable softmax.
            shifted = scores - np.max(scores)
            exp_scores = np.exp(shifted)
            scores = exp_scores / np.sum(exp_scores)

        return scores

    def _build_result(self, scores: np.ndarray) -> SpeciesResult | None:
        """Construct a SpeciesResult from softmax scores.

        Returns ``None`` when the best prediction is below the
        configured confidence threshold.
        """
        # Rank only the classes the taxonomy names.  Scores are left
        # un-renormalised, so a bird that is not one of these reads as a
        # low number against every one of them and is correctly rejected
        # rather than being redistributed into a confident wrong answer.
        candidates = self._label_indices[self._label_indices < len(scores)]
        if candidates.size == 0:
            return None

        candidate_scores = scores[candidates]
        k = min(self._top_k, candidate_scores.size)

        # argpartition is O(n) vs O(n log n) for full sort.
        top_positions = np.argpartition(candidate_scores, -k)[-k:]
        # Sort the top-k by descending score.
        top_positions = top_positions[
            np.argsort(candidate_scores[top_positions])[::-1]
        ]

        top_k_list: list[tuple[str, str, float]] = []
        for pos in top_positions:
            label_idx = int(candidates[pos])
            conf = float(scores[label_idx])
            info = self._taxonomy.label_to_species(label_idx)
            # Every candidate index came from the taxonomy, so this
            # lookup cannot miss.
            top_k_list.append((info.genus_species, info.common_name, conf))

        if not top_k_list:
            return None

        best_species, best_common, best_conf = top_k_list[0]

        if best_conf < self._min_confidence:
            logger.debug(
                "Top prediction %s (%.3f) below threshold %.3f",
                best_species,
                best_conf,
                self._min_confidence,
            )
            return None

        return SpeciesResult(
            species=best_species,
            common_name=best_common,
            confidence=best_conf,
            top_k=top_k_list,
        )


# ---------------------------------------------------------------------------
# Module-level helper for delegate loading
# ---------------------------------------------------------------------------


def _load_delegate(delegate_name: str) -> Any:
    """Try to load a TFLite delegate by library name.

    Attempts ``tflite_runtime`` first, then ``tensorflow``.

    Raises
    ------
    RuntimeError
        If the delegate cannot be loaded from either back-end.
    """
    if _TFLITE_SOURCE == "tflite_runtime":
        from tflite_runtime.interpreter import load_delegate  # type: ignore[import-untyped]

        return load_delegate(delegate_name)

    if _TFLITE_SOURCE == "tensorflow":
        import tensorflow as tf  # type: ignore[import-untyped]

        return tf.lite.experimental.load_delegate(delegate_name)

    raise RuntimeError(
        f"Cannot load delegate '{delegate_name}': no TFLite runtime available"
    )
