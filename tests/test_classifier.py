"""Tests for the TFLite species classifier.

These cover the three defects that made ``classifications=0`` a permanent
state of the running system rather than a property of what the cameras
saw: a destroyed input tensor, a softmax applied twice, and a top-1 taken
over classes the taxonomy cannot name.  Each is checked twice -- once as
the arithmetic invariant it violated, and once end to end on a real bird.

Skipped unless a TFLite runtime imports and the model is present.  The
weights are not committed (``models/*.tflite`` is in .gitignore); run
``scripts/download_models.sh`` first.

The two fixture crops are cut from the Open Images V7 validation images
this project already trains against, chosen because the classifier is
confident about both and they sit on opposite sides of the taxonomy:

* ``bird_cedar_waxwing.jpg``  -- Open Images 0d8d6cad9d3c0364, box
  (307, 206, 304, 258).  Label index 894, in ``config/species.yaml``.
* ``bird_common_hoopoe.jpg``  -- Open Images 007f1eeefb0c379d, box
  (430, 128, 447, 306).  Label index 491, deliberately *not* in the
  taxonomy: a real bird, confidently identified, that this system has no
  name for and must therefore decline to name.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ratcatcher.classification.classifier import (
    SpeciesClassifier,
    _TFLITE_AVAILABLE,
)
from ratcatcher.classification.taxonomy import Taxonomy


PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "mobilenet_v2_inat_bird_quant.tflite"
SPECIES_CONFIG = PROJECT_ROOT / "config" / "species.yaml"

pytestmark = pytest.mark.skipif(
    not _TFLITE_AVAILABLE or not MODEL_PATH.exists(),
    reason="needs a TFLite runtime and models/mobilenet_v2_inat_bird_quant.tflite",
)


@pytest.fixture(scope="module")
def taxonomy() -> Taxonomy:
    return Taxonomy(SPECIES_CONFIG)


@pytest.fixture(scope="module")
def classifier(taxonomy: Taxonomy) -> SpeciesClassifier:
    return SpeciesClassifier(
        model_path=str(MODEL_PATH),
        taxonomy=taxonomy,
        input_size=(224, 224),
        top_k=5,
        min_confidence=0.70,
    )


def _load(fixtures_dir: Path, name: str) -> np.ndarray:
    import cv2

    crop = cv2.imread(str(fixtures_dir / name))
    assert crop is not None, f"fixture {name} did not decode"
    return crop


# ---------------------------------------------------------------------------
# Input quantisation
# ---------------------------------------------------------------------------


def test_uint8_input_keeps_the_image(
    classifier: SpeciesClassifier, fixtures_dir: Path
) -> None:
    """The input tensor must still be a picture.

    This model quantises its input with scale 1/128 and zero_point 128.
    Applying q = real/scale + zero_point to raw 0-255 bytes gives
    rgb*128 + 128, which wraps mod 256 to two values: 128 for even
    pixels, 0 for odd ones.  The tensor became the parity bit of each
    pixel, and the model read it as "background".  A photograph resized
    to 224x224 has hundreds of distinct levels, not two.
    """
    crop = _load(fixtures_dir, "bird_cedar_waxwing.jpg")

    tensor = classifier._preprocess(crop)

    assert tensor.shape == (1, 224, 224, 3)
    assert tensor.dtype == np.uint8
    assert len(np.unique(tensor)) > 100, "input tensor lost its dynamic range"
    assert tensor.max() > 200, "input tensor never reaches the top of the range"


def test_preprocess_preserves_pixel_values(
    classifier: SpeciesClassifier, fixtures_dir: Path
) -> None:
    """For a uint8 image input the correct transform is the identity.

    Asserted against the resize directly so the property is pinned to
    something independent of the model's quantisation parameters.
    """
    import cv2

    crop = _load(fixtures_dir, "bird_cedar_waxwing.jpg")
    expected = cv2.cvtColor(
        cv2.resize(crop, (224, 224), interpolation=cv2.INTER_LINEAR),
        cv2.COLOR_BGR2RGB,
    )

    tensor = classifier._preprocess(crop)

    np.testing.assert_array_equal(tensor[0], expected)


# ---------------------------------------------------------------------------
# Output dequantisation
# ---------------------------------------------------------------------------


def test_quantised_softmax_is_not_softmaxed_again(
    classifier: SpeciesClassifier,
) -> None:
    """A quantised softmax sums to less than 1 and is still a softmax.

    965 classes at scale 1/256 round their whole tail to zero, so the
    vector sums to well under 1.  Reading that as "not probabilities"
    and applying softmax collapses the winner to 1/(1 + 964*e^-1), about
    0.0028 -- under any usable threshold, for every image, forever.
    """
    raw = np.zeros((1, 965), dtype=np.uint8)
    raw[0, 894] = 239  # 239/256 = 0.934

    scores = classifier._dequantize_output(raw)

    assert float(np.sum(scores)) < 0.99, "fixture should under-sum, as the model does"
    assert scores[894] == pytest.approx(0.934, abs=0.005)
    assert int(np.argmax(scores)) == 894


def test_logits_are_still_softmaxed(classifier: SpeciesClassifier) -> None:
    """The logit path must survive the fix.

    Negative values cannot be probabilities, so softmax still applies --
    that branch is what a float model with an unfused output needs.
    """
    logits = np.array([[-1.0, 3.0, 0.0, -2.0]], dtype=np.float32)

    scores = classifier._dequantize_output(logits)

    assert float(np.sum(scores)) == pytest.approx(1.0, abs=1e-5)
    assert int(np.argmax(scores)) == 1
    assert np.all(scores >= 0.0)


def test_oversumming_scores_are_softmaxed(classifier: SpeciesClassifier) -> None:
    """All-positive logits sum well past 1 and are not probabilities."""
    logits = np.array([[2.0, 5.0, 1.0, 3.0]], dtype=np.float32)

    scores = classifier._dequantize_output(logits)

    assert float(np.sum(scores)) == pytest.approx(1.0, abs=1e-5)
    assert int(np.argmax(scores)) == 1


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_known_bird_classifies_above_threshold(
    classifier: SpeciesClassifier, fixtures_dir: Path
) -> None:
    """The headline regression: a real bird must produce a species.

    With either of the arithmetic defects present this returns None, and
    did so for every crop the pipeline ever handed it.
    """
    crop = _load(fixtures_dir, "bird_cedar_waxwing.jpg")

    result = classifier.classify(crop)

    assert result is not None, "a clear in-taxonomy bird was not classified"
    assert result.species == "Bombycilla cedrorum"
    assert result.common_name == "Cedar Waxwing"
    assert result.confidence >= 0.70
    assert result.top_k[0][0] == result.species


def test_bird_outside_the_taxonomy_is_declined(
    classifier: SpeciesClassifier, fixtures_dir: Path
) -> None:
    """A confident answer this system has no name for must not be kept.

    The model puts 0.996 on Common Hoopoe here and under 0.001 on every
    one of the 50 species in the taxonomy.  Ranking across all 965
    classes and mapping afterwards used to emit
    ``species="unknown_491"``; scoring only mapped classes leaves nothing
    above threshold, which is the honest answer.
    """
    crop = _load(fixtures_dir, "bird_common_hoopoe.jpg")

    result = classifier.classify(crop)

    assert result is None


def test_every_reported_species_is_in_the_taxonomy(
    classifier: SpeciesClassifier, taxonomy: Taxonomy, fixtures_dir: Path
) -> None:
    """No placeholder identity may reach a caller.

    Covers the model's own "background" class (index 964), which is not a
    species and used to be reportable as one.
    """
    known = {s.genus_species for s in taxonomy.get_all_species()}

    for name in ("bird_cedar_waxwing.jpg", "bird_common_hoopoe.jpg"):
        result = classifier.classify(_load(fixtures_dir, name))
        if result is None:
            continue
        assert result.species in known
        for species, _common, _conf in result.top_k:
            assert species in known, f"{species} is not a taxonomy species"


def test_top_k_is_ordered_and_bounded(
    classifier: SpeciesClassifier, fixtures_dir: Path
) -> None:
    """top_k must be descending and no longer than the taxonomy."""
    crop = _load(fixtures_dir, "bird_cedar_waxwing.jpg")

    result = classifier.classify(crop)

    assert result is not None
    confidences = [conf for _s, _c, conf in result.top_k]
    assert confidences == sorted(confidences, reverse=True)
    assert 0 < len(result.top_k) <= 5
