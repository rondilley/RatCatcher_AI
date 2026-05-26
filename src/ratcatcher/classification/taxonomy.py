"""Species taxonomy management for RatCatcher AI.

Loads and indexes a YAML species configuration file that maps model
output class indices to biological species.  The YAML file contains
two top-level lists -- ``birds`` (species the classifier can identify)
and ``pests`` (non-bird animals the detection stage flags).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SpeciesInfo:
    """Immutable record for a single bird species.

    Attributes
    ----------
    genus_species:
        Binomial name, e.g. ``"Aphelocoma californica"``.
    common_name:
        English common name, e.g. ``"California Scrub-Jay"``.
    family:
        Taxonomic family, e.g. ``"Corvidae"``.
    label_index:
        Integer class index in the model softmax output.
    """

    genus_species: str
    common_name: str
    family: str
    label_index: int


class Taxonomy:
    """Species taxonomy loaded from a YAML configuration file.

    Parameters
    ----------
    species_config_path:
        Filesystem path to the species YAML file.  The file must
        contain a ``birds`` key with a list of species entries and
        optionally a ``pests`` key with pest animal entries.

    Raises
    ------
    FileNotFoundError
        If *species_config_path* does not exist.
    ValueError
        If the YAML cannot be parsed or is structurally invalid.
    """

    def __init__(self, species_config_path: str | Path) -> None:
        config_path = Path(species_config_path)

        try:
            with open(config_path, "r", encoding="utf-8") as fh:
                raw = yaml.safe_load(fh)
        except FileNotFoundError:
            logger.error("Species config file not found: %s", config_path)
            raise
        except yaml.YAMLError as exc:
            logger.error(
                "Failed to parse species YAML at %s: %s", config_path, exc
            )
            raise ValueError(
                f"Invalid YAML in species config {config_path}: {exc}"
            ) from exc

        if not isinstance(raw, dict):
            raise ValueError(
                f"Species config must be a YAML mapping, got {type(raw).__name__}"
            )

        # -- Build species records from the birds list -----------------------
        birds_raw = raw.get("birds")
        if not isinstance(birds_raw, list):
            raise ValueError(
                "Species config is missing a 'birds' list at the top level"
            )

        self._by_index: dict[int, SpeciesInfo] = {}
        self._by_name: dict[str, SpeciesInfo] = {}
        self._all_species: list[SpeciesInfo] = []

        for entry in birds_raw:
            species = SpeciesInfo(
                genus_species=str(entry["genus_species"]),
                common_name=str(entry["common_name"]),
                family=str(entry["family"]),
                label_index=int(entry["label_index"]),
            )
            self._by_index[species.label_index] = species
            self._by_name[species.genus_species] = species
            self._all_species.append(species)

        # -- Store pest categories verbatim ----------------------------------
        pests_raw = raw.get("pests")
        if isinstance(pests_raw, list):
            self._pests: list[dict] = list(pests_raw)
        else:
            self._pests = []

        logger.info(
            "Loaded taxonomy: %d bird species, %d pest entries from %s",
            len(self._all_species),
            len(self._pests),
            config_path,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def label_to_species(self, label_index: int) -> SpeciesInfo | None:
        """Map a model output class index to species data.

        Parameters
        ----------
        label_index:
            Zero-based integer index from the classifier softmax output.

        Returns
        -------
        The matching ``SpeciesInfo``, or ``None`` if the index is not in
        the taxonomy.
        """
        return self._by_index.get(label_index)

    def species_by_name(self, genus_species: str) -> SpeciesInfo | None:
        """Look up a species by its binomial name.

        Parameters
        ----------
        genus_species:
            Binomial name, e.g. ``"Aphelocoma californica"``.

        Returns
        -------
        The matching ``SpeciesInfo``, or ``None`` if the name is not in
        the taxonomy.
        """
        return self._by_name.get(genus_species)

    def get_all_species(self) -> list[SpeciesInfo]:
        """Return every bird species in the taxonomy.

        The list order matches the order they appear in the YAML file.
        """
        return list(self._all_species)

    @property
    def num_classes(self) -> int:
        """Number of bird species (model output vector length)."""
        return len(self._all_species)

    @property
    def pest_categories(self) -> list[dict]:
        """Return the pest species list from the YAML, as raw dicts.

        Each dict has keys ``genus_species``, ``common_name``, and
        ``category`` (e.g. ``"squirrel"``, ``"rat"``, ``"cat"``).
        """
        return list(self._pests)
