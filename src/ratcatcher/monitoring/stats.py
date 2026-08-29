"""Detection statistics for RatCatcher AI."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ratcatcher.storage.database import DetectionDatabase


@dataclass
class DetectionStats:
    """Summary statistics for a time window.

    ``species_counts`` answers "which birds", ``class_counts`` answers
    "how many of each kind of animal". They are separate because only
    the bird path fills in a species: counting animal kinds by species
    puts every squirrel, rat and cat into one "unknown" bucket.
    """

    window_label: str
    since: str
    total_detections: int
    species_counts: dict[str, int]
    detections_per_hour: float
    class_counts: dict[str, int] = field(default_factory=dict)

    @property
    def top_species(self) -> list[tuple[str, int]]:
        """Species sorted by count, descending."""
        return sorted(
            self.species_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )

    @property
    def bird_count(self) -> int:
        """Total detections the detector called a bird."""
        return self.class_counts.get("bird", 0)

    @property
    def pest_count(self) -> int:
        """Total detections of a non-bird animal."""
        return sum(
            count for class_name, count in self.class_counts.items()
            if class_name in _PEST_CLASSES
        )


@dataclass(frozen=True)
class ModalityCount:
    """How many times one category was seen and heard."""

    seen: int = 0
    heard: int = 0

    @property
    def total(self) -> int:
        return self.seen + self.heard


@dataclass(frozen=True)
class CategoryCounts:
    """Detection counts by animal category and by sense.

    ``seen`` counts come from the cameras, ``heard`` counts from the
    microphones. The two are independent observations of the same
    garden: one animal that is both seen and heard is counted twice,
    once in each column, because the video and audio paths never
    correlate their events.
    """

    since: str
    bird: ModalityCount = ModalityCount()
    rodent: ModalityCount = ModalityCount()
    other: ModalityCount = ModalityCount()

    @property
    def total(self) -> int:
        return self.bird.total + self.rodent.total + self.other.total


# Detector classes map straight onto the three display categories. The
# custom YOLO model emits exactly these five names.
_VIDEO_CATEGORY: dict[str, str] = {
    "bird": "bird",
    "squirrel": "rodent",
    "rat": "rodent",
    "cat": "other",
    "unknown_animal": "other",
}

# Every detector class that is not a bird. Derived from the map above
# rather than listed a second time: the pest count broke once already
# by keeping its own copy of this set, in the wrong column.
_PEST_CLASSES: frozenset[str] = frozenset(
    name for name in _VIDEO_CATEGORY if name != "bird"
)

# BirdNET labels that name a sound rather than an animal. These are
# discarded: a passing engine is not a visitor to the feeder.
_AUDIO_NON_ANIMAL: frozenset[str] = frozenset(
    {
        "Engine",
        "Environmental",
        "Fireworks",
        "Gun",
        "Human",
        "Human non-vocal",
        "Human vocal",
        "Human whistle",
        "Noise",
        "Power tools",
        "Siren",
    }
)

# Rodent genera that BirdNET can name. The detections_all view labels
# every audio row "bird" because the microphone path has no detector
# class, so the genus is the only signal available to correct it.
_AUDIO_RODENT_GENERA: frozenset[str] = frozenset(
    {
        "Glaucomys",
        "Marmota",
        "Mus",
        "Neotoma",
        "Peromyscus",
        "Rattus",
        "Sciurus",
        "Spermophilus",
        "Tamias",
        "Tamiasciurus",
        "Urocitellus",
    }
)

# Non-avian genera BirdNET can name that are not rodents either. These
# are discarded rather than counted: a cricket chirping in the grass is
# not a visitor to the feeder, the same reasoning that discards a
# passing engine. The volume is the argument -- 145 cricket windows in
# one evening, against a handful of real sightings, and filing them
# under "other" would only move the miscount one column over.
#
# Counting them as birds is what the fall-through did before: every
# binomial that was not a known rodent was assumed avian, so an evening
# of Allonemobius tinnulus reported 145 birds heard when the
# microphones heard no bird at all.
#
# Derived from models/BirdNET_v2.4_labels_en_us.txt, which contains 84
# entries in these genera. Genus is the only usable axis: the label
# file carries no taxonomic rank, and matching common names is a trap
# -- "Grasshopper Sparrow", "Cicadabird", "Squirrel Cuckoo" and
# "Cricket Longtail" are all birds. A BirdNET version bump can add
# taxa, so this list is tied to v2.4.
_AUDIO_NON_BIRD_GENERA: frozenset[str] = frozenset(
    {
        # Orthoptera: crickets, katydids, coneheads, trigs.
        "Allonemobius",
        "Amblycorypha",
        "Anaxipha",
        "Atlanticus",
        "Conocephalus",
        "Cyrtoxipha",
        "Eunemobius",
        "Gryllus",
        "Neoconocephalus",
        "Neonemobius",
        "Oecanthus",
        "Orchelimum",
        "Orocharis",
        "Phyllopalpus",
        "Pterophylla",
        "Scudderia",
        # Amphibians: frogs, toads, spadefoots.
        "Acris",
        "Anaxyrus",
        "Dryophytes",
        "Eleutherodactylus",
        "Gastrophryne",
        "Hyliola",
        "Incilius",
        "Lithobates",
        "Pseudacris",
        "Scaphiopus",
        "Spea",
        # Mammals that are not rodents.
        "Alouatta",
        "Canis",
        "Odocoileus",
    }
)


def categorize(
    modality: str,
    class_name: str | None,
    species: str | None,
) -> str | None:
    """Map one detection onto "bird", "rodent" or "other".

    Returns None for records that name no animal at all, which the
    caller must not count. Audio records are judged by species because
    the unified view reports their class as "bird" regardless -- a
    window with no species is one BirdNET could not identify, a stored
    sample rather than a bird, and counting it as one made every quiet
    evening look like a dawn chorus.

    None is also returned for the non-avian taxa BirdNET can name --
    insects, amphibians and non-rodent mammals -- for the same reason:
    they are identified sounds, but they are not visitors to the
    feeder. Rodents are the exception, and are counted, because a rat
    or a squirrel at the feeder is exactly what this system is for.
    """
    if modality == "audio":
        if species is None:
            return None
        if species in _AUDIO_NON_ANIMAL:
            return None
        genus = species.split()[0] if species else ""
        if genus in _AUDIO_RODENT_GENERA:
            return "rodent"
        if genus in _AUDIO_NON_BIRD_GENERA:
            return None
        # A BirdNET label with no space is a sound class, not a binomial.
        if " " not in species:
            return "other"
        return "bird"

    if class_name is None:
        return None
    return _VIDEO_CATEGORY.get(class_name, "other")


def get_category_counts(
    db: DetectionDatabase,
    since: str | None = None,
) -> CategoryCounts:
    """Count detections by category and by sense for a time window.

    Parameters
    ----------
    db:
        Open database connection.
    since:
        ISO-8601 lower bound. None counts every record ever stored.
    """
    seen: dict[str, int] = {"bird": 0, "rodent": 0, "other": 0}
    heard: dict[str, int] = {"bird": 0, "rodent": 0, "other": 0}

    for row in db.get_modality_counts(since=since):
        category = categorize(
            row["modality"], row["class_name"], row["species"]
        )
        if category is None:
            continue
        bucket = heard if row["modality"] == "audio" else seen
        bucket[category] += row["count"]

    return CategoryCounts(
        since=since if since is not None else "1970-01-01T00:00:00",
        bird=ModalityCount(seen=seen["bird"], heard=heard["bird"]),
        rodent=ModalityCount(seen=seen["rodent"], heard=heard["rodent"]),
        other=ModalityCount(seen=seen["other"], heard=heard["other"]),
    )


def get_stats(
    db: DetectionDatabase,
    hours: float | None = None,
    label: str | None = None,
) -> DetectionStats:
    """Compute detection statistics for a time window.

    Parameters
    ----------
    db:
        Open database connection.
    hours:
        Number of hours to look back. None means all time.
    label:
        Human-readable label for the time window.
    """
    if hours is not None:
        since_dt = datetime.now() - timedelta(hours=hours)
        since = since_dt.isoformat()
        window_hours = hours
        if label is None:
            label = f"last {hours:.0f}h"
    else:
        since = "1970-01-01T00:00:00"
        window_hours = 1.0
        if label is None:
            label = "all time"

    total = db.get_detection_count(since=since)
    counts = db.get_species_counts(since=since)
    classes = db.get_class_counts(since=since)

    if hours is not None and hours > 0:
        per_hour = total / hours
    elif total > 0:
        per_hour = total / max(window_hours, 1.0)
    else:
        per_hour = 0.0

    return DetectionStats(
        window_label=label,
        since=since,
        total_detections=total,
        species_counts=counts,
        detections_per_hour=per_hour,
        class_counts=classes,
    )


def format_stats(stats: DetectionStats) -> str:
    """Format statistics as a human-readable string."""
    lines = [
        f"Detection Statistics ({stats.window_label})",
        "=" * 50,
        f"Total detections: {stats.total_detections}",
        f"  Birds: {stats.bird_count}",
        f"  Pests: {stats.pest_count}",
        f"  Rate: {stats.detections_per_hour:.1f} detections/hour",
    ]

    if stats.class_counts:
        lines.append("")
        lines.append("By class:")
        for class_name, count in sorted(
            stats.class_counts.items(), key=lambda x: x[1], reverse=True
        ):
            lines.append(f"  {class_name}: {count}")

    # The "unknown" bucket is every row with no species, which is all
    # the pests as well as the birds the classifier passed on. The
    # class breakdown above already accounts for them, and listing the
    # bucket here would read as a species.
    named = [(s, c) for s, c in stats.top_species if s != "unknown"]
    if named:
        lines.append("")
        lines.append("By species:")
        for species, count in named:
            lines.append(f"  {species}: {count}")

    return "\n".join(lines)
