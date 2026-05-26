"""Detection statistics for RatCatcher AI."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from ratcatcher.storage.database import DetectionDatabase


@dataclass
class DetectionStats:
    """Summary statistics for a time window."""

    window_label: str
    since: str
    total_detections: int
    species_counts: dict[str, int]
    detections_per_hour: float

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
        """Total detections classified as any bird species."""
        pest_names = {"squirrel", "rat", "cat", "unknown_animal", "unknown"}
        return sum(
            count for species, count in self.species_counts.items()
            if species not in pest_names
        )

    @property
    def pest_count(self) -> int:
        """Total detections classified as pest animals."""
        pest_names = {"squirrel", "rat", "cat", "unknown_animal"}
        return sum(
            count for species, count in self.species_counts.items()
            if species in pest_names
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

    if stats.species_counts:
        lines.append("")
        lines.append("By species:")
        for species, count in stats.top_species:
            lines.append(f"  {species}: {count}")

    return "\n".join(lines)
