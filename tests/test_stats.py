"""Tests for the detection statistics the stats CLI prints.

No test doubles: a real SQLite database with real rows, and the real
formatter. The counts are asserted against rows shaped the way the
pipeline actually writes them -- in particular a pest row carries a
class name and no species, which is what broke the pest count.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ratcatcher.monitoring.stats import format_stats, get_stats
from ratcatcher.storage.database import DetectionDatabase


@pytest.fixture
def db(tmp_path: Path) -> DetectionDatabase:
    database = DetectionDatabase(tmp_path / "detections.db")
    yield database
    database.close()


def add(
    db: DetectionDatabase,
    class_name: str | None,
    count: int = 1,
    species: str | None = None,
    common_name: str | None = None,
) -> None:
    """Insert rows the way the storage loop does.

    Only the classification stage fills in a species. A squirrel is
    stored with a class name and nothing else, which is the whole
    reason a species-keyed pest count read zero.
    """
    for _ in range(count):
        db.insert_detection(
            timestamp="2026-08-28T10:00:00",
            camera_id=0,
            stage="classification" if species else "detection",
            class_name=class_name,
            species=species,
            common_name=common_name,
            confidence=0.8,
        )


def test_pests_are_counted_by_class_not_species(db: DetectionDatabase) -> None:
    """Pest rows carry no species. Counting by species reported zero."""
    add(db, "squirrel", count=12)
    add(db, "rat", count=3)
    add(db, "cat", count=1)

    stats = get_stats(db)

    assert stats.pest_count == 16
    assert stats.bird_count == 0


def test_unknown_animal_counts_as_a_pest(db: DetectionDatabase) -> None:
    """An animal that is not a bird is a pest, named or not."""
    add(db, "unknown_animal", count=4)

    assert get_stats(db).pest_count == 4


def test_birds_count_whether_or_not_a_species_was_named(db: DetectionDatabase) -> None:
    add(db, "bird", count=5, species="Junco hyemalis", common_name="Dark-eyed Junco")
    add(db, "bird", count=2)

    stats = get_stats(db)

    assert stats.bird_count == 7
    assert stats.pest_count == 0
    assert stats.species_counts["Junco hyemalis"] == 5
    assert stats.species_counts["unknown"] == 2


def test_birds_and_pests_sum_to_the_total(db: DetectionDatabase) -> None:
    add(db, "bird", count=9, species="Junco hyemalis")
    add(db, "squirrel", count=4)
    add(db, "cat", count=1)

    stats = get_stats(db)

    assert stats.total_detections == 14
    assert stats.bird_count + stats.pest_count == stats.total_detections


def test_motion_is_neither_bird_nor_pest_nor_total(db: DetectionDatabase) -> None:
    """Motion names no animal. It is a stored frame, not a sighting."""
    add(db, None, count=20)
    add(db, "bird", count=1)

    stats = get_stats(db)

    assert stats.total_detections == 1
    assert stats.bird_count == 1
    assert stats.pest_count == 0
    assert stats.class_counts == {"bird": 1}


def test_the_window_excludes_older_rows(db: DetectionDatabase) -> None:
    db.insert_detection(
        timestamp="2020-01-01T00:00:00",
        camera_id=0,
        stage="detection",
        class_name="squirrel",
    )

    assert get_stats(db, hours=1).pest_count == 0
    assert get_stats(db).pest_count == 1


def test_formatted_output_reports_pests_and_classes(db: DetectionDatabase) -> None:
    add(db, "squirrel", count=2)
    add(db, "bird", count=1, species="Junco hyemalis", common_name="Dark-eyed Junco")

    text = format_stats(get_stats(db, label="all time"))

    assert "Total detections: 3" in text
    assert "Birds: 1" in text
    assert "Pests: 2" in text
    assert "By class:" in text
    assert "  squirrel: 2" in text
    assert "  Junco hyemalis: 1" in text


def test_the_unknown_bucket_is_not_listed_as_a_species(db: DetectionDatabase) -> None:
    """Every pest has a NULL species. That bucket must not read as a bird."""
    add(db, "squirrel", count=2)

    text = format_stats(get_stats(db))

    assert "unknown" not in text
    assert "squirrel: 2" in text
