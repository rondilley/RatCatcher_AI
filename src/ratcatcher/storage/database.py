"""SQLite detection log with WAL mode for concurrent read support."""

from __future__ import annotations

import json
import logging
import math
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    camera_id INTEGER NOT NULL,
    stage TEXT NOT NULL,
    class_name TEXT,
    species TEXT,
    common_name TEXT,
    confidence REAL,
    bbox_x INTEGER,
    bbox_y INTEGER,
    bbox_w INTEGER,
    bbox_h INTEGER,
    clip_path TEXT,
    thumbnail_path TEXT,
    frame_width INTEGER,
    frame_height INTEGER,
    metadata TEXT
);
CREATE INDEX IF NOT EXISTS idx_detections_timestamp ON detections(timestamp);
CREATE INDEX IF NOT EXISTS idx_detections_species ON detections(species);
CREATE INDEX IF NOT EXISTS idx_detections_camera ON detections(camera_id);

-- Bird song identifications from the I2S microphones.
--
-- Audio detections live in their own table rather than sharing the
-- detections table above. They have no frame, no bounding box and no
-- camera, while they do have a channel, a duration and acoustic
-- measurements. Forcing both shapes into one table would mean a wide row
-- that is mostly NULL whichever modality wrote it, and would require
-- relaxing the NOT NULL constraint on camera_id. The detections_all view
-- below restores the unified query surface without that cost.
CREATE TABLE IF NOT EXISTS audio_detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    channel INTEGER NOT NULL,
    species TEXT,
    common_name TEXT,
    confidence REAL,
    duration_seconds REAL,
    band_rms_dbfs REAL,
    spectral_flatness REAL,
    peak_frequency_hz REAL,
    clip_path TEXT,
    metadata TEXT
);
CREATE INDEX IF NOT EXISTS idx_audio_timestamp ON audio_detections(timestamp);
CREATE INDEX IF NOT EXISTS idx_audio_species ON audio_detections(species);
CREATE INDEX IF NOT EXISTS idx_audio_channel ON audio_detections(channel);

-- Unified view across both modalities, so "what species were here today"
-- is one query rather than two. source_id is the camera for video rows
-- and the microphone channel for audio rows.
CREATE VIEW IF NOT EXISTS detections_all AS
    SELECT
        'video'      AS modality,
        id           AS id,
        timestamp    AS timestamp,
        camera_id    AS source_id,
        class_name   AS class_name,
        species      AS species,
        common_name  AS common_name,
        confidence   AS confidence,
        clip_path    AS clip_path
    FROM detections
    UNION ALL
    SELECT
        'audio'      AS modality,
        id           AS id,
        timestamp    AS timestamp,
        channel      AS source_id,
        'bird'       AS class_name,
        species      AS species,
        common_name  AS common_name,
        confidence   AS confidence,
        clip_path    AS clip_path
    FROM audio_detections;
"""


class DetectionDatabase:
    """Persistent SQLite store for wildlife detection events.

    Uses WAL journal mode so readers never block writers and vice-versa.
    Intended to be used as a context manager::

        with DetectionDatabase("detections.db") as db:
            db.insert_detection(...)
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            self._conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA_SQL)
            self._conn.commit()
        except sqlite3.Error as exc:
            logger.error("Failed to open database at %s: %s", self._db_path, exc)
            raise

        logger.info("Detection database opened at %s", self._db_path)

    # -- context manager ---------------------------------------------------

    def __enter__(self) -> DetectionDatabase:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self.close()

    # -- public API ---------------------------------------------------------

    def insert_detection(
        self,
        *,
        timestamp: str,
        camera_id: int,
        stage: str,
        class_name: str | None = None,
        species: str | None = None,
        common_name: str | None = None,
        confidence: float | None = None,
        bbox_x: int | None = None,
        bbox_y: int | None = None,
        bbox_w: int | None = None,
        bbox_h: int | None = None,
        clip_path: str | None = None,
        thumbnail_path: str | None = None,
        frame_width: int | None = None,
        frame_height: int | None = None,
        metadata: dict | None = None,
    ) -> int:
        """Insert a detection record and return the new row ID."""
        metadata_json = json.dumps(metadata) if metadata is not None else None
        try:
            cursor = self._conn.execute(
                """
                INSERT INTO detections (
                    timestamp, camera_id, stage, class_name, species,
                    common_name, confidence,
                    bbox_x, bbox_y, bbox_w, bbox_h,
                    clip_path, thumbnail_path,
                    frame_width, frame_height, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp, camera_id, stage, class_name, species,
                    common_name, confidence,
                    bbox_x, bbox_y, bbox_w, bbox_h,
                    clip_path, thumbnail_path,
                    frame_width, frame_height, metadata_json,
                ),
            )
            self._conn.commit()
            row_id: int = cursor.lastrowid  # type: ignore[assignment]
            logger.debug("Inserted detection row %d", row_id)
            return row_id
        except sqlite3.Error as exc:
            logger.error("Failed to insert detection: %s", exc)
            raise

    def insert_audio_detection(
        self,
        *,
        timestamp: str,
        channel: int,
        species: str | None = None,
        common_name: str | None = None,
        confidence: float | None = None,
        duration_seconds: float | None = None,
        band_rms_dbfs: float | None = None,
        spectral_flatness: float | None = None,
        peak_frequency_hz: float | None = None,
        clip_path: str | None = None,
        metadata: dict | None = None,
    ) -> int:
        """Insert a bird song identification and return the new row ID.

        The acoustic measurements are stored alongside the identification
        so the activity gate can be retuned against real recordings later
        rather than by guesswork.
        """
        metadata_json = json.dumps(metadata) if metadata is not None else None

        # SQLite has no concept of infinity for REAL columns, and a silent
        # channel legitimately measures as -inf dBFS. Store it as NULL.
        band_rms = _finite_or_none(band_rms_dbfs)

        try:
            cursor = self._conn.execute(
                """
                INSERT INTO audio_detections (
                    timestamp, channel, species, common_name, confidence,
                    duration_seconds, band_rms_dbfs, spectral_flatness,
                    peak_frequency_hz, clip_path, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp, channel, species, common_name, confidence,
                    duration_seconds, band_rms, spectral_flatness,
                    peak_frequency_hz, clip_path, metadata_json,
                ),
            )
            self._conn.commit()
            row_id: int = cursor.lastrowid  # type: ignore[assignment]
            logger.debug("Inserted audio detection row %d", row_id)
            return row_id
        except sqlite3.Error as exc:
            logger.error("Failed to insert audio detection: %s", exc)
            raise

    def get_audio_detections(
        self,
        since: str | None = None,
        channel: int | None = None,
        species: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query bird song identifications with optional filters.

        Parameters
        ----------
        since : str or None
            ISO-8601 timestamp lower bound (inclusive).
        channel : int or None
            Restrict to one microphone (0 is left, 1 is right).
        species : str or None
            Restrict to a single scientific name.
        limit : int
            Maximum number of rows to return (default 100).
        """
        clauses: list[str] = []
        params: list[object] = []

        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since)
        if channel is not None:
            clauses.append("channel = ?")
            params.append(channel)
        if species is not None:
            clauses.append("species = ?")
            params.append(species)

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        query = (
            f"SELECT * FROM audio_detections{where} "
            f"ORDER BY timestamp DESC LIMIT ?"
        )
        params.append(limit)

        try:
            rows = self._conn.execute(query, params).fetchall()
        except sqlite3.Error as exc:
            logger.error("Failed to query audio detections: %s", exc)
            raise

        return [_decode_metadata(dict(row)) for row in rows]

    def get_all_detections(
        self,
        since: str | None = None,
        modality: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query across both modalities via the detections_all view.

        Parameters
        ----------
        since : str or None
            ISO-8601 timestamp lower bound (inclusive).
        modality : str or None
            Restrict to "video" or "audio".
        limit : int
            Maximum number of rows to return (default 100).
        """
        if modality is not None and modality not in ("video", "audio"):
            raise ValueError(
                f"modality must be 'video', 'audio' or None, got {modality!r}"
            )

        clauses: list[str] = []
        params: list[object] = []

        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since)
        if modality is not None:
            clauses.append("modality = ?")
            params.append(modality)

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        query = (
            f"SELECT * FROM detections_all{where} "
            f"ORDER BY timestamp DESC LIMIT ?"
        )
        params.append(limit)

        try:
            rows = self._conn.execute(query, params).fetchall()
        except sqlite3.Error as exc:
            logger.error("Failed to query unified detections: %s", exc)
            raise

        return [dict(row) for row in rows]

    def get_detections(
        self,
        since: str | None = None,
        camera_id: int | None = None,
        species: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Query detections with optional filters.

        Parameters
        ----------
        since : str or None
            ISO-8601 timestamp lower bound (inclusive).
        camera_id : int or None
            Restrict to a single camera.
        species : str or None
            Restrict to a single species.
        limit : int
            Maximum number of rows to return (default 100).
        """
        clauses: list[str] = []
        params: list[object] = []

        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since)
        if camera_id is not None:
            clauses.append("camera_id = ?")
            params.append(camera_id)
        if species is not None:
            clauses.append("species = ?")
            params.append(species)

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        query = (
            f"SELECT * FROM detections{where} "
            f"ORDER BY timestamp DESC LIMIT ?"
        )
        params.append(limit)

        try:
            rows = self._conn.execute(query, params).fetchall()
        except sqlite3.Error as exc:
            logger.error("Failed to query detections: %s", exc)
            raise

        results: list[dict] = []
        for row in rows:
            record = dict(row)
            if record.get("metadata") is not None:
                try:
                    record["metadata"] = json.loads(record["metadata"])
                except (json.JSONDecodeError, TypeError):
                    pass  # leave as raw string if it cannot be decoded
            results.append(record)
        return results

    def get_species_counts(self, since: str | None = None) -> dict[str, int]:
        """Return a mapping of species name to detection count."""
        params: list[object] = []
        where = ""
        if since is not None:
            where = " WHERE timestamp >= ?"
            params.append(since)

        query = (
            f"SELECT species, COUNT(*) as cnt FROM detections{where} "
            f"GROUP BY species ORDER BY cnt DESC"
        )

        try:
            rows = self._conn.execute(query, params).fetchall()
        except sqlite3.Error as exc:
            logger.error("Failed to query species counts: %s", exc)
            raise

        counts: dict[str, int] = {}
        for row in rows:
            species_name = row["species"] if row["species"] is not None else "unknown"
            counts[species_name] = row["cnt"]
        return counts

    def get_modality_counts(self, since: str | None = None) -> list[dict]:
        """Return detection counts grouped by modality, class and species.

        Reads the ``detections_all`` view, so one call covers both the
        cameras and the microphones. Rows with no class name are motion
        events that never reached the detector, and are excluded: they
        record that something moved, not that an animal was identified.

        The species column is returned alongside the class name because
        the view labels every audio row ``bird``, and BirdNET also emits
        non-bird labels. The caller needs the species to tell those apart.

        Parameters
        ----------
        since : str or None
            ISO-8601 timestamp lower bound (inclusive).

        Returns
        -------
        A list of dicts with keys ``modality``, ``class_name``,
        ``species`` and ``count``.
        """
        params: list[object] = []
        clauses = ["class_name IS NOT NULL"]
        if since is not None:
            clauses.append("timestamp >= ?")
            params.append(since)

        query = (
            "SELECT modality, class_name, species, COUNT(*) AS cnt "
            f"FROM detections_all WHERE {' AND '.join(clauses)} "
            "GROUP BY modality, class_name, species"
        )

        try:
            rows = self._conn.execute(query, params).fetchall()
        except sqlite3.Error as exc:
            logger.error("Failed to query modality counts: %s", exc)
            raise

        return [
            {
                "modality": row["modality"],
                "class_name": row["class_name"],
                "species": row["species"],
                "count": int(row["cnt"]),
            }
            for row in rows
        ]

    def get_detection_count(self, since: str | None = None) -> int:
        """Return the total number of detections, optionally since a timestamp."""
        params: list[object] = []
        where = ""
        if since is not None:
            where = " WHERE timestamp >= ?"
            params.append(since)

        query = f"SELECT COUNT(*) as cnt FROM detections{where}"

        try:
            row = self._conn.execute(query, params).fetchone()
        except sqlite3.Error as exc:
            logger.error("Failed to count detections: %s", exc)
            raise

        return int(row["cnt"])

    def close(self) -> None:
        """Close the database connection."""
        try:
            self._conn.close()
            logger.info("Detection database closed")
        except sqlite3.Error as exc:
            logger.error("Error closing database: %s", exc)
            raise


def _finite_or_none(value: float | None) -> float | None:
    """Map non-finite floats to None so SQLite stores them as NULL."""
    if value is None:
        return None
    if not math.isfinite(value):
        return None
    return value


def _decode_metadata(record: dict) -> dict:
    """Decode a row's JSON metadata column in place, tolerating bad data."""
    if record.get("metadata") is not None:
        try:
            record["metadata"] = json.loads(record["metadata"])
        except (json.JSONDecodeError, TypeError):
            pass  # leave as raw string if it cannot be decoded
    return record
