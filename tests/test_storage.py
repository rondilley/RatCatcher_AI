"""Tests for storage subsystem: database, thumbnails, and retention.

Every test uses real SQLite databases, real JPEG encoding, and real
filesystem operations -- no fakes or substitutions.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from ratcatcher.storage.database import DetectionDatabase
from ratcatcher.storage.thumbnail import create_thumbnail
from ratcatcher.storage.retention import RetentionManager


# ---------------------------------------------------------------------------
# DetectionDatabase tests
# ---------------------------------------------------------------------------

class TestDetectionDatabase:

    def test_create_database(self, tmp_path: Path):
        """Creating a DetectionDatabase should produce a .db file on disk."""
        db_path = tmp_path / "det.db"
        db = DetectionDatabase(db_path)
        try:
            assert db_path.exists()
            assert db_path.stat().st_size > 0
        finally:
            db.close()

    def test_insert_and_retrieve(self, tmp_path: Path):
        """A round-trip insert then query should return all stored fields."""
        db_path = tmp_path / "det.db"
        db = DetectionDatabase(db_path)
        try:
            row_id = db.insert_detection(
                timestamp="2025-05-25T10:00:00",
                camera_id=0,
                stage="motion",
                class_name="bird",
                species="corvus_corax",
                common_name="Common Raven",
                confidence=0.92,
                bbox_x=100,
                bbox_y=200,
                bbox_w=50,
                bbox_h=60,
                clip_path="/tmp/clip1.mp4",
                thumbnail_path="/tmp/thumb1.jpg",
                frame_width=1920,
                frame_height=1080,
            )
            assert isinstance(row_id, int)
            assert row_id >= 1

            rows = db.get_detections()
            assert len(rows) == 1

            rec = rows[0]
            assert rec["timestamp"] == "2025-05-25T10:00:00"
            assert rec["camera_id"] == 0
            assert rec["stage"] == "motion"
            assert rec["class_name"] == "bird"
            assert rec["species"] == "corvus_corax"
            assert rec["common_name"] == "Common Raven"
            assert abs(rec["confidence"] - 0.92) < 1e-6
            assert rec["bbox_x"] == 100
            assert rec["bbox_y"] == 200
            assert rec["bbox_w"] == 50
            assert rec["bbox_h"] == 60
            assert rec["clip_path"] == "/tmp/clip1.mp4"
            assert rec["thumbnail_path"] == "/tmp/thumb1.jpg"
            assert rec["frame_width"] == 1920
            assert rec["frame_height"] == 1080
        finally:
            db.close()

    def test_insert_with_metadata(self, tmp_path: Path):
        """Metadata dict should survive JSON round-tripping through SQLite."""
        db_path = tmp_path / "det.db"
        db = DetectionDatabase(db_path)
        try:
            meta = {"tracker_id": 42, "notes": "seen near feeder", "tags": ["dawn", "repeat"]}
            db.insert_detection(
                timestamp="2025-05-25T10:05:00",
                camera_id=0,
                stage="classification",
                metadata=meta,
            )
            rows = db.get_detections()
            assert len(rows) == 1

            retrieved_meta = rows[0]["metadata"]
            assert isinstance(retrieved_meta, dict)
            assert retrieved_meta["tracker_id"] == 42
            assert retrieved_meta["notes"] == "seen near feeder"
            assert retrieved_meta["tags"] == ["dawn", "repeat"]
        finally:
            db.close()

    def test_get_detections_filter_by_camera(self, tmp_path: Path):
        """Filtering by camera_id should return only matching rows."""
        db_path = tmp_path / "det.db"
        db = DetectionDatabase(db_path)
        try:
            db.insert_detection(timestamp="2025-05-25T10:00:00", camera_id=0, stage="motion")
            db.insert_detection(timestamp="2025-05-25T10:01:00", camera_id=1, stage="motion")
            db.insert_detection(timestamp="2025-05-25T10:02:00", camera_id=0, stage="motion")

            cam0 = db.get_detections(camera_id=0)
            cam1 = db.get_detections(camera_id=1)

            assert len(cam0) == 2
            assert all(r["camera_id"] == 0 for r in cam0)
            assert len(cam1) == 1
            assert cam1[0]["camera_id"] == 1
        finally:
            db.close()

    def test_get_detections_filter_by_species(self, tmp_path: Path):
        """Filtering by species should isolate the correct rows."""
        db_path = tmp_path / "det.db"
        db = DetectionDatabase(db_path)
        try:
            db.insert_detection(
                timestamp="2025-05-25T10:00:00", camera_id=0,
                stage="classification", species="rattus_norvegicus",
            )
            db.insert_detection(
                timestamp="2025-05-25T10:01:00", camera_id=0,
                stage="classification", species="parus_major",
            )
            db.insert_detection(
                timestamp="2025-05-25T10:02:00", camera_id=0,
                stage="classification", species="rattus_norvegicus",
            )

            rats = db.get_detections(species="rattus_norvegicus")
            birds = db.get_detections(species="parus_major")

            assert len(rats) == 2
            assert all(r["species"] == "rattus_norvegicus" for r in rats)
            assert len(birds) == 1
            assert birds[0]["species"] == "parus_major"
        finally:
            db.close()

    def test_get_detections_filter_by_since(self, tmp_path: Path):
        """The since parameter should exclude detections before the cutoff."""
        db_path = tmp_path / "det.db"
        db = DetectionDatabase(db_path)
        try:
            db.insert_detection(timestamp="2025-05-25T08:00:00", camera_id=0, stage="motion")
            db.insert_detection(timestamp="2025-05-25T10:00:00", camera_id=0, stage="motion")
            db.insert_detection(timestamp="2025-05-25T12:00:00", camera_id=0, stage="motion")

            recent = db.get_detections(since="2025-05-25T09:00:00")
            assert len(recent) == 2
            # Results are ordered DESC; newest first.
            assert recent[0]["timestamp"] == "2025-05-25T12:00:00"
            assert recent[1]["timestamp"] == "2025-05-25T10:00:00"
        finally:
            db.close()

    def test_get_species_counts(self, tmp_path: Path):
        """get_species_counts should return correct tallies."""
        db_path = tmp_path / "det.db"
        db = DetectionDatabase(db_path)
        try:
            for _ in range(3):
                db.insert_detection(
                    timestamp="2025-05-25T10:00:00", camera_id=0,
                    stage="classification", species="rattus_norvegicus",
                )
            for _ in range(2):
                db.insert_detection(
                    timestamp="2025-05-25T10:00:00", camera_id=0,
                    stage="classification", species="parus_major",
                )

            counts = db.get_species_counts()
            assert counts["rattus_norvegicus"] == 3
            assert counts["parus_major"] == 2
        finally:
            db.close()

    def test_get_detection_count(self, tmp_path: Path):
        """get_detection_count should reflect total rows inserted."""
        db_path = tmp_path / "det.db"
        db = DetectionDatabase(db_path)
        try:
            assert db.get_detection_count() == 0

            for i in range(5):
                db.insert_detection(
                    timestamp=f"2025-05-25T10:0{i}:00",
                    camera_id=0,
                    stage="motion",
                )

            assert db.get_detection_count() == 5
        finally:
            db.close()

    def test_context_manager(self, tmp_path: Path):
        """DetectionDatabase should work as a context manager via with."""
        db_path = tmp_path / "det.db"
        with DetectionDatabase(db_path) as db:
            db.insert_detection(
                timestamp="2025-05-25T10:00:00",
                camera_id=0,
                stage="motion",
            )
            assert db.get_detection_count() == 1

        # After exiting the context the file should still exist on disk.
        assert db_path.exists()


# ---------------------------------------------------------------------------
# Thumbnail tests
# ---------------------------------------------------------------------------

class TestCreateThumbnail:

    def test_create_thumbnail_with_bbox(self, tmp_path: Path):
        """A JPEG thumbnail should be written with the bounding-box overlay."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add some non-black content so the thumbnail is not trivially empty.
        frame[100:200, 100:300] = (50, 120, 200)

        out = tmp_path / "thumb_bbox.jpg"
        result = create_thumbnail(frame, bbox=(100, 100, 200, 100), output_path=out)

        assert result == out
        assert out.exists()
        assert out.stat().st_size > 0

        # Read the thumbnail back and confirm it is a valid image.
        img = cv2.imread(str(out))
        assert img is not None
        assert img.ndim == 3

    def test_create_thumbnail_without_bbox(self, tmp_path: Path):
        """Thumbnail creation without a bbox should still succeed."""
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        out = tmp_path / "thumb_no_bbox.jpg"
        result = create_thumbnail(frame, bbox=None, output_path=out)

        assert result == out
        assert out.exists()
        assert out.stat().st_size > 0

    def test_create_thumbnail_respects_max_size(self, tmp_path: Path):
        """The longest edge of the thumbnail should equal max_size."""
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        out = tmp_path / "thumb_sized.jpg"
        max_size = 160
        create_thumbnail(frame, bbox=None, output_path=out, max_size=max_size)

        img = cv2.imread(str(out))
        assert img is not None
        h, w = img.shape[:2]
        longest = max(h, w)
        assert longest == max_size


# ---------------------------------------------------------------------------
# RetentionManager tests
# ---------------------------------------------------------------------------

class TestRetentionManager:

    def test_cleanup_old_files(self, tmp_path: Path):
        """Files older than retention_days should be deleted by cleanup()."""
        clip_dir = tmp_path / "clips"
        thumb_dir = tmp_path / "thumbs"
        clip_dir.mkdir()
        thumb_dir.mkdir()

        # Create files with modification times well in the past.
        old_clip = clip_dir / "old_clip.mp4"
        old_thumb = thumb_dir / "old_thumb.jpg"
        old_clip.write_bytes(b"\x00" * 1024)
        old_thumb.write_bytes(b"\x00" * 512)

        # Set their mtime to 60 days ago.
        sixty_days_ago = time.time() - (60 * 86400)
        os.utime(old_clip, (sixty_days_ago, sixty_days_ago))
        os.utime(old_thumb, (sixty_days_ago, sixty_days_ago))

        # Also create a recent file that should survive.
        recent = clip_dir / "recent_clip.mp4"
        recent.write_bytes(b"\x00" * 1024)

        mgr = RetentionManager(
            clip_dir=clip_dir,
            thumbnail_dir=thumb_dir,
            retention_days=30,
            max_disk_gb=100.0,  # large cap so only age matters
        )
        deleted = mgr.cleanup()

        assert deleted == 2
        assert not old_clip.exists()
        assert not old_thumb.exists()
        assert recent.exists()

    def test_disk_usage_calculation(self, tmp_path: Path):
        """get_disk_usage_gb should reflect file sizes in both directories."""
        clip_dir = tmp_path / "clips"
        thumb_dir = tmp_path / "thumbs"
        clip_dir.mkdir()
        thumb_dir.mkdir()

        # Write exactly 1 MiB into clips and 0.5 MiB into thumbs.
        mib = 1024 * 1024
        (clip_dir / "c1.bin").write_bytes(b"\x00" * mib)
        (thumb_dir / "t1.bin").write_bytes(b"\x00" * (mib // 2))

        mgr = RetentionManager(
            clip_dir=clip_dir,
            thumbnail_dir=thumb_dir,
            retention_days=30,
            max_disk_gb=100.0,
        )

        usage = mgr.get_disk_usage_gb()
        expected_gb = 1.5 * mib / (1024 ** 3)  # 1.5 MiB in GB
        assert abs(usage - expected_gb) < 1e-9
