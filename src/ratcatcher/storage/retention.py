"""Disk space management for clips and thumbnails."""

from __future__ import annotations

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class RetentionManager:
    """Delete old media files and enforce a disk-usage cap.

    Parameters
    ----------
    clip_dir : Path
        Directory containing video clip files.
    thumbnail_dir : Path
        Directory containing thumbnail images.
    retention_days : int
        Maximum age in days before a file is eligible for deletion.
    max_disk_gb : float
        Maximum combined size (in gigabytes) of clip_dir and thumbnail_dir.
    """

    def __init__(
        self,
        clip_dir: Path,
        thumbnail_dir: Path,
        retention_days: int,
        max_disk_gb: float,
    ) -> None:
        self._clip_dir = Path(clip_dir)
        self._thumbnail_dir = Path(thumbnail_dir)
        self._retention_days = retention_days
        self._max_disk_gb = max_disk_gb

    # -- public API ---------------------------------------------------------

    def cleanup(self) -> int:
        """Remove old or excess media files.

        1. Delete every file whose modification time is older than
           *retention_days*.
        2. If total disk usage still exceeds *max_disk_gb*, delete the
           oldest files one-by-one until the usage drops below the cap.

        Returns
        -------
        int
            The total number of files deleted.
        """
        deleted = 0

        # Phase 1 -- age-based cleanup
        cutoff = time.time() - (self._retention_days * 86400)
        deleted += self._delete_files_older_than(cutoff)

        # Phase 2 -- disk-cap cleanup
        if self.get_disk_usage_gb() > self._max_disk_gb:
            deleted += self._delete_until_under_cap()

        if deleted:
            logger.info("Retention cleanup removed %d file(s)", deleted)
        else:
            logger.debug("Retention cleanup: nothing to delete")

        return deleted

    def get_disk_usage_gb(self) -> float:
        """Return combined size of clip_dir and thumbnail_dir in gigabytes."""
        total_bytes = self._dir_size(self._clip_dir) + self._dir_size(self._thumbnail_dir)
        return total_bytes / (1024 ** 3)

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _dir_size(directory: Path) -> int:
        """Sum of file sizes (bytes) for every file under *directory*."""
        if not directory.exists():
            return 0
        total = 0
        try:
            for entry in directory.rglob("*"):
                if entry.is_file():
                    try:
                        total += entry.stat().st_size
                    except OSError:
                        # File may have been deleted between rglob and stat.
                        continue
        except OSError as exc:
            logger.warning("Error scanning directory %s: %s", directory, exc)
        return total

    def _all_media_files(self) -> list[Path]:
        """Collect every file from both managed directories, sorted oldest first."""
        files: list[Path] = []
        for directory in (self._clip_dir, self._thumbnail_dir):
            if not directory.exists():
                continue
            try:
                for entry in directory.rglob("*"):
                    if entry.is_file():
                        files.append(entry)
            except OSError as exc:
                logger.warning("Error listing files in %s: %s", directory, exc)

        # Sort by modification time, oldest first.
        files.sort(key=lambda p: self._safe_mtime(p))
        return files

    @staticmethod
    def _safe_mtime(path: Path) -> float:
        """Return the file's mtime, or 0.0 if stat fails."""
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    def _delete_files_older_than(self, cutoff: float) -> int:
        """Delete files with mtime before *cutoff*. Return count deleted."""
        deleted = 0
        for fpath in self._all_media_files():
            if self._safe_mtime(fpath) < cutoff:
                try:
                    fpath.unlink()
                    logger.debug("Deleted (age): %s", fpath)
                    deleted += 1
                except OSError as exc:
                    logger.warning("Could not delete %s: %s", fpath, exc)
        return deleted

    def _delete_until_under_cap(self) -> int:
        """Delete oldest files one-by-one until usage is within cap."""
        deleted = 0
        for fpath in self._all_media_files():
            if self.get_disk_usage_gb() <= self._max_disk_gb:
                break
            try:
                fpath.unlink()
                logger.debug("Deleted (disk cap): %s", fpath)
                deleted += 1
            except OSError as exc:
                logger.warning("Could not delete %s: %s", fpath, exc)
        return deleted
