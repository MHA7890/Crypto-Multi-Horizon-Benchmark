"""
forecasting.selection.archiver — Moves non-winning model artifacts into archive/{SYMBOL}/.
"""

from __future__ import annotations

import logging
from pathlib import Path
import shutil

logger = logging.getLogger(__name__)


class ModelArchiver:
    """Moves non-winning models to archive/{SYMBOL}/."""

    def __init__(self, archive_dir: Path | str = "archive"):
        self.archive_dir = Path(archive_dir)

    def archive_losing_models(
        self, symbol: str, coin_dir: Path, winner_model_name: str
    ) -> list[Path]:
        """Move all model files in coin_dir that do not belong to winner_model_name to archive_dir / symbol."""
        if not coin_dir.exists():
            return []

        target_archive = self.archive_dir / symbol
        target_archive.mkdir(parents=True, exist_ok=True)

        archived_files = []
        for filepath in coin_dir.glob("*"):
            if filepath.is_file():
                # Check if file belongs to winner
                if winner_model_name in filepath.name:
                    continue

                dest = target_archive / filepath.name
                shutil.move(str(filepath), str(dest))
                archived_files.append(dest)
                logger.debug("Archived %s -> %s", filepath.name, dest)

        return archived_files
