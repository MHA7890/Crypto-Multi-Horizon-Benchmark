"""
forecasting.training.checkpoint — Robust Checkpoint & Resume System.

Saves experiment progress after every completed model/coin to allow seamless
resuming after interruptions, hardware crashes, or manual pauses.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def compute_config_hash(config_dict: dict[str, Any]) -> str:
    """Generate MD5 hash string representing the experiment configuration."""
    raw_str = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.md5(raw_str.encode("utf-8")).hexdigest()


class CheckpointManager:
    """
    Manages experiment progress checkpointing and automatic resumption.
    """

    def __init__(self, checkpoint_path: Path | str, config_hash: str = ""):
        self.checkpoint_path = Path(checkpoint_path)
        self.config_hash = config_hash

        self.exp_id: str = self.checkpoint_path.parent.name
        self.created_at: str = datetime.now(timezone.utc).isoformat()
        self.last_updated_at: str = self.created_at

        self.total_coins: int = 0
        self.completed_coins: list[str] = []
        self.completed_pairs: set[str] = set()  # Set of "SYMBOL:MODEL"
        self.failed_coins: dict[str, str] = {}
        self.failed_models: dict[str, str] = {}
        self.elapsed_seconds: float = 0.0

        if self.checkpoint_path.exists():
            self.load()

    def is_model_completed(self, symbol: str, model_name: str) -> bool:
        return f"{symbol}:{model_name}" in self.completed_pairs

    def is_coin_completed(self, symbol: str) -> bool:
        return symbol in self.completed_coins

    def mark_model_completed(self, symbol: str, model_name: str) -> None:
        self.completed_pairs.add(f"{symbol}:{model_name}")
        self.save()

    def record_model_completion(self, symbol: str, model_name: str) -> None:
        self.mark_model_completed(symbol, model_name)

    def mark_coin_completed(self, symbol: str) -> None:
        if symbol not in self.completed_coins:
            self.completed_coins.append(symbol)
        self.save()

    def record_coin_completion(self, symbol: str, winner_model: str = "") -> None:
        self.mark_coin_completed(symbol)

    def record_failure(
        self, symbol: str, model_name: str, error_msg: str, traceback_str: str = ""
    ) -> None:
        pair_key = f"{symbol}:{model_name}" if model_name else symbol
        self.failed_models[pair_key] = f"{error_msg}\n{traceback_str}"
        self.save()

    def save(self) -> None:
        self.last_updated_at = datetime.now(timezone.utc).isoformat()
        data = {
            "exp_id": self.exp_id,
            "created_at": self.created_at,
            "last_updated_at": self.last_updated_at,
            "config_hash": self.config_hash,
            "total_coins": self.total_coins,
            "completed_coins": self.completed_coins,
            "completed_pairs": list(self.completed_pairs),
            "failed_coins": self.failed_coins,
            "failed_models": self.failed_models,
            "elapsed_seconds": self.elapsed_seconds,
        }
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.debug("Saved checkpoint to %s", self.checkpoint_path)

    def load(self) -> None:
        with open(self.checkpoint_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.exp_id = data.get("exp_id", self.exp_id)
        self.created_at = data.get("created_at", self.created_at)
        self.last_updated_at = data.get("last_updated_at", self.last_updated_at)
        self.config_hash = data.get("config_hash", self.config_hash)
        self.total_coins = data.get("total_coins", 0)
        self.completed_coins = data.get("completed_coins", [])
        self.completed_pairs = set(data.get("completed_pairs", []))
        self.failed_coins = data.get("failed_coins", {})
        self.failed_models = data.get("failed_models", {})
        self.elapsed_seconds = data.get("elapsed_seconds", 0.0)

        logger.info(
            "Resumed checkpoint from %s (%d coins, %d model runs completed)",
            self.checkpoint_path,
            len(self.completed_coins),
            len(self.completed_pairs),
        )
