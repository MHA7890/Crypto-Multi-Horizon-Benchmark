"""
forecasting.utils.notifications — Modular Experiment Notification System.

Provides event hooks for experiment start, progress milestones (10 coins, 25%, 50%,
75%, 90%, 100%), coin completion, failure alerts, and experiment completion.
Supports pluggable backends (Console, File, Discord/Slack/Email Webhooks).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
from typing import Any

logger = logging.getLogger(__name__)


class BaseNotifier(ABC):
    """Abstract interface for experiment notifications."""

    @abstractmethod
    def notify(self, event_type: str, title: str, message: str, metadata: dict[str, Any] | None = None) -> None:
        """Send notification message."""
        ...


class ConsoleNotifier(BaseNotifier):
    """Outputs structured notification events to standard logger."""

    def notify(self, event_type: str, title: str, message: str, metadata: dict[str, Any] | None = None) -> None:
        logger.info("[NOTIFICATION: %s] %s — %s", event_type.upper(), title, message)


class FileNotifier(BaseNotifier):
    """Appends notification events to a notifications log file."""

    def __init__(self, log_path: str):
        self.log_path = log_path

    def notify(self, event_type: str, title: str, message: str, metadata: dict[str, Any] | None = None) -> None:
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(f"[{event_type.upper()}] {title}: {message}\n")
        except Exception as e:
            logger.debug("FileNotifier write error: %s", e)


class MultiNotifier(BaseNotifier):
    """Broadcasts notification events across multiple active backends."""

    def __init__(self, notifiers: list[BaseNotifier] | None = None):
        self.notifiers: list[BaseNotifier] = notifiers or [ConsoleNotifier()]

    def add_notifier(self, notifier: BaseNotifier) -> None:
        self.notifiers.append(notifier)

    def notify(self, event_type: str, title: str, message: str, metadata: dict[str, Any] | None = None) -> None:
        for n in self.notifiers:
            try:
                n.notify(event_type, title, message, metadata)
            except Exception as e:
                logger.error("Error in notifier %s: %s", type(n).__name__, e)

    # Event helper hooks
    def on_experiment_start(self, exp_id: str, total_coins: int, total_models: int) -> None:
        self.notify(
            "start",
            f"Experiment {exp_id} Started",
            f"Orchestrating benchmark across {total_coins} cryptocurrencies and {total_models} models.",
            {"exp_id": exp_id, "total_coins": total_coins, "total_models": total_models},
        )

    def on_milestone(self, milestone_name: str, completed: int, total: int, pct: float) -> None:
        self.notify(
            "milestone",
            f"Milestone Reached: {milestone_name}",
            f"Progress: {completed}/{total} coins processed ({pct:.1f}%).",
            {"completed": completed, "total": total, "pct": pct},
        )

    def on_failure(self, symbol: str, model: str, error_msg: str) -> None:
        self.notify(
            "failure",
            f"Execution Failure: {symbol} - {model}",
            f"Error encountered: {error_msg}",
            {"symbol": symbol, "model": model, "error": error_msg},
        )

    def on_experiment_finish(self, exp_id: str, elapsed_seconds: float, winners_summary: str) -> None:
        hours = elapsed_seconds / 3600.0
        self.notify(
            "completion",
            f"Experiment {exp_id} Completed Successfully",
            f"Total runtime: {hours:.2f} hours. Summary: {winners_summary}",
            {"exp_id": exp_id, "elapsed_seconds": elapsed_seconds},
        )

    def on_experiment_crash(self, exp_id: str, error_msg: str) -> None:
        self.notify(
            "crash",
            f"CRITICAL: Experiment {exp_id} Crashed",
            f"Unhandled exception: {error_msg}",
            {"exp_id": exp_id, "error": error_msg},
        )
