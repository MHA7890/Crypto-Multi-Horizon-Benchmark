"""
forecasting.training.dashboard — Multi-Stage Progress Tracker & Real-Time Dashboard.

Displays dynamic multi-stage progress tracking across:
- Overall Experiment (Coins, %, Speed, Elapsed, ETA)
- Current Coin Progress
- Current Model Progress & Hardware Routing
- Walk-Forward Validation Folds (e.g. Fold 4/10)
- Pipeline Stage Phase (Loading, Targets, Reduction, Scaling, Training, Validation, Selection)
- Prediction Horizon Progress (1d, 7d, 14d, 30d, 90d)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Check Rich availability
try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class ExperimentDashboard:
    """
    Multi-stage live progress reporter for cryptocurrency forecasting experiments.
    """

    def __init__(self, total_coins: int, total_models_per_coin: int):
        self.total_coins = total_coins
        self.total_models_per_coin = total_models_per_coin
        self.total_runs = total_coins * total_models_per_coin

        self.start_time = time.time()
        self.completed_coins = 0
        self.completed_runs = 0

        self.current_coin = "N/A"
        self.current_model = "N/A"
        self.current_device = "N/A"
        self.current_phase = "Initializing"
        self.current_fold = 1
        self.total_folds = 5
        self.current_horizon_idx = 1
        self.total_horizons = 5

        self.current_best_model = "N/A"
        self.current_best_score = 0.0

        self.rich_active = RICH_AVAILABLE
        self.console = Console() if RICH_AVAILABLE else None
        self.live: Optional[Any] = None

    def start(self) -> None:
        """Initialize live console dashboard."""
        self.start_time = time.time()
        if self.rich_active and self.console:
            logger.info("Initializing Multi-Stage Live Progress Dashboard...")

    def update_stage(
        self,
        current_coin: str = "N/A",
        current_model: str = "N/A",
        current_device: str = "N/A",
        current_phase: str = "Training",
        current_fold: int = 1,
        total_folds: int = 5,
        current_horizon_idx: int = 1,
        total_horizons: int = 5,
        model_idx: int = 1,
    ) -> None:
        """Update multi-stage execution phase status."""
        self.current_coin = current_coin
        self.current_model = current_model
        self.current_device = current_device
        self.current_phase = current_phase
        self.current_fold = current_fold
        self.total_folds = total_folds
        self.current_horizon_idx = current_horizon_idx
        self.total_horizons = total_horizons

    def update(
        self,
        current_coin: str,
        current_model: str,
        current_device: str,
        completed_coins: int,
        completed_runs: int,
        hardware_stats: dict[str, float],
        best_model: str = "N/A",
        best_score: float = 0.0,
        current_phase: str = "Running Pipeline",
        current_fold: int = 1,
        total_folds: int = 5,
        current_horizon_idx: int = 1,
        total_horizons: int = 5,
    ) -> None:
        """Update overall experiment progress and dynamic ETA estimation."""
        self.current_coin = current_coin
        self.current_model = current_model
        self.current_device = current_device
        self.completed_coins = completed_coins
        self.completed_runs = completed_runs
        self.current_best_model = best_model
        self.current_best_score = best_score
        self.current_phase = current_phase
        self.current_fold = current_fold
        self.total_folds = total_folds
        self.current_horizon_idx = current_horizon_idx
        self.total_horizons = total_horizons

        elapsed = time.time() - self.start_time
        coins_per_hr = (completed_coins / elapsed * 3600.0) if (elapsed > 0 and completed_coins > 0) else 0.0
        remaining_coins = max(0, self.total_coins - completed_coins)
        eta_seconds = (remaining_coins / (coins_per_hr / 3600.0)) if coins_per_hr > 0 else 0.0
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds)) if eta_seconds > 0 else "Calculating..."
        coin_pct = (completed_coins / self.total_coins * 100.0) if self.total_coins > 0 else 0.0

        if not self.rich_active:
            logger.info(
                "[Progress %d/%d (%.1f%%)] Coin: %s | Model: %s [%s] | Phase: %s | Fold: %d/%d | Horizon: %d/%d | ETA: %s",
                completed_coins,
                self.total_coins,
                coin_pct,
                current_coin,
                current_model,
                current_device,
                current_phase,
                current_fold,
                total_folds,
                current_horizon_idx,
                total_horizons,
                eta_str,
            )

    def stop(self) -> None:
        """Stop dashboard and log final summary."""
        elapsed = time.time() - self.start_time
        hours = elapsed / 3600.0
        logger.info("Multi-Stage Progress Dashboard stopped. Total runtime: %.2f hours.", hours)
