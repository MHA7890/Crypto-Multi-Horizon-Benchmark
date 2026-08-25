"""
forecasting.optimization.tuner — Optuna wrapper for hyperparameter optimization.
"""

from __future__ import annotations

import logging
from forecasting.optimization.search_spaces import SEARCH_SPACES

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Optuna hyperparameter tuning scaffold."""

    def __init__(self, model_name: str, search_space: dict | None = None):
        self.model_name = model_name
        self.search_space = search_space or SEARCH_SPACES.get(model_name, {})

    def optimize(self, symbol: str, horizon: int, n_trials: int = 50) -> dict:
        """Run Optuna study and return best hyperparameter dict."""
        logger.info(
            "Hyperparameter optimization scaffold invoked for %s (%s, horizon=%d)",
            self.model_name,
            symbol,
            horizon,
        )
        return {}
