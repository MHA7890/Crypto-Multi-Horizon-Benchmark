"""
forecasting.training.trainer — Single unit trainer.
"""

from __future__ import annotations

import logging
import pandas as pd

from forecasting.evaluation.metrics import ForecastMetrics, MetricsCalculator
from forecasting.models.base import ForecastModel
from forecasting.utils.timing import Timer

logger = logging.getLogger(__name__)


class SingleUnitTrainer:
    """Trains a single model on train fold and evaluates on validation fold."""

    def __init__(self):
        self.metrics_calculator = MetricsCalculator()

    def train_and_evaluate(
        self,
        model: ForecastModel,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        horizon: int,
    ) -> ForecastMetrics:
        """Fit model on X_train/y_train and compute ForecastMetrics on X_val/y_val."""
        fit_result = model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        with Timer() as timer:
            prediction = model.predict(X_val, horizon=horizon)

        metrics = self.metrics_calculator.compute(
            y_true=y_val.values,
            prediction=prediction,
            fit_result=fit_result,
            inference_time=timer.elapsed,
        )

        return metrics
