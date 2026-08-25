"""
forecasting.selection.selector — Ranks candidates, picks winner, and triggers archiving.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

import numpy as np

from forecasting.evaluation.metrics import ForecastMetrics
from forecasting.evaluation.scorer import CompositeScorer
from forecasting.selection.archiver import ModelArchiver

logger = logging.getLogger(__name__)


class ModelSelector:
    """Selects best model per coin according to CompositeScorer and triggers archiving of non-winners."""

    def __init__(
        self,
        models_dir: Path | str = "models",
        archive_dir: Path | str = "archive",
        scorer: CompositeScorer | None = None,
    ):
        self.models_dir = Path(models_dir)
        self.archiver = ModelArchiver(archive_dir=archive_dir)
        self.scorer = scorer or CompositeScorer()

    def select_and_archive(
        self,
        symbol: str,
        model_metrics: dict[str, Union[ForecastMetrics, list[ForecastMetrics]]],
    ) -> str:
        """Pick highest scoring model and archive remaining candidates."""
        if not model_metrics:
            raise ValueError(f"No metrics provided for model selection on symbol {symbol}")

        single_metrics: dict[str, ForecastMetrics] = {}
        for m_name, m_val in model_metrics.items():
            if isinstance(m_val, list):
                if not m_val:
                    continue
                single_metrics[m_name] = ForecastMetrics(
                    rmse=float(np.mean([m.rmse for m in m_val])),
                    mae=float(np.mean([m.mae for m in m_val])),
                    median_ae=float(np.mean([m.median_ae for m in m_val])),
                    mape=float(np.mean([m.mape for m in m_val])),
                    picp=float(np.mean([m.picp for m in m_val])),
                    mpiw=float(np.mean([m.mpiw for m in m_val])),
                    direction_accuracy=float(np.mean([m.direction_accuracy for m in m_val])),
                    direction_precision=float(np.mean([m.direction_precision for m in m_val])),
                    direction_recall=float(np.mean([m.direction_recall for m in m_val])),
                    direction_f1=float(np.mean([m.direction_f1 for m in m_val])),
                    training_time_seconds=float(np.sum([m.training_time_seconds for m in m_val])),
                    inference_time_seconds=float(np.mean([m.inference_time_seconds for m in m_val])),
                    model_size_bytes=m_val[-1].model_size_bytes,
                )
            elif isinstance(m_val, ForecastMetrics):
                single_metrics[m_name] = m_val

        if not single_metrics:
            raise ValueError(f"No valid ForecastMetrics found for symbol {symbol}")

        rankings = self.scorer.rank(single_metrics)
        winner_name, winner_score = rankings[0]

        logger.info(
            "Selected winner for %s: %s (Composite Score: %.4f)",
            symbol,
            winner_name,
            winner_score,
        )

        coin_dir = self.models_dir / symbol
        self.archiver.archive_losing_models(
            symbol=symbol, coin_dir=coin_dir, winner_model_name=winner_name
        )

        return winner_name
