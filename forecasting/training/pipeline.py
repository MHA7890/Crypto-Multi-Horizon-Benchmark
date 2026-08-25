"""
forecasting.training.pipeline — Walk-forward training pipeline per symbol × model × horizon.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting.config.settings import ExperimentConfig
from forecasting.data.loader import DataLoader
from forecasting.data.target import TargetConstructor
from forecasting.data.splitter import WalkForwardSplitter
from forecasting.data.scaler import LeakproofScaler
from forecasting.data.reduction import FeatureReducer, ReductionReport
from forecasting.evaluation.metrics import ForecastMetrics
from forecasting.models.base import ForecastModel
from forecasting.training.trainer import SingleUnitTrainer

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of a single symbol × model × horizon training pipeline run."""

    symbol: str
    model_name: str
    horizon: int
    metrics: ForecastMetrics
    metrics_per_fold: list[ForecastMetrics]
    model_path: Path
    reduction_report: ReductionReport
    n_folds: int


class TrainingPipeline:
    """End-to-end training pipeline for ONE symbol × ONE model × ONE horizon."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.loader = DataLoader(features_dir=config.paths.features_dir)
        self.target_constructor = TargetConstructor(
            horizons_days=config.target.horizons,
            price_col=config.target.price_column,
        )
        self.splitter = WalkForwardSplitter(
            min_train_ratio=config.validation.min_train_ratio,
            val_size_ratio=config.validation.val_size_ratio,
            step_size_ratio=config.validation.step_size_ratio,
        )
        self.trainer = SingleUnitTrainer()

    def run(
        self,
        symbol: str,
        model: ForecastModel,
        horizon: int,
        output_dir: Path | None = None,
    ) -> PipelineResult:
        dataset = self.loader.load(symbol, price_col=self.config.target.price_column)
        targets = self.target_constructor.create_targets(dataset.features)

        if horizon not in targets:
            raise ValueError(f"Horizon {horizon} not found in constructed targets")

        target_info = targets[horizon]
        X_valid, y_valid, prices_valid = self.target_constructor.get_valid_features_and_target(
            dataset.features, target_info, drop_feature_nans=True
        )

        folds = self.splitter.split(X_valid.index)
        fold_metrics_list = []

        last_reducer = None
        last_scaler = None

        for fold in folds:
            X_train_fold = X_valid.loc[fold.train_indices]
            y_train_fold = y_valid.loc[fold.train_indices]

            X_val_fold = X_valid.loc[fold.val_indices]
            y_val_fold = y_valid.loc[fold.val_indices]

            # 1. Feature Reduction fit on fold train
            reducer = FeatureReducer(
                corr_threshold=self.config.reduction.correlation_threshold,
                variance_threshold=self.config.reduction.variance_threshold,
            )
            X_train_red, red_report = reducer.fit_transform(X_train_fold)
            X_val_red = reducer.transform(X_val_fold)
            last_reducer = reducer

            # 2. Scaling fit on fold train
            scaler = LeakproofScaler(quantile_range=self.config.scaling.quantile_range)
            X_train_scaled = scaler.fit_transform_train(X_train_red)
            X_val_scaled = scaler.transform_val(X_val_red)
            last_scaler = scaler

            # 3. Train & Evaluate
            fold_metrics = self.trainer.train_and_evaluate(
                model=model,
                X_train=X_train_scaled,
                y_train=y_train_fold,
                X_val=X_val_scaled,
                y_val=y_val_fold,
                horizon=horizon,
            )
            fold_metrics_list.append(fold_metrics)

        # Average metrics across folds
        avg_metrics = ForecastMetrics(
            rmse=float(np.mean([m.rmse for m in fold_metrics_list])),
            mae=float(np.mean([m.mae for m in fold_metrics_list])),
            median_ae=float(np.mean([m.median_ae for m in fold_metrics_list])),
            mape=float(np.mean([m.mape for m in fold_metrics_list])),
            picp=float(np.mean([m.picp for m in fold_metrics_list])),
            mpiw=float(np.mean([m.mpiw for m in fold_metrics_list])),
            direction_accuracy=float(np.mean([m.direction_accuracy for m in fold_metrics_list])),
            direction_precision=float(np.mean([m.direction_precision for m in fold_metrics_list])),
            direction_recall=float(np.mean([m.direction_recall for m in fold_metrics_list])),
            direction_f1=float(np.mean([m.direction_f1 for m in fold_metrics_list])),
            training_time_seconds=float(np.sum([m.training_time_seconds for m in fold_metrics_list])),
            inference_time_seconds=float(np.mean([m.inference_time_seconds for m in fold_metrics_list])),
            model_size_bytes=fold_metrics_list[-1].model_size_bytes if fold_metrics_list else 0,
        )

        # Retrain on full valid data for final artifact saving
        final_reducer = FeatureReducer(
            corr_threshold=self.config.reduction.correlation_threshold,
            variance_threshold=self.config.reduction.variance_threshold,
        )
        X_full_red, full_red_report = final_reducer.fit_transform(X_valid)
        final_scaler = LeakproofScaler(quantile_range=self.config.scaling.quantile_range)
        X_full_scaled = final_scaler.fit_transform_train(X_full_red)
        model.fit(X_full_scaled, y_valid)

        out_dir = output_dir or (self.config.paths.models_dir / symbol)
        model_path = model.save(out_dir)
        final_scaler.save(out_dir / f"{symbol}_{model.name}_scaler.joblib")
        final_reducer.save(out_dir / f"{symbol}_{model.name}_reducer.joblib")

        meta = {
            "symbol": symbol,
            "model_name": model.name,
            "horizon": horizon,
            "features_used": final_reducer.kept_features,
            "experiment_name": self.config.name,
        }
        with open(out_dir / f"{symbol}_{model.name}_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return PipelineResult(
            symbol=symbol,
            model_name=model.name,
            horizon=horizon,
            metrics=avg_metrics,
            metrics_per_fold=fold_metrics_list,
            model_path=model_path,
            reduction_report=full_red_report,
            n_folds=len(folds),
        )
