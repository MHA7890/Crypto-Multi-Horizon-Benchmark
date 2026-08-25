"""
End-to-End Integration Test for RandomForestModel.

Full Pipeline Flow:
  Load BTC
    ↓
  Generate Multi-Horizon Log-Return Targets (1, 7, 14, 30, 90 days)
    ↓
  Feature Reduction (Near-Zero Variance + Pair Correlation Filter fit on train)
    ↓
  Walk-Forward Validation Split (Expanding Window, zero data leakage)
    ↓
  Leakproof Robust Scaling (Fit on train ONLY)
    ↓
  Train RandomForestModel
    ↓
  Evaluate Metrics (Forecast Accuracy, Interval Quality, Directional, Efficiency)
    ↓
  Save Model & Metadata
    ↓
  Reload Model
    ↓
  Run Production Predictor Inference
    ↓
  Convert Log Returns to Price Prediction Intervals (P_lower < P_median < P_upper)
"""

import json
from pathlib import Path
import shutil
import unittest
import numpy as np
import pandas as pd

from forecasting.config.loader import _load_model_defaults
from forecasting.data.loader import DataLoader
from forecasting.data.target import TargetConstructor
from forecasting.data.splitter import WalkForwardSplitter
from forecasting.data.scaler import LeakproofScaler
from forecasting.data.reduction import FeatureReducer
from forecasting.evaluation.metrics import MetricsCalculator
from forecasting.inference.converter import ReturnToPriceConverter
from forecasting.inference.predictor import Predictor
from forecasting.models.random_forest import RandomForestModel
from forecasting.training.trainer import SingleUnitTrainer


class TestRandomForestE2EIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path("test_sandbox_rf_e2e")
        cls.models_dir = cls.test_dir / "models"

        # Verify presence of real BTC dataset
        cls.has_real_btc = Path("features/BTC_features.csv").exists()

    def setUp(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_random_forest_e2e_pipeline_real_btc(self):
        """Execute complete pipeline from raw BTC features CSV to price interval inference."""
        if not self.has_real_btc:
            self.skipTest("features/BTC_features.csv not present")

        # 1. CSV Loading
        loader = DataLoader(features_dir="features")
        dataset = loader.load("BTC", price_col="close")
        self.assertEqual(dataset.symbol, "BTC")
        self.assertGreater(dataset.n_rows, 1000)

        # 2. Target Construction (1, 7, 14, 30, 90 days)
        horizons_days = [1, 7, 14, 30, 90]
        target_constructor = TargetConstructor(
            horizons_days=horizons_days,
            price_col="close",
            steps_per_day=24,  # hourly candles
        )
        targets = target_constructor.create_targets(dataset.features)
        self.assertEqual(set(targets.keys()), set(horizons_days))

        # Select 1-day horizon for end-to-end execution
        target_1d = targets[1]
        X_valid, y_valid, prices_valid = target_constructor.get_valid_features_and_target(
            dataset.features, target_1d, drop_feature_nans=True
        )
        self.assertEqual(len(X_valid), len(y_valid))

        # 3. Walk-Forward Validation Splitter
        splitter = WalkForwardSplitter(
            min_train_ratio=0.7,
            val_size_ratio=0.1,
            step_size_ratio=0.1,
        )
        folds = splitter.split(X_valid.index)
        self.assertGreater(len(folds), 0)

        fold0 = folds[0]
        X_train_raw = X_valid.loc[fold0.train_indices]
        y_train = y_valid.loc[fold0.train_indices]
        X_val_raw = X_valid.loc[fold0.val_indices]
        y_val = y_valid.loc[fold0.val_indices]

        # Strict temporal order assertion
        self.assertLess(fold0.train_end, fold0.val_start)

        # 4. Feature Reduction (Fit on fold train ONLY)
        reducer = FeatureReducer(corr_threshold=0.95, variance_threshold=1e-6)
        X_train_red, reduction_report = reducer.fit_transform(X_train_raw)
        X_val_red = reducer.transform(X_val_raw)

        self.assertLessEqual(X_train_red.shape[1], X_train_raw.shape[1])

        # 5. Leakproof Robust Scaling (Fit on fold train ONLY)
        scaler = LeakproofScaler()
        X_train_scaled = scaler.fit_transform_train(X_train_red)
        X_val_scaled = scaler.transform_val(X_val_red)

        self.assertFalse(X_train_scaled.isna().any().any())
        self.assertFalse(X_val_scaled.isna().any().any())

        # 6. Train RandomForestModel using YAML defaults
        defaults = _load_model_defaults().get("random_forest", {})
        rf_params = defaults.copy()
        rf_params["n_estimators"] = 20  # Fast estimator count for integration test
        rf_params["random_state"] = 42

        rf_model = RandomForestModel(**rf_params)
        fit_result = rf_model.fit(X_train_scaled, y_train)

        self.assertTrue(rf_model.is_fitted)
        self.assertGreater(fit_result.training_time_seconds, 0)

        # 7. Evaluate Metrics
        trainer = SingleUnitTrainer()
        metrics = trainer.train_and_evaluate(
            model=rf_model,
            X_train=X_train_scaled,
            y_train=y_train,
            X_val=X_val_scaled,
            y_val=y_val,
            horizon=1,
        )

        self.assertGreater(metrics.rmse, 0)
        self.assertGreater(metrics.mae, 0)
        self.assertGreaterEqual(metrics.picp, 0.0)
        self.assertLessEqual(metrics.picp, 1.0)
        self.assertGreater(metrics.mpiw, 0)

        # 8. Save Model & Metadata
        btc_models_dir = self.models_dir / "BTC"
        saved_path = rf_model.save(btc_models_dir)
        scaler.save(btc_models_dir / "BTC_RandomForest_scaler.joblib")
        reducer.save(btc_models_dir / "BTC_RandomForest_reducer.joblib")

        # Write metadata sidecar expected by Predictor
        meta = {
            "symbol": "BTC",
            "model_name": "RandomForest",
            "horizon": 1,
            "features_used": reducer.kept_features,
            "experiment_name": "e2e_rf_test",
        }
        with open(btc_models_dir / "BTC_RandomForest_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        self.assertTrue(saved_path.exists())

        # 9. Reload Model
        reloaded_rf = RandomForestModel.load(btc_models_dir)
        self.assertTrue(reloaded_rf.is_fitted)

        # 10. Run Production Predictor Inference & Price Conversion
        predictor = Predictor(models_dir=self.models_dir)
        latest_btc_row = dataset.features.tail(1)
        current_price = float(latest_btc_row["close"].iloc[0])

        price_interval = predictor.predict_coin(
            symbol="BTC",
            X_latest=latest_btc_row,
            current_price=current_price,
            horizon=1,
        )

        # 11. Assertions on Price Prediction Interval
        self.assertEqual(price_interval.current_price, current_price)
        self.assertEqual(price_interval.horizon_days, 1)
        self.assertLess(price_interval.lower_price, price_interval.median_price)
        self.assertLess(price_interval.median_price, price_interval.upper_price)
        self.assertGreater(price_interval.lower_price, 0)


if __name__ == "__main__":
    unittest.main()
