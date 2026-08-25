"""
Unit tests for RandomForestModel and ForecastModel API compliance.
"""

import json
from pathlib import Path
import shutil
import unittest
import numpy as np
import pandas as pd
import yaml

from forecasting.config.loader import _load_model_defaults
from forecasting.models.base import ForecastModel, FitResult, PredictionResult
from forecasting.models.random_forest import RandomForestModel


class TestRandomForestModel(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("test_sandbox_rf")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(parents=True, exist_ok=True)

        np.random.seed(42)
        n_samples = 200
        n_features = 10

        self.X_train = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
            index=pd.date_range("2026-01-01", periods=n_samples, freq="1h"),
        )
        self.y_train = pd.Series(
            0.01 * np.random.randn(n_samples),
            index=self.X_train.index,
            name="target_log_return",
        )

        self.X_val = pd.DataFrame(
            np.random.randn(50, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
            index=pd.date_range("2026-01-10", periods=50, freq="1h"),
        )

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_forecast_model_interface_compliance(self):
        """Verify RandomForestModel inherits from ForecastModel and implements properties."""
        model = RandomForestModel()
        self.assertTrue(isinstance(model, ForecastModel))
        self.assertEqual(model.name, "RandomForest")
        self.assertEqual(model.file_extension, ".joblib")

    def test_hyperparameter_loading_and_set_get_params(self):
        """Test parameter initialization, get_params, set_params, and YAML default loading."""
        defaults = _load_model_defaults().get("random_forest", {})
        self.assertIn("n_estimators", defaults)

        model = RandomForestModel(**defaults)
        params = model.get_params()

        self.assertEqual(params["n_estimators"], defaults["n_estimators"])
        self.assertEqual(params["min_samples_split"], defaults["min_samples_split"])

        # Test set_params
        model.set_params(n_estimators=10, max_depth=5)
        updated_params = model.get_params()
        self.assertEqual(updated_params["n_estimators"], 10)
        self.assertEqual(updated_params["max_depth"], 5)

    def test_fit_and_predict_prediction_intervals(self):
        """Test model fit and prediction interval generation (5th, 50th, 95th percentiles)."""
        model = RandomForestModel(n_estimators=20, random_state=42)

        # Unfitted predict should raise RuntimeError
        with self.assertRaises(RuntimeError):
            model.predict(self.X_val)

        fit_res = model.fit(self.X_train, self.y_train)

        self.assertTrue(isinstance(fit_res, FitResult))
        self.assertTrue(model.is_fitted)
        self.assertGreater(fit_res.training_time_seconds, 0)
        self.assertGreater(fit_res.model_size_bytes, 0)

        # Predict
        pred_res = model.predict(self.X_val, horizon=7)

        self.assertTrue(isinstance(pred_res, PredictionResult))
        self.assertEqual(pred_res.horizon, 7)
        self.assertEqual(pred_res.confidence_level, 0.90)
        self.assertEqual(len(pred_res.point_forecast), len(self.X_val))
        self.assertEqual(len(pred_res.lower_bound), len(self.X_val))
        self.assertEqual(len(pred_res.upper_bound), len(self.X_val))

        # Check percentile order: lower_bound <= point_forecast <= upper_bound
        self.assertTrue(np.all(pred_res.lower_bound <= pred_res.point_forecast + 1e-6))
        self.assertTrue(np.all(pred_res.point_forecast <= pred_res.upper_bound + 1e-6))

    def test_multi_output_fitting(self):
        """Test multi-horizon 2D target matrix fitting."""
        y_multi = pd.DataFrame(
            {
                "log_ret_1d": 0.01 * np.random.randn(len(self.X_train)),
                "log_ret_7d": 0.02 * np.random.randn(len(self.X_train)),
            },
            index=self.X_train.index,
        )

        model = RandomForestModel(n_estimators=10, random_state=42)
        fit_res = model.fit(self.X_train, y_multi)
        self.assertTrue(model.is_fitted)

        pred_res = model.predict(self.X_val, horizon=1)
        self.assertEqual(len(pred_res.point_forecast), len(self.X_val))

    def test_save_and_load_serialization_metadata(self):
        """Test model save, load, and sidecar metadata JSON creation."""
        model = RandomForestModel(n_estimators=15, random_state=42)
        model.fit(self.X_train, self.y_train)

        saved_path = model.save(self.test_dir)
        self.assertTrue(saved_path.exists())

        meta_path = self.test_dir / "RandomForest_meta.json"
        self.assertTrue(meta_path.exists())

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.assertEqual(meta["model_name"], "RandomForest")
        self.assertEqual(meta["sample_count"], len(self.X_train))
        self.assertEqual(meta["feature_count"], self.X_train.shape[1])
        self.assertIn("timestamp_utc", meta)
        self.assertIn("parameters", meta)

        # Load model back
        loaded_model = RandomForestModel.load(self.test_dir)
        self.assertTrue(isinstance(loaded_model, RandomForestModel))
        self.assertTrue(loaded_model.is_fitted)

        # Predictions from original and reloaded model must match exactly
        orig_pred = model.predict(self.X_val)
        load_pred = loaded_model.predict(self.X_val)

        np.testing.assert_allclose(orig_pred.point_forecast, load_pred.point_forecast)
        np.testing.assert_allclose(orig_pred.lower_bound, load_pred.lower_bound)
        np.testing.assert_allclose(orig_pred.upper_bound, load_pred.upper_bound)

    def test_validation_and_error_handling(self):
        """Test robust error handling for empty matrices, NaNs, and missing files."""
        model = RandomForestModel()

        # Empty fit
        with self.assertRaises(ValueError):
            model.fit(pd.DataFrame(), pd.Series())

        # NaN features
        X_nan = self.X_train.copy()
        X_nan.iloc[0, 0] = np.nan
        with self.assertRaises(ValueError):
            model.fit(X_nan, self.y_train)

        # Load non-existent directory
        with self.assertRaises(FileNotFoundError):
            RandomForestModel.load(Path("non_existent_directory"))


if __name__ == "__main__":
    unittest.main()
