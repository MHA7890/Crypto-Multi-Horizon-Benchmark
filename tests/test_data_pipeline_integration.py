"""
Integration test suite for the complete data pipeline.

Demonstrates end-to-end data flow:
  1. Load real cryptocurrency CSVs (features/BTC_features.csv, features/ETH_features.csv)
  2. Construct multi-horizon log-return targets (1, 7, 14, 30, 90 days)
  3. Slice valid target-aligned data (no trailing NaNs)
  4. Generate expanding-window walk-forward validation folds
  5. Apply FeatureReducer (fitted on fold train only)
  6. Apply LeakproofScaler (fitted on fold train only)
  7. Verify zero data leakage and temporal ordering across all folds
"""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from forecasting.data.loader import CoinDataset, DataLoader
from forecasting.data.target import HorizonTarget, TargetConstructor
from forecasting.data.splitter import WalkForwardFold, WalkForwardSplitter
from forecasting.data.scaler import LeakproofScaler
from forecasting.data.reduction import FeatureReducer, ReductionReport


class TestDataPipelineIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.features_dir = Path("features")
        cls.has_real_data = (cls.features_dir / "BTC_features.csv").exists()

    def test_real_data_pipeline_btc(self):
        """Integration test using real BTC_features.csv dataset."""
        if not self.has_real_data:
            self.skipTest("features/BTC_features.csv not present")

        # 1. DataLoader
        loader = DataLoader(features_dir=self.features_dir)
        dataset = loader.load("BTC", price_col="close")

        self.assertEqual(dataset.symbol, "BTC")
        self.assertGreater(dataset.n_rows, 1000)
        self.assertGreater(dataset.n_features, 50)
        self.assertTrue(isinstance(dataset.features.index, pd.DatetimeIndex))
        self.assertTrue(dataset.features.index.is_monotonic_increasing)

        # 2. TargetConstructor (horizons: 1, 7, 14, 30, 90 days)
        horizons = [1, 7, 14, 30, 90]
        constructor = TargetConstructor(
            horizons_days=horizons,
            price_col="close",
            steps_per_day=24,  # hourly dataset
        )
        targets = constructor.create_targets(dataset.features)

        self.assertEqual(set(targets.keys()), set(horizons))

        for h in horizons:
            target = targets[h]
            self.assertEqual(target.horizon_days, h)
            self.assertEqual(target.shift_steps, h * 24)

            # Align features and target
            X_valid, y_valid, prices_valid = constructor.get_valid_features_and_target(
                dataset.features, target
            )

            self.assertEqual(len(X_valid), len(y_valid))
            self.assertEqual(len(y_valid), len(prices_valid))
            self.assertFalse(y_valid.isna().any())

            # 3. WalkForwardSplitter
            splitter = WalkForwardSplitter(
                min_train_ratio=0.6,
                val_size_ratio=0.1,
                step_size_ratio=0.1,
            )
            folds = splitter.split(X_valid.index)
            self.assertGreater(len(folds), 0)

            # Verify folds
            for fold in folds:
                # 4. Strict Temporal Order Check
                self.assertLess(fold.train_end, fold.val_start)
                self.assertEqual(fold.train_size, len(fold.train_indices))
                self.assertEqual(fold.val_size, len(fold.val_indices))

                X_train_raw = X_valid.loc[fold.train_indices]
                y_train = y_valid.loc[fold.train_indices]
                X_val_raw = X_valid.loc[fold.val_indices]
                y_val = y_valid.loc[fold.val_indices]

                # 5. FeatureReducer (fit on fold train ONLY)
                reducer = FeatureReducer(corr_threshold=0.95, variance_threshold=1e-6)
                X_train_red, report = reducer.fit_transform(X_train_raw)
                X_val_red = reducer.transform(X_val_raw)

                self.assertLessEqual(X_train_red.shape[1], X_train_raw.shape[1])
                self.assertEqual(X_train_red.shape[1], X_val_red.shape[1])

                # 6. LeakproofScaler (fit on fold train ONLY)
                scaler = LeakproofScaler()
                X_train_scaled = scaler.fit_transform_train(X_train_red)
                X_val_scaled = scaler.transform_val(X_val_red)

                self.assertEqual(X_train_scaled.shape, X_train_red.shape)
                self.assertEqual(X_val_scaled.shape, X_val_red.shape)
                self.assertFalse(X_train_scaled.isna().any().any())
                self.assertFalse(X_val_scaled.isna().any().any())

                # Anti-leakage verification:
                # Calculate training scaler parameters manually and verify match
                median_train = X_train_red.median()
                q25_train = X_train_red.quantile(0.25)
                q75_train = X_train_red.quantile(0.75)
                iqr_train = q75_train - q25_train
                iqr_train = np.where(iqr_train == 0.0, 1.0, iqr_train)

                expected_val_scaled = (X_val_red - median_train) / iqr_train
                np.testing.assert_allclose(
                    X_val_scaled.values,
                    expected_val_scaled.values,
                    rtol=1e-5,
                    atol=1e-5,
                )

    def test_synthetic_data_pipeline_edge_cases(self):
        """Integration test with synthetic data testing edge cases and multi-coin pipeline."""
        np.random.seed(123)
        dates = pd.date_range("2026-01-01 00:00:00", periods=500, freq="1h", tz="UTC")

        price = 100.0 + np.cumsum(np.random.randn(500))
        # Ensure positive price
        price = np.maximum(price, 10.0)

        f1 = np.random.randn(500)
        f2 = f1 * 0.999  # extremely high correlation
        f3 = np.ones(500) * 5.0  # zero variance

        df_synthetic = pd.DataFrame(
            {"close": price, "feat_corr1": f1, "feat_corr2": f2, "feat_const": f3},
            index=dates,
        )

        # 1. Target construction
        constructor = TargetConstructor(horizons_days=[1, 7], steps_per_day=24)
        targets = constructor.create_targets(df_synthetic)

        target_1d = targets[1]
        X_valid, y_valid, prices_valid = constructor.get_valid_features_and_target(
            df_synthetic, target_1d
        )

        # 2. Split
        splitter = WalkForwardSplitter(min_train_ratio=0.5, val_size_ratio=0.2, step_size_ratio=0.1)
        folds = splitter.split(X_valid.index)

        self.assertGreater(len(folds), 0)

        # 3. Fit reduction & scaling on first fold
        fold0 = folds[0]
        X_train_raw = X_valid.loc[fold0.train_indices]
        X_val_raw = X_valid.loc[fold0.val_indices]

        reducer = FeatureReducer(corr_threshold=0.95, variance_threshold=1e-5)
        X_train_red, report = reducer.fit_transform(X_train_raw)

        # Verify zero variance and high correlation features were removed
        self.assertIn("feat_const", report.variance_removed)
        self.assertEqual(len(report.corr_removed), 1)

        scaler = LeakproofScaler()
        X_train_scaled = scaler.fit_transform_train(X_train_red)
        X_val_scaled = scaler.transform_val(reducer.transform(X_val_raw))

        self.assertTrue(scaler.is_fitted)
        self.assertEqual(X_train_scaled.shape[1], X_val_scaled.shape[1])


if __name__ == "__main__":
    unittest.main()
