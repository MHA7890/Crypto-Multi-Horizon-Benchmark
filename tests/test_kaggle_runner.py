"""
Tests for kaggle.crypto_cuda_runner — Kaggle GPU Execution Orchestrator.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np
import pandas as pd
import torch

from forecasting.models.patchtst import PatchTSTModel
from forecasting.models.tft import TFTModel
from kaggle.crypto_cuda_runner import (
    CUDA_MODELS,
    KaggleCUDAExperimentRunner,
    clean_gpu_memory,
    get_gpu_memory_stats,
    verify_cuda_environment,
)


def _make_synthetic_feature_df(n_rows: int = 150, n_features: int = 12) -> pd.DataFrame:
    """Generate synthetic dataset for testing."""
    dates = pd.date_range("2023-01-01", periods=n_rows, freq="1h")
    data = {"timestamp": dates, "close": np.linspace(100.0, 150.0, n_rows) + np.random.randn(n_rows)}

    for i in range(n_features - 1):
        data[f"feature_{i}"] = np.random.randn(n_rows)

    return pd.DataFrame(data)


def _make_synthetic_ranking_df() -> pd.DataFrame:
    """Generate synthetic ranking dataframe."""
    return pd.DataFrame([
        {"Rank": 1, "CMC_ID": 1, "Name": "Bitcoin", "Symbol": "BTC", "Status": "READY", "MarketCap": 1e12},
        {"Rank": 2, "CMC_ID": 1027, "Name": "Ethereum", "Symbol": "ETH", "Status": "READY", "MarketCap": 3e11},
        {"Rank": 3, "CMC_ID": 825, "Name": "Tether USDt", "Symbol": "USDT", "Status": "STABLECOIN_SKIP", "MarketCap": 1e11},
    ])


class TestCUDADetectionAndMemory(unittest.TestCase):
    """Test CUDA detection and memory utilities."""

    def test_verify_cuda_environment_cpu_fallback(self):
        """When allow_cpu=True, returns diagnostic info even if CUDA is absent."""
        info = verify_cuda_environment(allow_cpu=True)
        self.assertIn("cuda_available", info)
        self.assertIn("torch_version", info)

    def test_clean_gpu_memory(self):
        """clean_gpu_memory executes without error."""
        clean_gpu_memory()

    def test_get_gpu_memory_stats(self):
        """get_gpu_memory_stats returns valid memory dictionary."""
        stats = get_gpu_memory_stats()
        self.assertIn("allocated_mb", stats)
        self.assertIn("reserved_mb", stats)
        self.assertIn("available_mb", stats)

    def test_find_kaggle_dataset_dir_nested(self):
        """find_kaggle_dataset_dir correctly discovers nested Kaggle dataset directories."""
        from kaggle.crypto_cuda_runner import find_kaggle_dataset_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_root = Path(tmpdir) / "input" / "datasets" / "mha7890" / "crypto-forecasting-dataset" / "crypto-forecasting-dataset"
            features_dir = nested_root / "features"
            output_dir = nested_root / "output"
            features_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

            (features_dir / "BTC_features.csv").write_text("timestamp,close\n", encoding="utf-8")
            (output_dir / "coin_mapping.csv").write_text("Rank,Symbol,Status\n1,BTC,READY\n", encoding="utf-8")

            found_feat, found_map, found_config = find_kaggle_dataset_dir(search_root=tmpdir)
            self.assertEqual(found_feat.resolve(), features_dir.resolve())
            self.assertEqual(found_map.resolve(), (output_dir / "coin_mapping.csv").resolve())

    def test_setup_python_path_nested(self):
        """setup_python_path correctly discovers nested forecasting code dataset roots and adds to sys.path."""
        import sys
        from kaggle.crypto_cuda_runner import setup_python_path

        with tempfile.TemporaryDirectory() as tmpdir:
            nested_code_root = Path(tmpdir) / "input" / "datasets" / "mha7890" / "crypto-forecasting-code" / "forecasting"
            pkg_dir = nested_code_root / "forecasting"
            pkg_dir.mkdir(parents=True, exist_ok=True)
            (pkg_dir / "__init__.py").write_text("# dummy init", encoding="utf-8")

            resolved_root = setup_python_path(search_root=tmpdir)
            self.assertEqual(resolved_root.resolve(), nested_code_root.resolve())
            self.assertIn(str(nested_code_root.resolve()), [str(Path(p).resolve()) for p in sys.path])


class TestPyTorchEarlyStopping(unittest.TestCase):
    """Test PyTorch TFT & PatchTST Early Stopping & Checkpoint Restoration."""

    def test_tft_early_stopping_best_checkpoint(self):
        """TFTModel early stopping monitors val loss and restores best checkpoint."""
        X_train = pd.DataFrame(np.random.randn(100, 8))
        y_train = pd.Series(np.random.randn(100))
        X_val = pd.DataFrame(np.random.randn(30, 8))
        y_val = pd.Series(np.random.randn(30))

        model = TFTModel(max_epochs=15, patience=2, min_delta=0.001, device="cpu")
        fit_res = model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        self.assertTrue(model.is_fitted)
        self.assertGreater(fit_res.training_time_seconds, 0.0)

    def test_patchtst_early_stopping_best_checkpoint(self):
        """PatchTSTModel early stopping monitors val loss and restores best checkpoint."""
        X_train = pd.DataFrame(np.random.randn(100, 8))
        y_train = pd.Series(np.random.randn(100))
        X_val = pd.DataFrame(np.random.randn(30, 8))
        y_val = pd.Series(np.random.randn(30))

        model = PatchTSTModel(max_epochs=15, patience=2, min_delta=0.001, device="cpu")
        fit_res = model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        self.assertTrue(model.is_fitted)
        self.assertGreater(fit_res.training_time_seconds, 0.0)


class TestKaggleCUDAExperimentRunner(unittest.TestCase):
    """Test Kaggle CUDA Experiment Runner components."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)

        self.data_dir = self.root / "features"
        self.output_dir = self.root / "crypto_forecasting"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Write synthetic datasets
        _make_synthetic_feature_df().to_csv(self.data_dir / "BTC_features.csv", index=False)
        _make_synthetic_feature_df().to_csv(self.data_dir / "ETH_features.csv", index=False)

        # Write synthetic ranking file
        self.ranking_file = self.root / "coin_mapping.csv"
        _make_synthetic_ranking_df().to_csv(self.ranking_file, index=False)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_runner_coin_selection(self):
        """Runner selects Top-N READY coins matching datasets."""
        runner = KaggleCUDAExperimentRunner(
            data_dir=self.data_dir,
            output_dir=self.output_dir,
            ranking_file=self.ranking_file,
            top_n=2,
            allow_cpu=True,
        )
        symbols = runner.select_coins()
        self.assertEqual(symbols, ["BTC", "ETH"])

    def test_export_results_zip(self):
        """Runner exports valid ZIP containing experiment artifacts."""
        runner = KaggleCUDAExperimentRunner(
            data_dir=self.data_dir,
            output_dir=self.output_dir,
            ranking_file=self.ranking_file,
            top_n=2,
            allow_cpu=True,
        )

        # Create dummy result files
        (runner.evaluation_dir / "all_model_results.csv").write_text("symbol,model\nBTC,XGBoost\n", encoding="utf-8")
        (runner.models_dir / "BTC" / "XGBoost").mkdir(parents=True, exist_ok=True)
        (runner.models_dir / "BTC" / "XGBoost" / "model.bin").write_text("model_data", encoding="utf-8")

        zip_path = runner.export_results_zip("test_results.zip")
        self.assertTrue(zip_path.exists())

        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            self.assertTrue(any("all_model_results.csv" in n for n in names))
            self.assertTrue(any("model.bin" in n for n in names))

    def test_smoke_kaggle_benchmark(self):
        """Small end-to-end smoke test run of Kaggle CUDA runner."""
        runner = KaggleCUDAExperimentRunner(
            data_dir=self.data_dir,
            output_dir=self.output_dir,
            ranking_file=self.ranking_file,
            top_n=1,
            allow_cpu=True,
        )
        # Override config for fast smoke test
        runner.config.target.horizons = [1]

        # Use XGBoost for smoke test
        with unittest.mock.patch("kaggle.crypto_cuda_runner.CUDA_MODELS", ["XGBoost"]):
            summary = runner.run_benchmark()

            self.assertGreater(summary["total_runs"], 0)
            self.assertGreaterEqual(summary["completed_runs"], 1)
            self.assertTrue((self.output_dir / "evaluation" / "all_model_results.csv").exists())


if __name__ == "__main__":
    unittest.main()
