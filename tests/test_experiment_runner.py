"""
Unit and integration tests for the Automated Master Experiment Orchestration System.
"""

import json
from pathlib import Path
import shutil
import unittest
import pandas as pd

from forecasting.config.settings import ExperimentConfig, PathConfig
from forecasting.evaluation.reports import GlobalReportGenerator
from forecasting.training.checkpoint import CheckpointManager
from forecasting.training.runner import ExperimentRunner
from forecasting.training.verifier import ExperimentVerifier
from forecasting.utils.device import DeviceManager


class TestExperimentRunnerSystem(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path("test_sandbox_runner")
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)

        self.models_dir = self.test_dir / "models"
        self.archive_dir = self.test_dir / "archive"
        self.evaluation_dir = self.test_dir / "evaluation"
        self.experiments_dir = self.test_dir / "experiments"
        self.logs_dir = self.test_dir / "logs"

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_device_manager_routing_and_hardware_info(self):
        """Verify DeviceManager hardware summary and model routing rules."""
        mgr = DeviceManager(use_cuda=True)
        info = mgr.hardware_info

        self.assertIn("cpu_model", info)
        self.assertIn("ram_total_gb", info)
        self.assertIn("cuda_available", info)

        self.assertEqual(mgr.get_device_for_model("RandomForest"), "cpu")
        self.assertEqual(mgr.get_device_for_model("ARIMA"), "cpu")

        xgb_device = mgr.get_device_for_model("XGBoost")
        self.assertIn(xgb_device, ["cuda", "cpu"])

    def test_checkpoint_manager_persistence_and_resumption(self):
        """Test CheckpointManager state saving, reading, and completion checks."""
        ckpt_path = self.experiments_dir / "exp_001" / "checkpoint.json"
        mgr = CheckpointManager(checkpoint_path=ckpt_path, config_hash="abc123hash")

        self.assertFalse(mgr.is_coin_completed("BTC"))
        self.assertFalse(mgr.is_model_completed("BTC", "RandomForest"))

        mgr.mark_model_completed("BTC", "RandomForest")
        mgr.mark_coin_completed("BTC")

        self.assertTrue(mgr.is_model_completed("BTC", "RandomForest"))
        self.assertTrue(mgr.is_coin_completed("BTC"))
        self.assertTrue(ckpt_path.exists())

        reloaded_mgr = CheckpointManager(checkpoint_path=ckpt_path)
        self.assertTrue(reloaded_mgr.is_coin_completed("BTC"))
        self.assertTrue(reloaded_mgr.is_model_completed("BTC", "RandomForest"))
        self.assertEqual(reloaded_mgr.config_hash, "abc123hash")

    def test_global_report_generator_and_plots(self):
        """Test generation of per-coin model_comparison.csv, master all_model_results.csv, and publication plots."""
        btc_eval_dir = self.evaluation_dir / "BTC"
        btc_eval_dir.mkdir(parents=True, exist_ok=True)

        df_metrics = pd.DataFrame(
            [
                {
                    "experiment": "exp_001",
                    "timestamp": "2026-08-07T12:00:00Z",
                    "symbol": "BTC",
                    "model": "RandomForest",
                    "horizon": 1,
                    "rmse": 0.02,
                    "mae": 0.015,
                    "mape": 0.05,
                    "picp": 0.92,
                    "mpiw": 0.08,
                    "dir_accuracy": 0.65,
                    "train_time_s": 1.2,
                    "inference_time_s": 0.05,
                    "model_size_bytes": 10240,
                },
                {
                    "experiment": "exp_001",
                    "timestamp": "2026-08-07T12:00:00Z",
                    "symbol": "BTC",
                    "model": "XGBoost",
                    "horizon": 1,
                    "rmse": 0.03,
                    "mae": 0.02,
                    "mape": 0.07,
                    "picp": 0.88,
                    "mpiw": 0.09,
                    "dir_accuracy": 0.58,
                    "train_time_s": 0.8,
                    "inference_time_s": 0.02,
                    "model_size_bytes": 5120,
                },
            ]
        )
        df_metrics.to_csv(btc_eval_dir / "metrics.csv", index=False)

        report_gen = GlobalReportGenerator(evaluation_dir=self.evaluation_dir)
        reports = report_gen.generate_reports(exp_id="exp_001", elapsed_seconds=120.0)

        # 1. Per-coin model_comparison.csv
        coin_comp_csv = btc_eval_dir / "model_comparison.csv"
        self.assertTrue(coin_comp_csv.exists())
        df_coin_comp = pd.read_csv(coin_comp_csv)
        self.assertIn("rank", df_coin_comp.columns)
        self.assertIn("is_winner", df_coin_comp.columns)
        self.assertEqual(df_coin_comp.iloc[0]["model"], "RandomForest")
        self.assertTrue(df_coin_comp.iloc[0]["is_winner"])

        # 2. Master all_model_results.csv
        self.assertTrue(reports["all_model_results"].exists())
        df_all = pd.read_csv(reports["all_model_results"])
        self.assertEqual(len(df_all), 2)

        # 3. Publication plots in evaluation/plots/
        plots_dir = reports["plots_dir"]
        self.assertTrue((plots_dir / "model_win_counts.png").exists())
        self.assertTrue((plots_dir / "composite_score_distribution.png").exists())
        self.assertTrue((plots_dir / "rmse_vs_mpiw_tradeoff.png").exists())
        self.assertTrue((plots_dir / "directional_accuracy_comparison.png").exists())
        self.assertTrue((plots_dir / "training_efficiency_comparison.png").exists())

    def test_verifier_engine(self):
        """Test ExperimentVerifier audit of production artifacts."""
        btc_models = self.models_dir / "BTC"
        btc_models.mkdir(parents=True, exist_ok=True)
        btc_eval = self.evaluation_dir / "BTC"
        btc_eval.mkdir(parents=True, exist_ok=True)

        (btc_models / "BTC_RandomForest.joblib").touch()
        (btc_models / "BTC_RandomForest_scaler.joblib").touch()
        (btc_models / "BTC_RandomForest_reducer.joblib").touch()
        (btc_models / "BTC_RandomForest_meta.json").touch()
        (btc_eval / "metrics.csv").touch()

        verifier = ExperimentVerifier(
            models_dir=self.models_dir,
            archive_dir=self.archive_dir,
            evaluation_dir=self.evaluation_dir,
        )

        res = verifier.verify_all(symbols=["BTC"], output_report_path=self.test_dir / "verification_report.txt")
        self.assertEqual(res["verified_count"], 1)
        self.assertEqual(res["failed_verification_count"], 0)
        self.assertTrue((self.test_dir / "verification_report.txt").exists())


if __name__ == "__main__":
    unittest.main()
