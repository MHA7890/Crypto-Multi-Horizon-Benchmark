"""
End-to-End integration test suite for Model Training, Evaluation, Winner Selection, and Archiving.
"""

import unittest
from pathlib import Path
import shutil
import pandas as pd

from forecasting.config.settings import ExperimentConfig, PathConfig
from forecasting.inference.predictor import Predictor
from forecasting.training.checkpoint import CheckpointManager
from forecasting.training.runner import ExperimentRunner


class TestEndToEndTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path("test_sandbox")
        cls.models_dir = cls.test_dir / "models"
        cls.archive_dir = cls.test_dir / "archive"
        cls.evaluation_dir = cls.test_dir / "evaluation"
        cls.experiments_dir = cls.test_dir / "experiments"

        cls.has_real_btc = Path("features/BTC_features.csv").exists()

    def setUp(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        self.test_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_end_to_end_coin_training_and_selection(self):
        """Test full training pipeline, evaluation logging, winner selection, archiving, and inference."""
        if not self.has_real_btc:
            self.skipTest("features/BTC_features.csv not present")

        config = ExperimentConfig(
            name="test_run",
            paths=PathConfig(
                features_dir=Path("features"),
                models_dir=self.models_dir,
                archive_dir=self.archive_dir,
                evaluation_dir=self.evaluation_dir,
                experiments_dir=self.experiments_dir,
                logs_dir=self.test_dir / "logs",
            ),
            coins=["BTC"],
            models=["ARIMA", "RandomForest", "XGBoost"],
            model_params={
                "RandomForest": {"n_estimators": 10, "max_depth": 4},
                "XGBoost": {"n_estimators": 10, "max_depth": 4},
                "ARIMA": {"maxiter": 10},
            },
        )
        config.target.horizons = [1]
        config.validation.min_train_ratio = 0.8
        config.validation.val_size_ratio = 0.1
        config.validation.step_size_ratio = 0.1

        runner = ExperimentRunner(config_path=None, resume=False)
        runner.config = config
        runner.pipeline.config = config
        runner.pipeline.target_constructor.horizons_days = [1]
        runner.selector.models_dir = self.models_dir
        runner.selector.archiver.archive_dir = self.archive_dir
        runner.reporter.evaluation_dir = self.evaluation_dir
        runner.checkpoint_mgr = CheckpointManager(
            checkpoint_path=self.experiments_dir / "checkpoint.json",
            config_hash=runner.config_hash,
        )
        runner.checkpoint_mgr.completed_coins = []
        runner.checkpoint_mgr.completed_pairs = set()

        res = runner.run_coin("BTC")

        winner_symbol = res["symbol"]
        winner_model = res["winner"]

        self.assertEqual(winner_symbol, "BTC")
        self.assertIn(winner_model, ["ARIMA", "RandomForest", "XGBoost"])

        eval_csv = self.evaluation_dir / "BTC" / "metrics.csv"
        self.assertTrue(eval_csv.exists())

        df_metrics = pd.read_csv(eval_csv)
        self.assertGreater(len(df_metrics), 0)
        self.assertIn("rmse", df_metrics.columns)
        self.assertIn("picp", df_metrics.columns)
        self.assertIn("dir_accuracy", df_metrics.columns)

        btc_models_dir = self.models_dir / "BTC"
        self.assertTrue(btc_models_dir.exists())

        winner_meta = btc_models_dir / f"BTC_{winner_model}_meta.json"
        self.assertTrue(winner_meta.exists())

        btc_archive_dir = self.archive_dir / "BTC"
        self.assertTrue(btc_archive_dir.exists())

        archived_files = list(btc_archive_dir.glob("*"))
        self.assertGreater(len(archived_files), 0)

        for arch_file in archived_files:
            self.assertNotIn(winner_model, arch_file.name)

        predictor = Predictor(models_dir=self.models_dir)
        df_btc = pd.read_csv("features/BTC_features.csv").tail(10)
        current_price = float(df_btc["close"].iloc[-1])

        price_interval = predictor.predict_coin(
            symbol="BTC",
            X_latest=df_btc,
            current_price=current_price,
            horizon=1,
        )

        self.assertEqual(price_interval.current_price, current_price)
        self.assertLess(price_interval.lower_price, price_interval.upper_price)
        self.assertGreater(price_interval.median_price, 0)


if __name__ == "__main__":
    unittest.main()
