"""
Unit tests for model registry.
"""

import unittest
from forecasting.config.model_registry import get_model_class, list_available_models


class TestModelRegistry(unittest.TestCase):
    def test_list_available_models(self):
        models = list_available_models()
        self.assertIn("ARIMA", models)
        self.assertIn("RandomForest", models)
        self.assertIn("XGBoost", models)
        self.assertIn("LightGBM", models)
        self.assertIn("TFT", models)
        self.assertIn("PatchTST", models)

    def test_get_model_class(self):
        # RandomForest is sklearn-backed so should always succeed
        rf_cls = get_model_class("RandomForest")
        self.assertEqual(rf_cls().name, "RandomForest")

        try:
            xgb_cls = get_model_class("XGBoost")
            self.assertEqual(xgb_cls().name, "XGBoost")
        except ImportError:
            pass  # Acceptable if xgboost is not installed in local python environment

        with self.assertRaises(KeyError):
            get_model_class("NonExistentModel")


if __name__ == "__main__":
    unittest.main()
