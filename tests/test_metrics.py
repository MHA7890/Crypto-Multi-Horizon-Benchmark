"""
Unit tests for MetricsCalculator and ForecastMetrics.
"""

import unittest
import numpy as np

from forecasting.evaluation.metrics import MetricsCalculator
from forecasting.models.base import FitResult, PredictionResult


class TestMetricsCalculator(unittest.TestCase):
    def test_metrics_calculator(self):
        calculator = MetricsCalculator()

        y_true = np.array([0.01, -0.02, 0.03, 0.04])
        prediction = PredictionResult(
            point_forecast=np.array([0.012, -0.015, 0.025, 0.045]),
            lower_bound=np.array([0.00, -0.03, 0.01, 0.02]),
            upper_bound=np.array([0.03, 0.00, 0.05, 0.06]),
            confidence_level=0.90,
            horizon=1,
        )
        fit_result = FitResult(training_time_seconds=1.5, model_size_bytes=2048)

        metrics = calculator.compute(
            y_true=y_true,
            prediction=prediction,
            fit_result=fit_result,
            inference_time=0.05,
        )

        self.assertGreater(metrics.rmse, 0)
        self.assertGreater(metrics.mae, 0)
        self.assertEqual(metrics.picp, 1.0)  # All true values are inside intervals
        self.assertEqual(metrics.direction_accuracy, 1.0)  # All signs match
        self.assertEqual(metrics.training_time_seconds, 1.5)
        self.assertEqual(metrics.inference_time_seconds, 0.05)


if __name__ == "__main__":
    unittest.main()
