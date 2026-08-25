"""
Unit tests for CompositeScorer.
"""

import unittest
from forecasting.evaluation.metrics import ForecastMetrics
from forecasting.evaluation.scorer import CompositeScorer, ScoringWeights


class TestCompositeScorer(unittest.TestCase):
    def test_composite_scorer(self):
        scorer = CompositeScorer(
            ScoringWeights(
                accuracy=0.40, interval_quality=0.35, directional=0.20, efficiency=0.05
            )
        )

        m1 = ForecastMetrics(
            rmse=0.01,
            mae=0.008,
            median_ae=0.007,
            mape=5.0,
            picp=0.95,
            mpiw=0.03,
            direction_accuracy=0.8,
            direction_precision=0.8,
            direction_recall=0.8,
            direction_f1=0.8,
            training_time_seconds=2.0,
            inference_time_seconds=0.01,
            model_size_bytes=1000,
        )

        m2 = ForecastMetrics(
            rmse=0.05,
            mae=0.04,
            median_ae=0.035,
            mape=20.0,
            picp=0.60,
            mpiw=0.08,
            direction_accuracy=0.5,
            direction_precision=0.5,
            direction_recall=0.5,
            direction_f1=0.5,
            training_time_seconds=10.0,
            inference_time_seconds=0.05,
            model_size_bytes=5000,
        )

        metrics = {"ModelA": m1, "ModelB": m2}
        rankings = scorer.rank(metrics)

        self.assertEqual(len(rankings), 2)
        # ModelA dominates ModelB across metrics so it must be ranked first
        self.assertEqual(rankings[0][0], "ModelA")
        self.assertGreater(rankings[0][1], rankings[1][1])


if __name__ == "__main__":
    unittest.main()
