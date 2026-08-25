"""
Unit tests for FeatureReducer.
"""

import unittest
import numpy as np
import pandas as pd

from forecasting.data.reduction import FeatureReducer


class TestFeatureReducer(unittest.TestCase):
    def test_feature_reducer(self):
        np.random.seed(42)
        col_zero_var = np.ones(100)
        col_a = np.random.randn(100)
        col_b = col_a + np.random.randn(100) * 1e-6  # Highly correlated with col_a

        df = pd.DataFrame(
            {"zero_var": col_zero_var, "col_a": col_a, "col_b": col_b}
        )

        reducer = FeatureReducer(corr_threshold=0.95, variance_threshold=1e-5)
        report = reducer.fit(df)

        self.assertIn("zero_var", report.variance_removed)
        self.assertEqual(len(report.corr_removed), 1)
        self.assertEqual(report.final_count, 1)


if __name__ == "__main__":
    unittest.main()
