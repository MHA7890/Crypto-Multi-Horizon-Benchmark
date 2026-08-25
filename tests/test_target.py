"""
Unit tests for TargetConstructor and log return target calculation.
"""

import unittest
import numpy as np
import pandas as pd

from forecasting.data.target import TargetConstructor


class TestTargetConstructor(unittest.TestCase):
    def test_target_constructor_log_returns(self):
        dates = pd.date_range("2026-01-01", periods=10, freq="1D")
        prices = [100.0, 105.0, 110.0, 108.0, 112.0, 115.0, 120.0, 118.0, 122.0, 125.0]
        df = pd.DataFrame({"close": prices}, index=dates)

        constructor = TargetConstructor(horizons_days=[1, 7], price_col="close", steps_per_day=1)
        targets = constructor.create_targets(df)

        self.assertIn(1, targets)
        self.assertIn(7, targets)

        t_1 = targets[1].log_returns
        expected_0 = np.log(105.0 / 100.0)
        self.assertTrue(np.isclose(t_1.iloc[0], expected_0))
        self.assertTrue(pd.isna(t_1.iloc[-1]))

        t_7 = targets[7].log_returns
        expected_7_0 = np.log(118.0 / 100.0)
        self.assertTrue(np.isclose(t_7.iloc[0], expected_7_0))
        self.assertTrue(pd.isna(t_7.iloc[-7]))


if __name__ == "__main__":
    unittest.main()
