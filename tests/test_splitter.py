"""
Unit tests for WalkForwardSplitter.
"""

import unittest
import pandas as pd

from forecasting.data.splitter import WalkForwardSplitter


class TestWalkForwardSplitter(unittest.TestCase):
    def test_walk_forward_splitter(self):
        dates = pd.date_range("2026-01-01", periods=100, freq="1D")
        splitter = WalkForwardSplitter(
            min_train_ratio=0.6, val_size_ratio=0.1, step_size_ratio=0.1
        )
        folds = splitter.split(dates)

        self.assertGreater(len(folds), 0)
        first_fold = folds[0]
        self.assertEqual(len(first_fold.train_indices), 60)
        self.assertEqual(len(first_fold.val_indices), 10)

        # Ensure no temporal overlap
        self.assertLess(first_fold.train_indices[-1], first_fold.val_indices[0])


if __name__ == "__main__":
    unittest.main()
