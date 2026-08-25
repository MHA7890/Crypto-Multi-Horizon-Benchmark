"""
Unit tests for LeakproofScaler.
"""

import unittest
import pandas as pd

from forecasting.data.scaler import LeakproofScaler


class TestLeakproofScaler(unittest.TestCase):
    def test_leakproof_scaler(self):
        df_train = pd.DataFrame({"feat1": [1.0, 2.0, 10.0, 4.0, 5.0]})
        df_val = pd.DataFrame({"feat1": [3.0, 6.0]})

        scaler = LeakproofScaler()

        # Transforming val before train should fail
        with self.assertRaises(RuntimeError):
            scaler.transform_val(df_val)

        scaled_train = scaler.fit_transform_train(df_train)
        scaled_val = scaler.transform_val(df_val)

        self.assertEqual(scaled_train.shape, df_train.shape)
        self.assertEqual(scaled_val.shape, df_val.shape)
        self.assertTrue(scaler.is_fitted)


if __name__ == "__main__":
    unittest.main()
