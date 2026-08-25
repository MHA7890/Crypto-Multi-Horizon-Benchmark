"""
Unit tests for ReturnToPriceConverter.
"""

import unittest
import numpy as np

from forecasting.inference.converter import ReturnToPriceConverter


class TestReturnToPriceConverter(unittest.TestCase):
    def test_return_to_price_converter(self):
        converter = ReturnToPriceConverter()
        current_price = 100.0
        point_forecast = 0.05
        lower_bound = 0.02
        upper_bound = 0.08

        res = converter.convert(
            current_price=current_price,
            point_forecast=point_forecast,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            horizon=1,
        )

        self.assertTrue(np.isclose(res.median_price, 100.0 * np.exp(0.05)))
        self.assertTrue(np.isclose(res.lower_price, 100.0 * np.exp(0.02)))
        self.assertTrue(np.isclose(res.upper_price, 100.0 * np.exp(0.08)))
        self.assertTrue(res.lower_price < res.median_price < res.upper_price)


if __name__ == "__main__":
    unittest.main()
