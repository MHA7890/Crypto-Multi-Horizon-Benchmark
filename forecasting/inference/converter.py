"""
forecasting.inference.converter — Converts log-return predictions to price prediction intervals.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class PriceInterval:
    """Human-readable price prediction interval."""

    current_price: float
    horizon_days: int
    lower_price: float
    median_price: float
    upper_price: float
    confidence_level: float


class ReturnToPriceConverter:
    """Converts log returns to price levels using P_t * exp(r)."""

    def convert(
        self,
        current_price: float,
        point_forecast: float,
        lower_bound: float,
        upper_bound: float,
        horizon: int,
        confidence_level: float = 0.90,
    ) -> PriceInterval:
        """Convert log returns to price levels."""
        return PriceInterval(
            current_price=current_price,
            horizon_days=horizon,
            lower_price=float(current_price * np.exp(lower_bound)),
            median_price=float(current_price * np.exp(point_forecast)),
            upper_price=float(current_price * np.exp(upper_bound)),
            confidence_level=confidence_level,
        )
