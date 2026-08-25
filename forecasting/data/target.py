"""
forecasting.data.target — Multi-horizon log-return target construction.

Constructs stationarity-preserving log-return target variables:
    r_{t, h} = ln(P_{t + h} / P_t)

Preserves reference prices (P_t) for downstream price prediction interval
back-conversion. Drops trailing NaN rows (rows where future price P_{t+h} is unavailable)
and initial indicator warm-up rows containing NaNs.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Dict, List, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class HorizonTarget:
    """
    Target values and reference metadata for a specific forecast horizon.

    Attributes
    ----------
    horizon_days : int
        Forecast horizon in days (e.g. 1, 7, 14, 30, 90).
    shift_steps : int
        Row shift offset corresponding to the horizon (e.g. 24 for 1 day hourly).
    log_returns : pd.Series
        Log-return series ln(P_{t+h} / P_t) indexed by DatetimeIndex.
    reference_prices : pd.Series
        Current price P_t used as denominator during training and reference during conversion.
    name : str
        Descriptive name (e.g. 'log_return_7d').
    """

    horizon_days: int
    shift_steps: int
    log_returns: pd.Series
    reference_prices: pd.Series
    name: str


class TargetConstructor:
    """
    Constructs multi-horizon forward log-return target variables.

    Mathematical Formulation
    ────────────────────────
    Log return:
        r_{t, h} = ln(close[t + shift] / close[t])

    For hourly candle datasets:
        1 day   → shift = 24 rows
        7 days  → shift = 168 rows
        14 days → shift = 336 rows
        30 days → shift = 720 rows
        90 days → shift = 2160 rows
    """

    def __init__(
        self,
        horizons_days: list[int] | None = None,
        price_col: str = "close",
        steps_per_day: int = 24,
    ):
        self.horizons_days = horizons_days or [1, 7, 14, 30, 90]
        self.price_col = price_col
        self.steps_per_day = steps_per_day

        if any(h <= 0 for h in self.horizons_days):
            raise ValueError(f"Horizons must be positive integers, got {self.horizons_days}")
        if steps_per_day <= 0:
            raise ValueError(f"steps_per_day must be positive, got {steps_per_day}")

    def create_targets(self, df: pd.DataFrame) -> dict[int, HorizonTarget]:
        """
        Create multi-horizon log return targets for all configured horizons.

        Parameters
        ----------
        df : pd.DataFrame
            Feature matrix containing the price column.

        Returns
        -------
        dict[int, HorizonTarget]
            Dictionary mapping horizon in days (e.g., 1, 7, 14, 30, 90) to HorizonTarget.

        Raises
        ------
        KeyError
            If price_col is not present in df.
        ValueError
            If price series contains non-positive values (<= 0).
        """
        if self.price_col not in df.columns:
            raise KeyError(
                f"Price column '{self.price_col}' not found in DataFrame. "
                f"Available columns: {list(df.columns[:5])}"
            )

        close = df[self.price_col]

        # Check positive price values for log calculation
        if (close <= 0).any():
            invalid_count = (close <= 0).sum()
            raise ValueError(
                f"Price column '{self.price_col}' contains {invalid_count} non-positive values (<= 0), "
                f"which cannot be used for log-return calculation."
            )

        targets = {}

        for h_days in self.horizons_days:
            shift_steps = h_days * self.steps_per_day

            # Shift close series backward in time to get future price P_{t+h} at index t
            future_close = close.shift(-shift_steps)

            # Compute log return ln(P_{t+h} / P_t)
            log_ret = np.log(future_close / close)
            log_ret.name = f"log_return_{h_days}d"

            target = HorizonTarget(
                horizon_days=h_days,
                shift_steps=shift_steps,
                log_returns=log_ret,
                reference_prices=close,
                name=f"log_return_{h_days}d",
            )
            targets[h_days] = target

            logger.debug(
                "Created target %s: horizon=%dd (%d steps), total_rows=%d, valid_targets=%d, trailing_nans=%d",
                target.name,
                h_days,
                shift_steps,
                len(log_ret),
                log_ret.dropna().shape[0],
                log_ret.isna().sum(),
            )

        return targets

    def get_valid_features_and_target(
        self,
        df: pd.DataFrame,
        target: HorizonTarget,
        drop_feature_nans: bool = True,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Align feature matrix and target series by removing invalid rows.

        Parameters
        ----------
        df : pd.DataFrame
            Full feature matrix.
        target : HorizonTarget
            Target variable object created by create_targets.
        drop_feature_nans : bool, default True
            If True, also drops head rows containing feature NaNs caused by initial indicator warm-up.

        Returns
        -------
        tuple[pd.DataFrame, pd.Series, pd.Series]
            (X_valid, y_valid, prices_valid) aligned and containing 0 NaNs.
        """
        valid_mask = target.log_returns.notna()

        if drop_feature_nans:
            feature_valid_mask = df.notna().all(axis=1)
            valid_mask = valid_mask & feature_valid_mask

        X_valid = df.loc[valid_mask].copy()
        y_valid = target.log_returns.loc[valid_mask].copy()
        prices_valid = target.reference_prices.loc[valid_mask].copy()

        if len(y_valid) == 0:
            raise ValueError(
                f"Dataset length ({len(df)} rows) is too small for shift of {target.shift_steps} steps. "
                f"No valid target rows remain."
            )

        return X_valid, y_valid, prices_valid
