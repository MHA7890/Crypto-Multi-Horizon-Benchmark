"""
forecasting.data.scaler — Leakproof RobustScaler Wrapper.

Prevents data leakage by restricting scaling parameter estimation strictly
to training fold data via API contract.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Union

import joblib
import pandas as pd
from sklearn.preprocessing import RobustScaler

logger = logging.getLogger(__name__)


class LeakproofScaler:
    """
    RobustScaler wrapper that guarantees zero data leakage.

    Design & Anti-Leakage Contract
    ──────────────────────────────
    1. fit_transform_train(X_train): Estimates median and interquartile range (IQR)
       from X_train ONLY, and transforms X_train.
    2. transform_val(X_val): Transforms X_val using the ALREADY-FITTED training
       parameters. Raises RuntimeError if called prior to fit_transform_train.

    RobustScaler Benefits:
    - Median and IQR are resistant to crypto price flash crashes and extreme outliers.
    """

    def __init__(self, quantile_range: tuple[float, float] | list[float] = (25.0, 75.0)):
        if isinstance(quantile_range, (list, tuple)):
            quantile_range = (float(quantile_range[0]), float(quantile_range[1]))
        self.quantile_range = quantile_range
        self.scaler = RobustScaler(quantile_range=quantile_range)
        self.is_fitted = False
        self.feature_columns: list[str] | None = None

    def fit_transform_train(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """
        Fit RobustScaler on training data and return scaled training DataFrame.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix.

        Returns
        -------
        pd.DataFrame
            Scaled training feature matrix with identical index and column names.
        """
        if X_train.empty:
            raise ValueError("Cannot fit scaler on empty training DataFrame")

        # Select numeric columns only
        numeric_cols = list(X_train.select_dtypes(include=["number"]).columns)
        if not numeric_cols:
            raise ValueError("Training DataFrame contains no numeric columns to scale")

        self.feature_columns = numeric_cols

        scaled_array = self.scaler.fit_transform(X_train[numeric_cols])
        self.is_fitted = True

        scaled_df = pd.DataFrame(
            scaled_array, index=X_train.index, columns=numeric_cols
        )

        # Include non-numeric columns if any
        non_numeric_cols = [c for c in X_train.columns if c not in numeric_cols]
        for c in non_numeric_cols:
            scaled_df[c] = X_train[c]

        return scaled_df[X_train.columns]

    def transform_val(self, X_val: pd.DataFrame) -> pd.DataFrame:
        """
        Transform validation data using parameters fitted on training data.

        Parameters
        ----------
        X_val : pd.DataFrame
            Validation feature matrix.

        Returns
        -------
        pd.DataFrame
            Scaled validation feature matrix using training parameters.

        Raises
        ------
        RuntimeError
            If called before fit_transform_train.
        ValueError
            If validation feature set does not match fitted training columns.
        """
        if not self.is_fitted or self.feature_columns is None:
            raise RuntimeError(
                "LeakproofScaler is not fitted yet. "
                "Call fit_transform_train(X_train) before calling transform_val(X_val)."
            )

        if X_val.empty:
            return X_val.copy()

        missing_cols = set(self.feature_columns) - set(X_val.columns)
        if missing_cols:
            raise ValueError(
                f"Validation DataFrame is missing columns present during fitting: {missing_cols}"
            )

        numeric_cols = self.feature_columns
        scaled_array = self.scaler.transform(X_val[numeric_cols])

        scaled_df = pd.DataFrame(
            scaled_array, index=X_val.index, columns=numeric_cols
        )

        non_numeric_cols = [c for c in X_val.columns if c not in numeric_cols]
        for c in non_numeric_cols:
            scaled_df[c] = X_val[c]

        return scaled_df[X_val.columns]

    def save(self, path: Union[Path, str]) -> Path:
        """Serialize scaler to disk via joblib."""
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.debug("Saved LeakproofScaler to %s", filepath)
        return filepath

    @classmethod
    def load(cls, path: Union[Path, str]) -> LeakproofScaler:
        """Deserialize scaler from disk via joblib."""
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Scaler file not found: {filepath}")
        scaler = joblib.load(filepath)
        if not isinstance(scaler, cls):
            raise TypeError(f"Loaded object is not instance of {cls.__name__}, got {type(scaler)}")
        return scaler
