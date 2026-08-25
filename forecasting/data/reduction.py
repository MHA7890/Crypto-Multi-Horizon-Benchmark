"""
forecasting.data.reduction — Model-Independent Feature Reduction.

Pipeline:
1. Near-Zero Variance Filter: Remove features with variance < threshold.
2. High Correlation Filter: For feature pairs with |correlation| > threshold (0.95),
   drop the feature with higher mean absolute correlation to all other features.

Fit strictly on training data only to avoid data leakage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import List, Tuple, Union

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ReductionReport:
    """
    Detailed audit log documenting all feature reduction decisions.

    Attributes
    ----------
    original_count : int
        Number of features prior to reduction.
    final_count : int
        Number of features retained after reduction.
    variance_removed : list[str]
        List of feature names removed due to near-zero variance.
    corr_removed : list[str]
        List of feature names removed due to high correlation.
    corr_pairs : list[tuple[str, str, float]]
        High-correlation pairs identified (feature1, feature2, absolute_corr).
    variance_threshold : float
        Variance cutoff threshold applied.
    corr_threshold : float
        Correlation cutoff threshold applied.
    """

    original_count: int
    final_count: int
    variance_removed: list[str] = field(default_factory=list)
    corr_removed: list[str] = field(default_factory=list)
    corr_pairs: list[tuple[str, str, float]] = field(default_factory=list)
    variance_threshold: float = 0.0
    corr_threshold: float = 0.0


class FeatureReducer:
    """
    Model-independent feature reduction tool.

    Execution Flow
    ──────────────
    1. Fit near-zero variance filter on X_train.
    2. Fit correlation filter on remaining features of X_train.
    3. Retain optimal feature subset.
    4. Transform any dataset (X_train, X_val, X_test) using retained features.
    """

    def __init__(
        self,
        corr_threshold: float = 0.95,
        variance_threshold: float = 1e-6,
    ):
        self.corr_threshold = corr_threshold
        self.variance_threshold = variance_threshold
        self.kept_features: list[str] | None = None
        self.is_fitted = False

    def fit(self, X_train: pd.DataFrame) -> ReductionReport:
        """
        Fit feature reduction parameters on training dataset X_train.
        """
        if X_train.empty:
            raise ValueError("Cannot fit FeatureReducer on empty DataFrame")

        numeric_df = X_train.select_dtypes(include=["number"])
        original_features = list(numeric_df.columns)

        if not original_features:
            raise ValueError("Training DataFrame contains no numeric features to reduce")

        # ── Step 1: Near-Zero Variance Filter ──
        variances = numeric_df.var(numeric_only=True, ddof=1).fillna(0.0)
        low_var_mask = variances <= self.variance_threshold
        variance_removed = list(variances[low_var_mask].index)

        features_after_var = [f for f in original_features if f not in variance_removed]

        if not features_after_var:
            logger.warning("Variance filter removed all features! Retaining single feature with max variance.")
            max_var_feat = variances.idxmax()
            features_after_var = [max_var_feat]
            variance_removed.remove(max_var_feat)

        # ── Step 2: High Correlation Filter ──
        df_var_filtered = numeric_df[features_after_var]
        corr_matrix = df_var_filtered.corr(method="pearson").abs().fillna(0.0)

        corr_removed = set()
        corr_pairs = []

        n_feats = len(features_after_var)
        feat_array = np.array(features_after_var)

        upper_mask = np.triu(np.ones((n_feats, n_feats), dtype=bool), k=1)
        mean_corrs = corr_matrix.mean(axis=1)

        for i in range(n_feats):
            for j in range(i + 1, n_feats):
                val = corr_matrix.iloc[i, j]
                if val > self.corr_threshold:
                    f1, f2 = feat_array[i], feat_array[j]
                    corr_pairs.append((f1, f2, float(val)))

                    if f1 in corr_removed or f2 in corr_removed:
                        continue

                    if mean_corrs[f1] >= mean_corrs[f2]:
                        corr_removed.add(f1)
                    else:
                        corr_removed.add(f2)

        corr_removed_list = sorted(list(corr_removed))
        self.kept_features = [f for f in features_after_var if f not in corr_removed]
        self.is_fitted = True

        report = ReductionReport(
            original_count=len(original_features),
            final_count=len(self.kept_features),
            variance_removed=variance_removed,
            corr_removed=corr_removed_list,
            corr_pairs=corr_pairs,
            variance_threshold=self.variance_threshold,
            corr_threshold=self.corr_threshold,
        )

        logger.debug(
            "Feature reduction completed: %d -> %d features (removed %d zero-variance, %d correlated)",
            report.original_count,
            report.final_count,
            len(report.variance_removed),
            len(report.corr_removed),
        )

        return report

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted or self.kept_features is None:
            raise RuntimeError("FeatureReducer must be fitted before calling transform")

        if X.empty:
            return X.copy()

        missing_cols = set(self.kept_features) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Input DataFrame is missing retained features: {missing_cols}")

        return X[self.kept_features].copy()

    def fit_transform(self, X_train: pd.DataFrame) -> tuple[pd.DataFrame, ReductionReport]:
        report = self.fit(X_train)
        return self.transform(X_train), report

    def save(self, path: Union[Path, str]) -> Path:
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)
        logger.debug("Saved FeatureReducer to %s", filepath)
        return filepath

    @classmethod
    def load(cls, path: Union[Path, str]) -> FeatureReducer:
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Reducer file not found: {filepath}")
        reducer = joblib.load(filepath)
        if not isinstance(reducer, cls):
            raise TypeError(f"Loaded object is not instance of {cls.__name__}, got {type(reducer)}")
        return reducer
