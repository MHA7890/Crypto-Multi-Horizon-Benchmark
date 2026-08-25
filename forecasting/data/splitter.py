"""
forecasting.data.splitter — Walk-Forward Expanding-Window Cross-Validation.

Generates expanding-window validation folds respecting temporal ordering strictly:
    Train window: [0 : T_train]
    Val window:   [T_train : T_train + T_val]

Advancing train window by step_size each fold. Ensures zero temporal leakage.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import List, Union

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardFold:
    """
    Single fold metadata and indices for walk-forward validation.

    Attributes
    ----------
    fold_index : int
        0-indexed fold number.
    train_start : pd.Timestamp
        Timestamp of the first training sample.
    train_end : pd.Timestamp
        Timestamp of the last training sample.
    val_start : pd.Timestamp
        Timestamp of the first validation sample.
    val_end : pd.Timestamp
        Timestamp of the last validation sample.
    train_indices : pd.Index
        DatetimeIndex (or Integer Index) for training slice.
    val_indices : pd.Index
        DatetimeIndex (or Integer Index) for validation slice.
    train_size : int
        Number of training rows in fold.
    val_size : int
        Number of validation rows in fold.
    """

    fold_index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    train_indices: pd.Index
    val_indices: pd.Index
    train_size: int
    val_size: int


class WalkForwardSplitter:
    """
    Expanding-Window Walk-Forward Cross-Validation Splitter.

    Design & Causality
    ──────────────────
    In financial time-series forecasting, standard K-Fold CV causes catastrophic
    data leakage because future observations appear in training folds. Walk-forward
    validation mimics actual production deployment:
      - Fold 0: Train on [0 : T_0], Evaluate on [T_0 : T_0 + V]
      - Fold 1: Train on [0 : T_0 + S], Evaluate on [T_0 + S : T_0 + S + V]
      ...
    where S is step_size and V is val_size.
    """

    def __init__(
        self,
        min_train_ratio: float = 0.6,
        val_size_ratio: float = 0.1,
        step_size_ratio: float = 0.1,
        min_train_size: int | None = None,
        val_size: int | None = None,
        step_size: int | None = None,
    ):
        self.min_train_ratio = min_train_ratio
        self.val_size_ratio = val_size_ratio
        self.step_size_ratio = step_size_ratio
        self.min_train_size = min_train_size
        self.val_size = val_size
        self.step_size = step_size

        if min_train_size is None and not (0.0 < min_train_ratio < 1.0):
            raise ValueError(f"min_train_ratio must be in (0, 1), got {min_train_ratio}")
        if val_size is None and not (0.0 < val_size_ratio < 1.0):
            raise ValueError(f"val_size_ratio must be in (0, 1), got {val_size_ratio}")
        if step_size is None and not (0.0 < step_size_ratio <= 1.0):
            raise ValueError(f"step_size_ratio must be in (0, 1], got {step_size_ratio}")

    def split(self, index: pd.Index) -> list[WalkForwardFold]:
        """
        Generate expanding-window walk-forward folds for a given pandas Index.

        Parameters
        ----------
        index : pd.Index
            DatetimeIndex or Index of the dataset. Must be sorted chronologically.

        Returns
        -------
        list[WalkForwardFold]
            List of WalkForwardFold metadata objects.

        Raises
        ------
        ValueError
            If dataset is too small to construct at least one fold.
        """
        n_samples = len(index)
        if n_samples < 10:
            raise ValueError(f"Dataset index length ({n_samples}) is too small for walk-forward validation")

        # Determine exact integer window sizes
        min_train = (
            self.min_train_size
            if self.min_train_size is not None
            else int(n_samples * self.min_train_ratio)
        )
        val_sz = (
            self.val_size
            if self.val_size is not None
            else max(1, int(n_samples * self.val_size_ratio))
        )
        step_sz = (
            self.step_size
            if self.step_size is not None
            else max(1, int(n_samples * self.step_size_ratio))
        )

        if min_train + val_sz > n_samples:
            raise ValueError(
                f"Dataset size ({n_samples} rows) is smaller than required initial train + val window "
                f"({min_train} + {val_sz} = {min_train + val_sz} rows)."
            )

        folds = []
        fold_idx = 0
        train_end_idx = min_train

        while train_end_idx + val_sz <= n_samples:
            train_idx = index[:train_end_idx]
            val_idx = index[train_end_idx : train_end_idx + val_sz]

            # Strict temporal check
            if len(train_idx) > 0 and len(val_idx) > 0:
                assert train_idx[-1] < val_idx[0], (
                    f"Temporal leakage detected in fold {fold_idx}: "
                    f"train_end ({train_idx[-1]}) >= val_start ({val_idx[0]})"
                )

            fold = WalkForwardFold(
                fold_index=fold_idx,
                train_start=train_idx[0],
                train_end=train_idx[-1],
                val_start=val_idx[0],
                val_end=val_idx[-1],
                train_indices=train_idx,
                val_indices=val_idx,
                train_size=len(train_idx),
                val_size=len(val_idx),
            )
            folds.append(fold)

            fold_idx += 1
            train_end_idx += step_sz

        logger.debug(
            "Generated %d walk-forward folds: initial_train=%d, val_size=%d, step_size=%d (total_samples=%d)",
            len(folds),
            min_train,
            val_sz,
            step_sz,
            n_samples,
        )

        return folds
