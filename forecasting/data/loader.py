"""
forecasting.data.loader — Feature CSV loading, schema validation, and detailed data auditing.

Loads cryptocurrency feature CSVs produced by upstream feature engineering.
Parses timestamps, validates schema, checks basic data integrity, and preserves
websocket gaps without imputation (per spec).
Includes in-memory caching to avoid redundant disk reads.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CoinDataset:
    """
    Immutable container for a single coin's loaded feature dataset.
    """

    symbol: str
    features: pd.DataFrame
    feature_names: list[str]
    n_rows: int
    n_features: int
    date_range: tuple[pd.Timestamp, pd.Timestamp]
    filepath: Path


class DataLoader:
    """
    Data loader for cryptocurrency feature CSVs with dataset caching.
    """

    def __init__(self, features_dir: Union[Path, str] = "features"):
        self.features_dir = Path(features_dir)
        self._cache: dict[str, CoinDataset] = {}

    def available_symbols(self) -> list[str]:
        if not self.features_dir.exists() or not self.features_dir.is_dir():
            logger.warning("Features directory does not exist: %s", self.features_dir)
            return []

        symbols = []
        for file in sorted(self.features_dir.glob("*_features.csv")):
            symbol = file.name.replace("_features.csv", "")
            symbols.append(symbol)

        logger.debug("Found %d available symbols in %s", len(symbols), self.features_dir)
        return symbols

    def load(self, symbol: str, price_col: str = "close") -> CoinDataset:
        symbol_upper = symbol.upper().strip()
        if symbol_upper in self._cache:
            return self._cache[symbol_upper]

        filepath = self.features_dir / f"{symbol_upper}_features.csv"

        if not filepath.exists():
            filepath_exact = self.features_dir / f"{symbol}_features.csv"
            if filepath_exact.exists():
                filepath = filepath_exact
            else:
                raise FileNotFoundError(
                    f"Feature dataset not found for symbol '{symbol}' at path: {filepath}"
                )

        logger.info("Loading feature dataset for %s from %s", symbol_upper, filepath)

        try:
            df = pd.read_csv(filepath)
        except Exception as err:
            raise ValueError(f"Failed to parse CSV file for {symbol_upper}: {err}") from err

        if df.empty:
            raise ValueError(f"Feature CSV for {symbol_upper} is empty: {filepath}")

        if "timestamp" not in df.columns:
            raise ValueError(
                f"Feature dataset for {symbol_upper} is missing required 'timestamp' column"
            )

        if price_col not in df.columns:
            raise ValueError(
                f"Feature dataset for {symbol_upper} is missing required price column '{price_col}'."
            )

        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        except Exception as err:
            raise ValueError(
                f"Failed to parse timestamps into DatetimeIndex for {symbol_upper}: {err}"
            ) from err

        dups = df["timestamp"].duplicated().sum()
        if dups > 0:
            logger.warning(
                "Dataset for %s contains %d duplicate timestamps; keeping first occurrence",
                symbol_upper,
                dups,
            )
            df = df.drop_duplicates(subset=["timestamp"], keep="first")

        df.set_index("timestamp", inplace=True)
        if not df.index.is_monotonic_increasing:
            logger.info("Sorting index chronologically for %s", symbol_upper)
            df.sort_index(inplace=True)

        feature_names = list(df.columns)
        date_range = (df.index[0], df.index[-1])

        # Detailed Infinite Value Audit & Reporting
        num_df = df.select_dtypes(include=[np.number])
        pos_inf_mask = np.isposinf(num_df)
        neg_inf_mask = np.isneginf(num_df)
        inf_count = pos_inf_mask.sum().sum() + neg_inf_mask.sum().sum()

        if inf_count > 0:
            logger.warning(
                "Dataset %s contains %d infinite values; coercing infinities to NaN",
                symbol_upper,
                inf_count,
            )
            for col in num_df.columns:
                pos_idx = num_df.index[pos_inf_mask[col]]
                neg_idx = num_df.index[neg_inf_mask[col]]
                for ts in pos_idx:
                    logger.warning(
                        "[%s Infinite Value Audit] Feature '%s' at timestamp %s contains +inf",
                        symbol_upper,
                        col,
                        ts.strftime("%Y-%m-%d %H:%M:%S UTC"),
                    )
                for ts in neg_idx:
                    logger.warning(
                        "[%s Infinite Value Audit] Feature '%s' at timestamp %s contains -inf",
                        symbol_upper,
                        col,
                        ts.strftime("%Y-%m-%d %H:%M:%S UTC"),
                    )

            df.replace([np.inf, -np.inf], np.nan, inplace=True)

        dataset = CoinDataset(
            symbol=symbol_upper,
            features=df,
            feature_names=feature_names,
            n_rows=len(df),
            n_features=len(feature_names),
            date_range=date_range,
            filepath=filepath,
        )

        logger.info(
            "Successfully loaded %s: %d rows, %d features, range %s to %s",
            symbol_upper,
            dataset.n_rows,
            dataset.n_features,
            date_range[0].strftime("%Y-%m-%d %H:%M:%S UTC"),
            date_range[1].strftime("%Y-%m-%d %H:%M:%S UTC"),
        )

        self._cache[symbol_upper] = dataset
        return dataset

    def load_all(self, price_col: str = "close") -> list[CoinDataset]:
        symbols = self.available_symbols()
        datasets = []
        for symbol in symbols:
            try:
                ds = self.load(symbol, price_col=price_col)
                datasets.append(ds)
            except Exception as err:
                logger.error("Skipping %s due to loading failure: %s", symbol, err)
        return datasets
