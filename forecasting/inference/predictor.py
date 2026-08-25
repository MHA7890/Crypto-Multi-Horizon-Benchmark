"""
forecasting.inference.predictor — Production prediction runner.
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from forecasting.data.scaler import LeakproofScaler
from forecasting.data.reduction import FeatureReducer
from forecasting.inference.converter import PriceInterval, ReturnToPriceConverter
from forecasting.models.base import ForecastModel
from forecasting.config.model_registry import get_model_class


class Predictor:
    """Loads production winner model and produces price intervals."""

    def __init__(self, models_dir: Path | str = "models"):
        self.models_dir = Path(models_dir)
        self.converter = ReturnToPriceConverter()

    def predict_coin(
        self,
        symbol: str,
        X_latest: pd.DataFrame,
        current_price: float,
        horizon: int = 1,
    ) -> PriceInterval:
        """Load coin's winning model, apply scaler & reducer, predict price interval."""
        coin_dir = self.models_dir / symbol
        if not coin_dir.exists():
            raise FileNotFoundError(f"No production model found for symbol {symbol} at {coin_dir}")

        meta_file = list(coin_dir.glob("*_meta.json"))
        if not meta_file:
            raise FileNotFoundError(f"Missing metadata JSON in {coin_dir}")

        with open(meta_file[0], "r", encoding="utf-8") as f:
            meta = json.load(f)

        model_name = meta["model_name"]
        model_cls = get_model_class(model_name)
        model = model_cls.load(coin_dir)

        scaler = LeakproofScaler.load(coin_dir / f"{symbol}_{model_name}_scaler.joblib")
        reducer = FeatureReducer.load(coin_dir / f"{symbol}_{model_name}_reducer.joblib")

        X_reduced = reducer.transform(X_latest)
        X_scaled = scaler.transform_val(X_reduced)

        prediction = model.predict(X_scaled, horizon=horizon)

        return self.converter.convert(
            current_price=current_price,
            point_forecast=float(prediction.point_forecast[-1]),
            lower_bound=float(prediction.lower_bound[-1]),
            upper_bound=float(prediction.upper_bound[-1]),
            horizon=horizon,
            confidence_level=prediction.confidence_level,
        )
