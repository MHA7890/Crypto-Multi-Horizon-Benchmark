"""
forecasting.config.model_registry — Centralised model registration.

Maps human-readable model names to their implementation classes via
lazy imports. This avoids loading heavy dependencies (torch,
xgboost, lightgbm, etc.) unless the model is actually requested.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from forecasting.models.base import ForecastModel

logger = logging.getLogger(__name__)


def _import_arima() -> type[ForecastModel]:
    from forecasting.models.arima import ARIMAModel
    return ARIMAModel


def _import_random_forest() -> type[ForecastModel]:
    from forecasting.models.random_forest import RandomForestModel
    return RandomForestModel


def _import_xgboost() -> type[ForecastModel]:
    from forecasting.models.xgboost_model import XGBoostModel
    return XGBoostModel


def _import_lightgbm() -> type[ForecastModel]:
    from forecasting.models.lightgbm_model import LightGBMModel
    return LightGBMModel


def _import_tft() -> type[ForecastModel]:
    from forecasting.models.tft import TFTModel
    return TFTModel


def _import_patchtst() -> type[ForecastModel]:
    from forecasting.models.patchtst import PatchTSTModel
    return PatchTSTModel


MODEL_REGISTRY: dict[str, callable] = {
    "ARIMA":        _import_arima,
    "RandomForest": _import_random_forest,
    "XGBoost":      _import_xgboost,
    "LightGBM":     _import_lightgbm,
    "TFT":          _import_tft,
    "PatchTST":     _import_patchtst,
}


def get_model_class(name: str) -> type[ForecastModel]:
    """
    Resolve a model name to its implementation class via lazy importing.

    Raises
    ------
    KeyError
        If model name is unknown.
    ImportError
        If required external library (xgboost, lightgbm, torch) is missing.
    """
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise KeyError(
            f"Unknown model '{name}'. Available models: {available}"
        )
    try:
        return MODEL_REGISTRY[name]()
    except ImportError as err:
        raise ImportError(
            f"Failed to import dependencies for model '{name}': {err}. "
            f"Ensure required package is installed."
        ) from err


def list_available_models() -> list[str]:
    """Return a sorted list of all registered model names."""
    return sorted(MODEL_REGISTRY.keys())
