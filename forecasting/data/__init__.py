"""
forecasting.data — Data ingestion, target construction, splitting, scaling, reduction.
"""

from forecasting.data.loader import CoinDataset, DataLoader
from forecasting.data.target import HorizonTarget, TargetConstructor
from forecasting.data.splitter import WalkForwardFold, WalkForwardSplitter
from forecasting.data.scaler import LeakproofScaler
from forecasting.data.reduction import FeatureReducer, ReductionReport

__all__ = [
    "CoinDataset",
    "DataLoader",
    "HorizonTarget",
    "TargetConstructor",
    "WalkForwardFold",
    "WalkForwardSplitter",
    "LeakproofScaler",
    "FeatureReducer",
    "ReductionReport",
]
