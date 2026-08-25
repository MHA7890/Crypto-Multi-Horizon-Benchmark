"""
forecasting.config — Configuration system.

Provides layered YAML + dataclass configuration with validation.
"""

from forecasting.config.settings import (
    DeviceConfig,
    ExperimentConfig,
    PathConfig,
    ReductionConfig,
    ScalingConfig,
    ScoringConfig,
    TargetConfig,
    TrainingConfig,
    ValidationConfig,
)
from forecasting.config.loader import load_config
from forecasting.config.model_registry import get_model_class, list_available_models

__all__ = [
    "DeviceConfig",
    "ExperimentConfig",
    "PathConfig",
    "ReductionConfig",
    "ScalingConfig",
    "ScoringConfig",
    "TargetConfig",
    "TrainingConfig",
    "ValidationConfig",
    "get_model_class",
    "list_available_models",
    "load_config",
]
