"""
forecasting.config.settings — Dataclass configuration definitions.

Every configurable parameter in the system is defined here as a typed
dataclass. These are the canonical source of defaults. YAML config
files and CLI flags override these defaults via the loader module.

Design decisions
────────────────
• Dataclasses over dicts: IDE autocomplete, type checking, immutable defaults.
• Flat hierarchy with nested composition: each concern gets its own dataclass,
  composed into a single ExperimentConfig root.
• Validation in __post_init__: catches invalid configs at load time, not at
  training time 2 hours later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ────────────────────────────────────────────────────────────
# PATH CONFIGURATION
# ────────────────────────────────────────────────────────────

@dataclass
class PathConfig:
    """All filesystem paths.  Centralised to prevent hardcoded paths in code."""

    features_dir: Path = field(default_factory=lambda: Path("features"))
    models_dir: Path = field(default_factory=lambda: Path("models"))
    archive_dir: Path = field(default_factory=lambda: Path("archive"))
    evaluation_dir: Path = field(default_factory=lambda: Path("evaluation"))
    predictions_dir: Path = field(default_factory=lambda: Path("predictions"))
    experiments_dir: Path = field(default_factory=lambda: Path("experiments"))
    logs_dir: Path = field(default_factory=lambda: Path("logs"))

    def __post_init__(self) -> None:
        # Ensure Path objects even when loaded from YAML strings
        for fld in self.__dataclass_fields__:
            val = getattr(self, fld)
            if not isinstance(val, Path):
                setattr(self, fld, Path(val))


# ────────────────────────────────────────────────────────────
# TARGET VARIABLE
# ────────────────────────────────────────────────────────────

@dataclass
class TargetConfig:
    """Target variable construction parameters."""

    horizons: list[int] = field(default_factory=lambda: [1, 7, 14, 30, 90])
    price_column: str = "close"
    target_type: str = "log_return"

    def __post_init__(self) -> None:
        if not self.horizons:
            raise ValueError("At least one forecast horizon is required")
        if any(h <= 0 for h in self.horizons):
            raise ValueError(f"All horizons must be positive, got {self.horizons}")
        if self.target_type not in ("log_return",):
            raise ValueError(
                f"Unsupported target_type '{self.target_type}'. "
                f"Supported: 'log_return'"
            )


# ────────────────────────────────────────────────────────────
# FEATURE REDUCTION
# ────────────────────────────────────────────────────────────

@dataclass
class ReductionConfig:
    """Feature reduction thresholds."""

    correlation_threshold: float = 0.95
    variance_threshold: float = 1e-6

    def __post_init__(self) -> None:
        if not 0.0 < self.correlation_threshold <= 1.0:
            raise ValueError(
                f"correlation_threshold must be in (0, 1], "
                f"got {self.correlation_threshold}"
            )
        if self.variance_threshold < 0:
            raise ValueError(
                f"variance_threshold must be non-negative, "
                f"got {self.variance_threshold}"
            )


# ────────────────────────────────────────────────────────────
# WALK-FORWARD VALIDATION
# ────────────────────────────────────────────────────────────

@dataclass
class ValidationConfig:
    """Walk-forward validation parameters."""

    min_train_ratio: float = 0.6
    val_size_ratio: float = 0.1
    step_size_ratio: float = 0.1

    def __post_init__(self) -> None:
        if not 0.0 < self.min_train_ratio < 1.0:
            raise ValueError(
                f"min_train_ratio must be in (0, 1), got {self.min_train_ratio}"
            )
        if not 0.0 < self.val_size_ratio < 1.0:
            raise ValueError(
                f"val_size_ratio must be in (0, 1), got {self.val_size_ratio}"
            )
        if self.min_train_ratio + self.val_size_ratio > 1.0:
            raise ValueError(
                f"min_train_ratio + val_size_ratio must be <= 1.0, "
                f"got {self.min_train_ratio + self.val_size_ratio}"
            )


# ────────────────────────────────────────────────────────────
# FEATURE SCALING
# ────────────────────────────────────────────────────────────

@dataclass
class ScalingConfig:
    """Feature scaling parameters."""

    method: str = "robust"
    quantile_range: tuple[float, float] = (25.0, 75.0)

    def __post_init__(self) -> None:
        if self.method != "robust":
            raise ValueError(
                f"Only 'robust' scaling is supported, got '{self.method}'"
            )
        lo, hi = self.quantile_range
        if not (0.0 <= lo < hi <= 100.0):
            raise ValueError(
                f"quantile_range must satisfy 0 <= lo < hi <= 100, "
                f"got ({lo}, {hi})"
            )


# ────────────────────────────────────────────────────────────
# EVALUATION SCORING
# ────────────────────────────────────────────────────────────

@dataclass
class ScoringConfig:
    """
    Composite scoring weights and prediction interval settings.

    Loaded from config — NOT hardcoded. Validated to sum to 1.0.
    """

    accuracy_weight: float = 0.40
    interval_weight: float = 0.35
    directional_weight: float = 0.20
    efficiency_weight: float = 0.05
    confidence_level: float = 0.90

    def __post_init__(self) -> None:
        total = (
            self.accuracy_weight
            + self.interval_weight
            + self.directional_weight
            + self.efficiency_weight
        )
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Scoring weights must sum to 1.0, got {total:.6f}. "
                f"Adjust weights in configs/experiment.yaml"
            )
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError(
                f"confidence_level must be in (0, 1), "
                f"got {self.confidence_level}"
            )


# ────────────────────────────────────────────────────────────
# HARDWARE / DEVICE
# ────────────────────────────────────────────────────────────

@dataclass
class DeviceConfig:
    """CUDA and hardware settings."""

    use_cuda: bool = True
    gpu_memory_fraction: float = 0.9
    mixed_precision: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.gpu_memory_fraction <= 1.0:
            raise ValueError(
                f"gpu_memory_fraction must be in (0, 1], "
                f"got {self.gpu_memory_fraction}"
            )


# ────────────────────────────────────────────────────────────
# TRAINING PARAMETERS
# ────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    """Global training parameters."""

    seed: int = 42
    n_jobs: int = 1
    verbose: bool = True
    save_all_folds: bool = False
    resume: bool = True


# ────────────────────────────────────────────────────────────
# ROOT CONFIGURATION
# ────────────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    """
    Root configuration object.

    Every configurable parameter in the system traces back to this
    object.  No magic numbers exist in the codebase — everything
    flows from here.
    """

    name: str = "default"

    paths: PathConfig = field(default_factory=PathConfig)
    target: TargetConfig = field(default_factory=TargetConfig)
    reduction: ReductionConfig = field(default_factory=ReductionConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    scaling: ScalingConfig = field(default_factory=ScalingConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # Per-model hyperparameters keyed by model name
    model_params: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Subset filters — None means "all"
    coins: list[str] | None = None
    models: list[str] | None = None
