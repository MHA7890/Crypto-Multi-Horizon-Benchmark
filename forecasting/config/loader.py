"""
forecasting.config.loader — YAML → dataclass configuration loading.

Implements the layered configuration precedence:
    CLI flags  →  experiment YAML  →  per-model default YAMLs  →  dataclass defaults

Design decisions
────────────────
• Deep-merge strategy: nested dicts are merged recursively so that
  specifying only ``scoring.accuracy_weight`` in the YAML doesn't
  wipe out the other scoring defaults.
• Path resolution: all paths in PathConfig are resolved relative to
  the project root (the directory containing ``features/``).
• Model param merging: per-model YAMLs in ``config/defaults/`` provide
  base hyperparameters; ``model_params`` in the experiment YAML
  overrides individual keys without replacing the whole dict.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import yaml

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

logger = logging.getLogger(__name__)

_SECTION_TYPES: dict[str, type] = {
    "paths": PathConfig,
    "target": TargetConfig,
    "reduction": ReductionConfig,
    "validation": ValidationConfig,
    "scaling": ScalingConfig,
    "scoring": ScoringConfig,
    "device": DeviceConfig,
    "training": TrainingConfig,
}

_DEFAULTS_DIR = Path(__file__).parent / "defaults"


def compute_config_hash(config_dict: dict[str, Any]) -> str:
    """Compute deterministic SHA-256 hash of configuration dictionary."""
    serialized = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def load_config(
    config_path: Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> ExperimentConfig:
    merged: dict[str, Any] = {}

    if config_path is not None:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}
        logger.info("Loaded experiment config from %s", config_path)
        merged = _deep_merge(merged, yaml_data)

    if cli_overrides:
        merged = _deep_merge(merged, cli_overrides)

    model_params = _load_model_defaults()
    if "model_params" in merged:
        for model_name, overrides in merged["model_params"].items():
            if model_name in model_params:
                model_params[model_name].update(overrides)
            else:
                model_params[model_name] = overrides
    merged["model_params"] = model_params

    config = _build_config(merged)
    logger.info("Configuration loaded: name='%s'", config.name)
    return config


def _deep_merge(base: dict, overlay: dict) -> dict:
    result = base.copy()
    for key, value in overlay.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_model_defaults() -> dict[str, dict[str, Any]]:
    model_params: dict[str, dict[str, Any]] = {}
    if not _DEFAULTS_DIR.is_dir():
        logger.warning("Model defaults directory not found: %s", _DEFAULTS_DIR)
        return model_params

    for yaml_file in sorted(_DEFAULTS_DIR.glob("*.yaml")):
        model_key = yaml_file.stem
        with open(yaml_file, "r", encoding="utf-8") as f:
            params = yaml.safe_load(f) or {}
        model_params[model_key] = params
        logger.debug("Loaded model defaults: %s (%d params)", model_key, len(params))

    return model_params


def _build_config(data: dict[str, Any]) -> ExperimentConfig:
    kwargs: dict[str, Any] = {}

    for key, value in data.items():
        if key in _SECTION_TYPES and isinstance(value, dict):
            section_cls = _SECTION_TYPES[key]
            kwargs[key] = _build_dataclass(section_cls, value)
        else:
            kwargs[key] = value

    valid_fields = {f.name for f in fields(ExperimentConfig)}
    filtered = {}
    for key, value in kwargs.items():
        if key in valid_fields:
            filtered[key] = value
        else:
            logger.warning("Ignoring unknown config key: '%s'", key)

    return ExperimentConfig(**filtered)


def _build_dataclass(cls: type, data: dict[str, Any]) -> Any:
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")

    valid_fields = {f.name for f in fields(cls)}
    filtered = {}
    for key, value in data.items():
        if key in valid_fields:
            filtered[key] = value
        else:
            logger.warning(
                "Ignoring unknown key '%s' in %s config", key, cls.__name__
            )

    return cls(**filtered)
