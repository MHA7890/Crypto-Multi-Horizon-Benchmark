"""
forecasting.optimization.search_spaces — Parameter search space definitions per model.
"""

from __future__ import annotations

SEARCH_SPACES: dict[str, dict[str, tuple[str, ...]]] = {
    "XGBoost": {
        "n_estimators": ("int", 100, 3000),
        "max_depth": ("int", 3, 12),
        "learning_rate": ("log_float", 0.005, 0.3),
        "subsample": ("float", 0.5, 1.0),
        "colsample_bytree": ("float", 0.3, 1.0),
        "min_child_weight": ("int", 1, 30),
        "reg_alpha": ("log_float", 1e-8, 10.0),
        "reg_lambda": ("log_float", 1e-8, 10.0),
    },
    "LightGBM": {
        "n_estimators": ("int", 100, 3000),
        "max_depth": ("int", -1, 15),
        "learning_rate": ("log_float", 0.005, 0.3),
        "num_leaves": ("int", 15, 255),
        "subsample": ("float", 0.5, 1.0),
        "colsample_bytree": ("float", 0.3, 1.0),
        "min_child_samples": ("int", 5, 100),
        "reg_alpha": ("log_float", 1e-8, 10.0),
        "reg_lambda": ("log_float", 1e-8, 10.0),
    },
    "RandomForest": {
        "n_estimators": ("int", 100, 1000),
        "min_samples_split": ("int", 2, 20),
        "min_samples_leaf": ("int", 1, 10),
    },
    "TFT": {
        "hidden_size": ("int", 16, 128),
        "learning_rate": ("log_float", 1e-4, 1e-2),
        "dropout": ("float", 0.05, 0.3),
    },
    "PatchTST": {
        "d_model": ("int", 32, 128),
        "n_heads": ("int", 2, 8),
        "learning_rate": ("log_float", 1e-4, 1e-2),
    },
}
