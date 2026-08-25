"""
forecasting.explainability.shap_analysis — SHAP feature importance for tree models.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
from forecasting.models.base import ForecastModel


class SHAPAnalyzer:
    """SHAP values for tree-based winning models."""

    def __init__(self, model: ForecastModel):
        self.model = model

    def feature_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculate mean absolute SHAP values per feature."""
        # SHAP calculation stub (runs after model selection)
        importance_df = pd.DataFrame(
            {"feature": X.columns, "importance": 1.0 / len(X.columns)}
        ).sort_values(by="importance", ascending=False)
        return importance_df

    def save_report(self, output_dir: Path | str) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "shap_importance.csv"
        # Save stub output
        pd.DataFrame({"feature": [], "importance": []}).to_csv(path, index=False)
        return path
