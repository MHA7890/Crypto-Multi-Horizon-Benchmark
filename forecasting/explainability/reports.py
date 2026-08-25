"""
forecasting.explainability.reports — High-level explainability runner and reporter.
"""

from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd

from forecasting.explainability.ablation import AblationAnalyzer
from forecasting.explainability.attention import AttentionAnalyzer
from forecasting.explainability.permutation import PermutationAnalyzer
from forecasting.explainability.shap_analysis import SHAPAnalyzer
from forecasting.models.base import ForecastModel

logger = logging.getLogger(__name__)


class ExplainabilityReporter:
    """Dispatches and aggregates explainability analysis post model selection."""

    def __init__(self, model: ForecastModel):
        self.model = model
        self.shap_analyzer = SHAPAnalyzer(model)
        self.permutation_analyzer = PermutationAnalyzer(model)
        self.ablation_analyzer = AblationAnalyzer(model)
        self.attention_analyzer = AttentionAnalyzer(model)

    def run_full_analysis(
        self, X: pd.DataFrame, y: pd.Series, output_dir: Path | str
    ) -> None:
        """Run appropriate feature importance and ablation analyses based on model type."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Running explainability analysis for model: %s", self.model.name)

        if self.model.name in ("RandomForest", "XGBoost", "LightGBM"):
            self.shap_analyzer.save_report(output_dir)
            self.permutation_analyzer.save_report(output_dir)
        elif self.model.name in ("TFT", "PatchTST"):
            self.attention_analyzer.save_report(output_dir)
            self.permutation_analyzer.save_report(output_dir)

        self.ablation_analyzer.save_report(output_dir)
        logger.info("Explainability reports saved to %s", output_dir)
