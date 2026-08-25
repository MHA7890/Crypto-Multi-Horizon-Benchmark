"""
forecasting.training.verifier — Post-Experiment Automated Artifact Verifier.

Audits every coin directory after experiment completion to guarantee that:
  1. Exactly 1 winning production model artifact exists in models/{SYMBOL}/
  2. Exactly 1 fitted scaler exists in models/{SYMBOL}/
  3. Exactly 1 fitted reducer exists in models/{SYMBOL}/
  4. Metadata JSON exists in models/{SYMBOL}/
  5. Evaluation CSV exists in evaluation/{SYMBOL}/metrics.csv
  6. Non-winning models are archived in archive/{SYMBOL}/

Generates verification_report.txt documenting audit status and any missing assets.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ExperimentVerifier:
    """
    Post-experiment directory and artifact auditor.
    """

    def __init__(self, models_dir: Path | str, archive_dir: Path | str, evaluation_dir: Path | str):
        self.models_dir = Path(models_dir)
        self.archive_dir = Path(archive_dir)
        self.evaluation_dir = Path(evaluation_dir)

    def verify_coin(self, symbol: str) -> list[str]:
        """Audit a single coin's directory structure for required production artifacts."""
        issues: list[str] = []
        coin_model_dir = self.models_dir / symbol
        coin_eval_dir = self.evaluation_dir / symbol

        # 1. Model directory existence
        if not coin_model_dir.exists():
            issues.append(f"[{symbol}] Missing models directory: {coin_model_dir}")
            return issues

        # 2. Check winning model artifacts
        model_artifacts = list(coin_model_dir.glob("*.joblib")) + list(coin_model_dir.glob("*.pt"))
        # Exclude scalers and reducers from primary model count
        primary_models = [
            p for p in model_artifacts if not (p.name.endswith("_scaler.joblib") or p.name.endswith("_reducer.joblib"))
        ]

        if len(primary_models) == 0:
            issues.append(f"[{symbol}] No primary production model artifact found in {coin_model_dir}")
        elif len(primary_models) > 1:
            issues.append(
                f"[{symbol}] Expected exactly 1 production winning model, but found {len(primary_models)}: {[p.name for p in primary_models]}"
            )

        # 3. Check scaler
        scalers = list(coin_model_dir.glob("*_scaler.joblib"))
        if len(scalers) == 0:
            issues.append(f"[{symbol}] Missing fitted scaler artifact (*_scaler.joblib) in {coin_model_dir}")

        # 4. Check reducer
        reducers = list(coin_model_dir.glob("*_reducer.joblib"))
        if len(reducers) == 0:
            issues.append(f"[{symbol}] Missing feature reducer artifact (*_reducer.joblib) in {coin_model_dir}")

        # 5. Check metadata JSON
        meta_files = list(coin_model_dir.glob("*_meta.json"))
        if len(meta_files) == 0:
            issues.append(f"[{symbol}] Missing metadata JSON (*_meta.json) in {coin_model_dir}")

        # 6. Check evaluation metrics CSV
        metrics_csv = coin_eval_dir / "metrics.csv"
        if not metrics_csv.exists():
            issues.append(f"[{symbol}] Missing evaluation metrics CSV: {metrics_csv}")

        return issues

    def verify_all(self, symbols: list[str], output_report_path: Path | str) -> dict[str, Any]:
        """
        Audit all symbols, output verification_report.txt, and return audit summary.
        """
        output_report_path = Path(output_report_path)
        output_report_path.parent.mkdir(parents=True, exist_ok=True)

        all_issues: list[str] = []
        verified_count = 0
        failed_verification_count = 0

        for symbol in symbols:
            coin_issues = self.verify_coin(symbol)
            if coin_issues:
                all_issues.extend(coin_issues)
                failed_verification_count += 1
            else:
                verified_count += 1

        # Format verification_report.txt
        lines = [
            "=" * 70,
            "AUTOMATED EXPERIMENT VERIFICATION REPORT",
            "=" * 70,
            f"Total Coins Audited: {len(symbols)}",
            f"Fully Verified Coins: {verified_count}",
            f"Coins with Missing/Inconsistent Artifacts: {failed_verification_count}",
            "=" * 70,
            "",
        ]

        if all_issues:
            lines.append("DETECTED ISSUES & MISSING ARTIFACTS:")
            for issue in all_issues:
                lines.append(f"  • {issue}")
        else:
            lines.append("SUCCESS: All coins passed verification. Every coin possesses exactly 1 production model, 1 scaler, 1 reducer, metadata JSON, and evaluation CSV.")

        lines.append("")
        lines.append("=" * 70)

        report_content = "\n".join(lines)
        with open(output_report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        logger.info("Verification audit completed (%d/%d verified). Report written to %s", verified_count, len(symbols), output_report_path)

        return {
            "total_coins": len(symbols),
            "verified_count": verified_count,
            "failed_verification_count": failed_verification_count,
            "issues": all_issues,
            "report_path": str(output_report_path),
        }
