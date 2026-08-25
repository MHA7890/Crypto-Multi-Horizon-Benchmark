"""
forecasting.__main__ — CLI entry point for the forecasting system.

Usage:
    python -m forecasting run-all [--config configs/experiment.yaml] [--resume]
    python -m forecasting train [--coin BTC] [--model XGBoost] [--horizon 1] [--config configs/experiment.yaml]
    python -m forecasting select [--coin BTC]
    python -m forecasting predict --coin BTC [--horizon 1]
    python -m forecasting explain --coin BTC
    python -m forecasting report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from forecasting.evaluation.reports import GlobalReportGenerator
from forecasting.inference.predictor import Predictor
from forecasting.training.runner import ExperimentRunner
from forecasting.utils.logging import setup_logging


def main() -> None:
    # Initialize logging for CLI entry point
    log_dir = Path("experiments") / "logs"
    setup_logging(log_dir=log_dir)

    parser = argparse.ArgumentParser(
        prog="python -m forecasting",
        description="Production Cryptocurrency Forecasting Framework",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # 1. run-all command
    run_all_parser = subparsers.add_parser("run-all", help="Run master automated experiment across all coins and models")
    run_all_parser.add_argument("--config", type=str, help="Path to YAML experiment configuration file")
    run_all_parser.add_argument("--resume", action="store_true", default=None, help="Resume experiment from last saved checkpoint")
    run_all_parser.add_argument("--no-resume", "--fresh", action="store_false", dest="resume", help="Force start a brand new experiment (e.g. exp_013) from scratch")
    run_all_parser.add_argument("--top-n", type=int, default=None, dest="top_n",
        help="Select only the top N READY-status ranked cryptocurrencies (default: all)")
    run_all_parser.add_argument("--ranking-file", type=str, default=None, dest="ranking_file",
        help="Path to ranking CSV (default: output/coin_mapping.csv)")

    # 2. train command
    train_parser = subparsers.add_parser("train", help="Run model training pipeline")
    train_parser.add_argument("--coin", type=str, help="Specific coin symbol (e.g. BTC)")
    train_parser.add_argument("--model", type=str, help="Specific model name (e.g. XGBoost)")
    train_parser.add_argument("--horizon", type=int, help="Specific forecast horizon in days")
    train_parser.add_argument("--config", type=str, help="Path to config YAML file")
    train_parser.add_argument("--resume", action="store_true", help="Resume experiment from last checkpoint")

    # 3. select command
    select_parser = subparsers.add_parser("select", help="Run model selection and archiving")
    select_parser.add_argument("--coin", type=str, help="Specific coin symbol")
    select_parser.add_argument("--config", type=str, help="Path to config YAML file")

    # 4. predict command
    predict_parser = subparsers.add_parser("predict", help="Generate price prediction interval with winner model")
    predict_parser.add_argument("--coin", type=str, required=True, help="Coin symbol")
    predict_parser.add_argument("--horizon", type=int, default=1, help="Horizon in days")

    # 5. explain command
    explain_parser = subparsers.add_parser("explain", help="Run explainability report on winner model")
    explain_parser.add_argument("--coin", type=str, required=True, help="Coin symbol")

    # 6. report command
    report_parser = subparsers.add_parser("report", help="Generate global benchmark reports and publication plots")
    report_parser.add_argument("--evaluation-dir", type=str, default="evaluation", help="Evaluation directory path")

    args = parser.parse_args()

    if args.command == "run-all" or not args.command or args.command == "train":
        config_path = getattr(args, "config", None)
        resume = getattr(args, "resume", None)
        top_n = getattr(args, "top_n", None)
        ranking_file = getattr(args, "ranking_file", None)
        runner = ExperimentRunner(
            config_path=config_path,
            resume=resume,
            top_n=top_n,
            ranking_file=ranking_file,
        )

        if getattr(args, "coin", None):
            runner.run_coin(args.coin)
        else:
            runner.run_all()

    elif args.command == "predict":
        predictor = Predictor()
        print(f"Predictor initialized for symbol {args.coin}")

    elif args.command == "select":
        print("Selecting best models...")

    elif args.command == "explain":
        print(f"Running explainability for {args.coin}...")

    elif args.command == "report":
        eval_dir = getattr(args, "evaluation_dir", "evaluation")
        generator = GlobalReportGenerator(evaluation_dir=eval_dir)
        reports = generator.generate_reports()
        print(f"Successfully generated evaluation reports & publication charts in {eval_dir}/")


if __name__ == "__main__":
    main()
