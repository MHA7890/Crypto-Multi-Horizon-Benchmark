from __future__ import annotations

import os
from pathlib import Path
import sys

# Crucial: Fix sys.path BEFORE any package imports occur
workspace_root = str(Path(__file__).resolve().parent.parent.parent)
_script_dir = str(Path(__file__).resolve().parent)

if os.path.normcase(_script_dir) in [os.path.normcase(p) for p in sys.path]:
    sys.path = [p for p in sys.path if os.path.normcase(os.path.abspath(p)) != os.path.normcase(_script_dir)]

if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

# Remove training module from sys.modules if it was mistakenly imported as top-level 'training'
if "training" in sys.modules and not hasattr(sys.modules["training"], "__path__"):
    del sys.modules["training"]

from forecasting.training.runner import ExperimentRunner
from forecasting.utils.logging import setup_logging
import argparse
import logging
import traceback

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run fully automated cryptocurrency forecasting benchmark across all models and coins."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML experiment configuration file (defaults to configs/experiment.yaml).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=None,
        help="Resume previous experiment from last saved checkpoint.",
    )
    parser.add_argument(
        "--no-resume",
        "--fresh",
        action="store_false",
        dest="resume",
        help="Force start a brand new experiment (e.g. exp_013) from scratch.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        dest="top_n",
        help="Select only the top N READY-status ranked cryptocurrencies (default: all).",
    )
    parser.add_argument(
        "--ranking-file",
        type=str,
        default=None,
        dest="ranking_file",
        help="Path to ranking CSV (default: output/coin_mapping.csv).",
    )
    args = parser.parse_args()

    log_dir = Path("experiments") / "logs"
    setup_logging(log_dir=log_dir, level=logging.INFO)

    print("[run_all] Initialising master experiment runner...", flush=True)

    try:
        runner = ExperimentRunner(
            config_path=args.config,
            resume=args.resume,
            top_n=args.top_n,
            ranking_file=args.ranking_file,
        )
        runner.run_all()
    except SystemExit:
        raise
    except Exception as exc:
        print(f"\n[FATAL] Experiment failed: {exc}", flush=True)
        traceback.print_exc()
        logger.critical("Fatal error during experiment execution: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
