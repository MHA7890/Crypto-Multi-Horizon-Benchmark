from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
import traceback

# Ensure workspace root is in sys.path
workspace_root = Path(__file__).resolve().parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from forecasting.utils.logging import setup_logging
from forecasting.training.runner import ExperimentRunner

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
        help="Resume previous experiment from last saved checkpoint.",
    )
    args = parser.parse_args()

    # Configure logging to stdout & file
    log_dir = Path("experiments") / "logs"
    setup_logging(log_dir=log_dir, level=logging.INFO)

    print("[run_all] Initialising master experiment runner...", flush=True)

    try:
        runner = ExperimentRunner(config_path=args.config, resume=args.resume)
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
