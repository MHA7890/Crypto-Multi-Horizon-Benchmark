"""
forecasting.training.runner — Automated Master Experiment Runner.

Orchestrates the entire cryptocurrency forecasting benchmark across all coins and models.
Features:
- Automatic hardware discovery & model device routing (CUDA/CPU)
- Progress checkpointing & resumption (checkpoint.json)
- Signal trapping for graceful shutdown (Ctrl+C)
- Failure tracking (evaluation/failures.csv)
- Automated winner selection & archiving
- Global report and publication chart generation
- Real-time tqdm progress bars with ETA
"""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
import sys
import time
import traceback
from typing import Any

from tqdm import tqdm

from forecasting.config.loader import compute_config_hash, load_config
from forecasting.config.model_registry import get_model_class, list_available_models
from forecasting.config.settings import ExperimentConfig
from forecasting.data.coin_selector import (
    load_ranking,
    print_selection_audit,
    save_snapshots,
    select_top_n,
)
from forecasting.data.loader import DataLoader
from forecasting.evaluation.reports import GlobalReportGenerator
from forecasting.evaluation.reporter import EvaluationReporter
from forecasting.evaluation.scorer import CompositeScorer
from forecasting.selection.selector import ModelSelector
from forecasting.training.checkpoint import CheckpointManager
from forecasting.training.pipeline import TrainingPipeline
from forecasting.training.verifier import ExperimentVerifier
from forecasting.utils.device import DeviceManager, clear_gpu_memory
from forecasting.utils.notifications import MultiNotifier
from forecasting.utils.signal_handler import GracefulInterruptHandler

logger = logging.getLogger(__name__)


# ───────────────────────── helpers ──────────────────────────

def _next_experiment_id(experiments_dir: Path) -> str:
    """Scan experiments directory and generate next exp_XXX identifier."""
    experiments_dir.mkdir(parents=True, exist_ok=True)
    existing = [d.name for d in experiments_dir.iterdir() if d.is_dir() and d.name.startswith("exp_")]
    if not existing:
        return "exp_001"

    numbers = []
    for exp_name in existing:
        try:
            num = int(exp_name.split("_")[1])
            numbers.append(num)
        except (IndexError, ValueError):
            continue

    next_num = max(numbers) + 1 if numbers else 1
    return f"exp_{next_num:03d}"


def _sanitize_for_yaml(data: Any) -> Any:
    if isinstance(data, Path):
        return str(data)
    elif hasattr(data, "__dataclass_fields__"):
        return {k: _sanitize_for_yaml(getattr(data, k)) for k in data.__dataclass_fields__}
    elif isinstance(data, dict):
        return {k: _sanitize_for_yaml(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [_sanitize_for_yaml(v) for v in data]
    return data


def _save_config_snapshot(config: ExperimentConfig, snapshot_path: Path) -> None:
    try:
        import yaml
        sanitized = _sanitize_for_yaml(config)
        with open(snapshot_path, "w", encoding="utf-8") as f:
            yaml.dump(sanitized, f, default_flow_style=False)
    except Exception as err:
        logger.warning("Failed to save config snapshot: %s", err)


def _fmt_elapsed(seconds: float) -> str:
    """Human-readable elapsed time."""
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


# ───────────────────────── main class ──────────────────────────

class ExperimentRunner:
    """
    Master automated experiment runner.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        resume: bool | None = None,
        top_n: int | None = None,
        ranking_file: str | Path | None = None,
    ):
        self.config = load_config(config_path)
        self.resume = self.config.training.resume if resume is None else resume
        self.top_n = top_n
        self.ranking_file = ranking_file

        self.device_mgr = DeviceManager(use_cuda=self.config.device.use_cuda)
        self.device_mgr.print_startup_summary()

        self.experiments_dir = self.config.paths.experiments_dir
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

        if self.resume:
            latest_exp = self._find_latest_experiment()
            if latest_exp:
                self.exp_id = latest_exp.name
                self.exp_dir = latest_exp
                logger.info("Resuming experiment from directory %s", self.exp_dir)
            else:
                self.exp_id = _next_experiment_id(self.experiments_dir)
                self.exp_dir = self.experiments_dir / self.exp_id
        else:
            self.exp_id = _next_experiment_id(self.experiments_dir)
            self.exp_dir = self.experiments_dir / self.exp_id

        self.exp_dir.mkdir(parents=True, exist_ok=True)

        config_snapshot_path = self.exp_dir / "config_snapshot.yaml"
        _save_config_snapshot(self.config, config_snapshot_path)
        config_dict = _sanitize_for_yaml(self.config)
        self.config_hash = compute_config_hash(config_dict)

        self.pipeline = TrainingPipeline(self.config)
        self.reporter = EvaluationReporter(evaluation_dir=self.config.paths.evaluation_dir)
        self.scorer = CompositeScorer(self.config.scoring)
        self.selector = ModelSelector(
            models_dir=self.config.paths.models_dir,
            archive_dir=self.config.paths.archive_dir,
            scorer=self.scorer,
        )
        self.loader = DataLoader(features_dir=self.config.paths.features_dir)
        self.checkpoint_mgr = CheckpointManager(
            checkpoint_path=self.exp_dir / "checkpoint.json",
            config_hash=self.config_hash,
        )
        self.notifier = MultiNotifier()
        self.report_generator = GlobalReportGenerator(evaluation_dir=self.config.paths.evaluation_dir)
        self.verifier = ExperimentVerifier(
            models_dir=self.config.paths.models_dir,
            archive_dir=self.config.paths.archive_dir,
            evaluation_dir=self.config.paths.evaluation_dir,
        )

        self.failures_csv = Path(self.config.paths.evaluation_dir) / "failures.csv"

    # ─────────────────── internal helpers ───────────────────

    def _find_latest_experiment(self) -> Path | None:
        if not self.experiments_dir.exists():
            return None
        dirs = [d for d in self.experiments_dir.iterdir() if d.is_dir() and d.name.startswith("exp_")]
        if not dirs:
            return None
        return max(dirs, key=lambda d: d.stat().st_mtime)

    def _record_failure(self, symbol: str, model_name: str, exc: Exception) -> None:
        tb_str = traceback.format_exc()
        self.checkpoint_mgr.record_failure(symbol, model_name, str(exc), tb_str)
        self.notifier.on_failure(symbol, model_name, str(exc))

        self.failures_csv.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.failures_csv.exists()

        with open(self.failures_csv, "a", encoding="utf-8") as f:
            if not file_exists:
                f.write("timestamp,symbol,model,error,traceback\n")
            ts = datetime.now(timezone.utc).isoformat()
            clean_exc = str(exc).replace(",", ";").replace("\n", " ")
            clean_tb = tb_str.replace(",", ";").replace("\n", " | ")
            f.write(f"{ts},{symbol},{model_name},{clean_exc},{clean_tb}\n")

    @staticmethod
    def _print_flush(msg: str) -> None:
        """Print a message and immediately flush stdout so it appears in real-time."""
        try:
            print(msg, flush=True)
        except UnicodeEncodeError:
            print(msg.encode("ascii", errors="replace").decode("ascii"), flush=True)

    # ─────────────────── per-coin runner ───────────────────

    def run_coin(self, symbol: str, coin_idx: int = 0, total_coins: int = 0) -> dict[str, str]:
        if self.checkpoint_mgr.is_coin_completed(symbol):
            logger.info("Skipping already completed coin %s (from checkpoint)", symbol)
            return {"symbol": symbol, "winner": "already_completed"}

        model_names = self.config.models or list_available_models()
        horizons = self.config.target.horizons
        total_models = len(model_names)
        total_horizons = len(horizons)
        total_steps = total_models * total_horizons

        coin_header = f"[{coin_idx}/{total_coins}]" if total_coins else ""
        self._print_flush(
            f"\n{'='*70}\n"
            f"  COIN {coin_header} {symbol} - {total_models} models x {total_horizons} horizons = {total_steps} pipeline runs\n"
            f"{'='*70}"
        )
        logger.info("Running experiment %s for coin %s across models: %s", self.exp_id, symbol, model_names)

        coin_metrics_records = []
        coin_model_metrics = {}
        step_counter = 0

        for m_idx, model_name in enumerate(model_names, 1):
            if self.checkpoint_mgr.is_model_completed(symbol, model_name):
                self._print_flush(f"  [OK] {model_name} - already completed (checkpoint)")
                step_counter += total_horizons
                continue

            target_device = self.device_mgr.get_device_for_model(model_name)

            try:
                model_cls = get_model_class(model_name)
                model_params = self.config.model_params.get(model_name, {}).copy()

                if "device" in model_cls.__init__.__code__.co_varnames:
                    model_params["device"] = target_device

                model = model_cls(**model_params)

                self._print_flush(
                    f"\n  >> Model {m_idx}/{total_models}: {model_name} ({model_cls.__name__}) "
                    f"[{target_device.upper()}]"
                )

                for h_idx, horizon in enumerate(horizons, 1):
                    step_counter += 1
                    step_start = time.time()
                    self._print_flush(
                        f"    +-- Horizon {h_idx}/{total_horizons} ({horizon}d) "
                        f"[step {step_counter}/{total_steps}] ..."
                    )

                    pipeline_res = self.pipeline.run(
                        symbol=symbol,
                        model=model,
                        horizon=horizon,
                        output_dir=self.config.paths.models_dir / symbol,
                    )

                    elapsed = time.time() - step_start
                    m = pipeline_res.metrics
                    self._print_flush(
                        f"    |  [OK] done in {_fmt_elapsed(elapsed)} - "
                        f"RMSE={m.rmse:.4f}  MAE={m.mae:.4f}  DirAcc={m.direction_accuracy:.2%}"
                    )

                    record = {
                        "experiment": self.exp_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "symbol": symbol,
                        "model": model_name,
                        "horizon": horizon,
                        "rmse": m.rmse,
                        "mae": m.mae,
                        "median_ae": m.median_ae,
                        "mape": m.mape,
                        "picp": m.picp,
                        "mpiw": m.mpiw,
                        "dir_accuracy": m.direction_accuracy,
                        "dir_precision": m.direction_precision,
                        "dir_recall": m.direction_recall,
                        "dir_f1": m.direction_f1,
                        "train_time_s": m.training_time_seconds,
                        "inference_time_s": m.inference_time_seconds,
                        "model_size_bytes": m.model_size_bytes,
                    }
                    coin_metrics_records.append(record)

                    if model_name not in coin_model_metrics:
                        coin_model_metrics[model_name] = []
                    coin_model_metrics[model_name].append(m)

                self.checkpoint_mgr.record_model_completion(symbol, model_name)
                self._print_flush(f"    \\-- [OK] {model_name} complete")

            except Exception as exc:
                logger.error("Error executing model %s for coin %s: %s", model_name, symbol, exc)
                self._print_flush(f"    \\-- [FAIL] {model_name} FAILED: {exc}")
                self._record_failure(symbol, model_name, exc)
                step_counter += max(0, total_horizons - (step_counter % total_horizons))

            clear_gpu_memory()

        if coin_metrics_records:
            self.reporter.save_coin_metrics(symbol, coin_metrics_records)

        if not coin_model_metrics:
            logger.info("No new model metrics recorded for %s", symbol)
            return {"symbol": symbol, "winner": "already_completed"}

        winner_model = self.selector.select_and_archive(symbol, coin_model_metrics)
        self.checkpoint_mgr.record_coin_completion(symbol, winner_model)

        self._print_flush(f"  ** Winner for {symbol}: {winner_model}")
        logger.info("Selected winner for %s: %s", symbol, winner_model)
        return {"symbol": symbol, "winner": winner_model}

    # ─────────────────── master orchestrator ───────────────────

    def run_all(self) -> None:
        # ── Coin selection: top-N or all ──
        all_available = self.config.coins or self.loader.available_symbols()

        if self.top_n is not None:
            ranking_df = load_ranking(self.ranking_file)
            result = select_top_n(
                top_n=self.top_n,
                ranking_df=ranking_df,
                available_symbols=all_available,
                ranking_source=str(self.ranking_file or "output/coin_mapping.csv"),
            )
            print_selection_audit(result)
            save_snapshots(self.exp_dir, result)
            symbols = result.trainable_symbols
        else:
            symbols = all_available

        total_coins = len(symbols)
        candidate_models = self.config.models or list_available_models()
        total_models = len(candidate_models)
        horizons = self.config.target.horizons
        total_horizons = len(horizons)
        grand_total = total_coins * total_models * total_horizons

        selection_note = ""
        if self.top_n is not None:
            selection_note = f" (top-{self.top_n} selection)"

        self._print_flush(
            f"\n{'='*70}\n"
            f"  MASTER EXPERIMENT ORCHESTRATOR - {self.exp_id}{selection_note}\n"
            f"  {total_coins} coins x {total_models} models x {total_horizons} horizons = "
            f"{grand_total} pipeline runs\n"
            f"{'='*70}"
        )

        logger.info("=" * 70)
        logger.info("STARTING MASTER EXPERIMENT ORCHESTRATOR %s", self.exp_id)
        logger.info("Discovered %d cryptocurrencies, %d candidate models per coin", total_coins, total_models)
        logger.info("=" * 70)

        self.notifier.on_experiment_start(self.exp_id, total_coins, total_models)

        milestones = [
            (int(total_coins * 0.25), "25%"),
            (int(total_coins * 0.50), "50%"),
            (int(total_coins * 0.75), "75%"),
            (int(total_coins * 0.90), "90%"),
        ]

        completed_coins = 0
        experiment_start = time.time()

        with GracefulInterruptHandler(
            checkpoint_mgr=self.checkpoint_mgr,
            device_mgr=self.device_mgr,
        ) as interrupt_handler:

            # ── Main progress bar (coins level) ──
            pbar = tqdm(
                symbols,
                desc="Coins",
                unit="coin",
                ncols=100,
                file=sys.stdout,
                dynamic_ncols=True,
                bar_format=(
                    "{l_bar}{bar}| {n_fmt}/{total_fmt} coins "
                    "[{elapsed}<{remaining}, {rate_fmt}]"
                ),
            )

            for idx, symbol in enumerate(pbar, 1):
                if interrupt_handler.interrupted:
                    self._print_flush(
                        "\n[!] Execution loop breaking due to graceful interrupt signal."
                    )
                    logger.warning("Execution loop breaking due to graceful interrupt signal.")
                    break

                if self.checkpoint_mgr.is_coin_completed(symbol):
                    completed_coins += 1
                    pbar.set_postfix_str(f"{symbol} (skip)")
                    continue

                pbar.set_postfix_str(f"{symbol}")

                try:
                    coin_start = time.time()
                    res = self.run_coin(symbol, coin_idx=idx, total_coins=total_coins)
                    coin_elapsed = time.time() - coin_start
                    completed_coins += 1

                    total_elapsed = time.time() - experiment_start
                    avg_per_coin = total_elapsed / completed_coins if completed_coins > 0 else 0
                    remaining = (total_coins - completed_coins) * avg_per_coin
                    pct = completed_coins / total_coins * 100

                    self._print_flush(
                        f"\n  [TIME] Coin {idx}/{total_coins} ({symbol}) done in {_fmt_elapsed(coin_elapsed)} - "
                        f"Overall: {pct:.1f}% ({completed_coins}/{total_coins}) - "
                        f"ETA: {_fmt_elapsed(remaining)}\n"
                    )

                    for m_count, m_label in milestones:
                        if completed_coins == m_count:
                            self.notifier.on_milestone(
                                m_label,
                                f"Completed {m_count}/{total_coins} coins ({pct:.1f}%).",
                            )

                    if completed_coins % 10 == 0:
                        self.notifier.on_milestone(
                            f"{completed_coins} Coins",
                            f"Checkpoint reached: {completed_coins}/{total_coins} coins complete.",
                        )

                except Exception as exc:
                    logger.error("Unhandled error processing coin %s: %s", symbol, exc, exc_info=True)
                    self._print_flush(f"\n  [FAIL] FAILED on coin {symbol}: {exc}\n")
                    self._record_failure(symbol, "ALL", exc)

            pbar.close()

        total_elapsed = time.time() - experiment_start
        self._print_flush(
            f"\n{'='*70}\n"
            f"  EXPERIMENT COMPLETE - {completed_coins}/{total_coins} coins in {_fmt_elapsed(total_elapsed)}\n"
            f"{'='*70}"
        )

        self._print_flush("\nGenerating global reports and publication plots...")
        logger.info("Generating global reports and publication plots...")
        try:
            self.report_generator.generate_all_reports()
            self._print_flush("  [OK] Global reports and publication plots generated.")
            logger.info("Global reports and publication plots successfully generated.")
        except Exception as err:
            logger.error("Failed to generate global reports: %s", err, exc_info=True)
            self._print_flush(f"  [FAIL] Failed to generate global reports: {err}")

        self._print_flush("Running automated verification audit...")
        logger.info("Running automated verification audit on production artifacts...")
        try:
            audit_report_path = self.exp_dir / "verification_report.txt"
            ver_res = self.verifier.verify(output_report_path=audit_report_path)
            self._print_flush(
                f"  [OK] Verification: {ver_res.verified_coins}/{ver_res.total_coins} coins verified."
            )
            logger.info(
                "Verification complete (%d/%d verified). Report: %s",
                ver_res.verified_coins,
                ver_res.total_coins,
                audit_report_path,
            )
        except Exception as err:
            logger.error("Failed to execute verification audit: %s", err, exc_info=True)
            self._print_flush(f"  [FAIL] Verification audit failed: {err}")

        self.notifier.on_experiment_complete(
            exp_id=self.exp_id,
            total_coins=completed_coins,
            total_models=completed_coins * total_models,
            summary_path=Path(self.config.paths.evaluation_dir) / "all_model_results.csv",
        )
