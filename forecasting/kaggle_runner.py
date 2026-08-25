"""
forecasting.kaggle_runner -- Kaggle GPU Master Execution Orchestrator.

Orchestrates CUDA-capable cryptocurrency forecasting benchmark on Kaggle:
- Strict CUDA hardware verification and VRAM monitoring
- Memory cleanup & CUDA Out-Of-Memory (OOM) handling with auto-retry
- Candidate models: XGBoost, LightGBM, TFT, PatchTST (ARIMA & RF excluded)
- Top-N coin selection using CoinSelector and ranking source
- Multi-horizon evaluation (1d, 7d, 14d, 30d, 90d)
- Checkpoint/resume system (/kaggle/working/crypto_forecasting/checkpoint.json)
- Horizon-level winner selection and archiving (winners to models/, losers to archive/)
- CSV reports & ZIP export (crypto_forecasting_results.zip)
"""

from __future__ import annotations

from datetime import datetime, timezone
import gc
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import Any
import zipfile

from forecasting.config.loader import load_config
from forecasting.config.model_registry import get_model_class
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

logger = logging.getLogger(__name__)

# Standard CUDA-capable models for Kaggle GPU execution
CUDA_MODELS = ["XGBoost", "LightGBM", "TFT", "PatchTST"]


# ────────────────────────────────────────────────────────────
# 1. CUDA HARDWARE VERIFICATION & MEMORY UTILITIES
# ────────────────────────────────────────────────────────────

def verify_cuda_environment(allow_cpu: bool = False) -> dict[str, Any]:
    """
    Verify CUDA availability and retrieve GPU diagnostics.

    Parameters
    ----------
    allow_cpu : bool, default=False
        If False, raises RuntimeError if CUDA is unavailable.

    Returns
    -------
    dict[str, Any]
        GPU diagnostic information.
    """
    import torch

    cuda_available = torch.cuda.is_available()

    info: dict[str, Any] = {
        "cuda_available": cuda_available,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if cuda_available else None,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "gpu_name": None,
        "total_vram_gb": 0.0,
    }

    if cuda_available:
        info["gpu_name"] = torch.cuda.get_device_name(0)
        total_vram_bytes = torch.cuda.get_device_properties(0).total_memory
        info["total_vram_gb"] = round(total_vram_bytes / (1024**3), 2)
        logger.info(
            "CUDA Hardware Verified: %s | VRAM: %.2f GB | CUDA Version: %s | PyTorch: %s",
            info["gpu_name"],
            info["total_vram_gb"],
            info["cuda_version"],
            info["torch_version"],
        )
    else:
        msg = (
            "CUDA is unavailable! This notebook is designed for Kaggle GPU execution. "
            "Please turn on GPU accelerator in Kaggle Settings (P100 / T4 GPU)."
        )
        if not allow_cpu:
            logger.critical(msg)
            raise RuntimeError(msg)
        else:
            logger.warning("CUDA unavailable. Proceeding on CPU because allow_cpu=True.")

    return info


def clean_gpu_memory() -> None:
    """Trigger Python garbage collection and clear PyTorch CUDA cache."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as err:
        logger.debug("Failed to clear CUDA cache: %s", err)


def get_gpu_memory_stats() -> dict[str, float]:
    """Get current GPU memory allocation stats in MB using CUDA driver API."""
    stats = {"allocated_mb": 0.0, "reserved_mb": 0.0, "available_mb": 0.0}
    try:
        import torch
        if torch.cuda.is_available():
            if hasattr(torch.cuda, "mem_get_info"):
                free_bytes, total_bytes = torch.cuda.mem_get_info(0)
                stats["available_mb"] = round(free_bytes / (1024**2), 2)
                stats["allocated_mb"] = round((total_bytes - free_bytes) / (1024**2), 2)
                stats["reserved_mb"] = round(torch.cuda.memory_reserved() / (1024**2), 2)
            else:
                stats["allocated_mb"] = round(torch.cuda.memory_allocated() / (1024**2), 2)
                stats["reserved_mb"] = round(torch.cuda.memory_reserved() / (1024**2), 2)
                total_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2)
                stats["available_mb"] = round(total_mb - stats["reserved_mb"], 2)
    except Exception:
        pass
    return stats


def setup_python_path(search_root: str | Path = "/kaggle/input") -> Path:
    """
    Locate the directory containing the 'forecasting' package and prepend it to sys.path.
    Prioritizes /kaggle/working if present so uploaded working files take precedence over input datasets.
    """
    # 1. Check if forecasting package exists in /kaggle/working or current directory
    working_dir = Path("/kaggle/working").resolve()
    if (working_dir / "forecasting" / "__init__.py").exists():
        if str(working_dir) not in sys.path:
            sys.path.insert(0, str(working_dir))
        logger.info("Configured sys.path with working root: %s", working_dir)
        return working_dir

    # 2. Search input datasets
    base = Path(search_root)
    found_root: Path | None = None

    if base.exists():
        for p in base.rglob("forecasting"):
            if p.is_dir() and (p / "__init__.py").exists():
                found_root = p.parent
                break

    if found_root is None:
        found_root = Path(".").resolve()

    if str(found_root) not in sys.path:
        sys.path.insert(0, str(found_root))

    logger.info("Configured sys.path with project root: %s", found_root)
    return found_root


def find_kaggle_dataset_dir(search_root: str | Path = "/kaggle/input") -> tuple[Path, Path, Path | None]:
    """
    Auto-discover features directory, coin_mapping.csv, and experiment.yaml on Kaggle.
    Supports nested Kaggle dataset paths such as:
    /kaggle/input/datasets/mha7890/crypto-forecasting-dataset/crypto-forecasting-dataset/
    /kaggle/input/datasets/mha7890/crypto-forecasting-code/configs/configs/experiment.yaml
    """
    base = Path(search_root)

    features_dir: Path | None = None
    mapping_file: Path | None = None
    config_file: Path | None = None

    if base.exists():
        # Search for directory named 'features' containing *_features.csv
        for p in base.rglob("features"):
            if p.is_dir() and any(p.glob("*_features.csv")):
                features_dir = p
                break

        # Search for coin_mapping.csv or top_coins.csv
        for p in base.rglob("coin_mapping.csv"):
            if p.is_file():
                mapping_file = p
                break

        # Search for experiment.yaml
        for p in base.rglob("experiment.yaml"):
            if p.is_file():
                config_file = p
                break

    if features_dir is None:
        features_dir = Path("features")
    if mapping_file is None:
        mapping_file = Path("output") / "coin_mapping.csv"
    if config_file is None and Path("configs/experiment.yaml").exists():
        config_file = Path("configs/experiment.yaml")

    logger.info("Auto-discovered Kaggle paths: features='%s', ranking='%s', config='%s'", features_dir, mapping_file, config_file)
    return features_dir, mapping_file, config_file


# ────────────────────────────────────────────────────────────
# 2. KAGGLE CUDA EXPERIMENT ORCHESTRATOR
# ────────────────────────────────────────────────────────────

class KaggleCUDAExperimentRunner:
    """
    Master Experiment Runner designed for Kaggle GPU execution.
    """

    def __init__(
        self,
        data_dir: str | Path | None = None,
        output_dir: str | Path = "crypto_forecasting",
        ranking_file: str | Path | None = None,
        top_n: int = 50,
        config_path: str | Path | None = None,
        resume: bool = True,
        allow_cpu: bool = False,
    ):
        # Configure sys.path for code imports
        setup_python_path()

        # Auto-discover Kaggle input paths if not explicitly provided or if default path doesn't exist
        auto_features, auto_ranking, auto_config = find_kaggle_dataset_dir()

        if data_dir is None or not Path(data_dir).exists():
            self.data_dir = auto_features
        else:
            self.data_dir = Path(data_dir)

        if ranking_file is None or not Path(ranking_file).exists():
            self.ranking_file = auto_ranking
        else:
            self.ranking_file = Path(ranking_file)

        if config_path is None or not Path(config_path).exists():
            resolved_config_path = auto_config
        else:
            resolved_config_path = Path(config_path)

        self.output_dir = Path(output_dir)
        self.top_n = top_n
        self.resume = resume
        self.allow_cpu = allow_cpu

        # Verify CUDA environment
        self.gpu_info = verify_cuda_environment(allow_cpu=self.allow_cpu)

        # Output subdirectories
        self.models_dir = self.output_dir / "models"
        self.archive_dir = self.output_dir / "archive"
        self.evaluation_dir = self.output_dir / "evaluation"
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.logs_dir = self.output_dir / "logs"

        for d in [self.models_dir, self.archive_dir, self.evaluation_dir, self.checkpoints_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Load or build configuration
        self.config = load_config(resolved_config_path) if resolved_config_path else ExperimentConfig()
        self.config.paths.features_dir = self.data_dir
        self.config.paths.models_dir = self.models_dir
        self.config.paths.archive_dir = self.archive_dir
        self.config.paths.evaluation_dir = self.evaluation_dir

        self.device_mgr = DeviceManager(use_cuda=self.gpu_info["cuda_available"])
        self.pipeline = TrainingPipeline(self.config)
        self.loader = DataLoader(features_dir=self.data_dir)
        self.reporter = EvaluationReporter(evaluation_dir=self.evaluation_dir)
        self.scorer = CompositeScorer(self.config.scoring)
        self.selector = ModelSelector(
            models_dir=self.models_dir,
            archive_dir=self.archive_dir,
            scorer=self.scorer,
        )
        self.report_generator = GlobalReportGenerator(evaluation_dir=self.evaluation_dir)

        # Checkpoint manager
        checkpoint_file = self.checkpoints_dir / "checkpoint.json"
        self.checkpoint_mgr = CheckpointManager(checkpoint_path=checkpoint_file)
        self.failures_csv = self.evaluation_dir / "failures.csv"

    def _record_failure(self, symbol: str, model_name: str, horizon: int, exc: Exception) -> None:
        tb_str = traceback.format_exc()
        self.checkpoint_mgr.record_failure(symbol, model_name, str(exc), tb_str)

        file_exists = self.failures_csv.exists()
        with open(self.failures_csv, "a", encoding="utf-8") as f:
            if not file_exists:
                f.write("timestamp,symbol,model,horizon,error,traceback\n")
            ts = datetime.now(timezone.utc).isoformat()
            clean_exc = str(exc).replace(",", ";").replace("\n", " ")
            clean_tb = tb_str.replace(",", ";").replace("\n", " | ")
            f.write(f"{ts},{symbol},{model_name},{horizon},{clean_exc},{clean_tb}\n")

    @staticmethod
    def _print_flush(msg: str) -> None:
        try:
            print(msg, flush=True)
        except UnicodeEncodeError:
            print(msg.encode("ascii", errors="replace").decode("ascii"), flush=True)

    def select_coins(self) -> list[str]:
        """Load ranking and return selected trainable coin symbols."""
        available = self.loader.available_symbols()
        if not available:
            raise FileNotFoundError(f"No feature CSV datasets found in {self.data_dir}")

        ranking_df = load_ranking(self.ranking_file)
        result = select_top_n(
            top_n=self.top_n,
            ranking_df=ranking_df,
            available_symbols=available,
            ranking_source=str(self.ranking_file),
        )
        print_selection_audit(result)
        save_snapshots(self.output_dir, result)
        return result.trainable_symbols

    def run_benchmark(self) -> dict[str, Any]:
        """
        Execute full Kaggle GPU Cryptocurrency Benchmark across CUDA models.
        """
        symbols = self.select_coins()
        candidate_models = CUDA_MODELS
        horizons = self.config.target.horizons

        total_coins = len(symbols)
        total_models = len(candidate_models)
        total_horizons = len(horizons)
        grand_total = total_coins * total_models * total_horizons

        device_count = self.gpu_info.get("device_count", 1)
        available_gpus = [f"cuda:{i}" for i in range(device_count)] if (self.gpu_info.get("cuda_available") and device_count > 0) else ["cpu"]

        self._print_flush(
            f"\n{'='*75}\n"
            f"  KAGGLE GPU CRYPTOCURRENCY FORECASTING BENCHMARK\n"
            f"  CUDA Devices : {available_gpus} ({self.gpu_info['gpu_name']} x {device_count})\n"
            f"  Workload     : {total_coins} coins x {total_models} models x {total_horizons} horizons = {grand_total} runs\n"
            f"  Output Dir   : {self.output_dir}\n"
            f"{'='*75}"
        )

        completed_runs = 0
        failed_runs = 0
        skipped_runs = 0
        run_counter = 0
        start_time = time.time()

        for coin_idx, symbol in enumerate(symbols, 1):
            self._print_flush(
                f"\n[{coin_idx}/{total_coins}] COIN: {symbol} "
                f"({total_models} models x {total_horizons} horizons)"
            )

            coin_model_metrics: dict[str, list[Any]] = {}
            coin_metrics_records: list[dict[str, Any]] = []

            for m_idx, model_name in enumerate(candidate_models, 1):
                model_cls = get_model_class(model_name)
                self._print_flush(f"  >> [{m_idx}/{total_models}] {model_name}")

                for h_idx, horizon in enumerate(horizons, 1):
                    if self.checkpoint_mgr.is_model_completed(symbol, f"{model_name}:{horizon}d"):
                        self._print_flush(f"    +-- {horizon}d horizon: [SKIP] (checkpoint)")
                        skipped_runs += 1
                        completed_runs += 1
                        continue

                    # Select GPU device (cuda:0, cuda:1, ...) for multi-GPU distribution
                    current_device = available_gpus[run_counter % len(available_gpus)]
                    run_counter += 1

                    model_params = self.config.model_params.get(model_name, {}).copy()
                    if "device" in model_cls.__init__.__code__.co_varnames:
                        model_params["device"] = current_device
                    if "device_type" in model_cls.__init__.__code__.co_varnames:
                        model_params["device_type"] = "cuda" if current_device.startswith("cuda") else "cpu"

                    # Log VRAM before training
                    mem = get_gpu_memory_stats()
                    self._print_flush(
                        f"    +-- Horizon {h_idx}/{total_horizons} ({horizon}d) [{current_device}] "
                        f"[VRAM Free: {mem['available_mb']} MB] ..."
                    )

                    try:
                        # Instantiate fresh model for each horizon
                        model = model_cls(**model_params)
                        step_start = time.time()

                        pipeline_res = self.pipeline.run(
                            symbol=symbol,
                            model=model,
                            horizon=horizon,
                            output_dir=self.models_dir / symbol / model_name / f"{horizon}d",
                        )

                        elapsed = time.time() - step_start
                        m = pipeline_res.metrics

                        self._print_flush(
                            f"    |   [OK] done in {elapsed:.1f}s - RMSE={m.rmse:.4f} "
                            f"MAE={m.mae:.4f} DirAcc={m.direction_accuracy:.2%}"
                        )

                        record = {
                            "experiment": "kaggle_cuda_v1",
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

                        self.checkpoint_mgr.record_model_completion(symbol, f"{model_name}:{horizon}d")
                        completed_runs += 1

                    except RuntimeError as err:
                        if "out of memory" in str(err).lower():
                            logger.error("CUDA Out of Memory on %s %s %dd: %s", symbol, model_name, horizon, err)
                            clean_gpu_memory()

                            # OOM Retry logic: attempt once with reduced batch size if applicable
                            if hasattr(model, "batch_size") and model.batch_size > 16:
                                model.batch_size = max(16, model.batch_size // 2)
                                self._print_flush(f"    |   [OOM RETRY] Reduced batch_size to {model.batch_size}...")
                                try:
                                    step_start = time.time()
                                    pipeline_res = self.pipeline.run(
                                        symbol=symbol,
                                        model=model,
                                        horizon=horizon,
                                        output_dir=self.models_dir / symbol / model_name / f"{horizon}d",
                                    )
                                    elapsed = time.time() - step_start
                                    m = pipeline_res.metrics
                                    self._print_flush(
                                        f"    |   [OK RETRY SUCCESS] done in {elapsed:.1f}s - RMSE={m.rmse:.4f}"
                                    )
                                    completed_runs += 1
                                    self.checkpoint_mgr.record_model_completion(symbol, f"{model_name}:{horizon}d")
                                    clean_gpu_memory()
                                    continue
                                except Exception as retry_err:
                                    err = retry_err

                            self._print_flush(f"    |   [FAIL] OOM Error: {err}")
                            self._record_failure(symbol, model_name, horizon, err)
                            failed_runs += 1
                        else:
                            logger.error("Execution error on %s %s %dd: %s", symbol, model_name, horizon, err)
                            self._print_flush(f"    |   [FAIL] Error: {err}")
                            self._record_failure(symbol, model_name, horizon, err)
                            failed_runs += 1
                    except Exception as exc:
                        logger.error("Error executing %s %s %dd: %s", symbol, model_name, horizon, exc)
                        self._print_flush(f"    |   [FAIL] Error: {exc}")
                        self._record_failure(symbol, model_name, horizon, exc)
                        failed_runs += 1

                    clean_gpu_memory()

            # Save per-coin metrics & select winner
            if coin_metrics_records:
                self.reporter.save_coin_metrics(symbol, coin_metrics_records)

            if coin_model_metrics:
                winner = self.selector.select_and_archive(symbol, coin_model_metrics)
                self.checkpoint_mgr.record_coin_completion(symbol, winner)
                self._print_flush(f"  ** Coin Winner for {symbol}: {winner}")

        # Post-experiment report generation
        self._print_flush("\nGenerating global report tables and publication plots...")
        try:
            self.report_generator.generate_reports()
            self._print_flush("  [OK] Reports and publication charts generated.")
        except Exception as err:
            logger.error("Failed to generate global reports: %s", err, exc_info=True)

        total_elapsed = time.time() - start_time
        summary = {
            "total_runs": grand_total,
            "completed_runs": completed_runs,
            "failed_runs": failed_runs,
            "skipped_runs": skipped_runs,
            "total_elapsed_seconds": total_elapsed,
        }

        self._print_flush(
            f"\n{'='*75}\n"
            f"  BENCHMARK COMPLETE\n"
            f"  Total Runs    : {grand_total}\n"
            f"  Completed     : {completed_runs}\n"
            f"  Failed        : {failed_runs}\n"
            f"  Skipped       : {skipped_runs}\n"
            f"  Elapsed Time  : {total_elapsed / 60:.2f} minutes\n"
            f"{'='*75}\n"
        )

        return summary

    def export_results_zip(self, zip_filename: str = "crypto_forecasting_results.zip") -> Path:
        """
        Package all benchmark evaluation CSVs, model winners, archive, and logs into a ZIP.

        Parameters
        ----------
        zip_filename : str
            Name of the ZIP file to create in /kaggle/working/ (or output root).

        Returns
        -------
        Path
            Path to the generated ZIP archive.
        """
        zip_path = self.output_dir.parent / zip_filename if self.output_dir.name in str(self.output_dir) else Path(zip_filename)
        if not zip_path.is_absolute():
            zip_path = self.output_dir / zip_filename

        self._print_flush(f"Creating export ZIP archive: {zip_path} ...")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(self.output_dir):
                for f in files:
                    full_path = Path(root) / f
                    if full_path == zip_path:
                        continue
                    arcname = full_path.relative_to(self.output_dir)
                    zf.write(full_path, arcname)

        self._print_flush(f"  [OK] Export ZIP created at: {zip_path} ({zip_path.stat().st_size / (1024**2):.2f} MB)")
        return zip_path
