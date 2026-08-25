"""
forecasting.utils.device — Automatic Hardware Acceleration & DeviceManager.

Detects hardware (CPU, RAM, CUDA GPU, VRAM) and routes each model family to the
optimal execution device with automatic memory management.
"""

from __future__ import annotations

import gc
import logging
import os
import platform
from typing import Any

logger = logging.getLogger(__name__)

# Check PyTorch availability
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore

# Check psutil availability
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None  # type: ignore


class DeviceManager:
    """
    Hardware discovery and device routing manager.
    """

    def __init__(self, use_cuda: bool = True):
        self.use_cuda = use_cuda
        self.hardware_info = self._detect_hardware()

    def _detect_hardware(self) -> dict[str, Any]:
        """Detect system CPU, RAM, CUDA GPU, VRAM, and OS info."""
        info: dict[str, Any] = {
            "os": platform.system(),
            "os_release": platform.release(),
            "python_version": platform.python_version(),
            "cpu_model": platform.processor() or "Unknown CPU",
            "cpu_cores_physical": os.cpu_count() or 1,
            "cpu_cores_logical": os.cpu_count() or 1,
            "ram_total_gb": 0.0,
            "ram_used_gb": 0.0,
            "ram_used_pct": 0.0,
            "cuda_available": False,
            "gpu_count": 0,
            "gpu_name": "N/A",
            "cuda_version": "N/A",
            "vram_total_gb": 0.0,
            "vram_used_gb": 0.0,
            "vram_used_pct": 0.0,
        }

        if PSUTIL_AVAILABLE:
            try:
                mem = psutil.virtual_memory()
                info["ram_total_gb"] = round(mem.total / (1024**3), 2)
                info["ram_used_gb"] = round(mem.used / (1024**3), 2)
                info["ram_used_pct"] = round(mem.percent, 1)
                info["cpu_cores_physical"] = psutil.cpu_count(logical=False) or info["cpu_cores_logical"]
                info["cpu_cores_logical"] = psutil.cpu_count(logical=True) or info["cpu_cores_logical"]
            except Exception as e:
                logger.debug("psutil info error: %s", e)

        if TORCH_AVAILABLE and self.use_cuda and torch.cuda.is_available():
            try:
                info["cuda_available"] = True
                info["gpu_count"] = torch.cuda.device_count()
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["cuda_version"] = torch.version.cuda or "Unknown"

                props = torch.cuda.get_device_properties(0)
                info["vram_total_gb"] = round(props.total_memory / (1024**3), 2)
                allocated = torch.cuda.memory_allocated(0)
                info["vram_used_gb"] = round(allocated / (1024**3), 2)
                if props.total_memory > 0:
                    info["vram_used_pct"] = round((allocated / props.total_memory) * 100, 1)
            except Exception as e:
                logger.debug("CUDA info error: %s", e)

        return info

    def print_startup_summary(self) -> None:
        """Log hardware startup configuration summary and model verification table."""
        h = self.hardware_info
        logger.info("=" * 75)
        logger.info("HARDWARE DISCOVERY SUMMARY")
        logger.info("=" * 75)
        logger.info("OS: %s %s (Python %s)", h["os"], h["os_release"], h["python_version"])
        logger.info("CPU: %s (%d cores)", h["cpu_model"], h["cpu_cores_logical"])
        logger.info("System RAM: %.2f GB (Used: %.1f%%)", h["ram_total_gb"], h["ram_used_pct"])

        if h["cuda_available"]:
            logger.info("GPU: %s (CUDA %s)", h["gpu_name"], h["cuda_version"])
            logger.info("VRAM: %.2f GB (Allocated: %.2f GB)", h["vram_total_gb"], h["vram_used_gb"])
        else:
            logger.info("GPU: None detected or CUDA disabled (Running on CPU)")

        logger.info("-" * 75)
        logger.info("MODEL EXECUTION & HARDWARE ROUTING VERIFICATION TABLE")
        logger.info("-" * 75)
        logger.info(f"{'Model':<15} | {'Implementation':<26} | {'Library':<15} | {'Device':<8}")
        logger.info("-" * 75)
        logger.info(f"{'ARIMA':<15} | {'ARIMA/SARIMAX':<26} | {'statsmodels':<15} | {'CPU':<8}")
        logger.info(f"{'Random Forest':<15} | {'RandomForestRegressor':<26} | {'scikit-learn':<15} | {'CPU':<8}")
        logger.info(f"{'XGBoost':<15} | {'XGBRegressor':<26} | {'xgboost':<15} | {('CUDA' if h['cuda_available'] else 'CPU'):<8}")
        logger.info(f"{'LightGBM':<15} | {'LGBMRegressor':<26} | {'lightgbm':<15} | {('CUDA' if h['cuda_available'] else 'CPU'):<8}")
        logger.info(f"{'TFT':<15} | {'PyTorchTFTNetwork':<26} | {'PyTorch':<15} | {('CUDA' if h['cuda_available'] else 'CPU'):<8}")
        logger.info(f"{'PatchTST':<15} | {'PyTorchPatchTSTNetwork':<26} | {'PyTorch':<15} | {('CUDA' if h['cuda_available'] else 'CPU'):<8}")
        logger.info("=" * 75)

    def get_device_for_model(self, model_name: str) -> str:
        model_upper = model_name.upper()
        cuda_active = self.hardware_info["cuda_available"]

        if "ARIMA" in model_upper or "RANDOMFOREST" in model_upper or "RF" in model_upper:
            return "cpu"
        elif "XGBOOST" in model_upper or "XGB" in model_upper:
            return "cuda" if cuda_active else "cpu"
        elif "LIGHTGBM" in model_upper or "LGBM" in model_upper:
            return "gpu" if cuda_active else "cpu"
        elif "TFT" in model_upper or "PATCHTST" in model_upper:
            return "cuda" if cuda_active else "cpu"

        return "cuda" if cuda_active else "cpu"

    def get_current_resource_usage(self) -> dict[str, float]:
        stats: dict[str, float] = {
            "cpu_pct": 0.0,
            "ram_pct": 0.0,
            "gpu_pct": 0.0,
            "vram_pct": 0.0,
        }
        if PSUTIL_AVAILABLE:
            try:
                stats["cpu_pct"] = psutil.cpu_percent(interval=None)
                stats["ram_pct"] = psutil.virtual_memory().percent
            except Exception:
                pass

        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                props = torch.cuda.get_device_properties(0)
                allocated = torch.cuda.memory_allocated(0)
                if props.total_memory > 0:
                    stats["vram_pct"] = round((allocated / props.total_memory) * 100, 1)
            except Exception:
                pass

        return stats


def get_device(use_cuda: bool = True) -> Any:
    mgr = DeviceManager(use_cuda=use_cuda)
    if mgr.hardware_info["cuda_available"] and TORCH_AVAILABLE:
        return torch.device("cuda")
    return "cpu"


def clear_gpu_memory() -> None:
    gc.collect()
    if TORCH_AVAILABLE and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
            torch.cuda.synchronize()
        except Exception as e:
            logger.debug("Error clearing GPU memory: %s", e)
