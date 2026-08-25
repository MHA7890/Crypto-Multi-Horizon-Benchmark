"""
forecasting.utils.serialization — Object serialization dispatching.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import joblib


def save_object(obj: Any, path: Path) -> None:
    """Serialize object to disk using joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_object(path: Path) -> Any:
    """Deserialize object from disk using joblib."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found for deserialization: {path}")
    return joblib.load(path)
