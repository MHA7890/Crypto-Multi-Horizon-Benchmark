"""
forecasting — Cryptocurrency forecasting system.

Production-quality ML framework for training, evaluating, and selecting
forecasting models across 136 cryptocurrencies with probabilistic
prediction intervals.

Usage:
    python -m forecasting train
    python -m forecasting train --coin BTC --model XGBoost
    python -m forecasting select
    python -m forecasting predict --coin BTC
    python -m forecasting explain --coin BTC
"""

__version__ = "0.1.0"
