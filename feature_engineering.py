"""
feature_engineering.py

Transforms validated historical/{BASE}.csv OHLCV data into
machine-learning-ready feature datasets. Pure feature engineering -
no labels, no model training.

Pipeline position: validator.py -> feature_engineering.py (you are here)

Design
------
FeatureEngineer        (orchestrator: mapping, BTC-first ordering, I/O)
    ↓ uses
IndicatorCalculator      (returns, trend, momentum, volatility, volume,
                           candle, time, rolling stats, relative strength)
FibonacciCalculator       (rolling-window algorithmic Fibonacci levels)
MarketStructure            (causal swing points, HH/HL/LH/LL, BOS/CHoCH)
OrderBlockDetector          (deterministic bullish/bearish order blocks)
FVGDetector                  (3-candle fair value gaps)
    ↓ writes through
FeatureWriter                 (incremental append, feature_summary.csv,
                                feature_metadata.json)

No look-ahead
-------------
Every feature uses pandas rolling()/ewm()/shift() operations, which are
inherently backward-looking (a rolling window ending at row i only
touches rows <= i). Structural features (swings, order blocks, FVGs)
are computed by scanning forward through time and only ever looking
at bars up to and including the current bar - a "swing" is only
recognized once enough later bars exist to confirm it causally at
that later bar, never retroactively injected before it existed.

Incremental processing
-----------------------
If {BASE}_features.csv already exists: read its last timestamp, drop
that last row (it may have been written while incomplete, same
principle used in history_downloader.py/websocket.py), reload only
the last LOOKBACK_CANDLES rows of historical data plus any new rows,
recompute features over that window, and append only rows strictly
newer than the previous run's last (post-drop) timestamp. Structural
state (swings/order blocks/FVGs) naturally only considers the
lookback window - older structure is treated as expired, which is
the explicit trade-off the spec accepts in exchange for bounded,
fast incremental runs.

BTC market context
-------------------
BTC is always processed first. A small context frame (close, return,
RSI14, ATR14, Bollinger width, trend, volume change, volatility) is
kept in memory and merged onto every other coin by exact timestamp.

Run:
    python feature_engineering.py
    python feature_engineering.py --symbol ETH
    python feature_engineering.py --force-full
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

MAPPING_FILE = Path("output/coin_mapping.csv")
HISTORICAL_DIR = Path("historical")
FEATURES_DIR = Path("features")
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "feature_engineering.log"

FEATURE_VERSION = "1.0.0"

# Largest rolling lookback used by any feature (Fibonacci=100,
# 30-day high/low=720). 250 candles of buffer, per spec, is not
# enough to cover the 30-day (720h) window on its own - so the
# incremental reload window is max(spec default, largest true need).
SPEC_LOOKBACK_CANDLES = 250
THIRTY_DAY_HOURS = 30 * 24
LOOKBACK_CANDLES = max(SPEC_LOOKBACK_CANDLES, THIRTY_DAY_HOURS + 10)

SWING_LOOKBACK = 5          # bars each side for causal fractal swing detection
OB_ATR_MULTIPLE = 2.0       # impulsive move threshold: >= 2x ATR ...
OB_PCT_MOVE = 0.03          # ... or >= 3% move (whichever framing fits)
OB_LOOKBACK_BARS = 10       # how far back to search for the origin candle
ZONE_MAX_AGE_BARS = 500     # order blocks / FVGs older than this expire
                             # from active tracking (keeps detection O(n))

REQUIRED_HIST_COLUMNS = [
    "timestamp", "open", "high", "low", "close",
    "volume", "quote_volume", "trades",
]

BTC_CONTEXT_COLUMNS = [
    "BTC_Close", "BTC_Return_1h", "BTC_RSI_14", "BTC_ATR_14",
    "BTC_BB_Width", "BTC_Trend", "BTC_Volume_Change", "BTC_Volatility_24",
]


# ============================================================
# LOGGING
# ============================================================

def build_logger() -> logging.Logger:

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("feature_engineering")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(LOG_FILE, encoding="utf8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger


# ============================================================
# INDICATOR CALCULATOR (groups 1-9)
# ============================================================

class IndicatorCalculator:
    """Stateless: every method takes a DataFrame and returns new
    columns computed purely from backward-looking operations."""

    @staticmethod
    def returns(df: pd.DataFrame) -> pd.DataFrame:

        c = df["close"]

        for n in (1, 3, 6, 12, 24):
            df[f"Return_{n}h"] = c.pct_change(n)

        df["Log_Return"] = np.log(c / c.shift(1))

        return df

    @staticmethod
    def trend(df: pd.DataFrame) -> pd.DataFrame:

        c = df["close"]

        df["SMA_10"] = c.rolling(10).mean()
        df["SMA_20"] = c.rolling(20).mean()
        df["SMA_50"] = c.rolling(50).mean()

        df["EMA_20"] = c.ewm(span=20, adjust=False).mean()
        df["EMA_50"] = c.ewm(span=50, adjust=False).mean()

        df["Close_SMA10_Ratio"] = c / df["SMA_10"]
        df["Close_SMA20_Ratio"] = c / df["SMA_20"]
        df["Close_EMA20_Ratio"] = c / df["EMA_20"]

        return df

    @staticmethod
    def momentum(df: pd.DataFrame) -> pd.DataFrame:

        c = df["close"]

        for period, col in ((7, "RSI_7"), (14, "RSI_14")):
            df[col] = IndicatorCalculator._rsi(c, period)

        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()

        df["MACD"] = macd
        df["MACD_Signal"] = signal
        df["MACD_Histogram"] = macd - signal

        low14 = df["low"].rolling(14).min()
        high14 = df["high"].rolling(14).max()
        percent_k = 100 * (c - low14) / (high14 - low14)

        df["Stoch_%K"] = percent_k
        df["Stoch_%D"] = percent_k.rolling(3).mean()

        df["ROC_12"] = c.pct_change(12) * 100

        return df

    @staticmethod
    def _rsi(close: pd.Series, period: int) -> pd.Series:

        delta = close.diff()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)

        rsi = 100 - (100 / (1 + rs))

        return rsi.where(avg_loss != 0, 100.0)

    @staticmethod
    def volatility(df: pd.DataFrame) -> pd.DataFrame:

        c, h, l = df["close"], df["high"], df["low"]
        prev_close = c.shift(1)

        tr = pd.concat([
            h - l,
            (h - prev_close).abs(),
            (l - prev_close).abs(),
        ], axis=1).max(axis=1)

        df["ATR_14"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()

        sma20 = c.rolling(20).mean()
        std20 = c.rolling(20).std()

        df["BB_Upper"] = sma20 + 2 * std20
        df["BB_Lower"] = sma20 - 2 * std20
        df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / sma20

        df["Rolling_Std_24"] = c.rolling(24).std()

        log_ret = np.log(c / c.shift(1))
        df["Historical_Volatility"] = log_ret.rolling(24).std() * np.sqrt(24)

        return df

    @staticmethod
    def volume(df: pd.DataFrame) -> pd.DataFrame:

        c, v = df["close"], df["volume"]

        direction = np.sign(c.diff()).fillna(0)
        df["OBV"] = (direction * v).cumsum()

        df["Volume_SMA20"] = v.rolling(20).mean()
        df["Volume_Ratio"] = v / df["Volume_SMA20"]
        df["Volume_Change_Pct"] = v.pct_change() * 100

        return df

    @staticmethod
    def candle(df: pd.DataFrame) -> pd.DataFrame:

        o, h, l, c = df["open"], df["high"], df["low"], df["close"]
        rng = (h - l).replace(0, np.nan)

        df["Body_Size"] = (c - o).abs()
        df["Upper_Wick"] = h - pd.concat([o, c], axis=1).max(axis=1)
        df["Lower_Wick"] = pd.concat([o, c], axis=1).min(axis=1) - l
        df["Body_Pct"] = df["Body_Size"] / rng
        df["Range_Pct"] = (h - l) / o
        df["High_Low_Pct"] = (h - l) / l * 100
        df["Bullish_Candle"] = (c > o).astype(int)
        df["Bearish_Candle"] = (c < o).astype(int)

        return df

    @staticmethod
    def time_features(df: pd.DataFrame) -> pd.DataFrame:

        ts = df["timestamp"]

        df["Hour"] = ts.dt.hour
        df["Day_Of_Week"] = ts.dt.dayofweek
        df["Month"] = ts.dt.month
        df["Weekend_Flag"] = (ts.dt.dayofweek >= 5).astype(int)

        return df

    @staticmethod
    def rolling_stats(df: pd.DataFrame) -> pd.DataFrame:

        c = df["close"]

        df["Rolling_Mean_24"] = c.rolling(24).mean()
        df["Rolling_Std_24_Close"] = c.rolling(24).std()
        df["Rolling_Max_24"] = c.rolling(24).max()
        df["Rolling_Min_24"] = c.rolling(24).min()

        return df

    @staticmethod
    def relative_strength(df: pd.DataFrame) -> pd.DataFrame:

        c, h, l = df["close"], df["high"], df["low"]

        ath = h.cummax()
        df["Distance_From_ATH"] = (c - ath) / ath

        window_30d = THIRTY_DAY_HOURS

        high_30d = h.rolling(window_30d, min_periods=1).max()
        low_30d = l.rolling(window_30d, min_periods=1).min()

        df["Distance_From_30D_High"] = (c - high_30d) / high_30d
        df["Distance_From_30D_Low"] = (c - low_30d) / low_30d

        return df


# ============================================================
# FIBONACCI (group 10)
# ============================================================

class FibonacciCalculator:

    RATIOS = [0.236, 0.382, 0.5, 0.618, 0.786]
    WINDOW = 100

    @classmethod
    def compute(cls, df: pd.DataFrame) -> pd.DataFrame:

        swing_high = df["high"].rolling(cls.WINDOW, min_periods=1).max()
        swing_low = df["low"].rolling(cls.WINDOW, min_periods=1).min()

        df["Fib_Swing_High"] = swing_high
        df["Fib_Swing_Low"] = swing_low

        span = swing_high - swing_low
        c = df["close"]

        for ratio in cls.RATIOS:

            label = f"{int(ratio * 1000)}"[:3]  # 236, 382, 500, 618, 786

            level = swing_low + span * ratio

            df[f"Fib_Level_{label}"] = level
            df[f"Fib_Distance_{label}"] = (c - level) / c.replace(0, np.nan)

        return df


# ============================================================
# MARKET STRUCTURE (group 11) - causal swing/BOS/CHoCH detection
# ============================================================

class MarketStructure:
    """
    A bar at position i is a *confirmed* swing high once SWING_LOOKBACK
    bars exist on both sides showing it was the local max - evaluated
    only once the current scan position has passed that confirmation
    point, so nothing here ever looks into the future relative to the
    row being written.
    """

    def __init__(self, lookback: int = SWING_LOOKBACK):
        self.lookback = lookback

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:

        n = len(df)
        L = self.lookback

        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()

        is_swing_high = np.zeros(n, dtype=bool)
        is_swing_low = np.zeros(n, dtype=bool)

        # A candidate at index i is confirmed only once we reach index
        # i + L (all bars used are <= current scan position).
        for i in range(L, n - L):
            window = high[i - L:i + L + 1]
            if high[i] == window.max() and np.argmax(window) == L:
                is_swing_high[i] = True

            window_l = low[i - L:i + L + 1]
            if low[i] == window_l.min() and np.argmin(window_l) == L:
                is_swing_low[i] = True

        last_swing_high = np.full(n, np.nan)
        last_swing_low = np.full(n, np.nan)
        prev_swing_high = np.full(n, np.nan)
        prev_swing_low = np.full(n, np.nan)

        hh = np.zeros(n, dtype=int)
        hl = np.zeros(n, dtype=int)
        lh = np.zeros(n, dtype=int)
        ll = np.zeros(n, dtype=int)
        trend = np.zeros(n, dtype=int)
        bos = np.zeros(n, dtype=int)
        choch = np.zeros(n, dtype=int)

        cur_high = np.nan
        cur_low = np.nan
        prior_high = np.nan
        prior_low = np.nan
        cur_trend = 0

        confirm_idx_high = None
        confirm_idx_low = None

        for i in range(n):

            # A swing at position i-L becomes known/confirmed at
            # position i (since confirmation requires L future bars
            # relative to the swing itself, but never future bars
            # relative to the CURRENT row i).
            confirm_pos = i - L

            if 0 <= confirm_pos < n:

                if is_swing_high[confirm_pos]:
                    prior_high = cur_high
                    cur_high = high[confirm_pos]
                    confirm_idx_high = confirm_pos

                    if not np.isnan(prior_high):
                        if cur_high > prior_high:
                            hh[i] = 1
                        else:
                            lh[i] = 1

                if is_swing_low[confirm_pos]:
                    prior_low = cur_low
                    cur_low = low[confirm_pos]
                    confirm_idx_low = confirm_pos

                    if not np.isnan(prior_low):
                        if cur_low > prior_low:
                            hl[i] = 1
                        else:
                            ll[i] = 1

            last_swing_high[i] = cur_high
            last_swing_low[i] = cur_low
            prev_swing_high[i] = prior_high
            prev_swing_low[i] = prior_low

            # Trend: uptrend needs a recent HH+HL pattern, downtrend
            # needs LH+LL. BOS = close breaks the last swing in the
            # direction of the established trend. CHoCH = close breaks
            # structure against the established trend (reversal).
            if hh[i] or hl[i]:
                if cur_trend >= 0:
                    cur_trend = 1
                elif not np.isnan(cur_high) and close[i] > cur_high:
                    choch[i] = 1
                    cur_trend = 1
            if lh[i] or ll[i]:
                if cur_trend <= 0:
                    cur_trend = -1
                elif not np.isnan(cur_low) and close[i] < cur_low:
                    choch[i] = 1
                    cur_trend = -1

            if cur_trend == 1 and not np.isnan(cur_high) and close[i] > cur_high:
                bos[i] = 1
            elif cur_trend == -1 and not np.isnan(cur_low) and close[i] < cur_low:
                bos[i] = 1

            trend[i] = cur_trend

        df["Higher_High"] = hh
        df["Higher_Low"] = hl
        df["Lower_High"] = lh
        df["Lower_Low"] = ll
        df["Current_Trend"] = trend
        df["Last_Swing_High"] = last_swing_high
        df["Last_Swing_Low"] = last_swing_low
        df["Distance_To_Swing_High"] = (close - last_swing_high) / close
        df["Distance_To_Swing_Low"] = (close - last_swing_low) / close
        df["Break_Of_Structure"] = bos
        df["Change_Of_Character"] = choch

        return df


# ============================================================
# ORDER BLOCKS (group 12)
# ============================================================

class OrderBlockDetector:
    """
    Bullish OB: last bearish candle before a strong impulsive bullish
    move that breaks the prior swing high. Bearish OB: mirror image.
    "Strong" = move >= OB_ATR_MULTIPLE x ATR_14, OR >= OB_PCT_MOVE.
    Entirely causal: at bar i we only look backward (open/close of
    bars <= i) to decide whether an OB was just created.
    """

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:

        n = len(df)

        close = df["close"].to_numpy()
        open_ = df["open"].to_numpy()
        atr = df["ATR_14"].to_numpy()
        swing_high = df["Last_Swing_High"].to_numpy()
        swing_low = df["Last_Swing_Low"].to_numpy()

        bull_ob_exists = np.zeros(n, dtype=int)
        bear_ob_exists = np.zeros(n, dtype=int)
        dist_bull_ob = np.full(n, np.nan)
        dist_bear_ob = np.full(n, np.nan)
        ob_age_bull = np.full(n, np.nan)
        ob_age_bear = np.full(n, np.nan)
        retest_bull = np.zeros(n, dtype=int)
        retest_bear = np.zeros(n, dtype=int)

        active_bull_obs: List[dict] = []  # {low, high, created_i, retests}
        active_bear_obs: List[dict] = []

        for i in range(1, n):

            move_pct = (close[i] - close[max(0, i - 3)]) / close[max(0, i - 3)] \
                if close[max(0, i - 3)] else 0
            atr_ref = atr[i] if not np.isnan(atr[i]) else 0
            move_abs = abs(close[i] - close[max(0, i - 3)])

            strong_up = (
                move_abs >= OB_ATR_MULTIPLE * atr_ref or move_pct >= OB_PCT_MOVE
            ) and close[i] > close[max(0, i - 3)]

            strong_down = (
                move_abs >= OB_ATR_MULTIPLE * atr_ref or move_pct <= -OB_PCT_MOVE
            ) and close[i] < close[max(0, i - 3)]

            breaks_up = not np.isnan(swing_high[i]) and close[i] > swing_high[i]
            breaks_down = not np.isnan(swing_low[i]) and close[i] < swing_low[i]

            if strong_up and breaks_up:
                origin = None
                for j in range(i - 1, max(-1, i - 1 - OB_LOOKBACK_BARS), -1):
                    if close[j] < open_[j]:  # bearish candle
                        origin = j
                        break
                if origin is not None:
                    active_bull_obs.append({
                        "low": df["low"].iloc[origin],
                        "high": df["high"].iloc[origin],
                        "created_i": i,
                        "retests": 0,
                    })

            if strong_down and breaks_down:
                origin = None
                for j in range(i - 1, max(-1, i - 1 - OB_LOOKBACK_BARS), -1):
                    if close[j] > open_[j]:  # bullish candle
                        origin = j
                        break
                if origin is not None:
                    active_bear_obs.append({
                        "low": df["low"].iloc[origin],
                        "high": df["high"].iloc[origin],
                        "created_i": i,
                        "retests": 0,
                    })

            price = close[i]

            active_bull_obs = [
                ob for ob in active_bull_obs if i - ob["created_i"] <= ZONE_MAX_AGE_BARS
            ]
            active_bear_obs = [
                ob for ob in active_bear_obs if i - ob["created_i"] <= ZONE_MAX_AGE_BARS
            ]

            for ob in active_bull_obs:
                if ob["low"] <= price <= ob["high"]:
                    ob["retests"] += 1

            for ob in active_bear_obs:
                if ob["low"] <= price <= ob["high"]:
                    ob["retests"] += 1

            if active_bull_obs:
                nearest = min(active_bull_obs, key=lambda o: abs(price - (o["low"] + o["high"]) / 2))
                bull_ob_exists[i] = 1
                dist_bull_ob[i] = (price - (nearest["low"] + nearest["high"]) / 2) / price
                ob_age_bull[i] = i - nearest["created_i"]
                retest_bull[i] = nearest["retests"]

            if active_bear_obs:
                nearest = min(active_bear_obs, key=lambda o: abs(price - (o["low"] + o["high"]) / 2))
                bear_ob_exists[i] = 1
                dist_bear_ob[i] = (price - (nearest["low"] + nearest["high"]) / 2) / price
                ob_age_bear[i] = i - nearest["created_i"]
                retest_bear[i] = nearest["retests"]

        df["Bullish_OB_Exists"] = bull_ob_exists
        df["Bearish_OB_Exists"] = bear_ob_exists
        df["Distance_To_Bullish_OB"] = dist_bull_ob
        df["Distance_To_Bearish_OB"] = dist_bear_ob
        df["Bullish_OB_Age"] = ob_age_bull
        df["Bearish_OB_Age"] = ob_age_bear
        df["Bullish_OB_Retest_Count"] = retest_bull
        df["Bearish_OB_Retest_Count"] = retest_bear

        return df


# ============================================================
# FAIR VALUE GAPS (group 13)
# ============================================================

class FVGDetector:
    """Classic 3-candle FVG: bullish gap when low[i] > high[i-2];
    bearish gap when high[i] < low[i-2]. Entirely causal (uses i-2,
    i-1, i only)."""

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:

        n = len(df)

        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()

        bull_fvg = np.zeros(n, dtype=int)
        bear_fvg = np.zeros(n, dtype=int)
        dist_bull_fvg = np.full(n, np.nan)
        dist_bear_fvg = np.full(n, np.nan)
        gap_filled = np.zeros(n, dtype=int)
        gap_age = np.full(n, np.nan)

        active_bull: List[dict] = []
        active_bear: List[dict] = []

        for i in range(2, n):

            if low[i] > high[i - 2]:
                bull_fvg[i] = 1
                active_bull.append({
                    "low": high[i - 2], "high": low[i], "created_i": i, "filled": False,
                })

            if high[i] < low[i - 2]:
                bear_fvg[i] = 1
                active_bear.append({
                    "low": high[i], "high": low[i - 2], "created_i": i, "filled": False,
                })

            price = close[i]

            active_bull = [
                g for g in active_bull
                if not g["filled"] and i - g["created_i"] <= ZONE_MAX_AGE_BARS
            ]
            active_bear = [
                g for g in active_bear
                if not g["filled"] and i - g["created_i"] <= ZONE_MAX_AGE_BARS
            ]

            for gap in active_bull:
                if not gap["filled"] and gap["low"] <= price <= gap["high"]:
                    gap["filled"] = True

            for gap in active_bear:
                if not gap["filled"] and gap["low"] <= price <= gap["high"]:
                    gap["filled"] = True

            unfilled_bull = [g for g in active_bull if not g["filled"]]
            unfilled_bear = [g for g in active_bear if not g["filled"]]

            if unfilled_bull:
                nearest = min(unfilled_bull, key=lambda g: abs(price - (g["low"] + g["high"]) / 2))
                dist_bull_fvg[i] = (price - (nearest["low"] + nearest["high"]) / 2) / price
                gap_age[i] = i - nearest["created_i"]

            if unfilled_bear:
                nearest = min(unfilled_bear, key=lambda g: abs(price - (g["low"] + g["high"]) / 2))
                dist_bear_fvg[i] = (price - (nearest["low"] + nearest["high"]) / 2) / price

            all_gaps = active_bull + active_bear
            if all_gaps:
                most_recent = max(all_gaps, key=lambda g: g["created_i"])
                gap_filled[i] = int(most_recent["filled"])

        df["Bullish_FVG"] = bull_fvg
        df["Bearish_FVG"] = bear_fvg
        df["Distance_To_Bullish_FVG"] = dist_bull_fvg
        df["Distance_To_Bearish_FVG"] = dist_bear_fvg
        df["FVG_Filled"] = gap_filled
        df["FVG_Age"] = gap_age

        return df


# ============================================================
# LIQUIDITY (group 14)
# ============================================================

class LiquidityCalculator:

    @staticmethod
    def compute(df: pd.DataFrame) -> pd.DataFrame:

        d = df.set_index("timestamp")

        daily_high = d["high"].resample("1D").max()
        daily_low = d["low"].resample("1D").min()

        prev_day_high = daily_high.shift(1).reindex(d.index, method="ffill")
        prev_day_low = daily_low.shift(1).reindex(d.index, method="ffill")

        weekly_high = d["high"].resample("1W").max()
        weekly_low = d["low"].resample("1W").min()

        prev_week_high = weekly_high.shift(1).reindex(d.index, method="ffill")
        prev_week_low = weekly_low.shift(1).reindex(d.index, method="ffill")

        close = d["close"]

        df["Distance_To_Prev_Day_High"] = ((close - prev_day_high) / close).to_numpy()
        df["Distance_To_Prev_Day_Low"] = ((close - prev_day_low) / close).to_numpy()
        df["Distance_To_Prev_Week_High"] = ((close - prev_week_high) / close).to_numpy()
        df["Distance_To_Prev_Week_Low"] = ((close - prev_week_low) / close).to_numpy()

        return df


# ============================================================
# FEATURE PIPELINE (runs all groups in order)
# ============================================================

class FeaturePipeline:

    def __init__(self):
        self.indicators = IndicatorCalculator()
        self.fibonacci = FibonacciCalculator()
        self.structure = MarketStructure()
        self.order_blocks = OrderBlockDetector()
        self.fvg = FVGDetector()
        self.liquidity = LiquidityCalculator()

    def run(self, df: pd.DataFrame) -> pd.DataFrame:

        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

        df = self.indicators.returns(df)
        df = self.indicators.trend(df)
        df = self.indicators.momentum(df)
        df = self.indicators.volatility(df)   # ATR_14 needed by OB detector
        df = self.indicators.volume(df)
        df = self.indicators.candle(df)
        df = self.indicators.time_features(df)
        df = self.indicators.rolling_stats(df)
        df = self.indicators.relative_strength(df)

        df = self.fibonacci.compute(df)
        df = self.structure.compute(df)       # swings needed by OB detector
        df = self.order_blocks.compute(df)
        df = self.fvg.compute(df)
        df = self.liquidity.compute(df)

        return df


# ============================================================
# FEATURE WRITER (incremental I/O)
# ============================================================

class FeatureWriter:

    def __init__(self, output_dir: Path, logger: logging.Logger):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger

    def output_path(self, base_asset: str) -> Path:
        return self.output_dir / f"{base_asset}_features.csv"

    def last_timestamp(self, base_asset: str) -> Optional[pd.Timestamp]:

        path = self.output_path(base_asset)

        if not path.exists() or path.stat().st_size == 0:
            return None

        with open(path, "rb") as f:

            f.seek(0, 2)
            file_size = f.tell()

            if file_size == 0:
                return None

            chunk = 1024
            data = b""
            pos = file_size

            while pos > 0:
                step = min(chunk, pos)
                pos -= step
                f.seek(pos)
                data = f.read(step) + data
                if data.count(b"\n") >= 2 or pos == 0:
                    break

            lines = [line for line in data.splitlines() if line.strip()]

            if not lines:
                return None

            last_line = lines[-1].decode("utf8")

        if last_line.startswith("timestamp"):
            return None  # header-only file

        try:
            return pd.to_datetime(last_line.split(",")[0], utc=True)
        except (ValueError, TypeError):
            return None

    def drop_last_row(self, base_asset: str):
        """The last written row may have been computed before its
        underlying candle was fully settled - drop it so it gets
        recomputed this run, mirroring the same principle used
        elsewhere in the pipeline for the currently-forming candle.

        Uses a byte-level tail truncation (not a full read+rewrite)
        so this stays O(1) regardless of how large the feature file
        has grown - critical for a dataset meant to run for months
        or years."""

        path = self.output_path(base_asset)

        if not path.exists():
            return

        with open(path, "r+b") as f:

            f.seek(0, 2)
            size = f.tell()

            if size == 0:
                return

            pos = size
            f.seek(pos - 1)

            if f.read(1) == b"\n":
                pos -= 1

            newline_pos = None
            search_pos = pos
            buffer = b""
            chunk = 4096

            while search_pos > 0:

                read_size = min(chunk, search_pos)
                search_pos -= read_size
                f.seek(search_pos)
                buffer = f.read(read_size) + buffer

                idx = buffer.rfind(b"\n")
                if idx != -1:
                    newline_pos = search_pos + idx
                    break

            if newline_pos is None:
                f.truncate(0)  # only the header (or one row) existed
            else:
                f.truncate(newline_pos + 1)

    def append(self, base_asset: str, df_new: pd.DataFrame, overwrite: bool = False) -> int:
        """
        Writes df_new to the coin's feature file.

        overwrite=False (default): appends - used for genuine
        incremental runs, where df_new is only the new rows that
        come after the file's existing last timestamp.

        overwrite=True: truncates and replaces the file first. This
        MUST be used whenever df_new represents a full recompute of
        the entire history rather than just the new tail - otherwise
        a full recompute (from --force-full, or from last_timestamp()
        being unable to read a prior position on an existing file)
        gets appended on top of what's already there instead of
        replacing it, silently duplicating the whole file every time
        it happens.
        """

        if df_new.empty:
            return 0

        path = self.output_path(base_asset)

        mode = "w" if overwrite else "a"

        write_header = overwrite or not (path.exists() and path.stat().st_size > 0)

        df_new.to_csv(path, mode=mode, header=write_header, index=False)

        return len(df_new)


# ============================================================
# COIN RESULT / SUMMARY
# ============================================================

@dataclass
class CoinFeatureResult:
    symbol: str
    status: str
    input_rows: int = 0
    output_rows: int = 0
    features_generated: int = 0
    elapsed_seconds: float = 0.0
    error: str = ""


# ============================================================
# FEATURE ENGINEER (orchestrator)
# ============================================================

class FeatureEngineer:

    def __init__(
        self,
        mapping_file: Path = MAPPING_FILE,
        historical_dir: Path = HISTORICAL_DIR,
        output_dir: Path = FEATURES_DIR,
        force_full: bool = False,
    ):
        self.mapping_file = Path(mapping_file)
        self.historical_dir = Path(historical_dir)
        self.output_dir = Path(output_dir)
        self.force_full = force_full

        self.logger = build_logger()
        self.pipeline = FeaturePipeline()
        self.writer = FeatureWriter(self.output_dir, self.logger)

        self.btc_context: Optional[pd.DataFrame] = None

    # --------------------------------------------------------

    def load_ready_coins(self) -> pd.DataFrame:

        if not self.mapping_file.exists():
            raise FileNotFoundError(f"Missing {self.mapping_file}.")

        df = pd.read_csv(self.mapping_file)

        ready = df[df["Status"] == "READY"].copy()

        ready.sort_values("Rank", inplace=True)

        return ready

    def load_historical(self, base_asset: str) -> pd.DataFrame:

        path = self.historical_dir / f"{base_asset}.csv"

        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")

        df = pd.read_csv(path)

        missing = [c for c in REQUIRED_HIST_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"{base_asset}.csv missing columns: {missing}")

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        return df

    # --------------------------------------------------------

    def _build_btc_context(self, features_df: pd.DataFrame) -> pd.DataFrame:

        ctx = pd.DataFrame({
            "timestamp": features_df["timestamp"],
            "BTC_Close": features_df["close"],
            "BTC_Return_1h": features_df["Return_1h"],
            "BTC_RSI_14": features_df["RSI_14"],
            "BTC_ATR_14": features_df["ATR_14"],
            "BTC_BB_Width": features_df["BB_Width"],
            "BTC_Trend": features_df["Current_Trend"],
            "BTC_Volume_Change": features_df["Volume_Change_Pct"],
            "BTC_Volatility_24": features_df["Historical_Volatility"],
        })

        return ctx

    def _merge_btc_context(self, df: pd.DataFrame) -> pd.DataFrame:

        if self.btc_context is None:
            df = pd.concat(
                [df, pd.DataFrame(np.nan, index=df.index, columns=BTC_CONTEXT_COLUMNS)],
                axis=1,
            )
            return df

        return df.merge(self.btc_context, on="timestamp", how="left")

    # --------------------------------------------------------

    def process_coin(self, base_asset: str) -> CoinFeatureResult:

        started = time.time()

        hist_df = self.load_historical(base_asset)
        input_rows = len(hist_df)

        last_output_ts = None

        if not self.force_full:
            last_output_ts = self.writer.last_timestamp(base_asset)

        if last_output_ts is not None:

            self.writer.drop_last_row(base_asset)

            cutoff_idx = hist_df["timestamp"].searchsorted(
                last_output_ts - pd.Timedelta(hours=LOOKBACK_CANDLES)
            )
            window_df = hist_df.iloc[max(0, cutoff_idx):].reset_index(drop=True)

            resume_from_ts = last_output_ts  # recompute this candle onward

        else:

            window_df = hist_df
            resume_from_ts = None

        if window_df.empty:
            return CoinFeatureResult(
                symbol=base_asset, status="SKIPPED", input_rows=input_rows,
                elapsed_seconds=time.time() - started,
            )

        features_df = self.pipeline.run(window_df)

        # BTC context is built from the full computed window
        # (pre-trim), so it's always available for merging onto other
        # coins even on a run where BTC itself has zero new rows to
        # write.
        if base_asset == "BTC":
            self.btc_context = self._build_btc_context(features_df)
        else:
            features_df = self._merge_btc_context(features_df)

        # Drop warm-up rows (rolling indicators still NaN) - only
        # relevant on the very first (full-history) run per coin.
        indicator_cols = [
            c for c in features_df.columns if c not in REQUIRED_HIST_COLUMNS
        ]
        features_df = features_df.dropna(subset=["SMA_50", "ATR_14", "RSI_14"])

        if last_output_ts is not None:
            features_df = features_df[features_df["timestamp"] > last_output_ts]

        features_df = features_df.drop_duplicates(subset="timestamp", keep="last")

        if last_output_ts is None:

            path = self.writer.output_path(base_asset)

            if path.exists() and path.stat().st_size > 0:
                self.logger.warning(
                    f"{base_asset}: performing a FULL recompute over an "
                    f"existing non-empty output file (force_full={self.force_full}). "
                    f"The existing file will be REPLACED, not appended to."
                )

        written = self.writer.append(
            base_asset, features_df, overwrite=(last_output_ts is None)
        )

        elapsed = time.time() - started

        self.logger.info(
            f"{base_asset}: input={input_rows} rows, "
            f"output_appended={written} rows, "
            f"features={len(indicator_cols)}, elapsed={elapsed:.1f}s"
        )

        return CoinFeatureResult(
            symbol=base_asset,
            status="OK",
            input_rows=input_rows,
            output_rows=written,
            features_generated=len(indicator_cols),
            elapsed_seconds=elapsed,
        )

    # --------------------------------------------------------

    def run(self, symbol_filter: Optional[str] = None) -> List[CoinFeatureResult]:

        coins = self.load_ready_coins()

        if symbol_filter:
            coins = coins[coins["BaseAsset"].str.upper() == symbol_filter.upper()]

        base_assets = coins["BaseAsset"].astype(str).tolist()

        # BTC must be processed first so its context is available for
        # every other coin, regardless of its rank position.
        if "BTC" in base_assets:
            base_assets.remove("BTC")
            base_assets.insert(0, "BTC")
        elif symbol_filter is None:
            self.logger.warning(
                "BTC not found among READY coins - BTC context columns "
                "will be NaN for all coins this run."
            )

        results = []

        started = time.time()

        for base_asset in tqdm(base_assets, desc="Generating features"):

            try:
                result = self.process_coin(base_asset)
            except Exception as exc:  # noqa: BLE001
                self.logger.error(f"{base_asset}: FAILED - {exc}")
                result = CoinFeatureResult(
                    symbol=base_asset, status="FAILED", error=str(exc)
                )

            results.append(result)

        self._write_summary(results)
        self._write_metadata(results, time.time() - started)
        self._print_summary(results, time.time() - started)

        return results

    # --------------------------------------------------------

    def _write_summary(self, results: List[CoinFeatureResult]):

        rows = [{
            "Coin": r.symbol,
            "Input Rows": r.input_rows,
            "Output Rows": r.output_rows,
            "Features Generated": r.features_generated,
            "Processing Time": round(r.elapsed_seconds, 2),
            "Status": r.status,
        } for r in results]

        pd.DataFrame(rows).to_csv(self.output_dir / "feature_summary.csv", index=False)

    def _write_metadata(self, results: List[CoinFeatureResult], elapsed: float):

        ok = [r for r in results if r.status == "OK"]

        metadata = {
            "feature_version": FEATURE_VERSION,
            "feature_count": max((r.features_generated for r in ok), default=0),
            "coins_processed": len(ok),
            "rows_written_this_run": sum(r.output_rows for r in ok),
            "lookback_candles_used": LOOKBACK_CANDLES,
            "generation_time_seconds": round(elapsed, 2),
            "generated_at": pd.Timestamp.now("UTC").isoformat(),
        }

        with open(self.output_dir / "feature_metadata.json", "w", encoding="utf8") as f:
            json.dump(metadata, f, indent=2)

    def _print_summary(self, results: List[CoinFeatureResult], elapsed: float):

        ok = [r for r in results if r.status == "OK"]
        skipped = [r for r in results if r.status == "SKIPPED"]
        failed = [r for r in results if r.status == "FAILED"]

        lines = [
            "", "=" * 60, "FEATURE ENGINEERING COMPLETE", "=" * 60,
            f"Coins processed : {len(results)}",
            f"Succeeded       : {len(ok)}",
            f"Skipped (up to date) : {len(skipped)}",
            f"Failed          : {len(failed)}",
            f"Rows written    : {sum(r.output_rows for r in ok)}",
            f"Elapsed         : {elapsed:.1f}s",
        ]

        if failed:
            lines.append("")
            lines.append("Failed coins:")
            for r in failed:
                lines.append(f"  {r.symbol}: {r.error}")

        report = "\n".join(lines)
        print(report)
        self.logger.info(report)


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Generate ML-ready feature datasets from validated historical OHLCV."
    )

    parser.add_argument("--mapping", default=str(MAPPING_FILE))
    parser.add_argument("--historical-dir", default=str(HISTORICAL_DIR))
    parser.add_argument("--output-dir", default=str(FEATURES_DIR))
    parser.add_argument("--symbol", default=None)

    parser.add_argument(
        "--force-full", action="store_true",
        help="Ignore existing feature files and regenerate from scratch.",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    engineer = FeatureEngineer(
        mapping_file=Path(args.mapping),
        historical_dir=Path(args.historical_dir),
        output_dir=Path(args.output_dir),
        force_full=args.force_full,
    )

    engineer.run(symbol_filter=args.symbol)


if __name__ == "__main__":
    main()
