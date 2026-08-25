"""
history_downloader.py

Downloads complete hourly OHLCV history from Binance Spot for every
coin marked Status == READY in coin_mapping.csv, and keeps that
history resumable across interrupted runs.

Design
------
HistoryDownloader
    ↓ uses
BinanceClient          (unmodified, reused as-is)
    ↓ writes through
StreamingCSVWriter      (append-only, page-at-a-time)
    ↓ validated by
CandleValidator          (continuity / numeric sanity / OHLC consistency)
    ↓ records progress via
HistoryLogger           (logs/history.log)

Key behaviours
--------------
* Streaming writes: each page is written to disk immediately and then
  discarded from memory. No coin's full history is held in memory
  during a normal incremental download.
* End-of-history is decided by comparing the last candle's timestamp
  to the current time - NOT by checking whether Binance returned
  fewer than `limit` candles. A short page no longer silently ends
  the download early (this was the root cause of gaps scattered
  throughout previously-downloaded files).
* Every page is checked for internal timestamp continuity before
  being written. If a page itself contains a gap, the missing
  sub-range is fetched immediately (bounded startTime/endTime
  request) and spliced in before anything is written.
* Zero-activity candles (volume == quote_volume == trades == 0)
  whose OHLC duplicates the previous candle are re-verified against
  Binance's own single-candle data before being trusted. If the
  source disagrees, the stored value is corrected from that source
  (never fabricated - always re-fetched from Binance).
* Any candle that is still numerically invalid after that (NaN,
  infinite, non-positive price, inconsistent OHLC) is rejected and
  never written. This deliberately leaves a gap rather than writing
  bad data - the gap is visible to validator.py / a future repair
  run instead of being silently masked.
* Resume: the last timestamp already on disk is read directly from
  the CSV (tail read, not a full load). If that last line exists but
  is unreadable (e.g. the process was killed mid-write), the coin is
  refused rather than silently restarted from scratch - a full
  restart with an append-mode writer would double the entire file.
* --repair-existing rescans already-downloaded files for both kinds
  of problem above (internal gaps and suspicious zero-activity rows)
  and heals them in place, since fixing the forward download loop
  does not retroactively fix data already written by the old logic.

Run:
    python history_downloader.py
    python history_downloader.py --symbol BTC
    python history_downloader.py --limit 5
    python history_downloader.py --force
    python history_downloader.py --repair-existing
    python history_downloader.py --repair-existing --symbol BTC
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from binance_client import BinanceClient


# ============================================================
# CONFIG
# ============================================================

MAPPING_FILE = Path("output/coin_mapping.csv")

HISTORICAL_DIR = Path("historical")

LOG_DIR = Path("logs")

LOG_FILE = LOG_DIR / "history.log"

INTERVAL = "1h"

INTERVAL_MS = {
    "1h": 60 * 60 * 1000,
}[INTERVAL]

PAGE_LIMIT = 1000

# How many consecutive page failures we tolerate on a single coin
# before giving up on it for this run.
MAX_PAGE_RETRIES = 5

REQUIRED_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trades",
]


# ============================================================
# LOGGING
# ============================================================

def build_logger() -> logging.Logger:

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("history_downloader")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


# ============================================================
# CANDLE VALIDATION (raw Binance kline rows, pre-DataFrame)
# ============================================================

class CorruptedHistoryFile(Exception):
    """Raised when an existing CSV's last row can't be trusted and
    resuming would risk silently duplicating the whole file."""


class CandleValidator:
    """
    Operates on raw Binance kline rows:
    [openTime, open, high, low, close, volume, closeTime,
     quoteVolume, trades, takerBuyBase, takerBuyQuote, ignore]

    All checks work on the raw list-of-lists Binance returns, before
    klines_to_dataframe() is called - so bad candles never reach the
    DataFrame/CSV stage at all.
    """

    @staticmethod
    def parse(candle) -> Optional[dict]:

        try:
            return {
                "open_time": int(candle[0]),
                "open": float(candle[1]),
                "high": float(candle[2]),
                "low": float(candle[3]),
                "close": float(candle[4]),
                "volume": float(candle[5]),
                "quote_volume": float(candle[7]),
                "trades": int(candle[8]),
            }
        except (TypeError, ValueError, IndexError):
            return None

    @classmethod
    def is_numerically_valid(cls, candle) -> bool:

        f = cls.parse(candle)

        if f is None:
            return False

        values = [f["open"], f["high"], f["low"], f["close"],
                   f["volume"], f["quote_volume"]]

        if any(math.isnan(v) or math.isinf(v) for v in values):
            return False

        if f["open"] <= 0 or f["high"] <= 0 or f["low"] <= 0 or f["close"] <= 0:
            return False

        if f["volume"] < 0 or f["quote_volume"] < 0 or f["trades"] < 0:
            return False

        return True

    @classmethod
    def is_ohlc_consistent(cls, candle) -> bool:

        f = cls.parse(candle)

        if f is None:
            return False

        o, h, l, c = f["open"], f["high"], f["low"], f["close"]

        return h >= o and h >= c and h >= l and l <= o and l <= c

    @classmethod
    def is_zero_activity(cls, candle) -> bool:

        f = cls.parse(candle)

        if f is None:
            return False

        return f["volume"] == 0 and f["quote_volume"] == 0 and f["trades"] == 0

    @classmethod
    def duplicates_previous_ohlc(cls, candle, previous_candle) -> bool:

        f, p = cls.parse(candle), cls.parse(previous_candle)

        if f is None or p is None:
            return False

        return (
            f["open"] == f["high"] == f["low"] == f["close"] == p["close"]
        )

    @staticmethod
    def find_internal_gaps(candles, interval_ms: int) -> List[Tuple[int, int, int]]:
        """
        Returns a list of (insert_index, gap_start_ms, gap_end_ms) for
        every place where consecutive candles in this page are more
        than one interval apart. gap_start/gap_end are the timestamps
        of the first and last MISSING candle in that internal gap.
        """

        gaps = []

        for i in range(1, len(candles)):

            prev_open = int(candles[i - 1][0])
            cur_open = int(candles[i][0])

            expected = prev_open + interval_ms

            if cur_open > expected:

                gap_start = expected
                gap_end = cur_open - interval_ms

                gaps.append((i, gap_start, gap_end))

        return gaps


# ============================================================
# STREAMING CSV WRITER
# ============================================================

class StreamingCSVWriter:
    """
    Appends DataFrame pages to a CSV one page at a time.

    Unlike BinanceClient.append_csv(), this never reads the
    existing file back into memory. It is only safe to use when
    the caller guarantees the incoming rows are strictly newer
    than what's already on disk (which is true here, since the
    downloader always resumes from last_timestamp + 1 candle).
    """

    def __init__(self, filepath: Path):

        self.filepath = Path(filepath)

        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        self._header_written = (
            self.filepath.exists() and self.filepath.stat().st_size > 0
        )

    def write(self, df: pd.DataFrame) -> int:

        if df is None or df.empty:
            return 0

        df = df[REQUIRED_COLUMNS]

        mode = "a" if self._header_written else "w"

        df.to_csv(
            self.filepath,
            mode=mode,
            header=not self._header_written,
            index=False,
        )

        self._header_written = True

        return len(df)


# ============================================================
# RESUME HELPERS
# ============================================================

def read_last_row_raw(filepath: Path) -> Optional[str]:
    """Reads only the last non-empty line of the CSV (raw text)."""

    filepath = Path(filepath)

    if not filepath.exists() or filepath.stat().st_size == 0:
        return None

    with open(filepath, "rb") as f:

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

        return lines[-1].decode("utf8")


def read_last_timestamp_ms(filepath: Path) -> Optional[int]:
    """
    Returns the last saved candle's timestamp in ms, or None if the
    file has no data rows yet (header only / doesn't exist).

    Raises CorruptedHistoryFile if the file has content but that
    content can't be trusted (wrong column count or unparsable
    fields) - this must never be silently treated as "no history".
    """

    last_line = read_last_row_raw(filepath)

    if last_line is None:
        return None

    if last_line.startswith("timestamp"):
        return None  # header-only file, no data yet

    fields = last_line.split(",")

    if len(fields) != len(REQUIRED_COLUMNS):
        raise CorruptedHistoryFile(
            f"{filepath}: last row has {len(fields)} fields, "
            f"expected {len(REQUIRED_COLUMNS)}. File is likely "
            f"truncated (e.g. process killed mid-write)."
        )

    try:
        ts = pd.to_datetime(fields[0], utc=True)
        for value in fields[1:]:
            float(value)
    except (ValueError, TypeError) as exc:
        raise CorruptedHistoryFile(
            f"{filepath}: last row failed to parse ({exc}). "
            f"File is likely truncated or corrupted."
        )

    return int(ts.timestamp() * 1000)


def truncate_last_row(filepath: Path) -> None:
    """
    Removes the last data row from the CSV in place (header is kept).

    Used on resume: the last saved candle may have been the
    currently-forming hour (incomplete) at the time it was written,
    so it's dropped here and re-downloaded fresh.
    """

    filepath = Path(filepath)

    with open(filepath, "r+b") as f:

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
            f.truncate(0)
        else:
            f.truncate(newline_pos + 1)


# ============================================================
# STATS
# ============================================================

@dataclass
class CoinResult:
    symbol: str
    status: str  # "OK", "SKIPPED", "FAILED"
    candles_downloaded: int = 0
    candles_rejected: int = 0
    candles_corrected: int = 0
    gaps_backfilled: int = 0
    elapsed_seconds: float = 0.0
    error: str = ""


@dataclass
class RunSummary:
    results: list = field(default_factory=list)
    started_at: float = 0.0

    def add(self, result: CoinResult):
        self.results.append(result)

    @property
    def total_candles(self) -> int:
        return sum(r.candles_downloaded for r in self.results)

    @property
    def ok(self):
        return [r for r in self.results if r.status == "OK"]

    @property
    def skipped(self):
        return [r for r in self.results if r.status == "SKIPPED"]

    @property
    def failed(self):
        return [r for r in self.results if r.status == "FAILED"]


# ============================================================
# HISTORY DOWNLOADER
# ============================================================

class HistoryDownloader:

    def __init__(
        self,
        mapping_file: Path = MAPPING_FILE,
        historical_dir: Path = HISTORICAL_DIR,
        interval: str = INTERVAL,
        force: bool = False,
    ):

        self.mapping_file = Path(mapping_file)
        self.historical_dir = Path(historical_dir)
        self.interval = interval
        self.force = force

        self.historical_dir.mkdir(parents=True, exist_ok=True)

        self.logger = build_logger()

        self.client = BinanceClient()

        self.mapping_df: Optional[pd.DataFrame] = None

    # --------------------------------------------------------
    # MAPPING I/O
    # --------------------------------------------------------

    def load_mapping(self) -> pd.DataFrame:

        if not self.mapping_file.exists():
            raise FileNotFoundError(
                f"Missing {self.mapping_file}. Run mapping.py first."
            )

        df = pd.read_csv(self.mapping_file)

        if "FirstHourlyCandle" not in df.columns:
            df["FirstHourlyCandle"] = ""

        self.mapping_df = df

        return df

    def get_ready_coins(self) -> pd.DataFrame:

        df = self.mapping_df

        ready = df[df["Status"] == "READY"].copy()

        ready.sort_values("Rank", inplace=True)

        return ready

    def save_mapping(self):

        self.mapping_df.to_csv(self.mapping_file, index=False)

    def record_first_candle(self, binance_symbol: str, first_ts_ms: int):

        iso = pd.to_datetime(
            first_ts_ms, unit="ms", utc=True
        ).isoformat()

        mask = self.mapping_df["BinanceSymbol"] == binance_symbol

        self.mapping_df.loc[mask, "FirstHourlyCandle"] = iso

        self.save_mapping()

    # --------------------------------------------------------
    # BOUNDED RANGE FETCH (used for gap backfill + verification)
    # --------------------------------------------------------

    def _fetch_range(self, binance_symbol: str, start_ms: int, end_ms: int) -> list:
        """Fetches candles in [start_ms, end_ms] inclusive. Used both
        for splicing internal page gaps and re-verifying a single
        suspicious candle. Never advances any cursor - purely a
        read; failures here are caught and logged by the caller."""

        try:
            return self.client.get_klines(
                symbol=binance_symbol,
                interval=self.interval,
                start_time=start_ms,
                end_time=end_ms,
                limit=PAGE_LIMIT,
            )
        except Exception as exc:  # noqa: BLE001

            self.logger.warning(
                f"{binance_symbol}: range fetch {start_ms}-{end_ms} failed: {exc}"
            )
            return []

    # --------------------------------------------------------
    # PAGE PROCESSING: continuity backfill + suspicious-candle
    # re-verification + hard rejection of invalid candles
    # --------------------------------------------------------

    def _ensure_continuity(
        self, binance_symbol: str, candles: list
    ) -> Tuple[list, int]:
        """Splices in any internally-missing sub-range using a bounded
        request. Returns (possibly-extended candle list, gaps_filled)."""

        gaps_filled = 0

        i = 1

        while i < len(candles):

            prev_open = int(candles[i - 1][0])
            cur_open = int(candles[i][0])

            expected = prev_open + INTERVAL_MS

            if cur_open > expected:

                gap_start = expected
                gap_end = cur_open - INTERVAL_MS

                self.logger.warning(
                    f"{binance_symbol}: internal page gap detected "
                    f"{pd.to_datetime(gap_start, unit='ms', utc=True)} -> "
                    f"{pd.to_datetime(gap_end, unit='ms', utc=True)}, "
                    f"backfilling..."
                )

                missing = self._fetch_range(binance_symbol, gap_start, gap_end)

                if missing:
                    candles[i:i] = missing
                    gaps_filled += 1
                    continue  # re-check from the same index against the splice
                else:
                    self.logger.error(
                        f"{binance_symbol}: could not backfill internal gap "
                        f"{gap_start}-{gap_end}; it will remain missing."
                    )

            i += 1

        return candles, gaps_filled

    def _verify_suspicious_candles(
        self, binance_symbol: str, candles: list
    ) -> Tuple[list, int]:
        """
        Re-verifies zero-activity candles whose OHLC duplicates the
        previous candle against Binance's own single-candle data.
        Corrects in place if the source disagrees. Never invents
        values - always re-fetched from Binance.
        """

        corrected = 0

        for i in range(1, len(candles)):

            candle = candles[i]

            if not CandleValidator.is_zero_activity(candle):
                continue

            if not CandleValidator.duplicates_previous_ohlc(candle, candles[i - 1]):
                continue

            open_time = int(candle[0])

            reverified = self._fetch_range(
                binance_symbol, open_time, open_time + INTERVAL_MS - 1
            )

            if len(reverified) == 1 and reverified[0] != candle:

                self.logger.warning(
                    f"{binance_symbol}: corrected suspicious zero-activity "
                    f"candle at {pd.to_datetime(open_time, unit='ms', utc=True)} "
                    f"using re-verified source data."
                )

                candles[i] = reverified[0]
                corrected += 1

            # If re-verification still shows zero activity, it's a
            # genuine no-trade candle - keep it as-is.

        return candles, corrected

    def _filter_invalid(
        self, binance_symbol: str, candles: list
    ) -> Tuple[list, int]:
        """Drops any candle that's still numerically invalid after
        continuity backfill + suspicious-candle verification. These
        are never written - the hour is left as a gap rather than
        writing bad data."""

        clean = []
        rejected = 0

        for candle in candles:

            if not CandleValidator.is_numerically_valid(candle):
                self.logger.error(
                    f"{binance_symbol}: REJECTED candle at {candle[0]} "
                    f"(non-finite or non-positive price / negative volume). "
                    f"Not written."
                )
                rejected += 1
                continue

            if not CandleValidator.is_ohlc_consistent(candle):
                self.logger.error(
                    f"{binance_symbol}: REJECTED candle at {candle[0]} "
                    f"(OHLC inconsistent). Not written."
                )
                rejected += 1
                continue

            clean.append(candle)

        return clean, rejected

    # --------------------------------------------------------
    # PER-COIN DOWNLOAD (STREAMING, RESUMABLE)
    # --------------------------------------------------------

    def download_coin(self, row: pd.Series) -> CoinResult:

        binance_symbol = str(row["BinanceSymbol"])
        base_asset = str(row["BaseAsset"])

        output_file = self.historical_dir / f"{base_asset}.csv"

        started = time.time()

        if self.force and output_file.exists():
            output_file.unlink()

        # --------------------------------------------------
        # Determine start point (resume or fresh discovery)
        # --------------------------------------------------

        last_ts = read_last_timestamp_ms(output_file)
        # raises CorruptedHistoryFile if the file exists but its last
        # row can't be trusted - propagates up and fails this coin
        # loudly instead of silently restarting in append mode.

        if last_ts is not None:

            truncate_last_row(output_file)

            start_time = last_ts

            self.logger.info(
                f"{base_asset} ({binance_symbol}): resuming from "
                f"{pd.to_datetime(start_time, unit='ms', utc=True)} "
                f"(re-fetching last candle in case it was incomplete)"
            )

        else:

            first_candle = self.client.get_first_kline(
                binance_symbol, interval=self.interval
            )

            if first_candle is None:
                raise ValueError(
                    f"No historical data available for {binance_symbol}"
                )

            start_time = int(first_candle[0])

            self.record_first_candle(binance_symbol, start_time)

            self.logger.info(
                f"{base_asset} ({binance_symbol}): starting fresh from "
                f"{pd.to_datetime(start_time, unit='ms', utc=True)}"
            )

        # --------------------------------------------------
        # Paginated streaming download
        # --------------------------------------------------

        writer = StreamingCSVWriter(output_file)

        total_candles = 0
        total_rejected = 0
        total_corrected = 0
        total_gaps_filled = 0

        consecutive_failures = 0

        while True:

            try:

                candles = self.client.get_klines(
                    symbol=binance_symbol,
                    interval=self.interval,
                    start_time=start_time,
                    limit=PAGE_LIMIT,
                )

                consecutive_failures = 0

            except Exception as exc:  # noqa: BLE001

                consecutive_failures += 1

                self.logger.warning(
                    f"{base_asset}: page request failed "
                    f"(attempt {consecutive_failures}/{MAX_PAGE_RETRIES}): {exc}"
                )

                if consecutive_failures >= MAX_PAGE_RETRIES:

                    raise

                time.sleep(2 ** consecutive_failures)

                continue

            if not candles:
                break  # genuinely nothing returned at all

            # 1. Fix any gap *within* this page before trusting it.
            candles, gaps_filled = self._ensure_continuity(binance_symbol, candles)
            total_gaps_filled += gaps_filled

            # 2. Re-verify suspicious zero-activity/duplicate-OHLC
            #    candles against Binance's own single-candle data.
            candles, corrected = self._verify_suspicious_candles(
                binance_symbol, candles
            )
            total_corrected += corrected

            # 3. Hard-reject anything still invalid. Never written.
            clean_candles, rejected = self._filter_invalid(binance_symbol, candles)
            total_rejected += rejected

            if clean_candles:

                df = self.client.klines_to_dataframe(clean_candles)

                written = writer.write(df)

                total_candles += written

            # Cursor always advances past the full (pre-rejection)
            # page, since we've already resolved/attempted every
            # candle up to this point - a rejected candle leaves a
            # visible gap rather than being retried forever.
            last_open = int(candles[-1][0])

            now_ms = int(time.time() * 1000)

            # End-of-history is decided by timestamp, never by
            # "fewer than PAGE_LIMIT candles returned" - a short page
            # for any other reason no longer ends the download early.
            if (now_ms - last_open) < INTERVAL_MS:
                break

            start_time = last_open + INTERVAL_MS

        elapsed = time.time() - started

        self.logger.info(
            f"{base_asset} ({binance_symbol}): finished. "
            f"+{total_candles} candles in {elapsed:.1f}s "
            f"(gaps backfilled: {total_gaps_filled}, "
            f"corrected: {total_corrected}, rejected: {total_rejected})"
        )

        return CoinResult(
            symbol=base_asset,
            status="OK",
            candles_downloaded=total_candles,
            candles_rejected=total_rejected,
            candles_corrected=total_corrected,
            gaps_backfilled=total_gaps_filled,
            elapsed_seconds=elapsed,
        )

    # --------------------------------------------------------
    # REPAIR PASS FOR ALREADY-DOWNLOADED FILES
    # --------------------------------------------------------

    def repair_existing_file(self, row: pd.Series) -> CoinResult:
        """
        Rescans an existing historical/{BASE}.csv for internal gaps
        and suspicious zero-activity/duplicate-OHLC rows left behind
        by the old (buggy) download logic, and heals them in place.

        This reads the whole file once - unlike the streaming
        incremental path, a repair pass needs the full picture to
        find gaps, and is expected to be run occasionally/on-demand
        rather than every run.
        """

        binance_symbol = str(row["BinanceSymbol"])
        base_asset = str(row["BaseAsset"])

        output_file = self.historical_dir / f"{base_asset}.csv"

        started = time.time()

        if not output_file.exists() or output_file.stat().st_size == 0:
            return CoinResult(symbol=base_asset, status="SKIPPED")

        df = pd.read_csv(output_file)

        if df.empty:
            return CoinResult(symbol=base_asset, status="SKIPPED")

        df["_ts"] = pd.to_datetime(df["timestamp"], utc=True)
        df.sort_values("_ts", inplace=True)
        df.drop_duplicates(subset="_ts", keep="first", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # ---- 1. Find and backfill internal gaps ----

        diffs = df["_ts"].diff()

        gap_positions = diffs[diffs > pd.Timedelta(milliseconds=INTERVAL_MS)].index

        gaps_filled = 0
        recovered_rows = []

        for pos in gap_positions:

            gap_start = df.loc[pos - 1, "_ts"] + pd.Timedelta(milliseconds=INTERVAL_MS)
            gap_end = df.loc[pos, "_ts"] - pd.Timedelta(milliseconds=INTERVAL_MS)

            self.logger.info(
                f"{base_asset}: repairing gap {gap_start} -> {gap_end}"
            )

            missing = self._fetch_range(
                binance_symbol,
                int(gap_start.timestamp() * 1000),
                int(gap_end.timestamp() * 1000),
            )

            if not missing:
                self.logger.error(
                    f"{base_asset}: gap {gap_start} -> {gap_end} "
                    f"could not be recovered from Binance."
                )
                continue

            missing, _ = self._verify_suspicious_candles(binance_symbol, missing)
            clean, rejected = self._filter_invalid(binance_symbol, missing)

            if clean:
                recovered_df = self.client.klines_to_dataframe(clean)
                recovered_rows.append(recovered_df)
                gaps_filled += 1

        # ---- 2. Re-verify existing suspicious zero-activity rows ----

        corrected = 0

        raw_rows = df[REQUIRED_COLUMNS].values.tolist()

        for i in range(1, len(raw_rows)):

            row_vals = raw_rows[i]
            prev_vals = raw_rows[i - 1]

            volume, quote_volume, trades = row_vals[5], row_vals[6], row_vals[7]
            o, h, l, c = row_vals[1], row_vals[2], row_vals[3], row_vals[4]
            prev_c = prev_vals[4]

            if not (volume == 0 and quote_volume == 0 and trades == 0):
                continue

            if not (o == h == l == c == prev_c):
                continue

            open_ms = int(df.loc[i, "_ts"].timestamp() * 1000)

            reverified = self._fetch_range(
                binance_symbol, open_ms, open_ms + INTERVAL_MS - 1
            )

            if len(reverified) == 1:

                f = CandleValidator.parse(reverified[0])

                if f and (
                    f["volume"] != volume
                    or f["quote_volume"] != quote_volume
                    or f["trades"] != trades
                    or f["open"] != o or f["high"] != h
                    or f["low"] != l or f["close"] != c
                ):

                    df.loc[i, "open"] = f["open"]
                    df.loc[i, "high"] = f["high"]
                    df.loc[i, "low"] = f["low"]
                    df.loc[i, "close"] = f["close"]
                    df.loc[i, "volume"] = f["volume"]
                    df.loc[i, "quote_volume"] = f["quote_volume"]
                    df.loc[i, "trades"] = f["trades"]

                    corrected += 1

                    self.logger.warning(
                        f"{base_asset}: corrected suspicious row at "
                        f"{df.loc[i, '_ts']} using re-verified source data."
                    )

        # ---- 3. Merge, sort, dedupe, write back ----

        if recovered_rows:

            all_df = pd.concat([df[REQUIRED_COLUMNS]] + recovered_rows, ignore_index=True)

        else:

            all_df = df[REQUIRED_COLUMNS]

        all_df["timestamp"] = pd.to_datetime(all_df["timestamp"], utc=True)
        all_df.sort_values("timestamp", inplace=True)
        all_df.drop_duplicates(subset="timestamp", keep="first", inplace=True)
        all_df.reset_index(drop=True, inplace=True)

        all_df.to_csv(output_file, index=False)

        elapsed = time.time() - started

        self.logger.info(
            f"{base_asset}: repair complete. gaps filled: {gaps_filled}, "
            f"rows corrected: {corrected}, elapsed: {elapsed:.1f}s"
        )

        return CoinResult(
            symbol=base_asset,
            status="OK",
            gaps_backfilled=gaps_filled,
            candles_corrected=corrected,
            elapsed_seconds=elapsed,
        )

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    def run(
        self,
        symbol_filter: Optional[str] = None,
        limit: Optional[int] = None,
        repair_existing: bool = False,
    ) -> RunSummary:

        self.load_mapping()

        coins = self.get_ready_coins()

        if symbol_filter:
            coins = coins[
                coins["BaseAsset"].str.upper() == symbol_filter.upper()
            ]

        if limit:
            coins = coins.head(limit)

        summary = RunSummary(started_at=time.time())

        mode = "repair" if repair_existing else "download"

        self.logger.info(
            f"Starting history {mode} for {len(coins)} coin(s) "
            f"(interval={self.interval})"
        )

        for _, row in tqdm(
            coins.iterrows(), total=len(coins), desc=f"{mode.capitalize()}ing history"
        ):

            base_asset = str(row["BaseAsset"])

            try:

                if repair_existing:
                    result = self.repair_existing_file(row)
                else:
                    result = self.download_coin(row)

            except CorruptedHistoryFile as exc:

                self.logger.error(f"{base_asset}: FAILED - {exc}")

                result = CoinResult(
                    symbol=base_asset, status="FAILED", error=str(exc)
                )

            except Exception as exc:  # noqa: BLE001

                self.logger.error(f"{base_asset}: FAILED - {exc}")

                result = CoinResult(
                    symbol=base_asset, status="FAILED", error=str(exc)
                )

            summary.add(result)

        self._print_summary(summary)

        return summary

    # --------------------------------------------------------
    # SUMMARY REPORT
    # --------------------------------------------------------

    def _print_summary(self, summary: RunSummary):

        elapsed = time.time() - summary.started_at

        total_rejected = sum(r.candles_rejected for r in summary.results)
        total_corrected = sum(r.candles_corrected for r in summary.results)
        total_gaps_filled = sum(r.gaps_backfilled for r in summary.results)

        lines = [
            "=" * 60,
            "HISTORY DOWNLOAD COMPLETE",
            "=" * 60,
            f"Coins processed : {len(summary.results)}",
            f"Succeeded       : {len(summary.ok)}",
            f"Skipped (up to date) : {len(summary.skipped)}",
            f"Failed          : {len(summary.failed)}",
            f"Total candles   : {summary.total_candles:,}",
            f"Gaps backfilled : {total_gaps_filled}",
            f"Candles corrected (source-verified) : {total_corrected}",
            f"Candles rejected (never written)    : {total_rejected}",
            f"Elapsed         : {elapsed:.1f}s",
        ]

        if summary.failed:

            lines.append("")
            lines.append("Failed coins:")

            for r in summary.failed:
                lines.append(f"  {r.symbol}: {r.error}")

        report = "\n".join(lines)

        print()
        print(report)

        self.logger.info(report)

    def close(self):
        self.client.close()


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Download and resume hourly OHLCV history from Binance."
    )

    parser.add_argument("--mapping", default=str(MAPPING_FILE))
    parser.add_argument("--output-dir", default=str(HISTORICAL_DIR))
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument(
        "--force", action="store_true",
        help="Delete existing history and redownload from scratch",
    )

    parser.add_argument(
        "--repair-existing", action="store_true",
        help="Rescan already-downloaded files for internal gaps and "
             "suspicious zero-activity rows, and heal them in place. "
             "Does not perform an incremental update in the same run.",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    downloader = HistoryDownloader(
        mapping_file=Path(args.mapping),
        historical_dir=Path(args.output_dir),
        force=args.force,
    )

    try:

        downloader.run(
            symbol_filter=args.symbol,
            limit=args.limit,
            repair_existing=args.repair_existing,
        )

    finally:

        downloader.close()


if __name__ == "__main__":
    main()
