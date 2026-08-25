"""
validator.py

Validates every historical hourly OHLCV dataset belonging to a coin
marked Status == READY in coin_mapping.csv, before that data is
handed off to feature_engineering.py.

Scope
-----
This module validates ONLY historical/*.csv (the hourly candle
datasets produced by history_downloader.py). live/*.csv is a
different schema (websocket ticker snapshots, irregular timing,
rolling 24h stats) and is intentionally out of scope here.

Checks performed (per coin)
----------------------------
* File exists
* File not empty
* Required columns present
* Timestamp format parses cleanly
* Duplicate timestamps
* Out-of-order timestamps
* Missing hourly candles (gaps in the expected hourly grid)
* NaN values
* Infinite values
* Negative prices
* Negative volume
* Negative trades
* OHLC consistency (high/low bound violations)
* Duplicate rows
* Large abnormal single-candle price jumps
* Dataset summary (row count, date range, coverage %)

Severity model
--------------
CRITICAL issues fail the coin (Status = FAIL): missing/empty/corrupt
files, missing columns, bad timestamps, NaNs, infinities, negative
values, OHLC violations, duplicate rows/timestamps, out-of-order
rows.

WARNING issues do not fail the coin (Status = WARN if otherwise
clean): missing hourly candles, large price jumps. These are
expected to happen occasionally in real market data (exchange
downtime, delistings, flash moves) and are surfaced for awareness,
not treated as corruption.

--fix behaviour
---------------
When --fix is passed, three (and only three) automatic repairs may
be applied, in this order:

    1. Remove fully duplicated rows.
    2. Remove duplicate timestamps (keep the first occurrence).
    3. Sort rows into chronological order.

The validator NEVER invents missing candles and NEVER modifies any
price, volume, or trade value. If --fix changes a file, the repaired
CSV is written back to disk and the change is logged.

Outputs
-------
validation/validation_report.csv   One row per coin: summary + status,
                                    including gap-level rollups
                                    (MissingCandles, GapCount,
                                    LargestGap, LargestGapStart,
                                    LargestGapEnd).
validation/details/<SYMBOL>_issues.csv   Full issue list, per coin
                                          with 1+ issues.
validation/details/<SYMBOL>_report.txt   Human-readable missing-candle
                                          diagnostic report, per coin
                                          with 1+ gaps.
validation/gaps/<SYMBOL>_gaps.csv        Every individual gap, per
                                          coin with 1+ gaps.
logs/validator.log                 Full run log.

Missing-candle gap reporting
-----------------------------
This is a purely diagnostic reporting layer built on top of the
existing MISSING_HOURLY_CANDLES check - it does not change what
counts as a gap or whether a coin passes/fails. Every contiguous
run of missing hourly timestamps is grouped into one "gap" and
reported with its boundaries (the last real candle before the gap
and the first real candle after it), so it's possible to tell at a
glance whether a gap looks like real exchange downtime (e.g. Binance
maintenance) or something a bug in the downloader would produce.
No missing candle is ever filled, interpolated, or invented - this
only describes gaps that already exist in the data on disk.

Run:
    python validator.py
    python validator.py --symbol BTC
    python validator.py --fix
    python validator.py --limit 5
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

MAPPING_FILE = Path("output/coin_mapping.csv")

HISTORICAL_DIR = Path("historical")

VALIDATION_DIR = Path("validation")

DETAILS_DIR = VALIDATION_DIR / "details"

GAPS_DIR = VALIDATION_DIR / "gaps"

# Above this many gaps, the per-coin .txt report switches from
# listing every missing hour inside each gap to a compact one-line
# summary per gap. The CSV always contains full detail either way.
MAX_GAPS_FOR_VERBOSE_TXT = 500

LOG_DIR = Path("logs")

LOG_FILE = LOG_DIR / "validator.log"

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

PRICE_COLUMNS = ["open", "high", "low", "close"]

NUMERIC_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trades",
]

EXPECTED_FREQ = "1h"

# A single-candle close move larger than this fraction of the
# previous close is flagged for review. Informational only - crypto
# markets do occasionally move this fast, so this is a WARNING, not
# a failure.
PRICE_JUMP_THRESHOLD = 0.50  # 50%


# ============================================================
# LOGGING
# ============================================================

def build_logger() -> logging.Logger:

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("validator")
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
# RESULT TYPES
# ============================================================

@dataclass
class ValidationIssue:
    code: str
    severity: str  # "CRITICAL" or "WARNING"
    message: str
    count: int = 1


@dataclass
class CoinValidationResult:
    symbol: str
    status: str = "PASS"  # PASS, WARN, FAIL
    row_count: int = 0
    start: Optional[str] = None
    end: Optional[str] = None
    coverage_pct: Optional[float] = None
    issues: list = field(default_factory=list)
    fixes_applied: list = field(default_factory=list)
    # Reporting-only: contiguous runs of missing hourly timestamps,
    # one dict per gap: {start, end, missing_hours, prev, next}.
    # Populated alongside the existing MISSING_HOURLY_CANDLES check
    # without altering that check's detection logic.
    gaps: list = field(default_factory=list)

    def add_issue(self, code: str, severity: str, message: str, count: int = 1):

        self.issues.append(ValidationIssue(code, severity, message, count))

        if severity == "CRITICAL":
            self.status = "FAIL"
        elif severity == "WARNING" and self.status == "PASS":
            self.status = "WARN"

    @property
    def critical_issues(self):
        return [i for i in self.issues if i.severity == "CRITICAL"]

    @property
    def warning_issues(self):
        return [i for i in self.issues if i.severity == "WARNING"]


@dataclass
class RunSummary:
    results: list = field(default_factory=list)
    started_at: float = 0.0

    def add(self, result: CoinValidationResult):
        self.results.append(result)

    @property
    def passed(self):
        return [r for r in self.results if r.status == "PASS"]

    @property
    def warned(self):
        return [r for r in self.results if r.status == "WARN"]

    @property
    def failed(self):
        return [r for r in self.results if r.status == "FAIL"]


# ============================================================
# VALIDATOR
# ============================================================

class HistoricalValidator:

    def __init__(
        self,
        mapping_file: Path = MAPPING_FILE,
        historical_dir: Path = HISTORICAL_DIR,
        validation_dir: Path = VALIDATION_DIR,
        fix: bool = False,
    ):

        self.mapping_file = Path(mapping_file)
        self.historical_dir = Path(historical_dir)
        self.validation_dir = Path(validation_dir)
        self.details_dir = self.validation_dir / "details"
        self.gaps_dir = self.validation_dir / "gaps"
        self.fix = fix

        self.validation_dir.mkdir(parents=True, exist_ok=True)
        self.details_dir.mkdir(parents=True, exist_ok=True)
        self.gaps_dir.mkdir(parents=True, exist_ok=True)

        self.logger = build_logger()

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

        self.mapping_df = df

        return df

    def get_ready_coins(self) -> pd.DataFrame:

        df = self.mapping_df

        ready = df[df["Status"] == "READY"].copy()

        ready.sort_values("Rank", inplace=True)

        return ready

    # --------------------------------------------------------
    # PER-COIN VALIDATION
    # --------------------------------------------------------

    def validate_coin(self, row: pd.Series) -> CoinValidationResult:

        base_asset = str(row["BaseAsset"])

        result = CoinValidationResult(symbol=base_asset)

        filepath = self.historical_dir / f"{base_asset}.csv"

        # ---- File exists ----
        if not filepath.exists():
            result.add_issue(
                "FILE_MISSING", "CRITICAL", f"{filepath} does not exist"
            )
            return result

        # ---- Empty file ----
        if filepath.stat().st_size == 0:
            result.add_issue(
                "FILE_EMPTY", "CRITICAL", "File exists but is empty (0 bytes)"
            )
            return result

        try:
            df = pd.read_csv(filepath)
        except Exception as exc:  # noqa: BLE001
            result.add_issue(
                "FILE_UNREADABLE", "CRITICAL", f"Could not parse CSV: {exc}"
            )
            return result

        if df.empty:
            result.add_issue(
                "NO_DATA_ROWS", "CRITICAL", "File has a header but zero data rows"
            )
            return result

        # ---- Required columns ----
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]

        if missing_cols:
            result.add_issue(
                "MISSING_COLUMNS",
                "CRITICAL",
                f"Missing required column(s): {missing_cols}",
            )
            return result  # nothing further can be safely checked

        # ---- Timestamp format ----
        parsed = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

        bad_timestamps = int(parsed.isna().sum())

        if bad_timestamps:

            result.add_issue(
                "BAD_TIMESTAMP_FORMAT",
                "CRITICAL",
                f"{bad_timestamps} row(s) with unparseable timestamps",
                count=bad_timestamps,
            )

            keep = parsed.notna()
            df = df.loc[keep].copy()
            parsed = parsed.loc[keep]

        if df.empty:
            result.add_issue(
                "NO_DATA_ROWS", "CRITICAL", "No rows remain with a valid timestamp"
            )
            return result

        df = df.copy()
        df["_ts"] = parsed.values

        # ---- Duplicate rows (full-row duplicates) ----
        dup_rows_mask = df.drop(columns="_ts").duplicated()
        dup_rows = int(dup_rows_mask.sum())

        if dup_rows:

            result.add_issue(
                "DUPLICATE_ROWS",
                "CRITICAL",
                f"{dup_rows} fully duplicated row(s)",
                count=dup_rows,
            )

            if self.fix:
                df = df.loc[~dup_rows_mask].copy()
                result.fixes_applied.append(f"Removed {dup_rows} duplicate row(s)")

        # ---- Duplicate timestamps ----
        dup_ts_mask = df["_ts"].duplicated()
        dup_ts = int(dup_ts_mask.sum())

        if dup_ts:

            result.add_issue(
                "DUPLICATE_TIMESTAMPS",
                "CRITICAL",
                f"{dup_ts} duplicate timestamp(s)",
                count=dup_ts,
            )

            if self.fix:
                df = df.loc[~dup_ts_mask].copy()  # keep first occurrence
                result.fixes_applied.append(
                    f"Removed {dup_ts} duplicate-timestamp row(s) (kept first)"
                )

        # ---- Out-of-order timestamps ----
        is_sorted = df["_ts"].is_monotonic_increasing

        if not is_sorted:

            out_of_order = int((df["_ts"].diff().dt.total_seconds() < 0).sum())

            result.add_issue(
                "OUT_OF_ORDER_TIMESTAMPS",
                "CRITICAL",
                f"{out_of_order} row(s) out of chronological order",
                count=out_of_order,
            )

            if self.fix:
                df = df.sort_values("_ts").reset_index(drop=True)
                result.fixes_applied.append("Sorted rows into chronological order")

        df = df.reset_index(drop=True)

        # ---- Missing hourly candles ----
        start_ts = df["_ts"].min()
        end_ts = df["_ts"].max()

        expected = pd.date_range(start_ts, end_ts, freq=EXPECTED_FREQ)
        missing = expected.difference(df["_ts"])

        if len(missing):
            result.add_issue(
                "MISSING_HOURLY_CANDLES",
                "WARNING",
                f"{len(missing)} missing hourly candle(s) between "
                f"{start_ts} and {end_ts}",
                count=len(missing),
            )
            # Intentionally never filled in - per spec, gaps are never invented.

            # Reporting only: group the missing timestamps into
            # contiguous gaps so each gap's real boundaries (the last
            # candle before it and the first candle after it) can be
            # reported. This does not change detection - it's the
            # same `missing` set, just organized for readability.
            result.gaps = self._group_missing_into_gaps(missing)

        # ---- NaN values ----
        nan_counts = df[NUMERIC_COLUMNS].isna().sum()
        total_nan = int(nan_counts.sum())

        if total_nan:
            detail = ", ".join(
                f"{col}={int(n)}" for col, n in nan_counts.items() if n
            )
            result.add_issue(
                "NAN_VALUES", "CRITICAL", f"NaN values found: {detail}", count=total_nan
            )

        # ---- Infinite values ----
        numeric_df = df[NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")
        inf_mask = np.isinf(numeric_df.to_numpy(dtype="float64", na_value=0.0))
        total_inf = int(inf_mask.sum())

        if total_inf:
            result.add_issue(
                "INFINITE_VALUES",
                "CRITICAL",
                f"{total_inf} infinite value(s) found",
                count=total_inf,
            )

        # ---- Negative prices ----
        neg_price_mask = (numeric_df[PRICE_COLUMNS] < 0).any(axis=1)
        neg_prices = int(neg_price_mask.sum())

        if neg_prices:
            result.add_issue(
                "NEGATIVE_PRICES",
                "CRITICAL",
                f"{neg_prices} row(s) with a negative price value",
                count=neg_prices,
            )

        # ---- Negative volume ----
        neg_volume_mask = (numeric_df["volume"] < 0) | (numeric_df["quote_volume"] < 0)
        neg_volume = int(neg_volume_mask.sum())

        if neg_volume:
            result.add_issue(
                "NEGATIVE_VOLUME",
                "CRITICAL",
                f"{neg_volume} row(s) with negative volume",
                count=neg_volume,
            )

        # ---- Negative trades ----
        neg_trades_mask = numeric_df["trades"] < 0
        neg_trades = int(neg_trades_mask.sum())

        if neg_trades:
            result.add_issue(
                "NEGATIVE_TRADES",
                "CRITICAL",
                f"{neg_trades} row(s) with a negative trade count",
                count=neg_trades,
            )

        # ---- OHLC consistency ----
        ohlc_bad_mask = (
            (numeric_df["high"] < numeric_df["low"])
            | (numeric_df["high"] < numeric_df["open"])
            | (numeric_df["high"] < numeric_df["close"])
            | (numeric_df["low"] > numeric_df["open"])
            | (numeric_df["low"] > numeric_df["close"])
        )
        ohlc_bad = int(ohlc_bad_mask.sum())

        if ohlc_bad:
            result.add_issue(
                "OHLC_INCONSISTENT",
                "CRITICAL",
                f"{ohlc_bad} row(s) violate OHLC bounds "
                f"(high/low do not bound open/close)",
                count=ohlc_bad,
            )

        # ---- Large abnormal price jumps ----
        close_pct_change = numeric_df["close"].pct_change().abs()
        jump_mask = close_pct_change > PRICE_JUMP_THRESHOLD
        jumps = int(jump_mask.sum())

        if jumps:
            worst = close_pct_change.max()
            result.add_issue(
                "LARGE_PRICE_JUMPS",
                "WARNING",
                f"{jumps} candle(s) with a >{PRICE_JUMP_THRESHOLD:.0%} close move "
                f"(largest={worst:.1%})",
                count=jumps,
            )

        # ---- Dataset summary ----
        result.row_count = len(df)
        result.start = str(start_ts)
        result.end = str(end_ts)

        expected_count = len(expected)
        result.coverage_pct = (
            round(100 * len(df) / expected_count, 3) if expected_count else None
        )

        # ---- Persist fixes ----
        if self.fix and result.fixes_applied:

            out_df = df.drop(columns="_ts")
            out_df.to_csv(filepath, index=False)

            self.logger.info(
                f"{base_asset}: fix applied -> {'; '.join(result.fixes_applied)}"
            )

        return result

    # --------------------------------------------------------
    # GAP GROUPING (reporting helper - not a validation check)
    # --------------------------------------------------------

    @staticmethod
    def _group_missing_into_gaps(missing: pd.DatetimeIndex) -> list:
        """
        Groups a DatetimeIndex of missing hourly timestamps (as
        already computed by the MISSING_HOURLY_CANDLES check) into
        contiguous gaps.

        Returns a list of dicts, one per gap, each with:
            start, end          - first/last missing timestamp in the gap
            missing_hours       - number of missing hours in the gap
            prev                - last real candle timestamp before the gap
            next                - first real candle timestamp after the gap
        """

        if len(missing) == 0:
            return []

        one_hour = pd.Timedelta(hours=1)

        gaps = []

        gap_start = missing[0]
        gap_prev = missing[0]

        for ts in missing[1:]:

            if ts - gap_prev == one_hour:
                # still inside the same contiguous run
                gap_prev = ts
                continue

            gaps.append((gap_start, gap_prev))
            gap_start = ts
            gap_prev = ts

        gaps.append((gap_start, gap_prev))

        gap_records = []

        for start, end in gaps:

            missing_hours = int((end - start) / one_hour) + 1

            gap_records.append(
                {
                    "start": start,
                    "end": end,
                    "missing_hours": missing_hours,
                    "prev": start - one_hour,
                    "next": end + one_hour,
                }
            )

        return gap_records

    # --------------------------------------------------------
    # REPORTING
    # --------------------------------------------------------

    def _write_gap_csv(self, result: CoinValidationResult):
        """
        Writes validation/gaps/<SYMBOL>_gaps.csv - one row per gap,
        with the full boundary detail. Always complete, regardless
        of how many gaps there are.
        """

        rows = []

        for i, gap in enumerate(result.gaps, start=1):

            rows.append(
                {
                    "GapNumber": i,
                    "GapStart": gap["start"],
                    "GapEnd": gap["end"],
                    "MissingHours": gap["missing_hours"],
                    "PreviousTimestamp": gap["prev"],
                    "NextTimestamp": gap["next"],
                }
            )

        pd.DataFrame(rows).to_csv(
            self.gaps_dir / f"{result.symbol}_gaps.csv", index=False
        )

    def _write_gap_txt_report(self, result: CoinValidationResult):
        """
        Writes validation/details/<SYMBOL>_report.txt - a
        human-readable missing-candle diagnostic, intended to make
        it obvious whether gaps look like real exchange downtime or
        a downloader bug.

        Below MAX_GAPS_FOR_VERBOSE_TXT gaps, every missing hour
        inside each gap is listed individually. Above that, each gap
        is summarized on a single line instead (the CSV always has
        full detail either way).
        """

        gaps = result.gaps

        total_missing = sum(g["missing_hours"] for g in gaps)
        largest = max(gaps, key=lambda g: g["missing_hours"])

        lines = [
            "=" * 60,
            "Missing Candle Summary",
            "=" * 60,
            f"Total Missing Candles : {total_missing}",
            f"Number of Gaps        : {len(gaps)}",
            f"Largest Gap           : {largest['missing_hours']} hour(s)",
            "=" * 60,
            "",
        ]

        verbose = len(gaps) <= MAX_GAPS_FOR_VERBOSE_TXT
        one_hour = pd.Timedelta(hours=1)

        for i, gap in enumerate(gaps, start=1):

            if verbose:

                lines.append(f"Gap {i}")
                lines.append("Previous Candle")
                lines.append(f"  {gap['prev']}")
                lines.append("Missing")

                ts = gap["start"]
                while ts <= gap["end"]:
                    lines.append(f"  {ts}")
                    ts += one_hour

                lines.append("Next Candle")
                lines.append(f"  {gap['next']}")
                lines.append(f"Length")
                lines.append(f"  {gap['missing_hours']} hour(s)")
                lines.append("-" * 36)

            else:

                lines.append(
                    f"Gap {i} | Previous: {gap['prev']} | "
                    f"Start: {gap['start']} | End: {gap['end']} | "
                    f"Next: {gap['next']} | "
                    f"Length: {gap['missing_hours']} hour(s)"
                )

        report_path = self.details_dir / f"{result.symbol}_report.txt"

        with open(report_path, "w", encoding="utf8") as f:
            f.write("\n".join(lines))

    def _write_detail_report(self, result: CoinValidationResult):

        detail_rows = [
            {
                "Code": i.code,
                "Severity": i.severity,
                "Count": i.count,
                "Message": i.message,
            }
            for i in result.issues
        ]

        pd.DataFrame(detail_rows).to_csv(
            self.details_dir / f"{result.symbol}_issues.csv", index=False
        )

    def _log_result(self, result: CoinValidationResult):

        if result.status == "PASS":

            self.logger.info(
                f"{result.symbol}: PASS "
                f"({result.row_count} rows, {result.coverage_pct}% coverage)"
            )

        elif result.status == "WARN":

            codes = [i.code for i in result.warning_issues]

            self.logger.warning(
                f"{result.symbol}: WARN - {len(codes)} warning(s): {codes}"
            )

        else:

            codes = [i.code for i in result.critical_issues]

            self.logger.error(
                f"{result.symbol}: FAIL - {len(codes)} critical issue(s): {codes}"
            )

    def _write_summary_csv(self, rows: list):

        df = pd.DataFrame(rows)

        path = self.validation_dir / "validation_report.csv"

        df.to_csv(path, index=False)

        self.logger.info(f"Validation report written to {path}")

    def _print_summary(self, summary: RunSummary):

        elapsed = time.time() - summary.started_at

        lines = [
            "=" * 60,
            "VALIDATION COMPLETE",
            "=" * 60,
            f"Coins checked : {len(summary.results)}",
            f"Passed        : {len(summary.passed)}",
            f"Warnings      : {len(summary.warned)}",
            f"Failed        : {len(summary.failed)}",
            f"Elapsed       : {elapsed:.1f}s",
        ]

        if summary.failed:

            lines.append("")
            lines.append("Failed coins:")

            for r in summary.failed:
                codes = ", ".join(sorted({i.code for i in r.critical_issues}))
                lines.append(f"  {r.symbol}: {codes}")

        report = "\n".join(lines)

        print()
        print(report)

        self.logger.info(report)

    # --------------------------------------------------------
    # RUN
    # --------------------------------------------------------

    def run(
        self,
        symbol_filter: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> RunSummary:

        self.load_mapping()

        coins = self.get_ready_coins()

        if symbol_filter:
            coins = coins[coins["BaseAsset"].str.upper() == symbol_filter.upper()]

        if limit:
            coins = coins.head(limit)

        summary = RunSummary(started_at=time.time())

        self.logger.info(
            f"Starting validation for {len(coins)} coin(s) (fix={self.fix})"
        )

        report_rows = []

        for _, row in tqdm(
            coins.iterrows(), total=len(coins), desc="Validating"
        ):

            base_asset = str(row["BaseAsset"])

            try:

                result = self.validate_coin(row)

            except Exception as exc:  # noqa: BLE001

                self.logger.error(f"{base_asset}: validation crashed - {exc}")

                result = CoinValidationResult(symbol=base_asset)
                result.add_issue("VALIDATION_ERROR", "CRITICAL", str(exc))

            summary.add(result)

            if result.gaps:
                missing_candles = sum(g["missing_hours"] for g in result.gaps)
                gap_count = len(result.gaps)
                largest_gap = max(result.gaps, key=lambda g: g["missing_hours"])
                largest_gap_size = largest_gap["missing_hours"]
                largest_gap_start = largest_gap["start"]
                largest_gap_end = largest_gap["end"]
            else:
                missing_candles = 0
                gap_count = 0
                largest_gap_size = None
                largest_gap_start = None
                largest_gap_end = None

            report_rows.append(
                {
                    "Symbol": result.symbol,
                    "Status": result.status,
                    "Rows": result.row_count,
                    "Start": result.start,
                    "End": result.end,
                    "CoveragePct": result.coverage_pct,
                    "MissingCandles": missing_candles,
                    "GapCount": gap_count,
                    "LargestGap": largest_gap_size,
                    "LargestGapStart": largest_gap_start,
                    "LargestGapEnd": largest_gap_end,
                    "CriticalIssues": len(result.critical_issues),
                    "WarningIssues": len(result.warning_issues),
                    "IssueCodes": ";".join(sorted({i.code for i in result.issues})),
                    "FixesApplied": ";".join(result.fixes_applied),
                }
            )

            if result.issues:
                self._write_detail_report(result)

            if result.gaps:
                self._write_gap_csv(result)
                self._write_gap_txt_report(result)

            self._log_result(result)

        self._write_summary_csv(report_rows)
        self._print_summary(summary)

        return summary


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Validate historical OHLCV datasets for READY coins."
    )

    parser.add_argument(
        "--mapping",
        default=str(MAPPING_FILE),
        help="Path to coin_mapping.csv",
    )

    parser.add_argument(
        "--historical-dir",
        default=str(HISTORICAL_DIR),
        help="Directory containing per-coin historical CSVs",
    )

    parser.add_argument(
        "--output-dir",
        default=str(VALIDATION_DIR),
        help="Directory to write validation_report.csv and details/",
    )

    parser.add_argument(
        "--symbol",
        default=None,
        help="Only validate a single BaseAsset symbol, e.g. BTC",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only validate the first N ready coins (useful for testing)",
    )

    parser.add_argument(
        "--fix",
        action="store_true",
        help=(
            "Remove duplicate rows, remove duplicate timestamps (keep "
            "first), and sort chronologically. Never fills gaps or "
            "modifies values."
        ),
    )

    return parser.parse_args()


def main():

    args = parse_args()

    validator = HistoricalValidator(
        mapping_file=Path(args.mapping),
        historical_dir=Path(args.historical_dir),
        validation_dir=Path(args.output_dir),
        fix=args.fix,
    )

    validator.run(symbol_filter=args.symbol, limit=args.limit)


if __name__ == "__main__":
    main()
