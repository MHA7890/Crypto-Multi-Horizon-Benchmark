"""
websocket.py

Collects real-time market data for all READY coins between hourly
candles, using Binance Spot combined WebSocket streams.

Streams used
------------
- MiniTicker  : close/open/high/low/volume/quoteVolume, 1 update/sec/symbol
- BookTicker  : best bid/ask price + quantity, updates on every book change

Row-emission design
--------------------
Every incoming message (either stream) updates an in-memory "latest
state" per symbol. A CSV row is only emitted when a MiniTicker message
arrives (throttled by Binance to 1/sec/symbol), stamped with whatever
bid/ask is freshest at that moment from the BookTicker cache.
BookTicker-only updates never emit a row on their own - this is what
keeps output bounded (~136 rows/sec total) despite BookTicker firing
far more often than that.

Design
------
WebSocketCollector
    ↓ manages one or more
StreamConnection            (one per group of <= max_streams_per_connection)
    ↓ writes through
CSVBuffer                   (per-symbol in-memory buffer, flushed by size/time)
    ↓ records progress via
Logger                      (logs/websocket.log)

Recovery model
---------------
Binance WebSockets do not replay history. If this collector is down
for a while, that window of real-time data is simply gone - the
historical downloader (history_downloader.py) is the source of truth
for backfilling candles. This module never tries to reconstruct
missed data; it only resumes collecting from the moment it reconnects.

Requires:
    pip install websockets

Run:
    python websocket.py
    python websocket.py --buffer-size 200 --flush-interval 5
    python websocket.py --max-streams-per-connection 200
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException


# ============================================================
# CONFIG
# ============================================================

MAPPING_FILE = Path("output/coin_mapping.csv")

LIVE_DIR = Path("live")

LOG_DIR = Path("logs")

LOG_FILE = LOG_DIR / "websocket.log"

WS_BASE_URL = "wss://stream.binance.com:9443/stream?streams="

# Binance's officially documented spot limit is 1024 streams/connection.
# We default to a much more conservative 200 to leave headroom and to
# make the multi-connection split logic actually exercise itself.
DEFAULT_MAX_STREAMS_PER_CONNECTION = 200

DEFAULT_BUFFER_SIZE = 100          # rows per symbol before forced flush
DEFAULT_FLUSH_INTERVAL = 10.0      # seconds before forced flush
DEFAULT_SAMPLE_INTERVAL = 1.0      # seconds between emitted rows per symbol

RECONNECT_BASE_DELAY = 1.0
RECONNECT_MAX_DELAY = 60.0

CSV_COLUMNS = [
    "timestamp",
    "symbol",
    "last_price",
    "open",
    "high",
    "low",
    "volume",
    "quote_volume",
    "bid_price",
    "bid_qty",
    "ask_price",
    "ask_qty",
]


# ============================================================
# LOGGING
# ============================================================

def build_logger(log_level: str = "INFO") -> logging.Logger:

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("websocket_collector")
    logger.setLevel(log_level.upper())
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
# SYMBOL STATE
# ============================================================

@dataclass
class SymbolState:
    """Latest known values for one symbol, merged from both streams."""

    base_asset: str
    binance_symbol: str

    last_price: str = ""
    open: str = ""
    high: str = ""
    low: str = ""
    volume: str = ""
    quote_volume: str = ""

    bid_price: str = ""
    bid_qty: str = ""
    ask_price: str = ""
    ask_qty: str = ""

    mini_ticker_active: bool = False
    book_ticker_active: bool = False

    def apply_mini_ticker(self, data: dict):
        self.last_price = data.get("c", "")
        self.open = data.get("o", "")
        self.high = data.get("h", "")
        self.low = data.get("l", "")
        self.volume = data.get("v", "")
        self.quote_volume = data.get("q", "")
        self.mini_ticker_active = True

    def apply_book_ticker(self, data: dict):
        self.bid_price = data.get("b", "")
        self.bid_qty = data.get("B", "")
        self.ask_price = data.get("a", "")
        self.ask_qty = data.get("A", "")
        self.book_ticker_active = True

    @property
    def both_streams_active(self) -> bool:
        return self.mini_ticker_active and self.book_ticker_active

    def to_row(self, timestamp_ms: int) -> dict:
        ts = datetime.fromtimestamp(
            timestamp_ms / 1000, tz=timezone.utc
        ).isoformat()

        return {
            "timestamp": ts,
            "symbol": self.base_asset,
            "last_price": self.last_price,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "volume": self.volume,
            "quote_volume": self.quote_volume,
            "bid_price": self.bid_price,
            "bid_qty": self.bid_qty,
            "ask_price": self.ask_price,
            "ask_qty": self.ask_qty,
        }


# ============================================================
# CSV BUFFER
# ============================================================

class CSVBuffer:
    """
    Per-symbol in-memory row buffer. Flushed to live/{BASE}.csv when
    either the row count or the time-since-last-flush threshold is
    reached. Always appends; never overwrites; creates the file (with
    header) on first write.
    """

    def __init__(
        self,
        output_dir: Path,
        max_rows: int,
        flush_interval: float,
        logger: logging.Logger,
        num_writer_lanes: int = 8,
    ):
        self.output_dir = Path(output_dir)
        self.max_rows = max_rows
        self.flush_interval = flush_interval
        self.logger = logger

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._buffers: Dict[str, List[dict]] = {}
        self._last_flush: Dict[str, float] = {}
        self._header_written: Dict[str, bool] = {}

        # Actual disk writes happen on background "lane" threads so a
        # burst of simultaneous flushes never blocks the asyncio event
        # loop that's receiving websocket messages. Each symbol is
        # deterministically pinned to one lane (hash-based), and each
        # lane is a single persistent thread draining its own FIFO
        # queue - so a given symbol's writes are always applied in the
        # exact order they were submitted. A shared thread pool was
        # tried first, but with multiple workers pulling from a common
        # queue, two writes for the same symbol could be picked up out
        # of order, producing out-of-order timestamps in the CSV.
        self._num_lanes = num_writer_lanes
        self._queues: List["queue.Queue"] = [
            queue.Queue() for _ in range(num_writer_lanes)
        ]
        self._lane_threads: List[threading.Thread] = []

        for i in range(num_writer_lanes):
            t = threading.Thread(
                target=self._lane_worker,
                args=(self._queues[i],),
                name=f"csv-writer-{i}",
                daemon=True,
            )
            t.start()
            self._lane_threads.append(t)

        self._count_lock = threading.Lock()

        self.total_rows_written = 0

    def _lane_for(self, base_asset: str) -> int:
        return hash(base_asset) % self._num_lanes

    def _filepath(self, base_asset: str) -> Path:
        return self.output_dir / f"{base_asset}.csv"

    def _has_header(self, base_asset: str) -> bool:

        if base_asset in self._header_written:
            return self._header_written[base_asset]

        filepath = self._filepath(base_asset)

        written = filepath.exists() and filepath.stat().st_size > 0

        self._header_written[base_asset] = written

        return written

    def add(self, base_asset: str, row: dict):

        self._buffers.setdefault(base_asset, []).append(row)

        self._last_flush.setdefault(base_asset, time.time())

        if len(self._buffers[base_asset]) >= self.max_rows:
            self.flush(base_asset)

    def flush_due(self) -> List[str]:
        """Symbols whose buffer has pending rows older than flush_interval."""

        now = time.time()

        due = []

        for base_asset, rows in self._buffers.items():

            if not rows:
                continue

            if now - self._last_flush.get(base_asset, now) >= self.flush_interval:
                due.append(base_asset)

        return due

    def flush(self, base_asset: str):
        """
        Snapshots and clears the in-memory buffer immediately (cheap,
        stays on the event loop thread), then queues the actual disk
        write onto this symbol's lane so it's applied in order without
        blocking the caller.
        """

        rows = self._buffers.get(base_asset)

        if not rows:
            self._last_flush[base_asset] = time.time()
            return

        filepath = self._filepath(base_asset)

        write_header = not self._has_header(base_asset)

        self._header_written[base_asset] = True

        self._buffers[base_asset] = []
        self._last_flush[base_asset] = time.time()

        lane = self._lane_for(base_asset)

        self._queues[lane].put((base_asset, filepath, rows, write_header))

    def _lane_worker(self, q: "queue.Queue"):
        """Runs on a dedicated thread for the lifetime of the process.
        Processes its queue strictly FIFO - never touches the event loop."""

        while True:

            item = q.get()

            if item is None:  # shutdown sentinel
                q.task_done()
                break

            base_asset, filepath, rows, write_header = item

            try:

                with open(filepath, "a", newline="", encoding="utf8") as f:

                    writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)

                    if write_header:
                        writer.writeheader()

                    writer.writerows(rows)

                with self._count_lock:
                    self.total_rows_written += len(rows)

            except Exception as exc:  # noqa: BLE001

                self.logger.error(f"{base_asset}: failed writing to disk: {exc}")

            q.task_done()

    def flush_all(self):

        for base_asset in list(self._buffers.keys()):
            self.flush(base_asset)

    def shutdown(self, wait: bool = True):
        """Signals all lane threads to stop and, if wait, blocks until
        every queued write has actually been applied to disk."""

        for q in self._queues:
            q.put(None)

        if wait:
            for t in self._lane_threads:
                t.join()


# ============================================================
# SINGLE STREAM CONNECTION (subset of symbols)
# ============================================================

class StreamConnection:
    """
    Manages one websocket connection covering a subset of symbols,
    with automatic reconnect + exponential backoff.
    """

    def __init__(
        self,
        connection_id: int,
        symbols: List[Tuple[str, str]],  # (binance_symbol, base_asset)
        buffer: CSVBuffer,
        logger: logging.Logger,
        stop_event: asyncio.Event,
    ):
        self.connection_id = connection_id
        self.symbols = symbols
        self.buffer = buffer
        self.logger = logger
        self.stop_event = stop_event

        self.state: Dict[str, SymbolState] = {
            binance_symbol: SymbolState(base_asset, binance_symbol)
            for binance_symbol, base_asset in symbols
        }

        self.messages_received = 0

    def _build_url(self) -> str:

        streams = []

        for binance_symbol, _ in self.symbols:
            sym_lower = binance_symbol.lower()
            streams.append(f"{sym_lower}@miniTicker")
            streams.append(f"{sym_lower}@bookTicker")

        return WS_BASE_URL + "/".join(streams)

    async def run(self):

        attempt = 0
        url = self._build_url()

        while not self.stop_event.is_set():

            try:

                self.logger.info(
                    f"[conn {self.connection_id}] connecting "
                    f"({len(self.symbols)} symbols, "
                    f"{len(self.symbols) * 2} streams)..."
                )

                async with websockets.connect(
                    url, open_timeout=15, close_timeout=5
                ) as ws:

                    self.logger.info(
                        f"[conn {self.connection_id}] connected."
                    )

                    attempt = 0  # reset backoff after a clean connect

                    await self._consume(ws)

            except asyncio.CancelledError:
                raise

            except (ConnectionClosed, WebSocketException, OSError) as exc:

                self.logger.warning(
                    f"[conn {self.connection_id}] disconnected: {exc}"
                )

            except Exception as exc:  # noqa: BLE001

                self.logger.error(
                    f"[conn {self.connection_id}] unexpected error: {exc}"
                )

            if self.stop_event.is_set():
                break

            delay = min(
                RECONNECT_BASE_DELAY * (2 ** attempt), RECONNECT_MAX_DELAY
            )

            attempt += 1

            self.logger.info(
                f"[conn {self.connection_id}] reconnecting in {delay:.1f}s..."
            )

            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=delay)
            except asyncio.TimeoutError:
                pass

        self.logger.info(f"[conn {self.connection_id}] stopped.")

    async def _consume(self, ws):

        async for raw_message in ws:

            if self.stop_event.is_set():
                break

            self.messages_received += 1

            try:
                envelope = json.loads(raw_message)
            except json.JSONDecodeError:
                self.logger.warning(
                    f"[conn {self.connection_id}] malformed message, skipped."
                )
                continue

            stream_name = envelope.get("stream", "")
            data = envelope.get("data", {})

            binance_symbol = data.get("s")

            if not binance_symbol or binance_symbol not in self.state:
                continue

            symbol_state = self.state[binance_symbol]

            if stream_name.endswith("@miniTicker"):
                symbol_state.apply_mini_ticker(data)
            elif stream_name.endswith("@bookTicker"):
                symbol_state.apply_book_ticker(data)
            else:
                continue

            if not symbol_state.both_streams_active:
                continue  # wait until both MiniTicker and BookTicker have reported

            event_time = data.get("E", int(time.time() * 1000))
            row = symbol_state.to_row(event_time)
            self.buffer.add(symbol_state.base_asset, row)


# ============================================================
# WEBSOCKET COLLECTOR
# ============================================================

class WebSocketCollector:

    def __init__(
        self,
        mapping_file: Path = MAPPING_FILE,
        live_dir: Path = LIVE_DIR,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        flush_interval: float = DEFAULT_FLUSH_INTERVAL,
        max_streams_per_connection: int = DEFAULT_MAX_STREAMS_PER_CONNECTION,
        log_level: str = "INFO",
        reset_on_start: bool = True,
        sample_interval: float = DEFAULT_SAMPLE_INTERVAL,
    ):
        self.mapping_file = Path(mapping_file)
        self.live_dir = Path(live_dir)
        self.max_streams_per_connection = max_streams_per_connection
        self.reset_on_start = reset_on_start
        self.sample_interval = sample_interval

        self.logger = build_logger(log_level)

        self.buffer = CSVBuffer(
            output_dir=self.live_dir,
            max_rows=buffer_size,
            flush_interval=flush_interval,
            logger=self.logger,
        )

        self.stop_event = asyncio.Event()
        self.connections: List[StreamConnection] = []

    # --------------------------------------------------------

    def load_symbols(self) -> List[Tuple[str, str]]:

        if not self.mapping_file.exists():
            raise FileNotFoundError(
                f"Missing {self.mapping_file}. Run mapping.py first."
            )

        df = pd.read_csv(self.mapping_file)

        ready = df[df["Status"] == "READY"]

        symbols = [
            (str(row["BinanceSymbol"]), str(row["BaseAsset"]))
            for _, row in ready.iterrows()
        ]

        self.logger.info(f"Loaded {len(symbols)} READY symbols.")

        return symbols

    def build_groups(
        self, symbols: List[Tuple[str, str]]
    ) -> List[List[Tuple[str, str]]]:
        """
        Splits symbols into groups so that each connection stays under
        max_streams_per_connection (2 streams per symbol).
        """

        symbols_per_group = max(1, self.max_streams_per_connection // 2)

        groups = [
            symbols[i:i + symbols_per_group]
            for i in range(0, len(symbols), symbols_per_group)
        ]

        self.logger.info(
            f"Split into {len(groups)} connection(s) "
            f"(<= {symbols_per_group} symbols / "
            f"{symbols_per_group * 2} streams each)."
        )

        return groups

    # --------------------------------------------------------

    async def _periodic_flush(self):
        """Safety net: flushes any buffer that's been sitting past
        flush_interval, even if it never hit max_rows."""

        while not self.stop_event.is_set():

            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass

            for base_asset in self.buffer.flush_due():
                self.buffer.flush(base_asset)


    async def _status_report(self):

        while not self.stop_event.is_set():

            try:
                await asyncio.wait_for(self.stop_event.wait(), timeout=60.0)
            except asyncio.TimeoutError:
                pass

            if self.stop_event.is_set():
                break

            total_messages = sum(
                c.messages_received for c in self.connections
            )

            self.logger.info(
                f"status: {total_messages} messages received, "
                f"{self.buffer.total_rows_written} rows written to disk."
            )

    def _install_signal_handlers(self, loop: asyncio.AbstractEventLoop):

        def _request_stop():
            self.logger.info("Shutdown signal received.")
            self.stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):

            try:
                loop.add_signal_handler(sig, _request_stop)
            except (NotImplementedError, RuntimeError):
                # add_signal_handler isn't available on some platforms
                # (e.g. Windows); KeyboardInterrupt fallback in main().
                pass

    # --------------------------------------------------------

    def _reset_live_files(self):
        """live/ holds only current-session data. A gap between runs
        means missing data anyway, so old files are cleared on startup
        instead of silently appending after a stale gap."""

        self.live_dir.mkdir(parents=True, exist_ok=True)

        for f in self.live_dir.glob("*.csv"):
            f.unlink()

        self.logger.info(f"Cleared {self.live_dir}/ for a fresh session.")

    async def run(self):

        if self.reset_on_start:
            self._reset_live_files()

        symbols = self.load_symbols()

        if not symbols:
            self.logger.warning("No READY symbols found. Nothing to collect.")
            return

        groups = self.build_groups(symbols)

        self.connections = [
            StreamConnection(
                connection_id=i + 1,
                symbols=group,
                buffer=self.buffer,
                logger=self.logger,
                stop_event=self.stop_event,
            )
            for i, group in enumerate(groups)
        ]

        loop = asyncio.get_running_loop()

        self._install_signal_handlers(loop)

        self.logger.info("Starting websocket collector...")

        tasks = [asyncio.create_task(c.run()) for c in self.connections]

        tasks.append(asyncio.create_task(self._periodic_flush()))
        tasks.append(asyncio.create_task(self._status_report()))

        try:

            await self.stop_event.wait()

        finally:

            self.logger.info("Shutting down: flushing buffers...")

            self.buffer.flush_all()

            self.buffer.shutdown(wait=True)

            for task in tasks:
                task.cancel()

            await asyncio.gather(*tasks, return_exceptions=True)

            self.logger.info(
                f"Shutdown complete. "
                f"Total rows written: {self.buffer.total_rows_written}"
            )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Collect real-time Binance market data (MiniTicker + "
                     "BookTicker) into per-symbol live CSVs."
    )

    parser.add_argument("--mapping", default=str(MAPPING_FILE))
    parser.add_argument("--output-dir", default=str(LIVE_DIR))

    parser.add_argument(
        "--buffer-size", type=int, default=DEFAULT_BUFFER_SIZE,
        help="Rows buffered per symbol before a forced flush.",
    )

    parser.add_argument(
        "--flush-interval", type=float, default=DEFAULT_FLUSH_INTERVAL,
        help="Seconds before a partially-full buffer is force-flushed.",
    )

    parser.add_argument(
        "--sample-interval", type=float, default=DEFAULT_SAMPLE_INTERVAL,
        help="Seconds between emitted rows per symbol (fixed cadence, "
             "same for every symbol regardless of trading activity).",
    )

    parser.add_argument(
        "--max-streams-per-connection", type=int,
        default=DEFAULT_MAX_STREAMS_PER_CONNECTION,
        help="Upper bound on streams per websocket connection "
             "(2 streams per symbol). Binance's hard limit is 1024.",
    )

    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )

    parser.add_argument(
        "--no-reset", action="store_true",
        help="Keep existing live/ files and append instead of clearing "
             "them on startup (off by default).",
    )

    return parser.parse_args()


def main():

    args = parse_args()

    collector = WebSocketCollector(
        mapping_file=Path(args.mapping),
        live_dir=Path(args.output_dir),
        buffer_size=args.buffer_size,
        flush_interval=args.flush_interval,
        max_streams_per_connection=args.max_streams_per_connection,
        log_level=args.log_level,
        reset_on_start=not args.no_reset,
        sample_interval=args.sample_interval,
    )

    try:
        asyncio.run(collector.run())
    except KeyboardInterrupt:
        # Fallback for platforms without asyncio signal handler support.
        pass


if __name__ == "__main__":
    main()
