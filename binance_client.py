"""
binance_client.py

Reusable Binance REST client.

Features
--------
✓ Automatic retries
✓ Rate limiting
✓ Historical klines
✓ Earliest candle lookup
✓ Pagination
✓ JSON handling
✓ Ready for hourly updater
✓ Ready for websocket integration

Author:
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests


class BinanceClient:

    BASE_URL = "https://api.binance.com"

    KLINES = "/api/v3/klines"

    EXCHANGE_INFO = "/api/v3/exchangeInfo"

    MAX_LIMIT = 1000

    RATE_LIMIT_SLEEP = 0.15

    MAX_RETRIES = 3

    TIMEOUT = 30

    # --------------------------------------------------------

    def __init__(self):

        self.session = requests.Session()

        self.session.headers.update({

            "User-Agent": (
                "Mozilla/5.0 "
                "(Windows NT 10.0; Win64; x64)"
            )

        })

    # --------------------------------------------------------

    def _request(self, endpoint, params=None):

        url = self.BASE_URL + endpoint

        for attempt in range(self.MAX_RETRIES):

            try:

                response = self.session.get(

                    url,

                    params=params,

                    timeout=self.TIMEOUT

                )

                response.raise_for_status()

                time.sleep(self.RATE_LIMIT_SLEEP)

                return response.json()

            except requests.RequestException:

                if attempt == self.MAX_RETRIES - 1:

                    raise

                time.sleep(

                    2 ** attempt

                )

    # --------------------------------------------------------

    def get_exchange_info(self):

        return self._request(

            self.EXCHANGE_INFO

        )

    # --------------------------------------------------------

    def get_klines(

        self,

        symbol,

        interval="1h",

        start_time=None,

        end_time=None,

        limit=1000

    ):

        params = {

            "symbol": symbol,

            "interval": interval,

            "limit": min(

                limit,

                self.MAX_LIMIT

            )

        }

        if start_time is not None:

            params["startTime"] = int(

                start_time

            )

        if end_time is not None:

            params["endTime"] = int(

                end_time

            )

        return self._request(

            self.KLINES,

            params

        )

    # --------------------------------------------------------

    def get_first_kline(

        self,

        symbol,

        interval="1h"

    ):

        """
        Returns the first available hourly candle.

        Returns None if symbol doesn't exist.
        """

        try:

            data = self.get_klines(

                symbol,

                interval=interval,

                start_time=0,

                limit=1

            )

        except:

            return None

        if len(data) == 0:

            return None

        return data[0]
    # --------------------------------------------------------
    # Download ALL historical klines
    # --------------------------------------------------------

    def download_all_history(
        self,
        symbol,
        interval="1h",
        start_time=None,
        end_time=None,
        verbose=True
    ):
        """
        Downloads all available historical klines for a symbol.

        Returns:
            pandas.DataFrame
        """

        # If no start time supplied, discover first candle
        if start_time is None:

            first = self.get_first_kline(
                symbol,
                interval
            )

            if first is None:
                raise ValueError(
                    f"No historical data for {symbol}"
                )

            start_time = int(first[0])

        all_rows = []

        page = 1

        while True:

            if verbose:
                print(
                    f"{symbol} | Page {page} | "
                    f"Starting: {pd.to_datetime(start_time, unit='ms')}"
                )

            candles = self.get_klines(
                symbol=symbol,
                interval=interval,
                start_time=start_time,
                end_time=end_time,
                limit=self.MAX_LIMIT
            )

            if len(candles) == 0:
                break

            all_rows.extend(candles)

            # If fewer than MAX_LIMIT returned,
            # we've reached the end.
            if len(candles) < self.MAX_LIMIT:
                break

            # Continue immediately after the last candle
            last_open = candles[-1][0]

            start_time = last_open + 1

            page += 1

        if verbose:
            print(
                f"Downloaded {len(all_rows)} candles."
            )

        return self.klines_to_dataframe(all_rows)
    # --------------------------------------------------------
    # Convert Binance klines -> DataFrame
    # --------------------------------------------------------

    def klines_to_dataframe(self, klines):

        columns = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore"
        ]

        df = pd.DataFrame(
            klines,
            columns=columns
        )

        # Keep only required columns
        df = df[
            [
                "timestamp",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "quote_volume",
                "trades"
            ]
        ]

        # Convert timestamp
        df["timestamp"] = pd.to_datetime(
            df["timestamp"],
            unit="ms",
            utc=True
        )

        # Numeric columns
        numeric = [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "quote_volume"
        ]

        for col in numeric:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df["trades"] = pd.to_numeric(
            df["trades"],
            errors="coerce",
            downcast="integer"
        )

        df.sort_values(
            "timestamp",
            inplace=True
        )

        df.drop_duplicates(
            subset="timestamp",
            inplace=True
        )

        df.reset_index(
            drop=True,
            inplace=True
        )

        return df


    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    def save_csv(
        self,
        df,
        filepath
    ):

        filepath = Path(filepath)

        filepath.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        df.to_csv(
            filepath,
            index=False
        )

        return filepath


    # --------------------------------------------------------
    # Append to existing CSV
    # --------------------------------------------------------

    def append_csv(
        self,
        df,
        filepath
    ):

        filepath = Path(filepath)

        filepath.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        if filepath.exists():

            old = pd.read_csv(
                filepath,
                parse_dates=["timestamp"]
            )

            combined = pd.concat(
                [old, df],
                ignore_index=True
            )

            combined.drop_duplicates(
                subset="timestamp",
                inplace=True
            )

            combined.sort_values(
                "timestamp",
                inplace=True
            )

            combined.to_csv(
                filepath,
                index=False
            )

            return combined

        else:

            df.to_csv(
                filepath,
                index=False
            )

            return df


    # --------------------------------------------------------
    # Last timestamp in CSV
    # --------------------------------------------------------

    def last_timestamp(
        self,
        filepath
    ):

        filepath = Path(filepath)

        if not filepath.exists():

            return None

        df = pd.read_csv(
            filepath,
            usecols=["timestamp"]
        )

        if df.empty:

            return None

        ts = pd.to_datetime(
            df.iloc[-1]["timestamp"],
            utc=True
        )

        return int(
            ts.timestamp() * 1000
        )


    # --------------------------------------------------------
    # CSV exists?
    # --------------------------------------------------------

    def has_history(
        self,
        filepath
    ):

        filepath = Path(filepath)

        return (
            filepath.exists()
            and filepath.stat().st_size > 0
        )

    # --------------------------------------------------------
    # Download directly to a CSV
    # --------------------------------------------------------

    def download_symbol(
        self,
        symbol,
        output_file,
        interval="1h",
        verbose=True
    ):
        """
        Download the complete history for a symbol
        and save it directly to a CSV.
        """

        df = self.download_all_history(
            symbol=symbol,
            interval=interval,
            verbose=verbose
        )

        self.save_csv(
            df,
            output_file
        )

        return df


    # --------------------------------------------------------
    # Resume download from existing CSV
    # --------------------------------------------------------

    def update_symbol(
        self,
        symbol,
        output_file,
        interval="1h",
        verbose=True
    ):
        """
        Continue downloading from the last candle in an
        existing CSV. If the CSV doesn't exist, download
        the full history.
        """

        output_file = Path(output_file)

        if not self.has_history(output_file):

            if verbose:
                print(f"{symbol}: no history found.")

            return self.download_symbol(
                symbol,
                output_file,
                interval,
                verbose
            )

        last = self.last_timestamp(output_file)

        if last is None:

            return self.download_symbol(
                symbol,
                output_file,
                interval,
                verbose
            )

        # Move forward exactly one candle (1 hour)
        next_start = last + (60 * 60 * 1000)

        if verbose:
            print(
                f"{symbol}: updating from "
                f"{pd.to_datetime(next_start, unit='ms', utc=True)}"
            )

        df = self.download_all_history(
            symbol=symbol,
            interval=interval,
            start_time=next_start,
            verbose=verbose
        )

        if df.empty:

            if verbose:
                print(f"{symbol}: already up to date.")

            return df

        self.append_csv(
            df,
            output_file
        )

        return df


    # --------------------------------------------------------
    # Close HTTP session
    # --------------------------------------------------------

    def close(self):

        self.session.close()


# ============================================================
# Simple test
# ============================================================

if __name__ == "__main__":

    client = BinanceClient()

    try:

        df = client.download_all_history(
            "BTCUSDT",
            verbose=True
        )

        print(df.head())

        print(df.tail())

        print(f"\nDownloaded {len(df):,} candles")

    finally:

        client.close()