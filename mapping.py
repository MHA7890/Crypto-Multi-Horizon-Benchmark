"""
mapping.py

Creates the master mapping between

CoinMarketCap
↓

Binance Spot

Outputs

output/
    coin_mapping.csv
    summary.txt

Author:
"""

from pathlib import Path
from datetime import datetime, timedelta
import json
import requests
import pandas as pd
from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

TOP_COINS = "output/top_coins.csv"

OUTPUT = Path("output")

CACHE = Path("cache")

CACHE.mkdir(exist_ok=True)

OUTPUT.mkdir(exist_ok=True)


EXCHANGE_INFO_URL = "https://api.binance.com/api/v3/exchangeInfo"

TICKER_URL = "https://api.binance.com/api/v3/ticker/24hr"


QUOTE_PRIORITY = [

    "USDT",

    "FDUSD",

    "USDC",

    "BTC",

    "ETH",

    "BNB",

    "TRY",

    "EUR",

    "BRL"

]


STABLECOINS = {

    "USDT",

    "USDC",

    "USDS",

    "USDE",

    "DAI",

    "FDUSD",

    "PYUSD",

    "FRAX",

    "TUSD",

    "USDG"

}


# ============================================================
# CLASS
# ============================================================


class MappingEngine:

    def __init__(self):

        self.exchange = []

        self.tickers = {}

        self.by_base = {}

        self.ready = []

        self.review = []



    # =======================================================
    # CACHE
    # =======================================================

    def cache_valid(self, file):

        if not file.exists():

            return False

        age = datetime.now() - datetime.fromtimestamp(
            file.stat().st_mtime
        )

        return age < timedelta(hours=12)



    # =======================================================
    # DOWNLOAD BINANCE DATA
    # =======================================================

    def load_exchange(self):

        cache_file = CACHE / "exchangeInfo.json"

        if self.cache_valid(cache_file):

            print("Using cached exchangeInfo...")

            with open(cache_file, "r", encoding="utf8") as f:

                data = json.load(f)

        else:

            print("Downloading exchangeInfo...")

            r = requests.get(EXCHANGE_INFO_URL, timeout=30)

            r.raise_for_status()

            data = r.json()

            with open(cache_file, "w", encoding="utf8") as f:

                json.dump(data, f)

        self.exchange = data["symbols"]



    # =======================================================
    # DOWNLOAD TICKERS
    # =======================================================

    def load_tickers(self):

        cache_file = CACHE / "ticker24.json"

        if self.cache_valid(cache_file):

            print("Using cached tickers...")

            with open(cache_file, "r", encoding="utf8") as f:

                data = json.load(f)

        else:

            print("Downloading tickers...")

            r = requests.get(TICKER_URL, timeout=30)

            r.raise_for_status()

            data = r.json()

            with open(cache_file, "w", encoding="utf8") as f:

                json.dump(data, f)

        self.tickers = {

            x["symbol"]: x

            for x in data

        }



    # =======================================================
    # BUILD INDEX
    # =======================================================

    def build_index(self):

        print("Building index...")

        self.by_base = {}

        for market in self.exchange:

            if market["status"] != "TRADING":

                continue

            base = market["baseAsset"]

            self.by_base.setdefault(base, []).append(market)

        print(

            f"Indexed {len(self.by_base)} assets."

        )



    # =======================================================
    # BEST MARKET
    # =======================================================

    def best_market(self, markets):

        for quote in QUOTE_PRIORITY:

            for market in markets:

                if market["quoteAsset"] == quote:

                    return market

        return markets[0]
    # =======================================================
    # CONFIDENCE SCORE
    # =======================================================

    def confidence(self, coin, market):

        score = 0
        reasons = []

        if coin["Symbol"].upper() == market["baseAsset"].upper():
            score += 50
            reasons.append("SYMBOL")

        if market["quoteAsset"] == "USDT":
            score += 20
            reasons.append("USDT")

        elif market["quoteAsset"] in ("FDUSD", "USDC"):
            score += 15
            reasons.append("STABLE_QUOTE")

        if market["status"] == "TRADING":
            score += 10
            reasons.append("TRADING")

        ticker = self.tickers.get(market["symbol"])

        if ticker:

            try:

                volume = float(
                    ticker.get("quoteVolume", 0)
                )

                if volume > 1000000:

                    score += 10
                    reasons.append("HIGH_VOLUME")

                elif volume > 100000:

                    score += 5
                    reasons.append("GOOD_VOLUME")

            except:
                pass

        if market["quoteAsset"] in QUOTE_PRIORITY:

            score += max(
                0,
                10 - QUOTE_PRIORITY.index(
                    market["quoteAsset"]
                )
            )

        if score >= 90:
            level = "HIGH"

        elif score >= 75:
            level = "MEDIUM"

        else:
            level = "LOW"

        return score, level, reasons



    # =======================================================
    # PROCESS
    # =======================================================

    def process(self):

        df = pd.read_csv(TOP_COINS)

        print(
            f"\nProcessing {len(df)} coins...\n"
        )

        for _, row in tqdm(
            df.iterrows(),
            total=len(df)
        ):

            coin = row.to_dict()

            symbol = str(
                coin["Symbol"]
            ).upper()

            coin["ConfidenceScore"] = 0
            coin["Confidence"] = ""
            coin["Reason"] = ""
            coin["BinanceSymbol"] = ""
            coin["BaseAsset"] = ""
            coin["QuoteAsset"] = ""
            coin["AvailableMarkets"] = ""
            coin["FirstHourlyCandle"] = ""

            # ------------------------------------
            # Skip stablecoins
            # ------------------------------------

            if symbol in STABLECOINS:

                coin["Status"] = "STABLECOIN_SKIP"

                coin["Reason"] = (
                    "Stablecoin"
                )

                self.review.append(coin)

                continue

            markets = self.by_base.get(
                symbol,
                []
            )

            # ------------------------------------
            # No Binance market
            # ------------------------------------

            if not markets:

                coin["Status"] = "NOT_LISTED"

                coin["Reason"] = (
                    "No Binance Spot Market"
                )

                self.review.append(coin)

                continue

            # ------------------------------------
            # Pick preferred quote asset
            # ------------------------------------

            best = self.best_market(
                markets
            )

            score, level, reasons = (
                self.confidence(
                    coin,
                    best
                )
            )

            coin["ConfidenceScore"] = score

            coin["Confidence"] = level

            coin["Reason"] = ",".join(
                reasons
            )

            coin["BinanceSymbol"] = (
                best["symbol"]
            )

            coin["BaseAsset"] = (
                best["baseAsset"]
            )

            coin["QuoteAsset"] = (
                best["quoteAsset"]
            )

            coin["AvailableMarkets"] = (
                ";".join(
                    sorted(
                        m["symbol"]
                        for m in markets
                    )
                )
            )

            ticker = self.tickers.get(
                best["symbol"],
                {}
            )

            coin["LastPrice"] = ticker.get(
                "lastPrice",
                ""
            )

            coin["24hVolume"] = ticker.get(
                "quoteVolume",
                ""
            )

            coin["Trades24h"] = ticker.get(
                "count",
                ""
            )

            coin["PriceChange24h"] = ticker.get(
                "priceChangePercent",
                ""
            )

            if best["quoteAsset"] == "USDT":

                coin["Status"] = "READY"

                self.ready.append(
                    coin
                )

            else:

                coin["Status"] = (
                    "NON_USDT_PAIR"
                )

                self.review.append(
                    coin
                )

    # =======================================================
    # SAVE OUTPUTS
    # =======================================================

    def save(self):

        ready_df = pd.DataFrame(self.ready)

        review_df = pd.DataFrame(self.review)

        mapping_df = pd.concat(
            [ready_df, review_df],
            ignore_index=True
        )

        if "Rank" in mapping_df.columns:

            mapping_df = mapping_df.sort_values(
                "Rank"
            )

        mapping_file = OUTPUT / "coin_mapping.csv"

        mapping_df.to_csv(
            mapping_file,
            index=False
        )

        if len(review_df):

            review_df.to_csv(
                OUTPUT / "review.csv",
                index=False
            )

        summary = []

        summary.append("=" * 60)
        summary.append("CRYPTO MAPPING SUMMARY")
        summary.append("=" * 60)
        summary.append("")

        summary.append(
            f"Total Coins : {len(mapping_df)}"
        )

        summary.append(
            f"Ready : {len(ready_df)}"
        )

        summary.append(
            f"Review : {len(review_df)}"
        )

        summary.append("")

        if len(review_df):

            summary.append(
                "Review Breakdown"
            )

            summary.append("-" * 30)

            counts = review_df[
                "Status"
            ].value_counts()

            for status, count in counts.items():

                summary.append(
                    f"{status:<25}{count}"
                )

        with open(
            OUTPUT / "summary.txt",
            "w",
            encoding="utf8"
        ) as f:

            f.write(
                "\n".join(summary)
            )

        print()

        print("=" * 60)

        print("MAPPING COMPLETE")

        print("=" * 60)

        print(
            f"Ready : {len(ready_df)}"
        )

        print(
            f"Review : {len(review_df)}"
        )

        print()

        print(
            "Generated:"
        )

        print(
            "  output/coin_mapping.csv"
        )

        if len(review_df):

            print(
                "  output/review.csv"
            )

        print(
            "  output/summary.txt"
        )



    # =======================================================
    # RUN
    # =======================================================

    def run(self):

        self.load_exchange()

        self.load_tickers()

        self.build_index()

        self.process()

        self.save()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if not Path(TOP_COINS).exists():

        raise FileNotFoundError(
            f"\nMissing file:\n{TOP_COINS}\n\nRun ranking.py first."
        )

    print()

    print("=" * 60)
    print("CRYPTO MAPPING ENGINE")
    print("=" * 60)

    engine = MappingEngine()

    engine.run()

    print()

    print("Finished.")