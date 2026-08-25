"""
ranking.py

Downloads the Top N cryptocurrencies from CoinMarketCap
and saves them as output/top_coins.csv
"""

from pathlib import Path

import pandas as pd
import requests


class CoinMarketCapRanking:

    URL = "https://api.coinmarketcap.com/data-api/v3/cryptocurrency/listing"

    def __init__(self, limit=200):
        self.limit = limit

    def fetch(self):

        params = {
            "start": 1,
            "limit": self.limit,
            "sortBy": "rank",
            "sortType": "desc",
            "convert": "USD,BTC,ETH",
            "cryptoType": "all",
            "tagType": "all",
            "audited": "false",
            "aux": (
                "ath,atl,high24h,low24h,"
                "num_market_pairs,"
                "cmc_rank,"
                "date_added,"
                "max_supply,"
                "circulating_supply,"
                "total_supply,"
                "volume_7d,"
                "volume_30d,"
                "self_reported_circulating_supply,"
                "self_reported_market_cap,"
                "socials"
            ),
        }

        headers = {
            "accept": "application/json, text/plain, */*",
            "origin": "https://coinmarketcap.com",
            "referer": "https://coinmarketcap.com/",
            "platform": "web",
            "user-agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/150.0.0.0 Safari/537.36"
            ),
        }

        response = requests.get(
            self.URL,
            params=params,
            headers=headers,
            timeout=30,
        )

        response.raise_for_status()

        data = response.json()["data"]["cryptoCurrencyList"]

        # Remove CMC20 and any future sponsored rows
        coins = [
            coin
            for coin in data
            if coin.get("cmcRank") is not None
        ]

        # Safety sort
        coins.sort(key=lambda x: x["cmcRank"])

        return coins

    def to_dataframe(self, coins):

        rows = []

        for coin in coins:

            usd = next(
                (q for q in coin["quotes"] if q["name"] == "USD"),
                {}
            )

            rows.append({

                "Rank": coin["cmcRank"],
                "CMC_ID": coin["id"],
                "Name": coin["name"],
                "Symbol": coin["symbol"],
                "Slug": coin["slug"],

                "Price_USD": usd.get("price"),
                "MarketCap": usd.get("marketCap"),
                "Volume24h": usd.get("volume24h"),

                "CirculatingSupply":
                    coin.get("circulatingSupply"),

                "TotalSupply":
                    coin.get("totalSupply"),

                "MaxSupply":
                    coin.get("maxSupply"),

                "DateAdded":
                    coin.get("dateAdded")

            })

        return pd.DataFrame(rows)

    def save(self, df):

        output = Path("output")

        output.mkdir(exist_ok=True)

        filename = output / "top_coins.csv"

        df.to_csv(filename, index=False)

        return filename


if __name__ == "__main__":

    print("Downloading Top Coins...\n")

    cmc = CoinMarketCapRanking(limit=200)

    coins = cmc.fetch()

    df = cmc.to_dataframe(coins)

    path = cmc.save(df)

    print(df.head(10))

    print("\n----------------------------")
    print(f"Coins Downloaded : {len(df)}")
    print(f"First Coin       : {df.iloc[0]['Name']}")
    print(f"Last Coin        : {df.iloc[-1]['Name']}")
    print(f"Saved To         : {path}")