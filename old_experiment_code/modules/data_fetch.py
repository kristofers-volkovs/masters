import csv
import datetime
from binance.client import Client
from pycoingecko import CoinGeckoAPI
from dataclasses import dataclass, field

from modules.data_classes import CoinData

api_key = "43FADLTZVHLj3mvvvg2I50vYhxwf5rg1RQ1bA2hgqZwUZ4h26l4c9OCq6rIXVpOx"
api_secret = "pwjmNa8Uo2QjsuMMk5ujPBNM7y4BpQa0QWoDkLkxJjhIacLIvkJie8qNfdZ83vMP"


def datetime_range(start, end, delta):
    current = start
    while current < end:
        yield current
        current += delta


def get_coin_abbrs():
    cg = CoinGeckoAPI()
    coins_market = cg.get_coins_markets(vs_currency='usd')

    coin_abbrs = [row["symbol"].upper() for row in coins_market]
    return coin_abbrs


# Candle data example
# [
#   [
#       {int} 1657458000000       Open time
#       {str} '21307.23000000'    Open
#       {str} '21313.99000000'    High
#       {str} '21235.61000000'    Low
#       {str} '21251.99000000'    Close
#       {str} '1601.88561000'     Volume
#       {int} 1657458899999       Close time
#       {str} '34072568.91701190' Quote asset volume
#       {int} 29618               Number of trades
#       {str} '806.16662000'      Taker buy base asset volume
#       {str} '17147644.31621670' Taker buy quote asset volume
#       {str} '0'                 Ignore
#   [
# [
def fetch_coin_data(interval: str = None, tickers=None, total_csv=1, days_saved_in_csv=30):
    # api_key = os.environ.get('binance_api')
    # api_secret = os.environ.get('binance_secret')
    # print(f"{api_key}, {api_secret}")

    client = Client(api_key, api_secret)
    client.API_URL = 'https://api.binance.com/api'

    if interval == "1min":
        kline_interval = client.KLINE_INTERVAL_1MINUTE
        minutes = 1
    elif interval == "15min":
        kline_interval = client.KLINE_INTERVAL_15MINUTE
        minutes = 15
    else:
        kline_interval = client.KLINE_INTERVAL_15MINUTE
        minutes = 15
    date_now = datetime.datetime.now()

    if tickers is None:
        # fetches top 100 coin abbreviations from coingecko
        coin_abbrs = get_coin_abbrs()
        against_money = "USDT"
        tickers = [abbr + against_money for abbr in coin_abbrs]

    # There's a different csv for each week of data
    for idx in range(0, total_csv):
        date_to = date_now - datetime.timedelta(days=idx * days_saved_in_csv)
        date_from = date_to - datetime.timedelta(days=days_saved_in_csv)
        print(f"from: {date_from}, to: {date_to}")

        # binance doesn't provide data for the 3 latest hours
        date_to_corrected = date_to - datetime.timedelta(hours=18, minutes=5)
        reading_timestamps = [dt.strftime('%Y-%m-%dT%H:%MZ') for dt in
                              datetime_range(date_from, date_to_corrected, datetime.timedelta(minutes=minutes))]

        coin_data_list = []
        for ticker in tickers:
            try:
                candle_data = client.get_historical_klines(ticker, kline_interval, str(date_from), str(date_to))
                if len(candle_data) > 0:
                    # TODO be able to save different candle parameters
                    candle_open_data1 = [row[1] for row in candle_data]
                    coin_data_list.append(CoinData(ticker, coin_open_data=candle_open_data1))
                else:
                    raise Exception("No data in the fetch")
            except:
                print(f"INFO: {ticker} is not available on Binance")

        print(f"Amount of successful fetches: {len(coin_data_list)}")

        output_dir = "data"
        file_name = "from_" + date_from.strftime("%Y-%d-%m") + "_to_" + date_to.strftime("%Y-%d-%m") + "_ETHBTC"
        print(f"Creating csv in dir '/{output_dir}' with name {file_name}")

        with open(f'{output_dir}/{file_name}.csv', 'w') as f:
            writer = csv.writer(f)

            timestamp_row = [""] + reading_timestamps
            writer.writerow(timestamp_row)

            for coin in coin_data_list:
                row = [coin.ticker] + coin.coin_open_data
                writer.writerow(row)
