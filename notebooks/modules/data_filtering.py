import numpy as np

class DataFiltering:
    @staticmethod
    def filter_stablecoins(df, tickers):
        new_tickers = []
        for ticker in tickers:
            df_log = np.log(df.loc[:, ticker])
            coin_mean = np.mean(df_log)
            if -0.15 < coin_mean < 0.13:
                df.drop(ticker, axis='columns', inplace=True)
            else:
                new_tickers.append(ticker)

        return df, new_tickers
