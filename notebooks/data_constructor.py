from data_warehouse import DataSource
import datetime as dt
from tqdm import tqdm
import numpy as np
import pandas as pd

from modules.models import TimePeriod, PreprocessingParams
from modules.data_filtering import DataFiltering
from modules.data_processing import DataProcessor

    
class DataConstructor:
    ds = DataSource()
    date_format = '%Y-%m-%d %H:%M:%S'
    
    def get_agg_1m_data(self, time_period: TimePeriod, fetch_col='price'):
        # Selects all aggregated data tables
        ds_tables = self.ds.contain_tables()
        tickers = [t for t in ds_tables if t.endswith('USDT_agg_1m')]
        
        # Calculates the data amount
        dt_from = time_period.dt_from
        dt_to = time_period.dt_to
        diff_seconds = (dt_to - dt_from).total_seconds()
        data_amount = diff_seconds // 60 + 1 # minutes
        
        # Creates dataframe from a date array
        dt_range = pd.date_range(dt_from, dt_to, freq='1Min')
        df = pd.DataFrame(dt_range, columns=['dtm'])

        for ticker in tqdm(tickers, desc='Constructing dataframe'):
            df_fetch = self.ds.as_pandas(ticker, columns=['dtm', fetch_col], time_period=time_period)
            
            # If dataframe is empty then continue
            if df_fetch.empty:
                continue

            # Checks if there is any mising data
            if df_fetch.shape[0] != data_amount:

                # Checks if there is a gap between dt_now and first recorded price
                if not (df_fetch['dtm']  == dt_from).any():
                    diff_data_gap = (df_fetch['dtm'].iloc[0] - dt_from).total_seconds()
                    gap_minutes = diff_data_gap // 60

                    # If gap is bigger than 1 day then skip coin
                    if gap_minutes > 60 * 24:
                        continue

                # If the ending date is missing then it's added   
                if not (df_fetch['dtm']  == dt_to).any():
                    df_fetch.loc[len(df_fetch), 'dtm'] = dt_to

                # Reindexes the dtm column to add the missing dates
                df_fetch = df_fetch.set_index('dtm')
                df_fetch = df_fetch.reindex(dt_range, method='ffill', copy=False)
                df_fetch = df_fetch.reset_index()

                # Forward fills in NaN values
                df_fetch[fetch_col].fillna(method = 'ffill', inplace = True)

                # Back fills NaN values if there are any missing values 
                # There can be a gap in the initial values of data 
                df_fetch[fetch_col].fillna(method = 'bfill', inplace = True)

            # Add fetched price column to original df  
            extracted_col = df_fetch[fetch_col]
            df = df.join(extracted_col)
            df.rename(columns={fetch_col: ticker.replace('_agg_1m', '')}, inplace = True)

        return df, list(df.columns[1:])

    def get_ticker_average_vol(self, time_period: TimePeriod, tickers: list):
        df_vol = pd.DataFrame([], columns=['vol_quote'])

        for ticker in tqdm(tickers, desc='Collecting coin volume means'):
            df_fetch =self.ds.table_column_means(ticker + '_agg_1m', columns=['vol_quote'], time_period=time_period)
            
            df_vol.loc[ticker] = float(df_fetch['vol_quote'])

        return df_vol
        
    
    def get_tickers_ordered_by_vol(self, tickers, time_period=None):
        if time_period is None:
            time_period = TimePeriod(dt_from=dt.datetime(2022, 1, 1), dt_to=dt.datetime(2023, 1, 1))
        
        print('=== Ordering tickers ===')
        df_vol = self.get_ticker_average_vol(time_period, tickers)
        df_vol = df_vol.sort_values(by=['vol_quote'], ascending=False)
        
        return list(df_vol.index)
    
    def construct_df(self, time_period: TimePeriod, preprocessing_params = PreprocessingParams(), filter_stablecoins = False, preprocess_data=True, fetch_col='price', order_by_volume=True):
        df, tickers = self.get_agg_1m_data(time_period, fetch_col=fetch_col)
        
        if filter_stablecoins:
            df, tickers = DataFiltering.filter_stablecoins(df, tickers)
        
        if preprocess_data:
            df = DataProcessor.data_preprocessor(df=df, tickers=tickers, params=preprocessing_params)
        
        if order_by_volume:
            tickers = self.get_tickers_ordered_by_vol(tickers, time_period=time_period)
        
        return df, tickers
