import datetime as dt
import pandas as pd

class DataIterator:
    timestamp_col = 'dtm'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    start_idx = -1
    idx = 0
    end_idx = -1
    
    """
        df - pandas.DataFrame, columns: ["dtm", "BTCUSDT", ...]
    """
    def __init__(self, df, reading_step=1, reading_window=60 * 24 * 7):
        self.df = df.copy()
        self.tickers = list(self.df.columns)
        self.tickers.remove(self.timestamp_col)

        self.reading_step = reading_step  # time step
        self.reading_window = reading_window  # window length
        
    def set_time_range(self, time_from, time_to):
        date_from = dt.datetime.strptime(time_from, self.date_format)
        date_to = dt.datetime.strptime(time_to, self.date_format)
        
        idx_from = min(
            range(len(self.df)),
            key=lambda i: abs(dt.datetime.strptime(str(self.df.at[i, self.timestamp_col]), self.date_format) - date_from)
        )
        
        idx_to = min(
            range(len(self.df)),
            key=lambda i: abs(dt.datetime.strptime(str(self.df.at[i, self.timestamp_col]), self.date_format) - date_to)
        )
        
        if idx_to + self.reading_step > len(self.df):
            raise ValueError("End date is out of bounds")
        
        self.start_idx = idx_from
        self.idx = idx_from
        self.end_idx = idx_to
    
    def get_timestamps(self):
        return self.df[self.timestamp_col]
    
    def get_tickers(self):
        return self.tickers
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            start = int(idx.start) * self.reading_step
            stop = int(idx.stop) * self.reading_step
        else:
            start = idx * self.reading_step
            stop = start + self.reading_window
        
        if self.start_idx != -1:
            start += self.start_idx
            stop += self.start_idx
                
        if stop > len(self.df):
            raise ValueError("End index is out of bounds")
            
        return self.df.iloc[start:stop]
        
    def __iter__(self):
        return self
    
    def __len__(self):
        if self.end_idx != -1:
            return int((self.end_idx - self.start_idx - self.reading_window + 2) / self.reading_step) 
        else:
            return int((len(self.df) - self.reading_window) / self.reading_step) + 1
    
    def __next__(self):
        window_end_idx = self.idx + self.reading_window - 1
        is_window_outside_df_range = window_end_idx > len(self.df) - 1
        is_idx_outside_range = self.end_idx != -1 and window_end_idx > self.end_idx
        if is_window_outside_df_range or is_idx_outside_range:
            self.idx = self.start_idx
            raise StopIteration
    
        df_slice = self.df.iloc[self.idx:self.idx + self.reading_window]
        self.idx += self.reading_step
        
        return df_slice
    
    def print_state(self):
        print('=== Data shapes ===')
        print(f'df: {self.df.shape}, tickers: {self.tickers}, tickers_len: {len(self.tickers)}')
        print("=== Iteration info ===")
        print(f'reading_step: {self.reading_step}, reading_window: {self.reading_window}')
        print(f'start_idx: {self.start_idx}, end_idx: {self.end_idx}, idx: {self.idx}')
        
        if self.start_idx != -1 and self.end_idx != -1:
            start_date = str(self.df.loc[self.start_idx, 'dtm'])
            end_date = str(self.df.loc[self.end_idx, 'dtm'])
        else:
            start_date = str(self.df.loc[0, 'dtm'])
            end_date = str(self.df.loc[len(self) - 1, 'dtm'])
        current_date = str(self.df.loc[self.idx, 'dtm'])
        
        print(f'start_date: {start_date}, end_date: {end_date}, current_date: {current_date}')
        print(f'iter amount: {self.__len__()}')

        
if __name__ == '__main__':
    # === Class examples ===
    
    df = pd.DataFrame({
        'dtm': [f"2022-01-01 00:0{i}:00" for i in range(10)], 
        'BTCUSDT': np.arange(10, 20),
        'BTC2USDT': np.arange(20, 30),
        'BTC3USDT': np.arange(30, 40),
    })
    df_iter = DataIterator(df, reading_window=6)
    df_iter.print_state()

    # === Regular iter ===

    for df_slice in df_iter:
        print('===')
        print(df_slice)
    print('===')

    # === Ranged iter ===

    date_from = "2022-01-01 00:02:00"
    date_to = "2022-01-01 00:08:00"
    print(f"Data range update: from{date_from}, to: {date_to}")
    df_iter.set_time_range(time_from=date_from, time_to=date_to)
    df_iter.print_state()

    for df_slice in df_iter:
        print('===')
        print(df_slice)
    print('===')

    # === Select individual ===

    print(f'=== 2rd to 7th row: ===\n{df_iter[2]}')

    date_from = "2022-01-01 00:02:00"
    date_to = "2022-01-01 00:08:00"
    print(f"Data range update: from{date_from}, to: {date_to}")
    df_iter.set_time_range(time_from=date_from, time_to=date_to)
    print(f'=== 4th to 9th row: ===\n{df_iter[2]}')
        