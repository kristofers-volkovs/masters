import datetime as dt
import os

from data_constructor import DataConstructor
from modules.data_iterator import DataIterator
from modules.data_processing import DataProcessor
from modules.component_manager import ComponentManager
from modules.models import TimePeriod


def process_windowed_pca_components():
    dc = DataConstructor()

    time_period = TimePeriod(dt_from=dt.datetime(2022, 1, 1), dt_to=dt.datetime(2023, 1, 1))
    df, tickers = dc.construct_df(time_period, filter_stablecoins=True)
    
    df_iter = DataIterator(df, reading_window=60 * 24, reading_step=15)

    # date_from = "2022-01-01 00:01:00"
    # date_to = "2022-02-01 00:00:00"
    # df_iter.set_time_range(time_from=date_from, time_to=date_to)

    file_dir = 'data/components_step15_window1day_2022-01-01_2023-01-01'
    manager = ComponentManager(file_dir)

    manager.proc_list(df_iter)
    manager.order_component_steps()
    
    print(f'Processed file dir: {file_dir}')
    

if __name__ == '__main__':
    process_windowed_pca_components()
