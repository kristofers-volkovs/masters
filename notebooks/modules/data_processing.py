import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm
from joblib import delayed
import datetime as dt
import os

from modules.models import PreprocessingParams, DiffParams
from modules.progress_parallel import ProgressParallel
    
    
class DataProcessor:
    """
    Params
        df - pandas.DataFrame
    """
    @staticmethod
    def data_preprocessor(df, tickers, params=PreprocessingParams(), log_steps=True):
        df_out = df.copy()
        df_tickers = df_out.loc[:, tickers]
        if params.log:
            if log_steps:
                print('=== Log ===')
            df_tickers = np.log(df_tickers)
        if params.diff.is_true:
            if log_steps:
                print('=== Diff ===')
            df_tickers = df_tickers.diff(axis = 0, periods = params.diff.period)

            df_idx = df_tickers.index[0]
            rows_to_drop = np.arange(df_idx, df_idx + params.diff.period)
            
            df_tickers = df_tickers.drop(rows_to_drop)
            df_out = df_out.drop(rows_to_drop)
            
            df_out = df_out.reset_index(drop=True)
            df_tickers = df_tickers.reset_index(drop=True)
        if params.std:
            if log_steps:
                print('=== Std normalization ===')
            std_val = np.std(df_tickers)
            std_val.replace(0.0, 1e-8, inplace=True)

            df_tickers /= std_val

        df_out.loc[:, tickers] = df_tickers

        return df_out
    
    """
    Params
        df - pandas.DataFrame
    """
    @staticmethod
    def data_process_pca(df, tickers, n_components=None):
        df_tickers = df.loc[:, tickers]

        pca = PCA(n_components=n_components)
        factors = pca.fit_transform(df_tickers)
        weights = pca.components_
        variance = pca.explained_variance_ratio_
        
        return factors, weights, variance

    """
    Params
       factors.shape - (windows, factors)
       weights.shape - (factors, coins)
    """
    @staticmethod
    def data_reconstruct_pca(factors, weights):
        return np.dot(factors, weights)
    
    """
    Params
       factors.shape - (windows, factors)
    """
    @staticmethod
    def integrate_data(factors):
        return np.cumsum(factors, axis=0)
    
    @staticmethod
    def variance_round(variance):
        return list(np.round_(variance * 100, decimals=2))
    
    @staticmethod
    def calc_derivative_variance_over_steps(df, tickers: list[str], period_from: int, period_to: int):
        ticker_variance_dict = {}
        for ticker in tqdm(tickers, desc='Calculating variance for each ticker'):

            variance_list = []
            for period in range(period_from, period_to):
                if period == 0:
                    variance = np.sqrt(np.std(df[ticker].copy()))
                    variance_list.append(variance)
                else:
                    df_diff = df[ticker].copy().diff(periods=period) 

                    df_idx = df_diff.index[0]
                    rows_to_drop = np.arange(df_idx, df_idx + period)
                    df_diff = df_diff.drop(rows_to_drop)

                    variance = np.sqrt(np.std(df_diff))
                    variance_list.append(variance)

            ticker_variance_dict[ticker] = variance_list

        return ticker_variance_dict
    
    @staticmethod
    def calc_derivative_pca_over_steps(df, tickers: list[str], period_from: int, period_to: int):        
        dict_variance_list = {'factor_1': [], 'factor_2': [], 'factor_3': []}
        for period in tqdm(range(period_from, period_to), desc='Calculating PCA for each diff period'):
            diff_params = DiffParams(period=period)
            proc_params = PreprocessingParams(log=False, diff=diff_params)
            df_proc = DataProcessor.data_preprocessor(df.loc[:, tickers], tickers, params=proc_params, log_steps=False)

            _factors, _weights, variance = DataProcessor.data_process_pca(df_proc, tickers)

            dict_variance_list['factor_1'].append(variance[0])
            dict_variance_list['factor_2'].append(variance[1])
            dict_variance_list['factor_3'].append(variance[2])
        
        return dict_variance_list

    # @staticmethod
    # def proc_pca_component(df, tickers, n_components):
    #     factors, weights, variance = DataProcessor.data_process_pca(df, tickers)
    #     return ComponentWindow(
    #         timestamps=np.array(list(df['dtm'])),
    #         factors=factors,
    #         weights=weights,
    #         variance=variance,
    #     )
    
    @staticmethod
    def proc_pca_components(idx, df, tickers, data_storage, n_components):
        factors, weights, variance = DataProcessor.data_process_pca(df, tickers)
        timestamps = np.array([dt.datetime.timestamp(t) for t in list(df['dtm'])])
        
        proc_component = {
            'timestamps': timestamps,
            'factors': factors,
            'weights': weights,
            'variance': variance,
        }
        
        data_storage.add_item(idx, proc_component)
    
    @staticmethod
    def calc_parallel_component_windows(df_iter, data_storage, n_components):
        tickers = df_iter.get_tickers()
        num_cores = 24
        
        parallel_params = {
            'n_jobs': num_cores,
            'total': len(df_iter),
            'desc': 'Processing components',
        }
        
        os.environ["JOBLIB_TEMP_FOLDER"] = "/home/jovyan/work/Kristofers/data/tmp"
        
        ProgressParallel(**parallel_params)(delayed(DataProcessor.proc_pca_components)(idx=idx, df=df_window, tickers=tickers, data_storage=data_storage, n_components=n_components) for idx, df_window in enumerate(df_iter))
    
    @staticmethod
    def calc_sequential_component_windows(df_iter, data_storage, n_components):
        tickers = df_iter.get_tickers()
        
        for idx, df_window in enumerate(tqdm(df_iter, desc='Processing components')):
            DataProcessor.proc_pca_component(
                idx=idx, 
                df=df_window, 
                tickers=tickers, 
                data_storage=data_storage, 
                n_components=n_components,
            )
    
    @staticmethod
    def calc_pca_component_windows(df_iter, data_storage, n_components=None):
        if len(df_iter) > 100:
            # Processes all the windows in parallel and saves it to disc
            DataProcessor.calc_parallel_component_windows(
                df_iter=df_iter, 
                data_storage=data_storage, 
                n_components=n_components,
            )
        else:
            # It's faster to process data one by one when the amount of data is small
            DataProcessor.calc_sequential_component_windows(
                df_iter=df_iter, 
                data_storage=data_storage, 
                n_components=n_components,
            )
        