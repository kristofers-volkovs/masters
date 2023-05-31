from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from tqdm import tqdm
import datetime as dt

from modules.data_processing import DataProcessor
from modules.data_iterator import DataIterator
from modules.models import ComponentParams


class DataVisualizer:
    
    @staticmethod
    def gen_color():
        return np.random.random(), np.random.random(), np.random.random()
    
    @staticmethod
    def plot_long_range_data(df, tickers, amount):
        fig = plt.figure(figsize=(13, 8))
        ax = fig.add_subplot(111)    # The big subplot
        
        # Turn off axis lines and ticks of the big subplot
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
        
        # Set common labels
        ax.set_xlabel('Laiks', fontsize=14)
        ax.set_ylabel('Likme', fontsize=14)
        
        axs = fig.subplots(amount, 1)
        
        x = df['dtm']
        for idx in tqdm(range(amount), desc='Constructing graphs'):
            ticker = tickers[idx]
            col = DataVisualizer.gen_color()
            y = DataProcessor.integrate_data(df[ticker])
            
            axs[idx].plot(x, y, label=ticker, c=col)
            axs[idx].legend(fontsize=16)
            
            axs[idx].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            axs[idx].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()
        
    
    @staticmethod
    def plot_raw_rate_invese():
        fig = plt.figure(figsize=(8, 7))
        axs = fig.subplots(2, 1)
        
        col = DataVisualizer.gen_color()
        
        x = np.arange(1, 51)
        y = np.arange(1, 51)
        
        axs[0].plot(x, y, label='y = r', c=col, linewidth=3.0)
        axs[1].plot(x, 1/y, label='y = 1/r', c=col, linewidth=2.0)
        axs[0].legend(fontsize=16)
        axs[1].legend(fontsize=16)
        
        axs[0].set_title('ETHBTC', fontsize=16)
        axs[1].set_title('BTCETH', fontsize=16)
        
        plt.tight_layout()
        plt.show()
        
    
    @staticmethod
    def plot_log_rate_invese():
        fig = plt.figure(figsize=(8, 7))
        axs = fig.subplots(2, 1)
        
        col = DataVisualizer.gen_color()
        
        x = np.arange(1, 51)
        y = np.arange(1, 51)
        
        axs[0].plot(x, np.log(y), label='y = ln(r)', c=col, linewidth=3.0)
        axs[1].plot(x, np.log(1/y), label='y = ln(1/r)', c=col, linewidth=3.0)
        axs[0].legend(fontsize=16)
        axs[1].legend(fontsize=16)
        
        axs[0].set_title('ETHBTC', fontsize=16)
        axs[1].set_title('BTCETH', fontsize=16)
        
        plt.tight_layout()
        plt.show()
        
        
    @staticmethod
    def plot_log_exp(df, tickers):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ticker = 'ETHBTC'
        col = DataVisualizer.gen_color()
        
        x = df['dtm'].loc[300960:325440]        
        y = df[ticker].loc[300960:325440]
        
        ax.plot(x, y, label=ticker, c=col)
        ax.legend(fontsize=16)
        
        ax.set_ylabel('Likme', fontsize=16)

        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=7))

        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_bar_chart(data):
        fig = plt.figure()
        x = np.arange(len(data))
        
        plt.bar(x, data, width=0.5)
        for i in range(len(data)):
            plt.text(i, data[i], data[i], ha = 'center')
            
        plt.show()
    
    @staticmethod
    def plot_average_correlation(data):
        fig = plt.figure(figsize=(10, 5))
        x = np.arange(len(data))
        
        plt.plot(x, data)
        
        x_ticks = np.arange(0, len(data), 5)
        x_labels = [f'{t + 1}. faktors' for t in x_ticks]
        
        plt.xticks(x_ticks, x_labels, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.grid()
        plt.show()
    
    @staticmethod
    def plot_variance(data):
        fig = plt.figure(figsize=(12, 5))
        ax = fig.add_subplot(111)    # The big subplot
        
        # Turn off axis lines and ticks of the big subplot
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

        
        # Set common labels
        ax.set_xlabel('Faktori', fontsize=10)
        ax.set_ylabel('Izklaidrotā dispersija procentos', fontsize=10)
        
        axs = fig.subplots(1, 2)
        
        x = np.arange(len(data))
        axs[0].plot(x, data)
        axs[0].set_title('Grafiks ar visu faktoru dispersijām', fontsize=14)
        
        data_cut = data[:10]
        x = np.arange(len(data_cut))
            
        axs[1].bar(x, data_cut)
        for i in range(len(data_cut)):
            plt.text(i, data_cut[i], data_cut[i], ha = 'center')

        axs[1].set_title('Pirmo 10 faktoru dispersiju vērtības', fontsize=14)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_derivative_variance(data_dict, period_from, period_to):
        fig = plt.figure(figsize=(8, 5))
        axs = fig.subplots(2, 3)
        
        x = range(period_from, period_to)
        for idx, (ticker, var_list) in enumerate(data_dict.items()):
            idx_row = int(idx / 3)
            idx_col = idx % 3

            axs[idx_row, idx_col].plot(x, var_list, label=ticker)
            axs[idx_row, idx_col].legend()

        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_derivative_pca_variance(data):
        fig = plt.figure(figsize=(8, 3))
        axs = fig.subplots(1, 3)

        for idx, (key, var_list) in enumerate(data.items()):
            x = np.arange(len(var_list))

            axs[idx].plot(x, var_list, label=key)
            axs[idx].legend()

        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_pca_factors(df, factors, amount):
        fig = plt.figure(figsize=(15, 8))
        axs = fig.subplots(amount, 1)
        
        x = df['dtm']
        for idx in range(amount):
            label = f'{idx + 1}. faktors'
            col = DataVisualizer.gen_color()
            y = DataProcessor.integrate_data(factors[:, idx])
            
            axs[idx].plot(x, y, label=label, c=col)
            axs[idx].legend(fontsize=16)
            
            axs[idx].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            axs[idx].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()
        
    @staticmethod
    def plot_factor_reconstruction(df, reconstructed_data, tickers, amount):
        fig = plt.figure(figsize=(12, 20))
        ax = fig.add_subplot(111)    # The big subplot
        
        # Turn off axis lines and ticks of the big subplot
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
        
        # Set common labels
        ax.set_xlabel('Laiks', fontsize=14)
        ax.set_ylabel('Likmes mainīgums', fontsize=14)
        
        axs = fig.subplots(amount * 2, 1)
        
        x = df['dtm']
        for idx in range(0, amount * 2, 2):
            ticker = tickers[int(idx / 2)]
            col = DataVisualizer.gen_color()
            # y_orig = DataProcessor.integrate_data(df[ticker])
            # y_rec = DataProcessor.integrate_data(reconstructed_data[:, idx])
            
            y_orig = df[ticker]
            y_rec = reconstructed_data[:, idx]
            
            axs[idx].plot(x, y_orig, label='Oriģinālais grafiks', c=col)
            axs[idx + 1].plot(x, y_rec, label='Rekonstruētais grafiks', c=col)
            axs[idx].set_title(ticker, fontsize=16)
            
        for idx in range(amount * 2):
            axs[idx].legend(fontsize=12)
            axs[idx].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            axs[idx].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_coin_cut_factors(dtm, factors, factors_cut, amount):
        fig = plt.figure(figsize=(15, 8))
        axs = fig.subplots(amount, 1)
        
        for idx in range(amount):
            label = f'{idx + 1}. faktors'
            col_orig = DataVisualizer.gen_color()
            col_cut = DataVisualizer.gen_color()
            y_orig = DataProcessor.integrate_data(factors[:, idx])
            y_cut = DataProcessor.integrate_data(factors_cut[:, idx])
            
            axs[idx].plot(dtm, y_orig, label='Oriģinālais faktors', c=col_orig)
            axs[idx].plot(dtm, y_cut, label='Reducētu datu faktors', c=col_cut)            
            axs[idx].set_title(label, fontsize=16)
            axs[idx].legend(fontsize=12)
            
            axs[idx].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            axs[idx].xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_pca_factor_windows(components, ts_format, integrate=False):
        fig = plt.figure(figsize=(15, 10))
        axs = fig.subplots(2, 2)
        
        for component in tqdm(components, desc='Constructing graphs'):
            ts_param = ComponentParams.timestamps.value
            factors_param = ComponentParams.factors.value
            
            x = [dt.datetime.fromtimestamp(ts) for ts in component[ts_param]]
            factors = component[factors_param][:, :4]
            if integrate:
                factors = DataProcessor.integrate_data(factors)
            
            col = DataVisualizer.gen_color()
            
            for idx in range(4):  # Plotting the 4 factors with highest variance
                idx_row = int(idx / 2)
                idx_col = idx % 2

                axs[idx_row, idx_col].plot(x, factors[:, idx], c=col)
            
        for idx in range(4):  
            idx_row = int(idx / 2)
            idx_col = idx % 2
            axs[idx_row, idx_col].xaxis.set_major_formatter(mdates.DateFormatter(ts_format))
            # axs[idx_row, idx_col].xaxis.set_major_locator(mdates.HourLocator(interval=24))
            axs[idx_row, idx_col].set_title(f"{idx + 1}. faktors", fontsize=16)

        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()  
    
    @staticmethod
    def plot_pca_weight_windows(components, tickers, n_currencies=10, plot_mid_line=False):
        fig = plt.figure(figsize=(15, 10))
        
        weights_param = ComponentParams.weights.value
        
        # shape = (num_components, weights, currencies)
        weights = np.array([c[weights_param] for c in components])
        for idx in tqdm(range(n_currencies), desc='Constructing graph'):
            col = DataVisualizer.gen_color()
            weight_graph = weights[:, :2, idx].T
            ticker = tickers[idx]
            plt.plot(weight_graph[1], weight_graph[0], '-o', c=col, label=ticker)
        
        if plot_mid_line:
            x = [0] * 2
            min_weight = np.min(weights[:, 0, :10])
            max_weight = np.max(weights[:, 0, :10])
            padding = abs(min_weight * 0.05)
            y = [min_weight - padding, max_weight + padding]
            plt.plot(x, y, '--', c='r')
        
        plt.legend(fontsize=16)
        plt.tight_layout()
        plt.show()
        
    @staticmethod        
    def plot_factors_weights_over_time(components, df_iter, tickers, tickers_to_plot, amount=5, n_currencies=10):
        ts_param = ComponentParams.timestamps.value
        weights_param = ComponentParams.weights.value
        factors_param = ComponentParams.factors.value
        
        col_1st = DataVisualizer.gen_color()
        col_2nd = DataVisualizer.gen_color()
        col_coins = [DataVisualizer.gen_color() for _ in range(n_currencies)]
        
        coin_weights = [[], []]  # shape = (x, y)
        for comp in components:
            for c_idx in range(n_currencies):
                weights = comp[weights_param]  # shape = (weights, currencies)
                c_map_idx = np.where(tickers == tickers_to_plot[c_idx])[0]
                weights = weights[:2, c_map_idx]
                coin_weights[0].append(weights[0])
                coin_weights[1].append(weights[1])

        weight_x_lim = [np.min(coin_weights[0]), np.max(coin_weights[0])]  # (min, max)
        weight_y_lim = [np.min(coin_weights[1]), np.max(coin_weights[1])]  # (min, max)
        
        max_iter = int(len(components) / amount)
        for idx_iter in tqdm(range(max_iter), desc='Constructing graphs'):
            fig = plt.figure(figsize=(15, 7))
            axs = fig.subplots(3, amount)
            
            for idx in range(amount):
                df_window = df_iter[idx + idx_iter * amount]
                comp = components[idx + idx_iter * amount]
                factors = comp[factors_param]  # shape = (window, factors)

                for c_idx in range(n_currencies):

                    start = 0
                    if idx_iter > 0:
                        start = idx_iter * amount - 4 + idx
                    end = idx + amount * idx_iter + 1
                    
                    coin_line = [[], []]  # shape = (x, y)
                    for w_idx in range(start, end):
                        w_comp = components[w_idx]
                        weights = w_comp[weights_param]  # shape = (weights, currencies)
                        c_map_idx = np.where(tickers == tickers_to_plot[c_idx])[0]
                        weights = weights[:2, c_map_idx]
                        coin_line[0].append(weights[0])
                        coin_line[1].append(weights[1])
                    
                    col = col_coins[c_idx]
                    ticker = tickers_to_plot[c_idx]
                    
                    axs[0, idx].set_xlim(weight_x_lim)
                    axs[0, idx].set_ylim(weight_y_lim)
                    axs[0, idx].plot(coin_line[0], coin_line[1], '-o', c=col, label=ticker)

                x = [dt.datetime.fromtimestamp(ts) for ts in comp[ts_param]]

                fac1_data = DataProcessor.integrate_data(comp['factors'][:, 0])
                fac2_data = DataProcessor.integrate_data(comp['factors'][:, 1])

                axs[1, idx].plot(x, fac1_data, c=col_1st)
                axs[2, idx].plot(x, fac2_data, c=col_2nd)

                for plot_idx in range(1, 3):    
                    axs[plot_idx, idx].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    axs[plot_idx, idx].xaxis.set_major_locator(mdates.DayLocator(interval=1))
            
            axs[0, 0].set_ylabel('Svari', fontsize=16)
            axs[1, 0].set_ylabel('1. faktors', fontsize=16)
            axs[2, 0].set_ylabel('2. faktors', fontsize=16)
            
            fig.autofmt_xdate()
            plt.tight_layout()
            plt.savefig(f'factor_weight_anim_{idx_iter+1}')
        
    @staticmethod        
    def plot_diff_component_window_size_weights(manager_a, manager_b, time_range):
        # TODO refactor this function
        # collect all the needed variables before plotting the graphs
        
        fig = plt.figure()
        # fig = plt.figure(figsize=(15, 10))
        axs = fig.subplots(2, 3)
        
        time_from = time_range.dt_from
        time_to = time_range.dt_to
        weight_key = ComponentParams.weights.value
        # Calculates idx step for days and weeks
        day_step = int((60 / manager_a.reading_step) * 24)
        week_step = day_step * 7
        
        time_diff_weeks = int((time_to - time_from).total_seconds() / (60 * 60 * 24 * 7))
        
        # Creates lists with idxes for weekly and daily samples
        week_idxes = [d * week_step for d in range(time_diff_weeks)]
        day_idxes = [d * day_step for d in range((time_diff_weeks - 1) * 7 + 1)]
        
        managers = [manager_a, manager_b]
        time_step_idxes = [week_idxes, day_idxes]
        label_text = ['1 nedēļas logs', '1 dienas logs']
        for idx in range(len(managers)): 
            time_idxes = time_step_idxes[idx]
            idx_from = manager_a.get_component_idx_by_timestamp(time_from)
            
            weight_list = []
            for step_idx in time_idxes:
                time_idx = idx_from + step_idx
                weights = managers[idx].get_item(time_idx, [weight_key])
                weights = weights[weight_key]
                
                if idx == 0:
                    weights[0] = -weights[0]  # The first factor is flipped

                weight_list.append(weights[:2, :6])

            weight_list = np.array(weight_list)
            
            for coin_idx in range(len(weight_list.T)):
                idx_row = int(coin_idx / 3)
                idx_col = coin_idx % 3

                coin_weights = weight_list.T[coin_idx]
                x = coin_weights[0]
                y = coin_weights[1]

                axs[idx_row, idx_col].plot(x, y, '.-', label=label_text[idx])
                axs[idx_row, idx_col].set_title(f'{manager_a.tickers[coin_idx]} svaru grafiks')
                axs[idx_row, idx_col].legend()
                # axs[idx_row, idx_col].legend(fontsize=16)
                
                if idx == 0:
                    for idx_day in range(len(x)):
                        axs[idx_row, idx_col].text(x[idx_day], y[idx_day] + 0.001, str(idx_day), ha='center')
                else:
                    for idx_day in range(int(len(x) / 4) - 1):
                        axs[idx_row, idx_col].text(x[idx_day * 7], y[idx_day * 7] + 0.001, str(idx_day), ha='center')
            
        plt.tight_layout()
        plt.show()
        
    @staticmethod
    def plot_3d_point_cloud(data):
        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(projection='3d')
        
        x = data[:, 0]
        y = data[:, 1]
        z = data[:, 2]
        ax.scatter3D(x, y, z, s=45)
        
        ax.set_xlabel('X', fontsize=15)
        ax.set_ylabel('Y', fontsize=15)
        ax.set_zlabel('Z', fontsize=15)
        
        plt.tight_layout()
        plt.show()
        
    @staticmethod
    def plot_3d_labeled_point_cloud(data, labels):
        def plot_3d_scatter(data, color):
            x = label_data[:, 0]
            y = label_data[:, 1]
            z = label_data[:, 2]
            ax.scatter3D(x, y, z, color=color, s=45)
        
        fig = plt.figure(figsize=(12, 8))
        ax = plt.axes(projection='3d')
        
        label_count = max(labels) + 1
        for label in range(label_count):
            idxes = np.where(labels == label)[0]
            label_data = data[idxes]
            
            plot_3d_scatter(label_data, color=DataVisualizer.gen_color())
        
        if -1 in labels:
            idxes = np.where(labels == -1)[0]
            label_data = data[idxes]
            
            plot_3d_scatter(label_data, color='gray')
        
        ax.set_xlabel('X', fontsize=15)
        ax.set_ylabel('Y', fontsize=15)
        ax.set_zlabel('Z', fontsize=15)
        
        fig.canvas.capture_scroll = True
        fig.canvas.toolbar_visible = True
        plt.tight_layout()
        plt.show()
