import csv
import datetime
import time
import multiprocessing
from joblib import Parallel, delayed
from enum import Enum
import numpy as np
from dataclasses import dataclass, field
from tqdm import tqdm
import matplotlib

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.image
import matplotlib.dates as mdates

import plotly.express as px
import plotly.graph_objects as go

from modules.utils import reconstruct_data_with_pca, generate_color, calculate_component_with_pca, data_preprocessing
from modules.data_fetch import fetch_coin_data
from modules.data_loader import DataLoader
from modules.component_storage import ComponentStorage
from modules.data_classes import Triangle


def crypto_triangulation():
    dataloader = DataLoader(file_path="data/from_2022-09-07_to_2022-08-08.csv", filter_coins=False)
    # time_from = "2022-06-25T18:09Z"
    # time_to = "2022-07-08T18:09Z"
    # dataloader.set_from_to_timestamp(time_from=time_from, time_to=time_to)

    tickers = ["ETHUSDT", "BTCUSDT", "ETHBTC"]
    (coin_ab, coin_bc, coin_ca) = tickers
    ab_idx = np.where(coin_ab == dataloader.ticker_data)[0][0]
    bc_idx = np.where(coin_bc == dataloader.ticker_data)[0][0]
    ca_idx = np.where(coin_ca == dataloader.ticker_data)[0][0]

    triangle_sequence = []
    for (coin_data, timestamp) in tqdm(dataloader):
        data_log = np.log(np.array(coin_data))
        # data_diff = np.diff(data_log.T)
        std_val = np.std(data_log.T, axis=1)
        (ab_len, bc_len, ca_len) = (std_val[ab_idx], std_val[bc_idx], std_val[ca_idx])

        # Checks if the triangle inequality condition is met
        ab_condition = bc_len + ca_len > ab_len
        bc_condition = ab_len + ca_len > bc_len
        ca_condition = ab_len + bc_len > ca_len

        if ab_condition and bc_condition and ca_condition:
            # Calculates the triangle angles using the law of cosines
            angle_A = np.arccos((ab_len ** 2 + ca_len ** 2 - bc_len ** 2) / (2 * ab_len * ca_len))
            angle_B = np.arccos((ab_len ** 2 + bc_len ** 2 - ca_len ** 2) / (2 * ab_len * bc_len))
            angle_C = np.pi - angle_A - angle_B

            point_A = [0, 0]
            c_x = (ca_len * np.tan(angle_C)) / (np.tan(angle_A) + np.tan(angle_C))
            c_y = c_x * np.tan(angle_A)
            point_B = [c_x, c_y]
            point_C = [ca_len, 0]

            points = np.array([point_A, point_B, point_C])
            angles = np.array([angle_A, angle_B, angle_C])
            lengths = np.array([ab_len, bc_len, ca_len])

            triangle_sequence.append(Triangle(points=points, side_lengths=lengths, angles=angles, timestamp=timestamp))
        else:
            raise ValueError(f"Triangle inequality condition was not met, timestamp: {timestamp}")

    angles = np.array([t.angles for t in triangle_sequence])
    angles = np.round_(angles / np.pi * 180, 2)
    a_angles = [a[0] for a in angles]
    b_angles = [a[1] for a in angles]
    c_angles = [a[2] for a in angles]
    a_angle_min = np.min(a_angles)
    b_angle_min = np.min(b_angles)
    c_angle_min = np.min(c_angles)
    a_angle_max = np.max(a_angles)
    b_angle_max = np.max(b_angles)
    c_angle_max = np.max(c_angles)
    a_angle_mean = np.round_(np.mean(a_angles), 2)
    b_angle_mean = np.round_(np.mean(b_angles), 2)
    c_angle_mean = np.round_(np.mean(c_angles), 2)

    print(f"Min angles: A: {a_angle_min}, B: {b_angle_min}, C: {c_angle_min}")
    print(f"Mean angles: A: {a_angle_mean}, B: {b_angle_mean}, C: {c_angle_mean}")
    print(f"Max angles: A: {a_angle_max}, B: {b_angle_max}, C: {c_angle_max}")

    x_max = np.max([t.points[:, 0] for t in triangle_sequence])
    y_max = np.max([t.points[:, 1] for t in triangle_sequence])

    # edge_labels = ["A", "B", "C"]
    # xytext = np.array([[-10, -15], [0, 10], [10, -15]])
    line_idxes = [[0, 1], [1, 2], [2, 0]]
    # text_rotation = [angle_A / np.pi * 180, -abs(angle_C - 180) / np.pi * 180, 0]

    start = 0
    stop = len(triangle_sequence)
    step = 4 * 24

    col_count = 7
    row_count = int((stop - start) / step / col_count)
    # col_count = 5
    # row_count = 1
    idx_frame = start
    fig, ax = plt.subplots(row_count, col_count, figsize=(20, 10))
    # fig, ax = plt.subplots(row_count, col_count, figsize=(25, 6))
    for row_idx in range(row_count):
        idx_gen = range(col_count)
        if row_idx % 2 == 1:
            idx_gen = reversed(range(col_count))

        for col_idx in idx_gen:
            ax[row_idx, col_idx].set_xlim([-0.0002, x_max])
            ax[row_idx, col_idx].set_ylim([-0.0002, y_max])

            triangle = triangle_sequence[idx_frame]
            for edge_idx in range(0, 3):
                (idx_1, idx_2) = line_idxes[edge_idx]
                point_1 = triangle.points[idx_1]
                point_2 = triangle.points[idx_2]
                line = np.array([point_1, point_2]).T
                midpoint = [(point_1[0] + point_2[0]) / 2, (point_1[1] + point_2[1]) / 2]
                angle = np.round_(triangle.angles[edge_idx] / np.pi * 180, 2)
                side_len = np.round_(triangle.side_lengths[edge_idx], 4)
                ts = datetime.datetime.strptime(triangle.timestamp, '%Y-%m-%dT%H:%MZ').strftime('%d-%m-%Y')
                ts_pos_bad = (0.01, 0.034)
                ts_pos_good = (0.0005, 0.0035)

                ax[row_idx, col_idx].plot(line[0], line[1], label=f"{tickers[edge_idx]}", lw=2)
                ax[row_idx, col_idx].plot(point_1[0], point_1[1], 'o', color='black')
                ax[row_idx, col_idx].text(ts_pos_bad[0], ts_pos_bad[1], f"{ts}")  # timestamp
                # ax[row_idx, col_idx].text(midpoint[0], midpoint[1], f"{side_len}")  # edge angle
                # ax[row_idx, col_idx].text(point_1[0], point_1[1], f"{angle}")  # lengths

            # ax[row_idx, col_idx].legend()

            idx_frame += step

    plt.tight_layout()
    plt.savefig('./img/bad_triangle.png')

    exit()

    fig = plt.figure(figsize=(15, 10))
    ax = plt.axes(xlim=(-0.0002, x_max), ylim=(-0.0002, y_max))
    line_plots = [ax.plot([], [], label=f"{tickers[i]}", lw=3) for i in range(0, 3)]
    # edge_label_plots = [ax.text(0, 0, f'{edge_labels[i]}') for i in range(0, 3)]
    edge_point_plots = [ax.plot([], [], 'o', color='black') for _ in range(0, 3)]
    edge_angle_plots = [ax.text(0, 0, '') for _ in range(0, 3)]
    lengths_text_plots = [ax.text(0, 0, '') for _ in range(0, 3)]
    ax.legend()

    def update(idx_frame):
        triangle = triangle_sequence[idx_frame]
        for idx_plot in range(len(line_plots)):
            (idx_1, idx_2) = line_idxes[idx_plot]
            point_1 = triangle.points[idx_1]
            point_2 = triangle.points[idx_2]
            line = np.array([point_1, point_2]).T
            midpoint = [(point_1[0] + point_2[0]) / 2, (point_1[1] + point_2[1]) / 2]
            angle = np.round_(triangle.angles[idx_plot] / np.pi * 180, 2)
            side_len = np.round_(triangle.side_lengths[idx_plot], 4)

            line_plots[idx_plot][0].set_data(line[0], line[1])
            edge_point_plots[idx_plot][0].set_data(point_1[0], point_1[1])
            lengths_text_plots[idx_plot].set(x=midpoint[0], y=midpoint[1], text=f"{side_len}")
            # edge_label_plots[idx_plot].set(x=point_1[0], y=point_1[1])
            edge_angle_plots[idx_plot].set(x=point_1[0], y=point_1[1], text=f"{angle}")

    animation_idxes = np.arange(0, len(triangle_sequence), 1)
    idx_step = int(len(triangle_sequence) / 3)
    for idx in tqdm(range(3)):
        end_idx = (idx * idx_step) + idx_step
        if end_idx >= len(animation_idxes):
            end_idx = -1

        ani = FuncAnimation(fig, update, animation_idxes[idx * idx_step:end_idx], interval=200)
        writer = PillowWriter(fps=25)
        ani.save(f"coin_dynamics_{idx}.gif", writer=writer)


# def correlation_for_idx(dataloader, idx):
#     time_res = int(60 / 15) * 24 * 7
#     correlation_map = np.zeros(time_res)
#
#     for t in range(3, time_res):
#         (data, _) = dataloader[idx:idx + t]
#
#         norm_data = data_preprocessing(data[:, 0])
#         # eth_volume = data[:, 1]
#         # usd_volume = data[:, 2]
#         # volume_data = np.sqrt(eth_volume * usd_volume)[:-1]
#         norm_no_diff_data = data_preprocessing(data[:, 0], diff=False)[:-1]
#
#         data_corr = np.corrcoef(norm_data, norm_no_diff_data)
#         correlation_map[t] = data_corr[0, -1] ** 2
#
#     return correlation_map


class PreProcessingComb(Enum):
    TYPE_A = 1
    TYPE_B = 2
    TYPE_C = 3


def lists_of_stds(dataloader, combination: PreProcessingComb) -> (list, list):
    list_a = []
    list_b = []
    for (data, _) in dataloader:
        data_a = []
        data_b = []
        if combination == PreProcessingComb.TYPE_A:
            data_a = data_preprocessing(data[:, 0], std=False)
            data_b = data_preprocessing(data[:, 0], diff=False, std=False)
        if combination == PreProcessingComb.TYPE_B:
            data_a = data_preprocessing(data[:, 0], std=False)
            data_b = np.sqrt(data[:, 1] * data[:, 2])  # data[:, 1] == eth_volume, data[:, 2] == usd_volume
        if combination == PreProcessingComb.TYPE_C:
            data_a = data_preprocessing(data[:, 0], diff=False, std=False)
            data_b = np.sqrt(data[:, 1] * data[:, 2])  # data[:, 1] == eth_volume, data[:, 2] == usd_volume

        data_a_std = np.std(data_a, axis=0)
        data_b_std = np.std(data_b, axis=0)

        list_a.append(data_a_std)
        list_b.append(data_b_std)

    return list_a, list_b


def plot_window_stds():
    # dataloader = DataLoader(file_path="data/from_2021-15-02_to_2022-17-08.csv")
    # dataloader = DataLoader(file_path="data/from_2021-16-02_to_2022-18-08_ETHUSDT_volume.csv")
    dataloader = DataLoader(reading_step=15, file_path="data/from_2021-21-02_to_2022-23-08_ETHUSDT_volume.csv")
    # time_from = "2022-06-15T18:09Z"
    # time_to = "2022-07-01T18:09Z"
    # dataloader.set_from_to_timestamp(time_from=time_from, time_to=time_to)

    combination = PreProcessingComb.TYPE_A
    readings_day = 60 * 24
    # readings_day = int(60 / 15) * 24
    # readings_day = int(60 / 15) + 10
    num_cores = multiprocessing.cpu_count()
    corr_list = Parallel(n_jobs=num_cores)(
        delayed(correlation_for_idx)(dataloader=dataloader, idx=idx, combination=combination)
        for idx in tqdm(range(3, readings_day), desc="Calculating corr"))

    ind = np.arange(0, len(corr_list))
    idx_amount = ind[np.array(corr_list) > 0.7][0] + 3

    print(f'First plot window len: {idx_amount}')

    dataloader.set_reading_amount(idx_amount)
    (list_a, list_b) = lists_of_stds(dataloader, combination)

    plt.scatter(list_a, list_b, marker='.')
    plot_str = ""
    x_label = ""
    y_label = ""
    if combination == PreProcessingComb.TYPE_A:
        x_label = "stdev(diff ln(ETHUSDT))"
        y_label = "stdev(ln(ETHUSDT))"
        plot_str = "pointA_scatter"
    if combination == PreProcessingComb.TYPE_B:
        x_label = "stdev(diff ln(ETHUSDT))"
        y_label = "sqrt(volume(ETHUSDT, in ETH)*volume(ETHUSDT, in USDT))"
        plot_str = "pointB_scatter"
    if combination == PreProcessingComb.TYPE_C:
        x_label = "stdev(ln(ETHUSDT))"
        y_label = "sqrt(volume(ETHUSDT, in ETH)*volume(ETHUSDT, in USDT))"
        plot_str = "pointC_scatter"
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f"Window len: {idx_amount / 60} h")
    plt.tight_layout()
    plt.savefig(f"{plot_str}_len{idx_amount}_plot.png")

    fig = plt.figure(figsize=(20, 12))
    axs = fig.subplots(2, 3)
    for idx, amount in enumerate(list(np.arange(1, 7) * readings_day)):
        dataloader.set_reading_amount(amount)
        (list_a, list_b) = lists_of_stds(dataloader, combination)

        axs[int(idx / 3), idx % 3].scatter(list_a, list_b)
        axs[int(idx / 3), idx % 3].set_title(f"Window len: {amount / 60} h")
        axs[int(idx / 3), idx % 3].set_xlabel(x_label)
        axs[int(idx / 3), idx % 3].set_ylabel(y_label)

    plt.tight_layout()
    # plt.show()
    fig.savefig(f"{plot_str}_plot.png")


def correlation_for_idx(dataloader, idx, combination: PreProcessingComb):
    dataloader.set_reading_amount(idx)
    (list_a, list_b) = lists_of_stds(dataloader, combination)

    data_corr = np.corrcoef(list_a, list_b)
    return data_corr[0, -1]


def calculate_correlation_of_methods():
    # dataloader = DataLoader(file_path="data/from_2021-15-02_to_2022-17-08.csv")
    # dataloader = DataLoader(file_path="data/from_2021-16-02_to_2022-18-08_ETHUSDT_volume.csv")
    dataloader = DataLoader(reading_step=15, file_path="data/from_2021-21-02_to_2022-23-08_ETHUSDT_volume.csv")
    # time_from = "2022-06-15T18:09Z"
    # time_to = "2022-07-01T18:09Z"
    # dataloader.set_from_to_timestamp(time_from=time_from, time_to=time_to)

    # time_res = int(60 / 15) * 24 * 7
    time_res = 60 * 24
    num_cores = multiprocessing.cpu_count()
    corr_list = Parallel(n_jobs=num_cores)(
        delayed(correlation_for_idx)(dataloader=dataloader, idx=idx, combination=PreProcessingComb.TYPE_C)
        for idx in tqdm(range(3, time_res), desc="Calculating corr"))

    max_idx = np.argmax(corr_list) + 3
    print(f"Max idx: {max_idx}")
    print(f"Max corr: {corr_list[max_idx - 3]}")

    # --- Correlation plot
    plt.figure(figsize=(20, 20))
    x = np.arange(3, time_res) / (60 * 24)
    plt.plot(x, corr_list)
    plt.show()

    # split_amount = 10
    # total_len = len(dataloader) - time_res
    # step = int(total_len / split_amount)
    # for i in range(0, split_amount):
    #     start = i * step
    #     end = start + step
    #     if i + 1 == split_amount:
    #         end = len(dataloader) - time_res
    #
    #     correlation_map = np.zeros((time_res, end - start))
    #     num_cores = multiprocessing.cpu_count()
    #     processed_map = Parallel(n_jobs=num_cores)(
    #         delayed(correlation_for_idx)(dataloader=dataloader, idx=idx)
    #         for idx in tqdm(range(start, end), desc="Calculating map"))
    #
    #     for idx in tqdm(range(len(processed_map)), desc="Arranging map"):
    #         correlation_map[:, idx] = processed_map[idx]
    #
    #     # Flip so that the negative k values are at the bottom
    #     correlation_map = np.flip(correlation_map, axis=0)
    #
    #     # --- Plot pixel perfect correlation maps
    #     my_dpi = 96
    #     (height, width) = (len(correlation_map), len(correlation_map[0]))
    #
    #     fig = plt.figure(figsize=(int(width / my_dpi), int(height / my_dpi)), dpi=my_dpi)
    #     ax = fig.add_axes([0, 0, 1, 1])
    #     ax.axis('off')
    #     ax.imshow(correlation_map, interpolation='nearest')
    #     ax.set(xlim=[-0.5, width - 0.5], ylim=[height - 0.5, -0.5], aspect=1)
    #     fig.savefig(f'method_correlation{i}_plot.png', dpi=my_dpi)
    #
    #     # --- Save x axis dates
    #     # plt.figure(figsize=(int(width / my_dpi), int(height / my_dpi)), dpi=my_dpi)
    #     #
    #     # dates = dataloader.get_timestamps()[start:end]
    #     # date_idxes = np.linspace(0, end - start - 1, 20)
    #     # dates = [dates[int(d_idx)] for d_idx in date_idxes]
    #     # x = [datetime.datetime.strptime(d, '%Y-%m-%dT%H:%MZ').date() for d in dates]
    #     # y = range(len(x))
    #     # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%dT%H:%MZ'))
    #     # plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    #     # plt.plot(x, y)
    #     # plt.gcf().autofmt_xdate()
    #     # plt.tight_layout()
    #     # plt.savefig(f"method_correlation{i}_plot_dates", dpi=my_dpi)
    #
    # # --- Plot maps in a graph, get colorbar
    # # plt.figure(figsize=(20, 20))
    # # plt.axis('off')
    # # im = plt.imshow(correlation_map, interpolation='nearest')
    # # plt.colorbar(im, orientation="horizontal")
    # # plt.show()


def calculate_correlation_for_idx(dataloader, idx):
    def calculate_correlation_for_every_k(k_idxes, correlation_map, cut_week):
        for k_idx, k in enumerate(k_idxes):
            if idx - int(c * 2 ** k) < 0 or idx + int(c * 2 ** k) > len(dataloader):
                break

            if not cut_week:
                k_idx += int(k_end / k_step) + 1

            (data_week, _) = dataloader[idx - c:idx + c]
            (data_cut, _) = dataloader[idx - int(c * 2 ** k):idx + int(c * 2 ** k)]

            component_week = calculate_component_with_pca(coin_data=data_week)
            component_cut = calculate_component_with_pca(coin_data=data_cut)
            factors_week = component_week.factors
            factors_cut = component_cut.factors

            reading_gap = int(abs(len(factors_week) - len(factors_cut)) / 2)
            if cut_week:
                factors_week = factors_week[reading_gap:-reading_gap]
            else:
                factors_cut = factors_cut[reading_gap:-reading_gap]

            for idx_factor in range(count_factors):
                factor_corr = np.corrcoef(factors_cut[:, idx_factor], factors_week[:, idx_factor])
                correlation_map[idx_factor, k_idx, 0] = factor_corr[0, -1] ** 2

    c = int(int(60 / 15) * 24 * 3.5)  # half a week worth of readings
    (k_start, k_end, k_step) = (-2, 2, 0.1)  # k changes temporal resolution
    # generates the ranges for positive and negative k values
    k_neg = np.arange(k_start, 0, k_step)
    k_pos = np.arange(k_step, k_end, k_step)
    # the base 2 calculation system produces odd floating point numbers so they are rounded
    k_neg = np.round_(k_neg, 1)
    k_pos = np.round_(k_pos, 1)

    count_factors = 4

    correlation_map = np.zeros((count_factors, int(k_end / k_step) * 2, 1))
    calculate_correlation_for_every_k(k_idxes=k_neg, correlation_map=correlation_map, cut_week=True)
    calculate_correlation_for_every_k(k_idxes=k_pos, correlation_map=correlation_map, cut_week=False)

    # the same resolution will correlate perfectly with itself
    correlation_map[:, [int(k_end / k_step)], 0] = 1

    return correlation_map


def calculate_factor_ranges():
    c = int(int(60 / 15) * 24 * 3.5)  # half a week worth of readings
    (k_start, k_end, k_step) = (-2, 2, 0.1)  # k changes temporal resolution

    count_factors = 4

    dataloader = DataLoader()
    # time_from = "2022-06-25T18:09Z"
    # time_to = "2022-07-08T18:09Z"
    # dataloader.set_from_to_timestamp(time_from=time_from, time_to=time_to)
    correlation_map = np.zeros((count_factors, abs(k_end * 20), len(dataloader) - c * 2))

    num_cores = multiprocessing.cpu_count()
    processed_map = Parallel(n_jobs=num_cores)(
        delayed(calculate_correlation_for_idx)(dataloader=dataloader, idx=idx)
        for idx in tqdm(range(c, len(dataloader) - c), desc="Calculating map"))

    for idx in tqdm(range(len(processed_map)), desc="Arranging map"):
        correlation_map[:, :, idx] = processed_map[idx][:, :, 0]

    # Flip so that the negative k values are at the bottom
    correlation_map = np.flip(correlation_map, axis=1)

    # --- Plot pixel perfect correlation maps
    # for idx in range(count_factors):
    #     matplotlib.image.imsave(f"factor{idx}_test_plot.png", correlation_map[idx])

    # --- Plot maps in a graph, get colorbar
    fig = plt.figure(figsize=(20, 20))
    axs = fig.subplots(4)
    for idx in range(count_factors):
        axs[idx].axis('off')
        im = axs[idx].imshow(correlation_map[idx], interpolation='nearest')
        axs[idx].set_title(f"Factor {idx + 1}")
        if idx == 0:
            fig.colorbar(im, orientation="horizontal")
    plt.show()
    # fig.savefig("4factor_r2_plot")


def calculate_factors_over_time():
    # start = time.time()

    # dataloader = DataLoader(reading_step=int(60 / 15) * 24 * 6, reading_amount=int(60 / 15) * 24 * 6)
    dataloader = DataLoader()
    time_from = "2022-06-25T18:09Z"
    time_to = "2022-07-09T18:09Z"
    # time_to = "2022-07-01T18:09Z"
    dataloader.set_from_to_timestamp(time_from=time_from, time_to=time_to)
    storage = ComponentStorage()
    reading_step = dataloader.reading_step
    reading_amount = dataloader.reading_amount

    wrapped_dataloader = tqdm(dataloader, desc="Storing components")
    storage.add_data(dataloader=wrapped_dataloader)

    storage.add_tickers(tickers=dataloader.ticker_data)
    storage.sort_timestamps()
    storage.sort_step_order(reading_step=reading_step)
    factor_list, weight_list = storage.get_ordered_component_list(integrate=True, normalize=True)
    timestamp_list = storage.get_timestamp_list()

    # end = time.time()
    # print(f"Time: {end - start}")

    # ----

    # time_from = "2022-06-25T18:09Z"
    # # time_to = "2022-07-09T18:09Z"
    # time_to = "2022-07-01T18:09Z"
    # dataloader.set_from_to_timestamp(time_from=time_from, time_to=time_to)
    # wrapped_dataloader = tqdm(dataloader, desc="Storing components")
    # storage.add_data(dataloader=wrapped_dataloader, n_components=4)
    #
    # storage.add_tickers(tickers=dataloader.ticker_data)
    # week_factors, week_weights = storage.get_ordered_component_list(ignore_order=True)
    # norm_week_factors, norm_week_weights = storage.get_ordered_component_list(ignore_order=True, integrate=True,
    #                                                                           normalize=True, n_cut=4)
    #
    # storage_daily = ComponentStorage()
    # dataloader_daily = DataLoader(reading_step=int(60 / 15) * 24, reading_amount=int(60 / 15) * 24)
    #
    # # time_to = "2022-07-02T18:09Z"
    # dataloader_daily.set_from_to_timestamp(time_from=time_from, time_to=time_to)
    # wrapped_dataloader = tqdm(dataloader_daily, desc="Storing components")
    # storage_daily.add_data(dataloader=wrapped_dataloader)
    #
    # storage_daily.add_tickers(tickers=dataloader_daily.ticker_data)
    # daily_factors, daily_weights = storage_daily.get_ordered_component_list(ignore_order=True, n_cut=4)
    # daily_factors = np.transpose(daily_factors, axes=(0, 2, 1))
    # norm_daily_factors, norm_daily_weights = storage_daily.get_ordered_component_list(ignore_order=True, integrate=True,
    #                                                                                   normalize=True, n_cut=4)
    # norm_daily_factors = np.transpose(norm_daily_factors, axes=(0, 2, 1))

    # ---

    # plt.figure(figsize=(15, 10))
    #
    # col_list = [generate_color() for _ in range(4)]
    # norm_week_weights = norm_week_weights[:, :2, :4]  # take each weight list first two values
    # for idx in range(len(norm_week_weights[0, 0])):
    #     x = norm_week_weights[:, 1, idx]
    #     y = norm_week_weights[:, 0, idx]
    #     plt.plot(x, y, "-D", c=col_list[idx], label=f"{storage.get_ticker(idx)}")
    #     for idx_point in range(len(x)):
    #         plt.text(x[idx_point], y[idx_point], f"{idx_point}")
    #
    # norm_daily_weights = norm_daily_weights[:, :2, :4]  # take each weight list first two values
    # for idx in range(len(norm_daily_weights[0, 0])):
    #     x = norm_daily_weights[:, 1, idx]
    #     y = norm_daily_weights[:, 0, idx]
    #     plt.plot(x, y, "-o", c=col_list[idx], label=f"{storage.get_ticker(idx)}")
    #     for idx_point in range(len(x)):
    #         plt.text(x[idx_point], y[idx_point], f"{idx_point}")
    #
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # fig = plt.figure(figsize=(15, 10))
    # axs = fig.subplots(2, 2)
    #
    # col = generate_color()
    # for idx in range(len(norm_week_factors[0].T)):
    #     x = np.arange(0, len(norm_week_factors[0, :, idx]))
    #
    #     axs[int(idx / 2), idx % 2].plot(x, norm_week_factors[0, :, idx], c=col)
    #     axs[int(idx / 2), idx % 2].set_title(f"Factor {idx + 1}")
    #
    # for idx_1, factors in enumerate(norm_daily_factors):
    #     col = generate_color()
    #     for idx_2 in range(len(factors)):
    #         x = np.arange(idx_1 * dataloader_daily.reading_step,
    #                       idx_1 * dataloader_daily.reading_step + dataloader_daily.reading_amount - 1)
    #
    #         axs[int(idx_2 / 2), idx_2 % 2].plot(x, factors[idx_2], c=col)
    #
    # plt.tight_layout()
    # plt.show()

    # fig = plt.figure(figsize=(15, 10))
    # axs = fig.subplots(2, 2)
    #
    # for idx in range(4):
    #     daily_factor_cut = daily_factors[:, idx]
    #     week_factor_cut = []
    #     for idx_day in range(0, len(daily_factor_cut)):
    #         factor_cut = week_factors[0].T[idx][
    #                      idx_day * dataloader_daily.reading_step:idx_day * dataloader_daily.reading_step + dataloader_daily.reading_amount - 1]
    #         week_factor_cut.append(factor_cut)
    #
    #     correlation_coeffs = np.corrcoef(daily_factor_cut, week_factor_cut)
    #     correlation_coeffs = correlation_coeffs[:int(len(correlation_coeffs) / 2), int(len(correlation_coeffs) / 2):]
    #
    #     im = axs[int(idx / 2), idx % 2].imshow(correlation_coeffs, interpolation='nearest')
    #     fig.colorbar(im, ax=axs[int(idx / 2), idx % 2])
    #     axs[int(idx / 2), idx % 2].set_title(f"Factor {idx + 1}")
    #
    # plt.show()

    # ---- creates animation with top 10 coin weights, BTC and ETH rates and top 2 factors

    factor_list = factor_list[:, :, :2]  # first two factors
    weight_list = weight_list[:, :2, :10]  # take each weight list first two values

    coin_data = dataloader.get_all_coin_data().T[:2]
    processed_coin_data = data_preprocessing(coin_data, std_axis=1)
    i_coin_data = np.cumsum(processed_coin_data, axis=1)
    # norm_coin_data = ((i_coin_data.T - np.mean(i_coin_data, axis=1)) / np.std(i_coin_data)).T
    coin_rate_list = []
    for idx_step in range(0, len(i_coin_data[0]) - reading_amount + 2, reading_step):
        coin_rate_list.append(i_coin_data[:, idx_step:idx_step + reading_amount - 1])
    coin_rate_list = np.array(coin_rate_list)

    fig = plt.figure(figsize=(20, 15))
    axs = fig.subplots(3, 2)

    gs = axs[0, 0].get_gridspec()
    for ax in axs[0, 0:]:
        ax.remove()
    ax_big = fig.add_subplot(gs[0, 0:])

    ax_big.set_xlim((np.min(weight_list[:, 1]), np.max(weight_list[:, 1])))
    ax_big.set_ylim((np.min(weight_list[:, 0]), np.max(weight_list[:, 0])))
    for i in range(2, 4):
        axs[int(i / 2), i % 2].set_xlim((0, reading_amount - 1))
        axs[int(i / 2), i % 2].set_ylim((np.min(coin_rate_list), np.max(coin_rate_list)))
    for i in range(4, 6):
        axs[int(i / 2), i % 2].set_xlim((0, reading_amount - 1))
        axs[int(i / 2), i % 2].set_ylim((-3, 4))

    weight_plots = [ax_big.plot([], [], "-o", c=generate_color(), label=f"{storage.get_ticker(i)}") for i in
                    range(0, 10)]
    text_plot = ax_big.text(-0.2, 0.11, '', fontsize=15)
    coin_rate_plots = [axs[1, i].plot([], [], c=generate_color(), label=f"{storage.get_ticker(i)}") for i in
                       range(0, 2)]
    factor_plots = [axs[2, i].plot([], [], c=generate_color(), label=f"Factor {i + 1}") for i in range(0, 2)]
    x_readings = np.arange(0, reading_amount - 1)

    # axs[0, 0].legend(bbox_to_anchor=(1, 1), loc="upper left")
    ax_big.legend()
    for i in range(2, 6):
        axs[int(i / 2), i % 2].legend()

    def update(idx_frame):
        # Plot time as text
        dt_from = datetime.datetime.strptime(timestamp_list[idx_frame], storage.date_format)
        dt_to = dt_from + datetime.timedelta(days=7)
        # dt_to = datetime.datetime.strptime(timestamp_list[idx + int(reading_amount / reading_step)], storage.date_format)
        text_plot.set_text(f"Time: {dt_from.strftime('%Y-%m-%d %H:%M')} - {dt_to.strftime('%Y-%m-%d %H:%M')}")

        # Plot the constructed weight values
        trail_idx = 0
        trail = 30
        if idx_frame > trail:
            trail_idx = idx_frame - trail
        for idx_plot, plot in enumerate(weight_plots):
            x = weight_list[trail_idx:idx_frame + 1, 1, idx_plot]
            y = weight_list[trail_idx:idx_frame + 1, 0, idx_plot]
            plot[0].set_data((x, y))

        # Plot coin price rates
        for idx_plot, plot in enumerate(coin_rate_plots):
            y = coin_rate_list[idx_frame, idx_plot]
            plot[0].set_data((x_readings, y))

        # Plot constructed factors
        for idx_plot, plot in enumerate(factor_plots):
            y = factor_list[idx_frame, :, idx_plot]
            plot[0].set_data((x_readings, y))

    animation_idxes = np.arange(0, len(weight_list), 1)
    idx_step = int(len(weight_list) / 10)
    for idx in tqdm(range(10)):
        end_idx = (idx * idx_step) + idx_step
        if end_idx >= len(animation_idxes):
            end_idx = -1

        ani = FuncAnimation(fig, update, animation_idxes[idx * idx_step:end_idx], interval=200)
        writer = PillowWriter(fps=25)
        ani.save(f"coin_dynamics_{idx}.gif", writer=writer)

    # ---- creates a 3d interactive figure

    # first_weight_cut = weight_list[:, :3, :]
    # x = first_weight_cut[:, 1, 0]
    # y = first_weight_cut[:, 0, 0]
    # z = first_weight_cut[:, 2, 0]
    #
    # fig = go.Figure(data=[go.Scatter3d(x=x, y=y, z=z, mode="markers")])
    # fig.show()

    # ---- creates a top 10 weights animation that changes through time

    # TODO weight cut is incorrect
    # first_weight_cut = weight_list[:, :2, :]
    # fig = plt.figure(figsize=(15, 10))
    # ax = plt.axes(xlim=(-0.6, 0.3), ylim=(0.1, 0.15))
    # weight_plots = [ax.plot([], [], "-o", c=generate_color(), label=f"{storage.get_ticker(i)}") for i in range(0, 10)]
    # text_plot = ax.text(-0.55, 0.1015, '', fontsize=15)
    # ax.legend()
    #
    # def init():
    #     for plot in weight_plots:
    #         plot[0].set_data([], [])
    #     return weight_plots
    #
    # def update(idx):
    #     dt = datetime.datetime.strptime(timestamp_list[idx], storage.date_format)
    #     text_plot.set_text(dt.strftime('%Y-%m-%d'))
    #
    #     train_idx = 0
    #     trail = 20
    #     if idx > trail:
    #         train_idx = idx - trail
    #     for idx_plot, plot in enumerate(weight_plots):
    #         x = first_weight_cut[train_idx:idx + 1, 1, idx_plot]
    #         y = first_weight_cut[train_idx:idx + 1, 0, idx_plot]
    #         plot[0].set_data((x, y))
    #     return weight_plots
    #
    # ani = FuncAnimation(fig, update, np.arange(0, len(first_weight_cut), 2), init_func=init, interval=200)
    # writer = PillowWriter(fps=25)
    # ani.save("coin_weight.gif", writer=writer)

    # ---- creates a factor plot where factors are plotted over each other though time

    # plt.figure(figsize=(80, 10))
    # i_factor_list = np.cumsum(np.transpose(factor_list, axes=[0, 2, 1]), axis=2)[:, 0, :]
    # # norm_factors = (i_factor_list - np.mean(i_factor_list)) / np.std(i_factor_list)
    # for idx, factor in enumerate(i_factor_list):
    #     x = np.arange(idx * reading_step, idx * reading_step + reading_amount - 1)
    #     col = generate_color()
    #     norm_factor = (factor - np.mean(factor)) / np.std(factor)
    #     plt.plot(x, norm_factor, c=col)
    #
    # plt.tight_layout()
    # plt.show()

    # ---- creates a top 10 coin weight plot

    # plt.figure(figsize=(15, 10))
    # for idx_1 in range(0, 10):
    #     col = generate_color()
    #     weight_line = weight_list[:, :2, idx_1].T
    #     ticker = storage.get_ticker(idx_1)
    #     plt.plot(weight_line[1], weight_line[0], "-o", c=col, label=f"{ticker}")
    #
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    # ---- creates top 4 factor plot plotted over each other

    # fig = plt.figure(figsize=(15, 10))
    # axs = fig.subplots(2, 2)
    # for idx_1, factors in enumerate(factor_list):
    #     i_factors = np.cumsum(factors.T, axis=1)
    #
    #     col = generate_color()
    #     for idx_2 in range(len(factors[0, :4])):
    #         x = np.arange(idx_1 * reading_step, idx_1 * reading_step + reading_amount - 1)
    #
    #         corrected_factors = []
    #         for factor in i_factors:
    #             corrected_factors.append((factor - np.mean(factor)) / np.std(factor))
    #
    #         axs[int(idx_2 / 2), idx_2 % 2].plot(x, corrected_factors[idx_2], c=col)
    #         axs[int(idx_2 / 2), idx_2 % 2].set_title(f"Factor {idx_2 + 1}")
    #
    # plt.tight_layout()
    # plt.show()

    # ---- creates a plot with top 4 factors, 4 time steps in separate sub-plots

    # fig = plt.figure(figsize=(15, 10))
    # axs = fig.subplots(4, 4)
    # for idx_1, factors in enumerate(factor_list):
    #     factor_min = np.min(factors) - 0.2
    #     factor_max = np.max(factors) + 0.2
    #     factors_T = factors.T
    #
    #     x = np.arange(0, len(factors_T[0]))
    #     for idx_2 in range(4):
    #         col = generate_color()
    #
    #         axs[idx_1, idx_2].plot(x, factors_T[idx_2], c=col)
    #         axs[idx_1, idx_2].set_title(f"Factor {idx_2 + 1}")
    #         axs[idx_1, idx_2].set_ylim(factor_min, factor_max)
    #
    # plt.tight_layout()
    # plt.show()


def plot_coin_data():
    dataloader = DataLoader(file_path="data/from_2023-21-04_to_2023-21-05_ETHBTC.csv", filter_coins=False)
    print()



if __name__ == '__main__':
    # fetch_coin_data(interval="1min", tickers=["ETHBTC"])
    # fetch_coin_data(tickers=["BTCUSDT", "ETHUSDT", "ETHBTC"])
    # fetch_coin_data(tickers=["XMRETH", "XMRBTC", "ETHBTC", "ETHUSDT", "BTCUSDT", "XMRUSDT"])
    # fetch_coin_data(tickers=["ETHUSDT"], days_saved_in_csv=548)
    # fetch_coin_data(interval="1min", tickers=["ETHUSDT"], days_saved_in_csv=548)
    # TODO data visualization functions don't work after the new csv update
    plot_coin_data()
    # experiments_with_pca()
    # --
    # calculate_factors_over_time()
    # calculate_factor_ranges()
    # crypto_triangulation()
    # crypto_complete_graph()
    # calculate_correlation_of_methods()
    # plot_window_stds()
