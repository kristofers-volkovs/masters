import matplotlib.pyplot as plt
import numpy as np
from utils import data_preprocessing, generate_color, read_data_from_csv, reconstruct_data_with_pca


def plot_coin_data():
    file_path = "data/from_2022-04-07_to_2022-11-07.csv"
    coin_data = read_data_from_csv(file_path)

    plt.figure(figsize=(15, 10))
    x = np.arange(0, len(coin_data[0]) - 1)
    for idx, row in enumerate(coin_data):
        ticker = row[0]
        coin_open_data = [float(val) for val in row[1:]]

        y = data_preprocessing(coin_open_data)
        col = (np.random.random(), np.random.random(), np.random.random())

        plt.plot(x, y, c=col)

    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_factors_over_each_other(factors_a, factors_b):
    fig = plt.figure(figsize=(15, 10))
    axs = fig.subplots(1, 3)
    x = np.arange(0, len(factors_a[0]))
    factor_min = np.min((np.min(factors_a), np.min(factors_b))) - 0.2
    factor_max = np.max((np.max(factors_a), np.max(factors_b))) + 0.2
    for idx in range(len(factors_a)):
        col1 = generate_color()
        col2 = generate_color()

        axs[idx].plot(x, factors_a[idx], c=col1, label="factor_a")
        axs[idx].plot(x, factors_b[idx], c=col2, label="factor_b")
        axs[idx].set_title(f"Factor {idx + 1}")
        axs[idx].set_ylim(factor_min, factor_max)
        axs[idx].legend()
    plt.tight_layout()
    plt.show()


def plot_last_coins(coin_data, new_coin_data, ticker_data):
    # Visualize last crypto coins
    old_data_cut = []
    for row in coin_data.T:
        old_data_cut.append(data_preprocessing(row))
    old_data_cut = np.array(old_data_cut)[-6:]
    new_data_cut = new_coin_data.T[-6:]
    tickers = ticker_data[-6:]

    old_data_cut = old_data_cut[::-1]
    new_data_cut = new_data_cut[::-1]
    tickers = tickers[::-1]

    fig = plt.figure(figsize=(15, 10))
    axs = fig.subplots(2, 3)

    x = np.arange(0, len(old_data_cut[0]))
    idx = 0
    for idx_1 in range(2):
        for idx_2 in range(3):
            col1 = generate_color()
            col2 = generate_color()

            axs[idx_1, idx_2].plot(x, old_data_cut[idx], c=col1, label="original")
            axs[idx_1, idx_2].plot(x, new_data_cut[idx], c=col2, label="reconstructed")
            axs[idx_1, idx_2].set_title(tickers[idx])
            axs[idx_1, idx_2].legend()
            idx += 1

    plt.tight_layout()
    plt.show()


def plot_original_vs_reconstructed(coin_data, new_coin_data, ticker_data):
    # Visualize original and reconstructed data against each other
    old_data_cut = []
    for row in coin_data.T:
        old_data_cut.append(data_preprocessing(row))
    old_data_cut = np.array(old_data_cut)[:6]
    new_data_cut = new_coin_data.T[:6]
    new_data_cut = np.cumsum(new_data_cut, axis=1)

    fig = plt.figure(figsize=(15, 10))
    axs = fig.subplots(2, 3)

    x_old = np.arange(0, len(old_data_cut[0]))
    x_new = np.arange(0, len(new_data_cut[0]))
    idx = 0
    for idx_1 in range(2):
        for idx_2 in range(3):
            col1 = generate_color()
            col2 = generate_color()

            axs[idx_1, idx_2].plot(x_old, old_data_cut[idx], c=col1, label="original")
            axs[idx_1, idx_2].plot(x_new, new_data_cut[idx], c=col2, label="reconstructed")
            axs[idx_1, idx_2].set_title(ticker_data[idx])
            axs[idx_1, idx_2].legend()
            idx += 1

    plt.tight_layout()
    plt.show()


def plot_factors(factors):
    # Visualize the factors
    fig = plt.figure(figsize=(15, 10))
    axs = fig.subplots(1, 3)
    i_factors = np.cumsum(factors, axis=1)
    x = np.arange(0, len(i_factors[0]))
    factor_min = np.min(i_factors) - 0.2
    factor_max = np.max(i_factors) + 0.2
    for idx in range(len(i_factors)):
        col = generate_color()
        axs[idx].plot(x, i_factors[idx], c=col)
        axs[idx].set_title(f"Factor {idx + 1}")
        axs[idx].set_ylim(factor_min, factor_max)
    plt.tight_layout()
    plt.show()


def plot_original_vs_reconstructed_side(coin_data, new_coin_data):
    # Visualize the original and reconstructed data
    old_data_cut = []
    for row in coin_data.T:
        old_data_cut.append(data_preprocessing(row))
    old_data_cut = np.array(old_data_cut)[:5]
    new_data_cut = new_coin_data.T[:5]
    new_data_cut = np.cumsum(new_data_cut, axis=1)

    fig = plt.figure(figsize=(15, 10))
    axs = fig.subplots(1, 2)

    x_old = np.arange(0, len(old_data_cut[0]))
    x_new = np.arange(0, len(new_data_cut[0]))
    for idx in range(len(old_data_cut)):
        col = generate_color()

        axs[0].plot(x_old, old_data_cut[idx], c=col)
        axs[1].plot(x_new, new_data_cut[idx], c=col)

    axs[0].set_title("Original data plot")
    axs[1].set_title("Reconstructed data plot")
    plt.tight_layout()
    plt.show()


def experiments_with_pca():
    # from sklearn.preprocessing import StandardScaler

    file_path = "data/from_2022-04-07_to_2022-11-07.csv"
    coin_data = read_data_from_csv(file_path)

    # Rearranging and parsing to float
    ticker_data = np.array(coin_data)[:, :1].T[0]
    coin_data = np.array(coin_data)[:, 1:].T
    coin_data_less = np.array(coin_data)[:, 10:]

    coin_data = coin_data.astype(float)
    coin_data_less = coin_data_less.astype(float)

    (new_coin_data, factors_orig, weights_orig) = reconstruct_data_with_pca(coin_data, n_components=3)
    (new_coin_less_data, factors_less, weights_less) = reconstruct_data_with_pca(coin_data_less, n_components=3)

    # plot_factors_over_each_other(factors_orig.T, factors_less.T)
    # plot_last_coins(coin_data, new_coin_data, ticker_data)
    # plot_original_vs_reconstructed(coin_data, new_coin_data, ticker_data)
    plot_factors(factors_orig.T)
    # plot_original_vs_reconstructed_side(coin_data, new_coin_data)

    print()
