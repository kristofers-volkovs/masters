import csv
import numpy as np
from sklearn.decomposition import PCA

from modules.data_classes import CoinComponents


@staticmethod
def data_preprocessing(data, log=True, diff=True, std=True, std_axis=0) -> np.ndarray:
    data_out = np.array(data)
    if log:
        data_out = np.log(data_out)
    if diff:
        data_out = np.diff(data_out)
    if std:
        std_val = np.std(data_out, axis=std_axis)
        if isinstance(std_val, list):
            std_val[std_val == 0.0] = 1e-8
        else:
            std_val = 1e-8
        data_out = (data_out.T / std_val).T
    return data_out


def calculate_component_with_pca(coin_data, timestamp=None, n_components=None) -> CoinComponents:
    norm_coin_data = data_preprocessing(coin_data.T, std_axis=1).T

    pca = PCA(n_components=n_components)
    factors = pca.fit_transform(norm_coin_data)
    weights = pca.components_

    # print(f"Variance: {list(np.round_(pca.explained_variance_ratio_, decimals=4))}")

    return CoinComponents(
        factors=factors,
        weights=weights,
        timestamp=timestamp,
    )


def sort_with_list(to_sort, sort_with) -> list:
    return [x for _, x in sorted(zip(sort_with, to_sort))]


def generate_color():
    return np.random.random(), np.random.random(), np.random.random()


def read_data_from_csv(file_path):
    with open(file_path, 'r') as f:
        reader = csv.reader(f)

        data = []
        for row in reader:
            data.append(row)

        return data


# TODO have only one function that calculates components
def reconstruct_data_with_pca(coin_data, n_components=None):
    norm_coin_data = []
    for row in coin_data.T:
        data = np.log(np.array(row))
        data = np.diff(data)
        data = data / np.std(data)
        norm_coin_data.append(data)
    norm_coin_data = np.array(norm_coin_data).T

    pca = PCA(n_components=n_components)
    factors = pca.fit_transform(norm_coin_data)
    weights = pca.components_
    new_coin_data = np.dot(factors, weights)

    # print(f"Variance: {list(np.round_(pca.explained_variance_ratio_, decimals=4))}")

    # metric = np.mean((norm_coin_data - new_coin_data) ** 2)
    # print(f"mse: {metric}")

    return new_coin_data, factors, weights


def calc_improvement(old_val, new_val):
    old_val = 3.43
    new_val = 4.48

    percent = 100 - new_val / old_val * 100

    print(f"Percent: {percent}")
