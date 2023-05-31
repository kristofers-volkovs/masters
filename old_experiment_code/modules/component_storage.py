import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import datetime
from tqdm import tqdm

from modules.data_classes import CoinComponents
from modules.utils import calculate_component_with_pca, sort_with_list


class ComponentStorage:
    def __init__(self):
        self.data: list[CoinComponents] = []
        self.ticker_data: list[str] = []
        self.date_format = '%Y-%m-%dT%H:%MZ'
        self.max_coeff_threshold = 0.7
        self.min_coeff_threshold = -0.7

    def add_tickers(self, tickers) -> None:
        self.ticker_data = tickers

    def get_ticker(self, idx) -> str:
        return self.ticker_data[idx]

    def add_data(self, dataloader, n_components=None) -> None:
        num_cores = multiprocessing.cpu_count()
        processed_data = Parallel(n_jobs=num_cores)(
            delayed(calculate_component_with_pca)(
                coin_data=coin_data, timestamp=timestamp, n_components=n_components
            ) for (coin_data, timestamp) in dataloader)

        self.data = processed_data

    def add(self, coin_data, timestamp) -> None:
        component = calculate_component_with_pca(coin_data=coin_data, timestamp=timestamp)
        self.data.append(component)

    def sort_timestamps(self) -> None:
        self.data = sorted(self.data, key=lambda c: datetime.datetime.strptime(c.timestamp, self.date_format))

    def sort_step_order(self, reading_step) -> None:
        def flip_factors_and_weights(component: CoinComponents) -> CoinComponents:
            factors = component.factors.T
            weights = component.weights
            factors[idx_coeff] = -factors[idx_coeff]
            weights[idx_coeff] = -weights[idx_coeff]
            component.factors = factors.T
            component.weights = weights
            return component

        for idx in tqdm(range(len(self.data)), desc="Ordering steps"):
            if idx == 0:  # initializes original order
                first = self.data[idx]
                first.step_order = np.arange(0, len(first.weights))
                self.data[idx] = first
            else:
                # Selects the current and previous component to be able to sort it relative to the previous one
                component = self.data[idx]
                prev_component = self.data[idx - 1]

                factor_overlap = component.factors.T[:, :-reading_step]
                prev_factor_overlap = prev_component.factors.T[:, reading_step:]
                # Sorts factors into their original order
                prev_factor_overlap = sort_with_list(to_sort=prev_factor_overlap,
                                                          sort_with=prev_component.step_order)

                correlation_coeffs = np.corrcoef(factor_overlap, prev_factor_overlap)
                # Selects the part where each row describes current factors correlation to each previous factor
                correlation_coeffs = correlation_coeffs[:len(factor_overlap), len(factor_overlap):]

                # Constructs a new factor order relative to the original
                step_order = []
                # Factor correlation idxes that fall below the threshold get filtered later
                unsure_idxes = []
                for idx_coeff in range(len(correlation_coeffs)):
                    idx_max = np.argmax(abs(correlation_coeffs[idx_coeff]))

                    if idx_max in step_order:
                        step_order.append(-1)
                        unsure_idxes.append(idx_coeff)
                    elif correlation_coeffs[idx_coeff, idx_max] >= self.max_coeff_threshold:
                        step_order.append(idx_max)
                    elif correlation_coeffs[idx_coeff, idx_max] <= self.min_coeff_threshold:
                        step_order.append(idx_max)

                        # Flips factors and weights and saves them in the component
                        component = flip_factors_and_weights(component=component)
                    else:
                        step_order.append(-1)
                        unsure_idxes.append(idx_coeff)

                for idx_coeff in unsure_idxes:
                    # Sorts coeff row in ascending order, coeff values get replaced with idxes
                    coeff_row_idxes = np.argsort(abs(correlation_coeffs[idx_coeff]))

                    # Puts the index in the next available spot (might not be the highest correlation)
                    for idx_max in reversed(coeff_row_idxes):
                        if idx_max not in step_order:
                            step_order[idx_coeff] = idx_max

                            if correlation_coeffs[idx_coeff, idx_max] < 0:
                                # Flips factors and weights and saves them in the component
                                component = flip_factors_and_weights(component=component)
                            break

                if len(step_order) != len(correlation_coeffs) or len(step_order) != len(set(step_order)):
                    print("Problem with 'sort_step_order'")

                component.step_order = step_order
                self.data[idx] = component

    def get_ordered_component_list(self, ignore_order=False, integrate=False, normalize=False, n_cut=-1
                                   ) -> (np.ndarray, np.ndarray):
        factor_list = []
        weight_list = []

        for component in tqdm(self.data, desc="Composing component list"):
            factors = component.factors.T
            weights = component.weights
            if not ignore_order:  # ignore_order=True will make the function return the lists as is without ordering
                factors = sort_with_list(to_sort=factors, sort_with=component.step_order)
                weights = sort_with_list(to_sort=weights, sort_with=component.step_order)
            if integrate:
                factors = np.cumsum(factors, axis=1)
            if normalize:
                factors = ((factors.T - np.mean(factors, axis=1)) / np.std(factors, axis=1)).T
                # factors = (abs(factors).T / np.sum(abs(factors), axis=1)).T
                # factors = ((factors.T - np.min(factors, axis=1)) / (np.max(factors, axis=1) - np.min(factors, axis=1))).T
            factor_list.append(np.array(factors).T)
            weight_list.append(weights)

        np_factor_list = np.array(factor_list)
        np_weight_list = np.array(weight_list)

        # cuts out a specific amount of weights and returns them
        if not n_cut < 0:
            # np_factor_list.shape = (num_components, readings, factors)
            np_factor_list = np_factor_list[:, :, :n_cut]
            # np_weight_list.shape = (num_components, weights, coins)
            np_weight_list = np_weight_list[:, :n_cut]

        return np_factor_list, np_weight_list

    def get_timestamp_list(self) -> np.ndarray:
        return np.array([component.timestamp for component in tqdm(self.data, desc="Composing timestamp list")])

