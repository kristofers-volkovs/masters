import csv
import numpy as np
import datetime


class DataLoader:
    def __init__(self, file_path=None, filter_coins=True, reading_step=1, reading_amount=int(60 / 15) * 24 * 7):
        if file_path is None:
            file_path = "data/from_2022-18-06_to_2022-18-07.csv"

        data = []
        with open(file_path, 'r') as f:
            reader = csv.reader(f)

            for row in reader:
                data.append(row)
        data = np.array(data)

        self.timestamps = data[0, 1:]
        self.ticker_data = data[1:, :1].T[0]
        self.coin_data = data[1:, 1:].T
        self.coin_data = self.coin_data.astype(float)

        if filter_coins:
            # Filter out stable coins
            filter_idxes = []
            for idx_ticker in range(len(self.ticker_data)):
                coin_mean = np.mean(self.coin_data.T[idx_ticker])
                if 0.9 < coin_mean < 1.1:
                    filter_idxes.append(idx_ticker)

            for idx_filter in reversed(filter_idxes):
                self.coin_data = np.delete(self.coin_data, idx_filter, 1)
                self.ticker_data = np.delete(self.ticker_data, idx_filter, 0)

        self.date_format = '%Y-%m-%dT%H:%MZ'

        self.start_idx = -1
        self.idx = 0
        self.end_idx = -1
        self.reading_step = reading_step  # amount of readings per time step
        self.reading_amount = reading_amount  # amount of readings to be analysed

    def get_all_coin_data(self):
        return self.coin_data

    def set_reading_amount(self, amount):
        self.reading_amount = amount

    def set_reading_step(self, step):
        self.reading_step = step

    def set_from_to_timestamp(self, time_from: str, time_to: str) -> None:
        date_from = datetime.datetime.strptime(time_from, self.date_format)
        date_to = datetime.datetime.strptime(time_to, self.date_format)

        idx_from = min(range(len(self.timestamps)),
                       key=lambda i: abs(datetime.datetime.strptime(self.timestamps[i], self.date_format) - date_from))
        idx_to = min(range(len(self.timestamps)),
                     key=lambda i: abs(datetime.datetime.strptime(self.timestamps[i], self.date_format) - date_to))

        if idx_to + self.reading_step > len(self.coin_data):
            raise ValueError("End date is out of bounds")

        self.start_idx = idx_from
        self.idx = idx_from
        self.end_idx = idx_to

    def get_timestamps(self):
        if self.start_idx >= 0 and self.end_idx >= 0:
            return self.timestamps[self.start_idx:self.end_idx]
        else:
            return self.timestamps

    def __getitem__(self, idx) -> (np.ndarray, str):
        if isinstance(idx, slice):
            (start, stop) = (int(idx.start) * self.reading_step, int(idx.stop) * self.reading_step)
        else:
            (start, stop) = (idx * self.reading_step, idx * self.reading_step + self.reading_amount)

        if self.end_idx != -1:
            start += self.idx
            stop += self.idx

        if stop > len(self.coin_data):
            raise ValueError("End index is out of bounds")

        data = self.coin_data[start:stop]
        timestamp = self.timestamps[start]

        return data, timestamp

    def __iter__(self):
        return self

    def __len__(self):
        if self.end_idx != -1:
            return self.end_idx - self.start_idx
        else:
            return len(self.timestamps) - self.reading_amount + 1

    def __next__(self) -> (np.ndarray, str):
        if self.idx + self.reading_amount > len(self.coin_data) \
                or (self.end_idx != -1 and self.idx > self.end_idx):
            self.idx = 0
            self.start_idx = -1
            self.end_idx = -1
            raise StopIteration

        data = self.coin_data[self.idx:self.idx + self.reading_amount]
        timestamp = self.timestamps[self.idx]
        self.idx += self.reading_step

        return data, timestamp
