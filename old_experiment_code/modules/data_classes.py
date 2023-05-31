import numpy as np
from dataclasses import dataclass, field


@dataclass
class CoinComponents:
    timestamp: str
    factors: np.ndarray
    weights: np.ndarray
    step_order: list[int] = field(default_factory=list)


@dataclass
class CoinData:
    ticker: str
    coin_open_data: list[str] = field(default_factory=list)


@dataclass
class Triangle:
    points: np.ndarray
    side_lengths: np.ndarray
    angles: np.ndarray
    timestamp: str
