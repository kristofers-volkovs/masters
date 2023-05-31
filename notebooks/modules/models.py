from dataclasses import dataclass, field, asdict
import datetime as dt
import pandas as pd
import numpy as np
from enum import Enum

from modules.data_sorting import DataSorting


@dataclass
class TimePeriod:
    dt_from: dt
    dt_to: dt

    
@dataclass
class DiffParams:
    is_true: bool
    period: int
    
    def __init__(self, is_true=True, period=1):
        self.is_true = is_true
        self.period = period
    
    
@dataclass
class PreprocessingParams:
    log: bool
    diff: DiffParams
    std: bool
    
    def __init__(self, log=True, diff=DiffParams(), std=True):
        self.log = log
        self.diff = diff
        self.std = std
       
    
class ComponentParams(Enum):
    timestamps = 'timestamps'
    factors = 'factors'
    weights = 'weights'
    variance = 'variance'
    correlation = 'correlation'
    step_order = 'step_order'
    
    @staticmethod
    def aslist():
        return [c.value for c in ComponentParams]
    
    @staticmethod
    def asdict(default_value=True):
        return {c.value: default_value for c in ComponentParams}
    
    
# @dataclass
# class ComponentWindow:        
#     timestamps: np.ndarray
#     factors: np.ndarray  # shape - (windows, factors)
#     weights: np.ndarray  # shape - (weights, currencies)
#     variance: np.ndarray 
#     correlation: np.ndarray = field(default_factory=lambda: np.array([]))  # Note - first component correlation is empty
#     step_order: np.ndarray = field(default_factory=lambda: np.array([]))
    
#     def flip_params(self, params_to_flip):
#         # PCA can converge on a possitive or negative result
#         # Reasoning example: 2 * 5 = -2 * -5
#         # To simplify working with factors and weights they need to be flipped back
        
#         factors = self.factors.T
#         weights = self.weights
#         factors[params_to_flip] = -factors[params_to_flip]
#         weights[params_to_flip] = -weights[params_to_flip]
#         self.factors = factors.T
#         self.weights = weights
    
#     def order_component(self):
#         if len(self.step_order) == 0:
#             raise ValueError('Component cannot be ordered, step_order is empty')
        
#         factors = DataSorting.sort_with_list(to_sort=self.factors.T, sort_with=self.step_order).T
#         weights = DataSorting.sort_with_list(to_sort=self.weights, sort_with=self.step_order)
#         variance = DataSorting.sort_with_list(to_sort=self.variance, sort_with=self.step_order)
#         correlation = self.correlation
#         if len(self.correlation) != 0:
#             correlation = DataSorting.sort_with_list(to_sort=correlation, sort_with=self.step_order)
    
#         return ComponentWindow(
#             timestamps=self.timestamps,
#             factors=factors,
#             weights=weights,
#             variance=variance,
#             correlation=correlation,
#             step_order=self.step_order,
#         )
    

class Metric(Enum):
    DEFAULT = 'default'
    IRMS = 'irms'
    