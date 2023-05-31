from tqdm import tqdm
import numpy as np
import datetime as dt
from scipy.spatial.distance import euclidean

from modules.models import TimePeriod

class ComponentAnalyzer:
    def __init__(self):
        pass
    
    @staticmethod
    def diff_window_size_weigt_distances(manager_a, manager_b, time_step, time_period: TimePeriod):
        time_from = time_period.dt_from
        time_to = time_period.dt_to        
        time_diff = int((time_to - time_from).total_seconds() / time_step.total_seconds())
        
        ts_list = [time_from + time_step * d for d in range(time_diff)]

        weight_key = ComponentParams.weights.value
        weight_list = []
        for ts in ts_list:
            a_idx = manager_a.get_component_idx_by_timestamp(ts)
            b_idx = manager_b.get_component_idx_by_timestamp(ts)
            
            a_weight = manager_a.get_item(a_idx, [weight_key])
            b_weight = manager_b.get_item(b_idx, [weight_key])
            
            # how much coins to select? all of them?
            
            ab_dist = euclidean()
            
            # collect distances in a list
        
        pass
