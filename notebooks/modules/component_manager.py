from tqdm import tqdm
import numpy as np
import datetime as dt

from modules.data_storage import DataStorage
from modules.data_processing import DataProcessor
from modules.component_orderer import ComponentOrderer
from modules.models import ComponentParams, Metric, TimePeriod


class ComponentManager:
    components: DataStorage 
    date_format: str = None
    reading_step: int = None
    tickers: list[str] = []
    
    def __init__(self, file_dir):
        self.components = DataStorage(
            file_dir=file_dir, 
            group_structure=ComponentParams.aslist(),
        )
        
        metadata = self.components.get_metadata()
        for key, value in metadata.items():
            if key == 'components':
                continue
            if all([v.isdigit() for v in value]):
                value = [int(v) for v in value]
            if len(value) == 1:
                value = value[0]
            setattr(self, key, value)
        

    def proc_list(self, df_iter, n_components=None):
        self.reading_step = df_iter.reading_step
        self.date_format = df_iter.date_format
        self.tickers = df_iter.get_tickers()
        
        for key, value in self.__dict__.items():
            if not isinstance(value, list):
                value = [str(value)]
            self.components.add_metadata(key, value)
        
        DataProcessor.calc_pca_component_windows(
            df_iter=df_iter, 
            data_storage=self.components, 
            n_components=n_components,
        )
    
    def order_component_steps(self):
        orderer = ComponentOrderer(self.reading_step, self.date_format)        
        orderer.order_component_steps(self.components)
    
    def _is_ordered(self):
        if len(self.components) == 0:
            return False
        
        return self.components.does_all_groups_contain(ComponentParams.step_order.value)

    def avg_step_correlation(self, metric=Metric.IRMS):
        # TODO the small values are being overshadowed by a lot of big values
        # Think of some way to visualize/calculate average in a proper way
        
        if not self._is_ordered():
            raise ValueError('Components are not ordered')
        
        correlation_val = ComponentParams.correlation.value
        comp_correlation = []
        for idx in tqdm(range(1, len(self.components)), desc='Collecting component correlation'):
            component = self.components.get_item(idx, groups=[correlation_val])
            comp_correlation.append(component[correlation_val])
        comp_correlation = np.array(comp_correlation)
        
        avg_correlation = []
        if metric == Metric.DEFAULT:
            avg_correlation = np.mean(comp_correlation, axis=0)
        elif metric == Metric.IRMS:
            avg_correlation = 1.0 - np.sqrt(np.mean((1.0 - comp_correlation) ** 2.0, axis=0))
        
        return avg_correlation
    
    def get_item(self, idx, groups=ComponentParams.aslist()):
        return self.components.get_item(idx, groups)
    
    def get_item_list(self, idx_from, idx_to, idx_step=1, order_components=False, groups=ComponentParams.aslist()):
        if idx_to < idx_from:
                raise ValueError(f'Index from {idx_from} can not be bigger then index to {idx_to}')
            
        components_out = []
        for idx in tqdm(range(idx_from, idx_to, idx_step), desc='Collecting components'):
            comp = self.components.get_item(idx, groups)
            if order_components:
                comp = ComponentOrderer.order_component(comp)
                
            components_out.append(comp)
        
        return components_out   
    
    def get_window_len(self):
        factor_key = ComponentParams.factors.value
        first_factor = self.components.get_item(0, [factor_key])
        first_factor = first_factor[factor_key]
        
        return len(first_factor)
    
    def _get_timestamp_by_idx(self, idx):
        ts_key = ComponentParams.timestamps.value
        time_idx = self.components.get_item(idx, [ts_key])
        time_idx = dt.datetime.fromtimestamp(time_idx[ts_key][0])
    
        return time_idx
    
    """
        time_at - datetime
    """
    def get_component_idx_by_timestamp(self, time_at):
        comp_idx = -1
        idx_bottom = 0
        idx_top = len(self.components) - 1
        while comp_idx < 0:
            if idx_top - idx_bottom < 0:
                raise ValueError(f'Bound indexes have been inverted, idx_bottom: {idx_bottom}, idx_top: {idx_top}')
            
            current_idx = idx_bottom + int((idx_top - idx_bottom) / 2)
            
            time_idx = self._get_timestamp_by_idx(current_idx)
            time_diff = int((time_at - time_idx).total_seconds() / 60)
            
            # Checks nearby component time differences to find 
            # the closest component to the searched timestamp 
            if abs(time_diff) < self.reading_step:
                check_idx = [-1, 1]  # checks idx to the left and right of the current idx
                
                for idx in check_idx:
                    if current_idx + idx < 0 or current_idx + idx > len(self.components) - 1:
                        continue
                    
                    time_check_idx = self._get_timestamp_by_idx(current_idx + idx)
                    time_check_diff = int((time_at - time_check_idx).total_seconds() / 60)
                    
                    if abs(time_check_diff) < abs(time_diff):
                        comp_idx = current_idx + idx
                        break
                        
                # If nearby components are not closer to the searchable 
                # Then selects the current idx
                if comp_idx == -1:
                    comp_idx = current_idx
                    continue
            
            if time_diff > 0:
                idx_bottom = current_idx
            elif time_diff < 0:
                idx_top = current_idx
        
        return comp_idx
    
    def print_state(self):
        self.components.print_state()
        print('=== Orderer params ===')
        print(f'tickers: {self.tickers}, tickers_len: {len(self.tickers)}')
        print(f'reading_step: {self.reading_step}, date_format: {self.date_format}')
        print(f'is_ordered: {self._is_ordered()}')
