import numpy as np

from modules.component_manager import ComponentManger


class ComponentTemporalCorrelation:
    def __init__(self, manager: ComponentManger):
        self.comp_manager = manager
        self.k = np.arange(-2, 2, 0.1)
        self.c = int((60 / manager.reading_step) * 24 * 3.5)  # Half a week of readings

    def _calc_correlation_for_idx(self):
        pass
        
    def calc_correlation_map(self):
        pass
        
    
        