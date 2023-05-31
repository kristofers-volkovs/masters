import numpy as np
from tqdm import tqdm

from modules.models import ComponentParams
from modules.data_sorting import DataSorting


class ComponentOrderer:
    coeff_threshold = 0.7
    
    def __init__(self, reading_step, date_format):
        self.reading_step = reading_step
        self.date_format = date_format

    def _construct_step_order(self, correlation_coeffs):
        # Idx becomes unsure if the correlation does not exceed threshold or is not the highest in row 
        step_order = []
        unsure_idxes = []
        component_params_to_flip = []
        
        for idx in range(len(correlation_coeffs)):
            idx_max = np.argmax(abs(correlation_coeffs[idx]))
        
            idx_is_not_taken = idx_max not in step_order
            highest_row_correlation = np.argmax(abs(correlation_coeffs[:, idx_max])) == idx
            above_coeff_threshold = abs(correlation_coeffs[idx, idx_max]) >= self.coeff_threshold
            
            if idx_is_not_taken and highest_row_correlation and above_coeff_threshold:
                step_order.append(idx_max)
                
                # When correlation is negative it means that the factor and weights are flipped
                if correlation_coeffs[idx, idx_max] < 0:
                    component_params_to_flip.append(idx)
            else:
                step_order.append(-1)
                unsure_idxes.append(idx)
                
        for idx in unsure_idxes:
            # Sorts the correlation list in an ascending 
            # At the same time correlation values get replaced by their idx before ordering
            row_idxes = np.argsort(abs(correlation_coeffs[idx]))
            
            # The idx with the highest correlation that is not already taken is used as the step idx
            for idx_max in reversed(row_idxes):
                if idx_max in step_order:
                    continue
                
                step_order[idx] = idx_max
                
                # When correlation is negative it means that the factor and weights are flipped
                if correlation_coeffs[idx, idx_max] < 0:
                    component_params_to_flip.append(idx)
                    
                break
        
        return np.array(step_order), component_params_to_flip
    
    @staticmethod
    def _collect_step_correlation(correlation_coeffs, step_order):
        step_correlation = []
        for idx in range(len(correlation_coeffs)):
            step_correlation.append(abs(correlation_coeffs[idx, step_order[idx]]))
        
        return np.array(step_correlation)
    
    @staticmethod
    def _flip_params(component, params_to_flip):
        # PCA can converge on a possitive or negative result
        # Reasoning example: 2 * 5 = -2 * -5
        # To simplify working with factors and weights they need to be flipped back
        factor_val = ComponentParams.factors.value
        weight_val = ComponentParams.weights.value
        
        factors = component[factor_val].T
        weights = component[weight_val]
        factors[params_to_flip] = -factors[params_to_flip]
        weights[params_to_flip] = -weights[params_to_flip]
        component[factor_val] = factors.T
        component[weight_val] = weights
        
        return component
    
    def order_component_steps(self, data_storage):
        factor_val = ComponentParams.factors.value
        weight_val = ComponentParams.weights.value
        step_order_val = ComponentParams.step_order.value
        
        for idx in tqdm(range(len(data_storage)), desc='Determining component steps'):
            if idx == 0:  # first components order is taken as the origin
                component = data_storage.get_item(idx, groups=[weight_val])
                
                step_order = {step_order_val: np.arange(0, len(component[weight_val]))}
                data_storage.add_item(idx, step_order)
                continue

            current_component_groups = [factor_val, weight_val]
            previous_component_groups = [factor_val, weight_val, step_order_val]
            
            # Current component is ordered relative to the previous component factors
            current_component = data_storage.get_item(idx, groups=current_component_groups)
            previous_component = data_storage.get_item(idx - 1, groups=previous_component_groups)
            
            current_factor_overlap = current_component[factor_val].T[:, :-self.reading_step]
            previous_factor_overlap = previous_component[factor_val].T[:, self.reading_step:]
            
            # Sorts previous factors to the order that the origin has
            previous_factor_overlap = DataSorting.sort_with_list(to_sort=previous_factor_overlap, sort_with=previous_component[step_order_val])
            
            # Calculates correlation matrix
            # Selects a matrix that describes current factor correlation to previous factors
            # Each row is a current factor
            # Each column in the previous factor
            correlation_coeffs = np.corrcoef(current_factor_overlap, previous_factor_overlap)
            correlation_coeffs = correlation_coeffs[:len(current_factor_overlap), len(current_factor_overlap):]
            
            # Constructs factor order relative to the origin
            step_order, component_params_to_flip = self._construct_step_order(correlation_coeffs)

            # Collects each factor correlation values
            step_correlation = self._collect_step_correlation(correlation_coeffs=correlation_coeffs, step_order=step_order)
            
            # ===
            # print(f'Step: {idx}')
            # print(f'step order: {step_order[:4]}')
            # print(f'step correlation: {step_correlation[:4]}')
            # print('Correlation matrix')
            # for row in correlation_coeffs[:4]:
            #     print(row[:4])
            # print('===')
            # ===
            
            component_dict = {
                step_order_val: step_order,
                ComponentParams.correlation.value: step_correlation,
            }
            if len(component_params_to_flip) > 0:
                current_component = ComponentOrderer._flip_params(current_component, component_params_to_flip)
                component_dict[factor_val] = current_component[factor_val] 
                component_dict[weight_val] = current_component[weight_val] 
                
            data_storage.add_item(idx, component_dict)
    
    @staticmethod
    def order_component(component):
        factor_val = ComponentParams.factors.value
        weight_val = ComponentParams.weights.value
        variance_val = ComponentParams.variance.value
        correlation_val = ComponentParams.correlation.value
        step_order_val = ComponentParams.step_order.value
        
        if len(component[step_order_val]) == 0:
            raise ValueError('Component cannot be ordered, step_order is empty')
        
        step_order = component[step_order_val]
        
        component[factor_val] = DataSorting.sort_with_list(component[factor_val].T, step_order).T
        component[weight_val] = DataSorting.sort_with_list(component[weight_val], step_order)
        component[variance_val] = DataSorting.sort_with_list(component[variance_val], step_order)
        if correlation_val in component.keys():
            component[step_order_val] = DataSorting.sort_with_list(component[step_order_val], step_order)
        
        return component
    