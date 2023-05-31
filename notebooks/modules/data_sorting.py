import numpy as np

class DataSorting:
    @staticmethod
    def is_sorted(data, sort_condition):
        if len(data) == 0:
            raise ValueError('List is empty')

        return data == sorted(data, key=sort_condition)
    
    @staticmethod    
    def sort_with_list(to_sort, sort_with):
        if len(to_sort) == 0 or len(sort_with) == 0:
            raise ValueError(f'Lists are empty, len_to_sort: {len(to_sort)}, len_sort_with: {len(sort_with)}')
        if len(to_sort) != len(sort_with):
            raise ValueError(f'List lengths are not equal, len_to_sort: {len(to_sort)}, len_sort_with: {len(sort_with)}')
            
        return np.array([x for _, x in sorted(zip(sort_with, to_sort))])
