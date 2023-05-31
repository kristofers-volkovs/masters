import numpy as np
import zarr
import numcodecs

from modules.models import ComponentParams


class DataStorage:
    def __init__(self, file_dir, group_structure):
        self.root_map = zarr.open(file_dir, mode='a')
        self.group_structure = group_structure
        numcodecs.blosc.set_nthreads(12)  
    
    def add_metadata(self, key, value):
        self.root_map.array(key, data=value, dtype=object, object_codec=numcodecs.VLenUTF8())
    
    def get_metadata(self):
        metadata_dict = {}
        for key, value in self.root_map.arrays():
            metadata_dict[key] = value[:]
        
        return metadata_dict
    
    def add_item(self, idx, item):
        if str(idx) in self.root_map.group_keys():
            component_group = self.root_map[str(idx)]
        else:
            component_group = self.root_map.create_group(f'{idx}')
        
        for key, value in item.items():
            if key not in self.group_structure:
                raise ValueError(f'Provided key "{key}" not in model structure: {self.group_structure}')
            
            if key in component_group.array_keys():
                component_param = component_group[key]
            else:
                component_param = component_group.zeros(key, shape=value.shape, chunks=False, dtype='f8')
                
            component_param[:] = value[:]
    
    """
    Params
        idx - index of specific element to select
        groups - list of groups to select
    """
    def get_item(self, idx, groups):
        if idx > len(self):
            raise ValueError(f"Index is out of bounds, idx: {idx}, data_len: {len(self)}")
        
        if not set(groups).issubset(self.group_structure):
            raise ValueError(f"Groups {groups} are not in storage defined groups {self.group_structure}")
        
        item_dict = {}
        for key in groups:
            item_dict[key] = self.root_map[f'{idx}/{key}'][:]
        
        return item_dict
    
    def __len__(self):
        idx_list = [int(idx) for idx in self.root_map.group_keys()]
        return np.max(idx_list)
    
    def does_all_groups_contain(self, key):
        for idx in range(len(self)):
            if key not in self.root_map[str(idx)].array_keys():
                return False
        return True
    
    def print_state(self):
        print('=== Data info ===')
        print(f'data_len: {len(self)}')
        print(f'group_structure: {self.group_structure}')
        # print('=== Data example ===')
        # print(self.get_item(len(self) - 1, self.group_structure))
        