'''
This dataloader accepts a path for NC files, timesteps_in (Int) and timesteps_out (Int).
It returns three objects, two tensors with time data and one with time information for selected indexes.

AUTHOR: ANIKET PANT, GT MSE, 04/25/2023
'''

import os
import torch
import numpy as np
from torch.utils.data import Dataset
import netCDF4 as nc


def open_aqua_modis_file(file_path):
    '''
    Handler function for opening AQUA MODIS file.
    '''
    try:
        dataset = nc.Dataset(file_path)
        return dataset
    except Exception as e:
        print(f"Error opening file: {e}")
        return None
    
def prep_data_plot(dataset):
    chlor_a = dataset.variables['chlor_a']
    fill_value = chlor_a._FillValue
    valid_min = chlor_a.valid_min
    valid_max = chlor_a.valid_max

    masked_chlor_a = np.ma.masked_outside(chlor_a, valid_min, valid_max)
    masked_chlor_a = np.ma.masked_values(masked_chlor_a, fill_value)
    
    return {
        "chlor_a": masked_chlor_a,
        "display_min": chlor_a.display_min,
        "display_max": chlor_a.display_max,
    }

class ChloroDataset(Dataset):
    def __init__(self, files_directory, timesteps_in, timesteps_out):
        super(ChloroDataset, self).__init__()
        self.files_directory = files_directory # file directory where all NC files are located.
        self.data_files = sorted(os.listdir(self.files_directory)) # sort for chrono order
        self.data_files = [os.path.join(self.files_directory, x) for x in self.data_files] # add full path for loading
        self.timesteps_in = timesteps_in # amount of timesteps to feed to model
        self.timesteps_out = timesteps_out # amount of timesteps to predict out of model

    def __getitem__(self, index):
        start_index = index
        end_index_in = index + self.timesteps_in
        end_index_out = end_index_in + self.timesteps_out

        if end_index_out > len(self.data_files):
            raise IndexError("Sample index out of range")

        time_in_data = []
        time_out_data = []

        for i in range(start_index, end_index_in):
            dataset = open_aqua_modis_file(self.data_files[i]) # load dataset
            prep_data = prep_data_plot(dataset)["chlor_a"] # process dataset
            time_in_data.append(prep_data) # append dataset

        for i in range(end_index_in, end_index_out):
            dataset = open_aqua_modis_file(self.data_files[i]) # load dataset
            prep_data = prep_data_plot(dataset)["chlor_a"] # process dataset
            time_out_data.append(prep_data)

        time_in_tensor = torch.tensor(np.array(time_in_data), dtype=torch.float32)
        time_out_tensor = torch.tensor(np.array(time_out_data), dtype=torch.float32)

        return {
            "t_gt": time_in_tensor,
            "t_forecast": time_out_tensor,
            "t_info": ([start_index, end_index_in], [end_index_in, end_index_out])
        }
    
    def __len__(self):
        return len(self.data_files) - self.timesteps_in - self.timesteps_out