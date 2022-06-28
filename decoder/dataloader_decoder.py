import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import random
import json
import time
import gc
    
    
class decoder_dataset(Dataset):
    """
    Prepares the dataset, including normalization if specified
    """
    def __init__(self, input_data_file, target_data_file, target_freq='W', i_start_ds = 0, i_end_ds=-1, i_start_spec = 0, i_end_spec = 257,normalize_input = True, normalize_output = True, mean_normalization_json = None, std_normalization_json = None, mean_normalization_npy = None, std_normalization_npy = None, shuffle_pre_train = False):
        super(Dataset, self).__init__()
        self.i_start_ds = i_start_ds
        self.i_end_ds = i_end_ds
        self.target_dataset = h5py.File(target_data_file, 'r')[f'spectrum_{target_freq}_256']
        self.input_dataset = h5py.File(input_data_file,'r')['input_features']
        self.norm_file = mean_normalization_json
        self.norm_input = normalize_input
        self.norm_output = normalize_output
        self.i_start_spec = i_start_spec
        self.i_end_spec = i_end_spec
        if target_freq=='W':
            self.inds_input_features = [0,1,2,3,4,5,6,7,9,11]
        elif target_freq=='X':
            self.inds_input_features = [0,1,2,3,4,5,6,8,10,12]
        if not(mean_normalization_json is None):
            with open(mean_normalization_json) as mn:
                mean_norm = json.load(mn)
                self.target_mean_for_normalization = mean_norm[f'spectrum_{target_freq}_256']
            with open(std_normalization_json) as sn:
                std_norm = json.load(sn)
                self.target_std_for_normalization = std_norm[f'spectrum_{target_freq}_256']
            self.input_mean_for_normalization = torch.from_numpy(np.load(mean_normalization_npy)[self.inds_input_features]) 
            self.input_std_for_normalization = torch.from_numpy(np.load(std_normalization_npy)[self.inds_input_features])
            
        if shuffle_pre_train:
            self.shuffle = torch.randperm(self.input_dataset.shape[0])
        else:
            self.shuffle = None
        
    def __len__(self):
        return self.i_end_ds - self.i_start_ds


    def __getitem__(self, idx0):
        if not(self.shuffle is None):
            idx = self.shuffle[idx0]
        else:
            idx = idx0

        input_data = torch.from_numpy(self.input_dataset[idx+self.i_start_ds,self.inds_input_features])
        target_data = torch.from_numpy(self.target_dataset[idx+self.i_start_ds,self.i_start_spec:self.i_end_spec])
            
        if (not(self.norm_file is None) & self.norm_input) :
            input_data = (input_data-self.input_mean_for_normalization)/self.input_std_for_normalization
        if (not(self.norm_file is None) & self.norm_output):
            target_data = (target_data-self.target_mean_for_normalization)/self.target_std_for_normalization

        return (input_data.float(), target_data)


if __name__ == '__main__':
    # For some tests
    import json
    tic = time.time()
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    train_dataset = decoder_dataset("../input_newParams_w_minspec_log.h5","/data/acbr_spectra_dbcreation/newParams_all_wind_ts.h5",shuffle_pre_train = 0)
    inds = torch.arange(0,1000000)
    train_dataloader = DataLoader(torch.utils.data.Subset(train_dataset,inds[torch.randperm(len(inds))]), num_workers=32, batch_size=250, pin_memory=True)
    print(len(train_dataloader))
    tac = time.time()
    for batch_idx, data in enumerate(train_dataloader):
        print(batch_idx)
        if batch_idx==100:
            print(time.time()-tic)
            print(time.time()-tac)
            break
