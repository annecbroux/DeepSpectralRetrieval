import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import random
import json
import time
import gc

    
   
class decoder_dataset(Dataset):

    def __init__(self, input_data_file, target_data_file, target_freq='W', i_start_ds = 0, i_end_ds=-1, i_start_spec = 0, i_end_spec = 257, inds_input_features = [], normalize_input = True, normalize_output = True, mean_normalization_spec_json = None, std_normalization_spec_json = None, mean_normalization_input_npy = None, std_normalization_input_npy = None, shuffle_pre_train = False):
        super(Dataset, self).__init__()
        self.i_start_ds = i_start_ds
        self.i_end_ds = i_end_ds
        self.target_dataset = h5py.File(target_data_file, 'r')[f'spec{target_freq}']
        self.input_dataset = h5py.File(input_data_file,'r')['input_features']
        self.norm_file = mean_normalization_spec_json
        self.norm_input = normalize_input
        self.norm_output = normalize_output
        self.i_start_spec = i_start_spec
        self.i_end_spec = i_end_spec
        self.inds_input_features = inds_input_features
        # if target_freq=='W':
        #     self.inds_input_features = [0,1,2,5,8,9]
        # elif target_freq=='Ka':
        #     self.inds_input_features = [0,1,2,4,7,9]
        # elif target_freq=='X':
        #     self.inds_input_features = [0,1,2,3,6,9]
        if not(mean_normalization_spec_json is None):
            with open(mean_normalization_spec_json) as mn:
                mean_norm = json.load(mn)
                self.target_mean_for_normalization = mean_norm[f'spec{target_freq}']
            with open(std_normalization_spec_json) as sn:
                std_norm = json.load(sn)
                self.target_std_for_normalization = std_norm[f'spec{target_freq}']
            self.input_mean_for_normalization = torch.from_numpy(np.load(mean_normalization_input_npy)[self.inds_input_features]) 
            self.input_std_for_normalization = torch.from_numpy(np.load(std_normalization_input_npy)[self.inds_input_features])
            
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
#         print(idx)

        input_data = torch.from_numpy(self.input_dataset[idx+self.i_start_ds,self.inds_input_features])
        target_data = torch.from_numpy(self.target_dataset[idx+self.i_start_ds,self.i_start_spec:self.i_end_spec])
            
        if (not(self.norm_file is None) & self.norm_input) :
            input_data = (input_data-self.input_mean_for_normalization)/self.input_std_for_normalization
        if (not(self.norm_file is None) & self.norm_output):
            target_data = (target_data-self.target_mean_for_normalization)/self.target_std_for_normalization

        return (input_data.float(), target_data)


if __name__ == '__main__':
    import json
    tic = time.time()
    
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

    config = json.load(open('config_decoder.json'))

    train_dataset = decoder_dataset(config['input_ds_path'], config['syn_dataset_path'], config['target_freq'],
                                            i_start_ds = config['i_start_ts'], i_end_ds = config['i_start_ts']+config['train_size'],
                                            i_start_spec = config['i_start_spec'], i_end_spec = config['i_end_spec'],
                                            inds_input_features=config['inds_input_vars_'+config['target_freq']],
                                
                                            normalize_input = config['normalize_input'],
                                            normalize_output = config['normalize_output'],

                                            mean_normalization_spec_json = None, #config['normalize_path']+'/means_spec.json',
                                            std_normalization_spec_json = None, #config['normalize_path']+'/stds_spec.json',

                                            mean_normalization_input_npy = config['normalize_path']+'/means_input.npy',
                                            std_normalization_input_npy = config['normalize_path']+'/stds_input.npy',

                                            shuffle_pre_train = config['shuffle_pre_train'])
    
    
    train_dataloader = DataLoader(train_dataset, num_workers=32, batch_size=5,pin_memory=True, shuffle=True)

    # from torch.utils.data.sampler import RandomSampler
    # sampler = RandomSampler(train_dataset)#,num_samples=config['batch_size'])
    # train_dataloader = DataLoader(train_dataset, num_workers=32, batch_size = config['batch_size'], sampler=sampler,pin_memory=True,shuffle=False)

    print(len(train_dataloader))
    tac = time.time()
    for batch_idx, data in enumerate(train_dataloader):
        print(data.max())#[0].shape)
        print(batch_idx)
        if batch_idx==100:
            print(time.time()-tic)
            print(time.time()-tac)
            break