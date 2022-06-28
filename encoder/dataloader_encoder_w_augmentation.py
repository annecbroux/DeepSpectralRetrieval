"""
This script contains the data loader for the encoder model.

It prepares the dataset (originally stored in hdf5 with X and W-band spectra mapped on a common (time, range, velocity) grid.
The hdf5 file should include the fields "sW" and "sX"

The dataloader includes normalization if provided (should be normalized with the same statistics as the decoder part, i.e. containing mean and std deviation of spectra)
E.g. one JSON file containing the mean values and another containing the std (with each the fields "spectrum_W_256" and "spectrum_X_256")
"""


import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import random
import json
import time

rvel = np.linspace(-6.92,6.92,256, endpoint=False)
dv = rvel[1]-rvel[0]


class encoder_dataset(Dataset):
    
    def __init__(self, data_file, rgate_start = 25, rgate_end = 75, batch_shape=25, step=1, mean_normalization_json = None, std_normalization_json = None):
        with h5py.File(data_file,'r') as f:
            sW = f['sW'][:]
            sX = f['sX'][:]
            
        indices_start = torch.arange(rgate_start, rgate_end-batch_shape, step)

        self.input_data_W = torch.hstack([torch.from_numpy(sW[:,istart:istart+25,:])+10*np.log10(dv) for istart in indices_start]) # We have to add the dv term for consistency with synthetic dataset
        self.input_data_X = torch.hstack([torch.from_numpy(sX[:,istart:istart+25,:])+10*np.log10(dv) for istart in indices_start])
	
        self.calc_moments(batch_shape) # Full moments might not be necessary but we at least need the min
        
        if not (mean_normalization_json is None):
            self.normalize_input_data(mean_normalization_json, std_normalization_json)
            
        input_w_reshaped = self.input_data_W.view(-1,batch_shape,256)
        input_x_reshaped = self.input_data_X.view(-1,batch_shape,256)
        self.input_data = torch.stack((input_w_reshaped,input_x_reshaped),axis=1).float()
        
        
    def __len__(self):
        return self.input_data.shape[0]


    def __getitem__(self, idx):
        return self.input_data[idx,:,:,:], self.minW[idx,:], self.minX[idx,:], self.zeW_data[idx,:], self.zeX_data[idx,:], self.mdvW_data[idx,:], self.mdvX_data[idx,:]

    
    def normalize_input_data(self, mean_normalization_json, std_normalization_json):
        with open(mean_normalization_json) as mnj:
            mean_norm = json.load(mnj)
            mnsw = mean_norm['spectrum_W_256']
            mnsx = mean_norm['spectrum_X_256']
            print(mnsw)
        with open(std_normalization_json) as snj:
            std_norm = json.load(snj)
            snsw = std_norm['spectrum_W_256']
            snsx = std_norm['spectrum_X_256']
        
        self.specW_mean = mnsw
        self.specW_std = snsw
        self.specX_mean = mnsx
        self.specX_std = snsx
        
        self.input_data_W = (self.input_data_W-mnsw)/snsw
        self.input_data_X = (self.input_data_X-mnsx)/snsx
        
        
    def calc_moments(self, batch_shape):      
        vel = np.linspace(-6.92,6.92,256,endpoint=False)
        dv = vel[1]-vel[0]
        
        minW = torch.min(self.input_data_W,axis=-1).values
        self.minW = minW.view(-1,batch_shape).float()
        minX = torch.min(self.input_data_X,axis=-1).values
        self.minX = minX.view(-1,batch_shape).float()
        
        mdv_W = torch.mean(torch.pow(10,(self.input_data_W/10))*vel,axis=-1)
        mdv_X = torch.mean(torch.pow(10,(self.input_data_X/10))*vel,axis=-1)
        mdvW_mean = torch.mean(mdv_W)
        mdvX_mean = torch.mean(mdv_X)
        mdvW_std = torch.std(mdv_W)
        mdvX_std = torch.std(mdv_X)
        mdv_W = (mdv_W-mdvW_mean)/mdvW_std
        mdv_X = (mdv_X-mdvX_mean)/mdvX_std
        
        ze_W = 10*torch.log10(torch.pow(10,(self.input_data_W/10)).sum(axis=-1)*dv)
        ze_X = 10*torch.log10(torch.pow(10,(self.input_data_X/10)).sum(axis=-1)*dv)
        zew_mean = torch.mean(ze_W)
        zex_mean = torch.mean(ze_X)
        zew_std = torch.std(ze_W)
        zex_std = torch.std(ze_X)
        ze_W = (ze_W-zew_mean)/zew_std
        ze_X = (ze_X-zex_mean)/zex_std
        
        self.zeW_data = (ze_W.view(-1,batch_shape)).float()
        self.zeX_data = (ze_X.view(-1,batch_shape)).float()
        self.mdvW_data = (mdv_W.view(-1,batch_shape)).float() 
        self.mdvX_data = (mdv_X.view(-1,batch_shape)).float()
        self.mdvW_mean = mdvW_mean
        self.mdvX_mean = mdvX_mean
        self.mdvW_std = mdvW_std
        self.mdvX_std = mdvX_std
        self.zew_mean = zew_mean
        self.zex_mean = zex_mean
        self.zew_std = zew_std
        self.zex_std = zex_std
        
        print(self.zeW_data.shape)
        
        
    
if __name__ == '__main__':
    # For some tests
    tic = time.time()

    train_dataset = encoder_dataset('20210122_2200_2359_X_W_av_clean_corrZe',batch_shape=25)

    train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True, pin_memory=True) #

    tac = time.time()
    for batch_idx, data in enumerate(train_dataloader):
        print(batch_idx)
        print(data[0].shape)
        if batch_idx==500:
            print(time.time()-tic)
            print(time.time()-tac)
            break
