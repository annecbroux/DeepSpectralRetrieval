import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import random
import json
import time
import xarray as xr

class encoder_dataset(Dataset):
    def __init__(self, data_file, rgate_start = 0, rgate_end = -1, batch_shape=25, step_for_data_augmentation=10, mean_normalization_json = None, std_normalization_json = None, moments = False, vel_array = None):
        """
        data_file: path to the hdf5 file containing the data (sX, sKa, sW on same (time, range, vel grid))
        rgate_start: starting range gate (default 0, set to 25 to skip first 25 range gates e.g. because of clutter)
        rgate_end: ending range gate (default -1, set to a given number to skip last range gates e.g. no relevant data)
        batch_shape: number of range gates to include in each batch (default 25)
        step_augmentation: step for data augmentation (i.e. repeated data with a shift of step_augmentation range gates) (default 10). Set to < 0 to disable data augmentation.
        mean_normalization_json: path to json file containing the mean of the spectra in the synthetic dataset
        std_normalization_json: path to json file containing the standard deviation of the spectra in the synthetic dataset
        calc_moments: if True, calculate the moments of the spectra (default False)
        vel_array: array of velocities (default None, only neede if calc_moments is True)
        """

        with h5py.File(data_file,'r') as f:
            sX = f['sX'][:]
            sKa = f['sKa'][:]
            sW = f['sW'][:]
            
        if step_for_data_augmentation < 0:
            self.input_data_X = torch.from_numpy(sX[:,rgate_start:rgate_end,:])
            self.input_data_Ka = torch.from_numpy(sKa[:,rgate_start:rgate_end,:])
            self.input_data_W = torch.from_numpy(sW[:,rgate_start:rgate_end,:])
        else:
            indices_start = torch.arange(rgate_start, rgate_end-batch_shape, step_for_data_augmentation)
            self.input_data_X = torch.hstack([torch.from_numpy(sX[:,istart:istart+batch_shape,:]) for istart in indices_start])
            self.input_data_Ka = torch.hstack([torch.from_numpy(sKa[:,istart:istart+batch_shape,:]) for istart in indices_start])
            self.input_data_W = torch.hstack([torch.from_numpy(sW[:,istart:istart+batch_shape,:]) for istart in indices_start]) 

        self.moments = moments
        if moments:
            self.calc_moments(batch_shape)
            self.vel_array = vel_array # TO DO maybe this is in the data file
        
        if not (mean_normalization_json is None):
            self.normalize_input_data(mean_normalization_json, std_normalization_json)
            
        input_x_reshaped = self.input_data_X.view(-1,batch_shape,self.input_data_X.shape[-1])
        input_ka_reshaped = self.input_data_Ka.view(-1,batch_shape,self.input_data_Ka.shape[-1])
        input_w_reshaped = self.input_data_W.view(-1,batch_shape,self.input_data_W.shape[-1])

        self.input_data = torch.stack((input_x_reshaped, input_ka_reshaped, input_w_reshaped),axis=1).float()
        
        
    def __len__(self):
        return self.input_data.shape[0]


    def __getitem__(self, idx):
        if self.moments:
            return self.input_data[idx,:,:,:] , self.minW[idx,:], self.minKa[idx,:], self.minX[idx,:], self.zeW_data[idx,:], self.zeKa_data[idx,:], self.zeX_data[idx,:], self.mdvW_data[idx,:], self.mdvKa_data[idx,:], self.mdvX_data[idx,:]    
        else:
            return self.input_data[idx,:,:,:]

    
    def normalize_input_data(self, mean_normalization_json, std_normalization_json):
        with open(mean_normalization_json) as mnj:
            mean_norm = json.load(mnj)
            mnsx = mean_norm['specX']
            mnska = mean_norm['specKa']
            mnsw = mean_norm['specW']
            print(mnsw)
        with open(std_normalization_json) as snj:
            std_norm = json.load(snj)
            snsx = std_norm['specX']
            snska = std_norm['specKa']
            snsw = std_norm['specW']
        
        self.specX_mean = mnsx
        self.specX_std = snsx
        self.specKa_mean = mnska
        self.specKa_std = snska
        self.specW_mean = mnsw
        self.specW_std = snsw
        
        self.input_data_X = (self.input_data_X-mnsx)/snsx
        self.input_data_Ka = (self.input_data_Ka-mnska)/snska
        self.input_data_W = (self.input_data_W-mnsw)/snsw
        
        
    def calc_moments(self, batch_shape):      
        vel = self.vel_array
        dv = vel[1]-vel[0]
        
        minX = torch.min(self.input_data_X,axis=-1).values
        self.minX = minX.view(-1,batch_shape).float()
        minKa = torch.min(self.input_data_Ka,axis=-1).values
        self.minKa = minKa.view(-1,batch_shape).float()
        minW = torch.min(self.input_data_W,axis=-1).values
        self.minW = minW.view(-1,batch_shape).float()

        mdv_X = torch.mean(torch.pow(10,(self.input_data_X/10))*vel,axis=-1)
        mdv_Ka = torch.mean(torch.pow(10,(self.input_data_Ka/10))*vel,axis=-1)
        mdv_W = torch.mean(torch.pow(10,(self.input_data_W/10))*vel,axis=-1) 
        mdvX_mean = torch.mean(mdv_X)
        mdvKa_mean = torch.mean(mdv_Ka)
        mdvW_mean = torch.mean(mdv_W)
        mdvX_std = torch.std(mdv_X)
        mdvKa_std = torch.std(mdv_Ka)
        mdvW_std = torch.std(mdv_W)
        mdv_X = (mdv_X-mdvX_mean)/mdvX_std
        mdv_Ka = (mdv_Ka-mdvKa_mean)/mdvKa_std
        mdv_W = (mdv_W-mdvW_mean)/mdvW_std
                
        ze_X = 10*torch.log10(torch.pow(10,(self.input_data_X/10)).sum(axis=-1)*dv)
        ze_Ka = 10*torch.log10(torch.pow(10,(self.input_data_Ka/10)).sum(axis=-1)*dv)
        ze_W = 10*torch.log10(torch.pow(10,(self.input_data_W/10)).sum(axis=-1)*dv)

        zex_mean = torch.mean(ze_X)
        zeka_mean = torch.mean(ze_Ka)
        zew_mean = torch.mean(ze_W)
        
        zex_std = torch.std(ze_X)
        zeka_std = torch.std(ze_Ka)
        zew_std = torch.std(ze_W)

        ze_X = (ze_X-zex_mean)/zex_std
        ze_Ka = (ze_Ka-zeka_mean)/zeka_std
        ze_W = (ze_W-zew_mean)/zew_std
        
        ## reshape to batch shape and set as fields
        self.zeX_data = (ze_X.view(-1,batch_shape)).float()
        self.zeKa_data = (ze_Ka.view(-1,batch_shape)).float()
        self.zeW_data = (ze_W.view(-1,batch_shape)).float()

        self.mdvX_data = (mdv_X.view(-1,batch_shape)).float()
        self.mdvKa_data = (mdv_Ka.view(-1,batch_shape)).float()
        self.mdvW_data = (mdv_W.view(-1,batch_shape)).float() 

        self.mdvX_mean = mdvX_mean
        self.mdvKa_mean = mdvKa_mean
        self.mdvW_mean = mdvW_mean

        self.mdvX_std = mdvX_std
        self.mdvKa_std = mdvKa_std
        self.mdvW_std = mdvW_std

        self.zex_mean = zex_mean
        self.zeka_mean = zeka_mean
        self.zew_mean = zew_mean

        self.zex_std = zex_std
        self.zeka_std = zeka_std
        self.zew_std = zew_std
        
        print(self.zeW_data.shape)
        
        
    
if __name__ == '__main__':
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