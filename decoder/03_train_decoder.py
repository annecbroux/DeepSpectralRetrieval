from torch.utils.data import DataLoader
import dataloader_decoder as dataloader
from decoder_models import *
import torch
import numpy as np
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch import nn
from torch.nn import functional as F
import os
import shutil
import json
import matplotlib.pyplot as plt
import time
import gc

from torch.utils.tensorboard import SummaryWriter

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

import argparse


## Parse input and load config
parser = argparse.ArgumentParser(description='train Decoder model')
parser.add_argument('config_file', type=str)
args = parser.parse_args()

config = json.load(open(args.config_file))

directory = config['directory']
if not(os.path.exists(directory)):
    os.mkdir(directory)
try:
    if os.path.exists(directory+'/config.json'):
        check_if_replace = input('config.json already exists in {}. Do you want to replace it? (y/n)'.format(directory))
        if check_if_replace == 'n':
            print('Exiting')
            sys.exit()
    shutil.copy2(args.config_file, directory+'/config.json')
except shutil.SameFileError:
    pass


## Set-up tensorboard writer
writer = SummaryWriter(log_dir=directory)


## Prepare dataset
train_dataset = dataloader.decoder_dataset(config['input_ds_path'], config['syn_dataset_path'], config['target_freq'],
                                        i_start_ds = config['i_start_ts'], i_end_ds = config['i_start_ts']+config['train_size'],
                                        i_start_spec = config['i_start_spec'], i_end_spec = config['i_end_spec'],
                                        inds_input_features=config['inds_input_vars_'+config['target_freq']],
                            
                                        normalize_input = config['normalize_input'],
                                        normalize_output = config['normalize_output'],

                                        mean_normalization_spec_json = config['normalize_path']+'/means_spec.json',
                                        std_normalization_spec_json = config['normalize_path']+'/stds_spec.json',

                                        mean_normalization_input_npy = config['normalize_path']+'/means_input.npy',
                                        std_normalization_input_npy = config['normalize_path']+'/stds_input.npy',

                                        shuffle_pre_train = config['shuffle_pre_train'])

val_dataset = dataloader.decoder_dataset(config['input_ds_path'], config['syn_dataset_path'], config['target_freq'],
                                           i_start_ds = config['i_start_val'], i_end_ds = config['i_start_val']+config['val_size'],
                                           i_start_spec = config['i_start_spec'], i_end_spec = config['i_end_spec'],
                                           inds_input_features=config['inds_input_vars_'+config['target_freq']],

                                           normalize_input = config['normalize_input'],
                                           normalize_output = config['normalize_output'],

                                           mean_normalization_spec_json = config['normalize_path']+'/means_spec.json',
                                           std_normalization_spec_json = config['normalize_path']+'/stds_spec.json',

                                           mean_normalization_input_npy = config['normalize_path']+'/means_input.npy',
                                           std_normalization_input_npy = config['normalize_path']+'/stds_input.npy',

                                           shuffle_pre_train = config['shuffle_pre_train'])

use_subsampler = config['use_subsampler']

## Training parameters
nb_epochs = config['nb_epochs'] 
batch_size = config['batch_size']
val_batch_size = config['val_batch_size']
lr_initial = config['learning_rate_ini'] 

model_name = config['model_name']
n_input_features = config['n_input_features']
n_channels = config['n_channels']
n_hidden_layers = config['n_hidden_layers']

secondary_loss = config['secondary_loss']
normalize_loss = config['normalize_loss']

use_scheduler = config['scheduler']
if use_scheduler == 'step':
    lr_sched_step = config['lr_sched_step'] 
    lr_sched_gamma = config['lr_sched_gamma'] 
stop_decreasing_after = config['stop_decreasing_after']
    
checkpoint_name = directory+config['checkpoint_name']

ts_size_per_epoch = config['ts_size_per_epoch']
num_workers_loader = config['num_workers_loader']


    
## Set-up model, optimizer and scheduler
device = config['device']
model = locals()[model_name](n_channels,n_hidden_layers,n_input_features).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr_initial)
if use_scheduler=='step':
    scheduler = StepLR(optimizer, step_size=lr_sched_step, gamma=lr_sched_gamma)

init = config['init']
if init=='xavier_normal':
    def weights_init(m):
        if isinstance(m, nn.Conv1d):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)
        if isinstance(m, nn.ConvTranspose1d):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    model.apply(weights_init)
    
    
## Load the checkpoint if it exists
bc = 0
nb_epochs_finished = 0
N = len(train_dataset)#.input_dataset.shape[0]
epoch_loss_previous = 0
try:
    checkpoint = torch.load(checkpoint_name)
    nb_epochs_finished = checkpoint['nb_epochs_finished']
    bc = checkpoint['nb_batches_finished']
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    if (use_scheduler=='step') & (nb_epochs_finished<=stop_decreasing_after):
        scheduler.load_state_dict(checkpoint['scheduler_state'])
    print(f'Checkpoint loaded with {nb_epochs_finished} epochs finished.')
except FileNotFoundError:
    print('Starting from scratch.')
except Exception as e:
    print('Error when loading the checkpoint: ', e)
    exit(1)

    
### Perform the training
# The validation dataloader is defined before training
val_dataloader = DataLoader(val_dataset, num_workers=num_workers_loader, batch_size=val_batch_size, pin_memory=True)

# If we don't use the subsampler, the dataloader is defined once and for all
if not(use_subsampler):
    train_dataloader = DataLoader(train_dataset, num_workers=num_workers_loader, batch_size=batch_size,pin_memory=True, shuffle=config['shuffle_dataloader'])


for epoch in range(nb_epochs_finished,nb_epochs):
    epochloss = 0
    epochsize = 0
    if use_subsampler:
        inds = torch.arange((ts_size_per_epoch*epoch)%N, min(N,(ts_size_per_epoch*epoch)%N + ts_size_per_epoch))
        train_dataloader = DataLoader(torch.utils.data.Subset(train_dataset,inds[torch.randperm(len(inds))]), num_workers=num_workers_loader, batch_size=batch_size, pin_memory=True) #

        print(len(train_dataloader))
        
    for (input_data, target) in iter(train_dataloader):
        input_data = input_data.to(device)
        # input_data[input_data!=input_data]=0
        target = target.to(device)
        output = model.decode(input_data)[:,0,:]#.unsqueeze(1)
        loss = 0.5 * ((output[:,:] - target).pow(2).sum())/ output.size(0)
        epochsize+=output.size(0)
        epochloss+=loss*output.size(0)
        
        if bc%5==0:
            writer.add_scalar('batch_loss',loss,bc/5)
        bc+=1

        if secondary_loss=='full_peak':
            peak = torch.gt(target,.1+torch.min(target,axis=1).values[:,None])
            loss2 = 0.5*((target[peak] - output[peak]).pow(2).sum())/output.size(0)
            loss = loss+loss2
        if secondary_loss=='weighted_peak':
            peak = torch.gt(target,.1+torch.min(target,axis=-1).values[:,None])
            loss2= 0.25*((target - output)*peak).pow(2).sum()/output.size(0)
            loss = loss+loss2
        if secondary_loss=='finer_peak':
            minmax = torch.max(target,axis=-1).values[:,None]-torch.min(target,axis=-1).values[:,None]
            peak = torch.gt(target,torch.min(target,axis=-1).values[:,None]+.75*minmax)
            loss2= 0.125*((target - output)*peak).pow(2).sum()/output.size(0)
            loss = loss+loss2
        if secondary_loss=='finer_peak_old':
            peak = torch.gt(target,.5+torch.min(target,axis=-1).values[:,None])
            loss2= 0.125*((target - output)*peak).pow(2).sum()/output.size(0)
            loss = loss+loss2
        if normalize_loss:
            loss = loss/output.size(-1)
        
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        
        fig = plt.figure(figsize=(10,3))
        plt.plot(output[0][:].cpu().detach().numpy())
        plt.plot(target[0][:].cpu().detach().numpy())
        writer.add_figure('spectrum_train',fig,epoch)
        plt.close()
#     if (epoch > 0)*(epochloss/ts_size_per_epoch > 2*epoch_loss_previous):
#         print('LOSS EXPLODING')
#         break
        
    with torch.no_grad():
        writer.add_scalar('training_loss',epochloss/epochsize, epoch)
        print(epoch, epochloss/epochsize)
        
        if epoch%4==0:
            val_epochloss = 0
            for (input_data, target) in iter(val_dataloader):
                input_data = input_data.to(device)
                target = target.to(device)
                output = model.decode(input_data)[:,0,:]#.unsqueeze(1)
                val_loss = 0.5 * ((output[:,:] - target).pow(2).sum())/ output.size(0)
                val_epochloss+=val_loss*output.size(0)
            val_epochloss = val_epochloss/len(val_dataset)
    
            writer.add_scalar('validation_loss',val_epochloss, epoch)
            fig = plt.figure(figsize=(10,3))
            plt.plot(output[0][:].cpu().detach().numpy())
            plt.plot(target[0][:].cpu().detach().numpy())
            writer.add_figure('spectrum_val',fig,epoch)
            plt.close()

            def writer_histogram(m):
                if ((isinstance(m, nn.Conv1d)) | (isinstance(m, nn.ConvTranspose1d)) | (isinstance(m,nn.Linear))):
                    writer.add_histogram(str(m)+'.weight', m.weight, epoch)
                    writer.add_histogram(str(m)+'.weight.grad', m.weight.grad, epoch)

            model.apply(writer_histogram)

        
    epoch_loss_previous = epochloss/ts_size_per_epoch
    
    checkpoint = {
    'nb_epochs_finished': epoch+1,
    'nb_batches_finished':bc+1,
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    }

    if (use_scheduler=='step') & (epoch<=stop_decreasing_after):
        scheduler.step()
        checkpoint['scheduler_state']=scheduler.state_dict()
    elif (use_scheduler == 'plateau')& (epoch<=stop_decreasing_after):
        scheduler.step(epochloss)
        checkpoint['scheduler_state']=scheduler.state_dict()
    torch.save(checkpoint, checkpoint_name)
    
    if (epoch+1)%20==0:
        torch.save(checkpoint, checkpoint_name[:-4]+'_'+str(epoch)+'.pth')
        
#     gc.collect()