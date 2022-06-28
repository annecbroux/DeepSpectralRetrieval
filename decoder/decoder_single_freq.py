"""
This script performs the training of the decoder model.
JSON config file should be parsed as input (see example in the same directory), containing all the relevant info
(i.e. dataset to be used, architecture of the decoder, etc.)
"""

from torch.utils.data import DataLoader
import dataloader_decoder as dataloader
from decoder_architectures_single_freq import *
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
parser = argparse.ArgumentParser(description='train Decoder model with provided config parameters')
parser.add_argument('config_file', type=str)
args = parser.parse_args()
config = json.load(open(args.config_file))

directory = config['directory']
if not(os.path.exists(directory)):
    os.mkdir(directory)
try:
    shutil.copy2(args.config_file, directory)
except shutil.SameFileError:
    pass


## Set-up tensorboard writer
writer = SummaryWriter(log_dir=directory)


## Prepare dataset
train_dataset = dataloader.decoder_dataset(config['input_array'], config['input_h5'], config['target_freq'],
                                           i_start_ds = config['i_start_ds'], i_end_ds = config['i_end_ds'],
                                           i_start_spec = config['i_start_spec'], i_end_spec = config['i_end_spec'],
                                           mean_normalization_json = config['mean_normalization_json'],
                                           std_normalization_json = config['std_normalization_json'],
                                           mean_normalization_npy = config['mean_normalization_npy'],
                                           std_normalization_npy = config['std_normalization_npy'],
                                           shuffle_pre_train = config['shuffle_pre_train'],
                                           normalize_input = config['normalize_input'],
                                           normalize_output = config['normalize_output'])

val_dataset = dataloader.decoder_dataset(config['input_array'], config['input_h5'], config['target_freq'],
                                           i_start_ds = config['i_start_val_ds'], i_end_ds = config['i_end_val_ds'],
                                           i_start_spec = config['i_start_spec'], i_end_spec = config['i_end_spec'],
                                           mean_normalization_json = config['mean_normalization_json'],
                                           std_normalization_json = config['std_normalization_json'],
                                           mean_normalization_npy = config['mean_normalization_npy'],
                                           std_normalization_npy = config['std_normalization_npy'],
                                           shuffle_pre_train = config['shuffle_pre_train'],
                                           normalize_input = config['normalize_input'],
                                           normalize_output = config['normalize_output'])

use_subsampler = config['use_subsampler']
with_wind=config['with_wind'] # Old stuff, keeping for consistency

## Training and architecture parameters
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
elif use_scheduler == 'plateau':
    lr_factor = config['lr_factor']
    lr_patience = config['lr_patience']
    lr_threshold = config['lr_threshold']
    lr_min = config['lr_min']
    
checkpoint_name = directory+config['checkpoint_name']

ts_size_per_epoch = config['ts_size_per_epoch']
num_workers_loader = config['num_workers_loader']


    
## Set-up model, optimizer and scheduler
device = config['device']
model = locals()[model_name](n_channels,n_hidden_layers,n_input_features).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr_initial)
if use_scheduler=='step':
    scheduler = StepLR(optimizer, step_size=lr_sched_step, gamma=lr_sched_gamma)
elif use_scheduler=='plateau':
    scheduler = ReduceLROnPlateau(optimizer, factor = lr_factor, patience = lr_patience, threshold = lr_threshold, min_lr = lr_min)
stop_decreasing_after = config['stop_decreasing_after']

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
N = len(train_dataset)
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
    train_dataloader = DataLoader(train_dataset, num_workers=num_workers_loader, batch_size=batch_size,pin_memory=True)
    
for epoch in range(nb_epochs_finished,nb_epochs):
    epochloss = 0
    epochsize = 0
    # With the subsampler we do "pseudo-epochs" with only a part of the dataset (otherwise this is super long) - kind of "super batches". 
    # Not very meaningful per se but helps monitor training.
    if use_subsampler:
        inds = torch.arange((ts_size_per_epoch*epoch)%N, min(N,(ts_size_per_epoch*epoch)%N + ts_size_per_epoch))
        train_dataloader = DataLoader(torch.utils.data.Subset(train_dataset,inds[torch.randperm(len(inds))]), num_workers=num_workers_loader, batch_size=batch_size, pin_memory=True) #
        print(len(train_dataloader))
        
    for (input_data, target) in iter(train_dataloader):
        input_data = input_data.to(device)
        target = target.to(device)
        output = model.decode(input_data)[:,0,:]
        
        # This is the main loss i.e. MSE loss of target vs NN output
        loss = 0.5 * ((output[:,:] - target).pow(2).sum())/ output.size(0)
        epochsize+=output.size(0)
        epochloss+=loss*output.size(0)

        # Here optionally add a secondary loss (gives more weight to peak compared to noise level)
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
        if normalize_loss:
            loss = loss/output.size(-1)
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
       
        # Add stuff to tensorboard for monitoring
        if bc%5==0:
            writer.add_scalar('batch_loss',loss,bc/5)
        bc+=1
       
        fig = plt.figure(figsize=(10,3))
        plt.plot(output[0][:].cpu().detach().numpy())
        plt.plot(target[0][:].cpu().detach().numpy())
        writer.add_figure('spectrum_train',fig,epoch)
        plt.close()

        
    with torch.no_grad():
        # Add more stuff to tensorboard for monitoring
        writer.add_scalar('training_loss',epochloss/epochsize, epoch)
        print(epoch, epochloss/ts_size_per_epoch)

        # Validation every 4 epochs (quite slow)
        if epoch%4==0:
            val_epochloss = 0
            for (input_data, target) in iter(val_dataloader):
                input_data = input_data.to(device)
                target = target.to(device)
                output = model.decode(input_data)[:,0,:]
                val_loss = 0.5 * ((output[:,:] - target).pow(2).sum())/ output.size(0)
                val_epochloss+=val_loss*output.size(0)
            val_epochloss = val_epochloss/len(val_dataset)
    
            # Add more stuff to tensorboard for monitoring
            writer.add_scalar('validation_loss',val_epochloss, epoch)
            fig = plt.figure(figsize=(10,3))
            plt.plot(output[0][:].cpu().detach().numpy())
            plt.plot(target[0][:].cpu().detach().numpy())
            writer.add_figure('spectrum_val',fig,epoch)
            plt.close()

            def writer_histogram(m): # This plots the histograms of weights (helps check for possible vanishing gradients)
                if ((isinstance(m, nn.Conv1d)) | (isinstance(m, nn.ConvTranspose1d)) | (isinstance(m,nn.Linear))):
                    writer.add_histogram(str(m)+'.weight', m.weight, epoch)
                    writer.add_histogram(str(m)+'.weight.grad', m.weight.grad, epoch)

            model.apply(writer_histogram)

        
    epoch_loss_previous = epochloss/ts_size_per_epoch
    # Save the checkpoint
    checkpoint = {
    'nb_epochs_finished': epoch+1,
    'nb_batches_finished':bc+1,
    'model_state': model.state_dict(),
    'optimizer_state': optimizer.state_dict(),
    }

    # If need be, scheduler step
    if (use_scheduler=='step') & (epoch<=stop_decreasing_after):
        scheduler.step()
        checkpoint['scheduler_state']=scheduler.state_dict()
    elif (use_scheduler == 'plateau')& (epoch<=stop_decreasing_after):
        scheduler.step(epochloss)
        checkpoint['scheduler_state']=scheduler.state_dict()
    torch.save(checkpoint, checkpoint_name)
    
    if (epoch+1)%20==0:
        torch.save(checkpoint, checkpoint_name[:-4]+'_'+str(epoch)+'.pth')
        

