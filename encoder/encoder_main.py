"""
This script performs the training of the encoder model.
JSON config file should be parsed as input (see example in the same directory), containing all the relevant info
(i.e. dataset to be used, architecture of the encoder, location of the pretrained decoder models)
"""

import h5py
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
from torch.nn import functional as F
import json
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append('/home/billault/ml_design/decoder/')
from encoder_architectures import *
from decoder_architectures_single_freq import *
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import argparse
import shutil
import os
import warnings
warnings.filterwarnings("ignore")


## Parse input and load config
parser = argparse.ArgumentParser(description='train Encoder model with specified config')
parser.add_argument('config_file', type=str)
parser.add_argument('--data_file',type=str,default='')
args = parser.parse_args()

config = json.load(open(args.config_file))

directory = config['directory']
if not(os.path.exists(directory)):
    os.mkdir(directory)
try:
    shutil.copy2(args.config_file, directory)
except shutil.SameFileError:
    pass

#--------------------------------------------------

## Load decoder data and normalization parameters
device = config['device']

config_decW = json.load(open(config['config_W_decoder']))
config_decX = json.load(open(config['config_X_decoder']))

model_name_decW=config_decW['model_name'] 
nb_channels_decW, n_hidden_units_decW, n_input_features_decW = config_decW['n_channels'], config_decW['n_hidden_layers'], config_decW['n_input_features']
model_name_decX=config_decX['model_name'] 
nb_channels_decX, n_hidden_units_decX, n_input_features_decX = config_decX['n_channels'], config_decX['n_hidden_layers'], config_decX['n_input_features']

# We assume that the normalization parameters are the same for X and W decoders (that's what it should be)
mean_normalization_json = config_decW['mean_normalization_json']
std_normalization_json = config_decW['std_normalization_json']


# Load model states for W and X decoder
checkpoint_W = torch.load(config_decW['directory']+config_decW['checkpoint_name'])
model_W_state_dict = checkpoint_W['model_state']

checkpoint_X = torch.load(config_decX['directory']+config_decX['checkpoint_name'])
model_X_state_dict = checkpoint_X['model_state']

decoderW = locals()[model_name_decW](nb_channels_decW, n_hidden_units_decW, n_input_features_decW).to(device)
decoderW.load_state_dict(model_W_state_dict)
decoderX = locals()[model_name_decX](nb_channels_decX, n_hidden_units_decX, n_input_features_decX).to(device)
decoderX.load_state_dict(model_X_state_dict)
decoderW.eval()
decoderX.eval()


#--------------------------------------------------
## Prepare dataset and dataloader
if args.data_file=='':
    data_file = config['data_file']
else:
    data_file = args.data_file
rgate_start = config['rgate_start']
rgate_end = config['rgate_end']
num_workers_dataloader = config['num_workers_dataloader']
batch_size = config['batch_size']
batch_shape = config['batch_shape']
if config['augmented_dataset']:
    from dataloader_encoder_w_augmentation import encoder_dataset
    step_for_data_augmentation = config['step_for_data_augmentation']
    dataset = encoder_dataset(data_file, rgate_start=rgate_start, rgate_end=rgate_end, batch_shape = batch_shape, step=step_for_data_augmentation, mean_normalization_json = mean_normalization_json, std_normalization_json = std_normalization_json)
else: # This is deprecated - config file should be set to augmented_dataset = True
    from dataloader_encoder import encoder_dataset
    dataset = encoder_dataset(data_file, rgate_start=rgate_start, rgate_end=rgate_end, batch_shape = batch_shape, mean_normalization_json = mean_normalization_json, std_normalization_json = std_normalization_json)
    
force_min_noise_from_latent = config['force_min_noise_from_latent']
if force_min_noise_from_latent:
    mean_noise_W = np.load(config_decW['mean_normalization_npy'])[-2]
    mean_noise_X = np.load(config_decW['mean_normalization_npy'])[-1]
    std_noise_W = np.load(config_decW['std_normalization_npy'])[-2]
    std_noise_X = np.load(config_decW['std_normalization_npy'])[-1]
force_min_noise = config['force_min_noise']    

constrain_latent = config['constrain_latent']
epoch_start_secloss = config['epoch_start_secloss']
log_dm = config['log_dm']
if constrain_latent:
    mean_normalization_npy = config_decW['mean_normalization_npy']
    std_normalization_npy = config_decW['std_normalization_npy']
    mean_latent = torch.from_numpy(np.load(mean_normalization_npy)).to(device)
    std_latent = torch.from_numpy(np.load(std_normalization_npy)).to(device)
N = len(dataset)
dataloader = DataLoader(dataset, num_workers=num_workers_dataloader,batch_size=batch_size, shuffle=True, pin_memory=True) #


#---------------------------------------------------
## Prepare encoder model

# Set up model
nb_channels_enc = config['nb_channels_enc']
encoder_name = config['encoder_name']
                                 
encoder = locals()[encoder_name](nb_channels_enc).to(device)
init = config['init']
if init=='xavier_normal':
    def weights_init(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
            torch.nn.init.zeros_(m.bias)
    encoder.apply(weights_init)


    
#---------------------------------------------------
## Set up training parameters
    
# Training parameters and monitoring
lr_ini = config['lr_optimizer_initial']
eps = config['eps_optimizer']
optimizer = torch.optim.Adam(encoder.parameters(),lr=lr_ini, eps=eps)
writer = SummaryWriter(log_dir=directory)
use_scheduler = config['use_scheduler']
if use_scheduler:
    scheduler = StepLR(optimizer, step_size=config['scheduler_step'], gamma=config['scheduler_gamma'])
n_epochs = config['n_epochs']
loss_method = config['loss_method']
use_lin_loss= config['use_lin_loss']
start_lin_loss = config['start_lin_loss']

# Remove gradient update for decoder parameters 
# (I think this is not necessary per se because the optimizer only applies to the encoder part anyways, but this is for safety + seems to speed up)
for p in decoderW.parameters():
    p.requires_grad = False
for p in decoderX.parameters():
    p.requires_grad = False

    
#-----------------------------------------------
## If we use a pretrained model, we load the starting point here
if config["use_pretrained_model"]:
    pretrained_model_checkpoint = torch.load(config["pretrained_model_checkpoint"])
    encoder.load_state_dict(pretrained_model_checkpoint['model_state'])
    if config["use_pretrained_optimizer"]:
        optimizer.load_state_dict(pretrained_model_checkpoint['optimizer_state'])
    print('Loaded pretrained model')
    
    
#-----------------------------------------------
## Load the current checkpoint if it exists
checkpoint_name = directory+config['checkpoint_name']
bc = 0
nb_epochs_finished = 0
epoch_loss_previous = 0
try:
    checkpoint = torch.load(checkpoint_name)
    nb_epochs_finished = checkpoint['nb_epochs_finished']
    bc = checkpoint['nb_batches_finished']
    encoder.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    if use_scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state'])
    print(f'Checkpoint loaded with {nb_epochs_finished} epochs finished.')
except FileNotFoundError:
    print('Starting from scratch.')
except:
    print('Error when loading the checkpoint.')
    exit(1)
    
#---------------------------------------------
## Training loop
    
for epoch in range(nb_epochs_finished,n_epochs):
    epoch_loss = 0
    epoch_loss_x = 0
    epoch_loss_w = 0
    
    for sub_input_data, sub_minW, sub_minX, _, _, _, _ in iter(dataloader):
        sub_input_data = sub_input_data.to(device)
        target_w = sub_input_data[:,0,:,:]
        target_x = sub_input_data[:,1,:,:]
        
        latent = encoder.encode(sub_input_data)
        
        # Depending on the model used, the latent variables might be in the features or channel dimension, so we check if axis permutation is needed
        if latent.shape[-1]==1:
            latent = latent.transpose(1,3)

        # We select the indices of the latent variables to be used in the X and W emulators.
        if force_min_noise_from_latent:
            sub_minW = ((sub_minW-mean_noise_W)/std_noise_W).to(device)
            sub_minX = ((sub_minX-mean_noise_X)/std_noise_X).to(device)
            latent_w0 = latent[:,:,:,[0,1,2,3,4,5,6,7,9]].float()
            latent_x0 = latent[:,:,:,[0,1,2,3,4,5,6,8,10]].float()
            latent_w = torch.cat([latent_w0, sub_minW.unsqueeze(1).unsqueeze(-1)],dim=-1)
            latent_x = torch.cat([latent_x0, sub_minX.unsqueeze(1).unsqueeze(-1)],dim=-1)

        else:
            latent_w = latent[:,:,:,[0,1,2,3,4,5,6,7,9,11]].float()
            latent_x = latent[:,:,:,[0,1,2,3,4,5,6,8,10,12]].float()
        
        if constrain_latent: # Here we (optionally) implement the secondary loss which constrains the latent space values (to be in a realistic range)
            latent_denorm = latent*std_latent+mean_latent
            
            log10iwc = torch.log10(torch.exp(latent_denorm[:,:,:,6]))
            loss_iwc = (((log10iwc<(-6))*torch.pow(log10iwc-(-6),2)).sum() + ((log10iwc>(-2.))*torch.pow(log10iwc-(-2.),2)).sum()) /(batch_shape*batch_size)
            
            dm = latent_denorm[:,:,:,5]
            if log_dm:
                dm = torch.exp(dm)
            loss_dm = (((dm<0.)*torch.pow((dm),2)).sum()) /(batch_shape*batch_size)
            
            ar = latent_denorm[:,:,:,2]
            loss_ar = (((ar<0.1)*torch.pow((ar-.1),2)).sum() + ((ar>.95)*torch.pow(ar-.95,2)).sum()) /(batch_shape*batch_size)
            
            bms = latent_denorm[:,:,:,3]
            loss_bms = (((bms<0.5)*torch.pow((bms-.5),2)).sum() + ((bms>3.2)*torch.pow(bms-3.2,2)).sum()) /(batch_shape*batch_size)
            
            log10ams = torch.log10(torch.exp(latent_denorm[:,:,:,0]))
            loss_ams = (((log10ams<(-4))*torch.pow((log10ams-(-4)),2)).sum() + ((log10ams>2.5)*torch.pow(log10ams-2.5,2)).sum()) /(batch_shape*batch_size)
            
            bas = latent_denorm[:,:,:,4]
            loss_bas = (((bas<0)*torch.pow(bas,2)).sum() + ((bas>2.2)*torch.pow(bas-2.2,2)).sum()) /(batch_shape*batch_size)
            
            log10aas = torch.log10(torch.exp(latent_denorm[:,:,:,1]))
            loss_aas = (((log10aas<(-4.5))*torch.pow((log10aas-(-4.5)),2)).sum() + ((log10aas>1.5)*torch.pow(log10aas-1.5,2)).sum()) /(batch_shape*batch_size)

        
        output_w = decoderW.decode(latent_w.view((-1,10)))[:,0,:].view(sub_input_data[:,0,:,:].shape)
        output_x = decoderX.decode(latent_x.view((-1,10)))[:,0,:].view(sub_input_data[:,1,:,:].shape)
        
                
        # Here loss is simply mse
        loss_w = 0.5 * (output_w - target_w).pow(2).sum()/ (batch_shape*256*batch_size)
        loss_x = 0.5 * (output_x - target_x).pow(2).sum()/ (batch_shape*256*batch_size)
        loss = loss_w + loss_x 
        
        
        #That's the loss we want to track in the tensorboard instance
        writer.add_scalar('batch_loss',loss, bc)
        epoch_loss+=loss*output_w.size(0)
        epoch_loss_x+= loss_x*output_w.size(0)
        epoch_loss_w+= loss_w*output_w.size(0)
        bc+=1
        writer.add_scalar('batch_loss',loss, bc)
        
        if (epoch>265) & (loss.item()>0.17):
            print('LOSS TOO LARGE, EXITING')
            exit()
        
        #Several possibilities for secondary loss (optional)
        if loss_method == 'avg_mse_total_and_peak':
            peakW = torch.gt(target_w,.1+torch.min(target_w,axis=-1).values[:,:,None])
            peakX = torch.gt(target_x,.1+torch.min(target_x,axis=-1).values[:,:,None])
            loss2 = 0.5*((target_w[peakW] - output_w[peakW]).pow(2).sum() + (target_x[peakX] - output_x[peakX]).pow(2).sum())/(batch_shape*batch_size)
            loss = .25*(loss+loss2) 
            
        if (loss_method=='finer_peak') & (epoch > epoch_start_secloss): #Secondary loss can be activated after a certain number of epochs
            minmax = torch.max(target_w,axis=-1).values[:,:,None]-torch.min(target_w,axis=-1).values[:,:,None]
            peakW = torch.gt(target_w,torch.min(target_w,axis=-1).values[:,:,None]+.75*minmax)
            peakX = torch.gt(target_x,torch.min(target_x,axis=-1).values[:,:,None]+.75*minmax)
            loss2_w = .5*((target_w - output_w)*peakW).pow(2).sum()/(batch_shape*256*batch_size)
            loss2_x = .5*((target_x - output_x)*peakX).pow(2).sum()/(batch_shape*256*batch_size)
            loss = loss+.05*(loss2_w+loss2_x)
            
        if (use_lin_loss) & (epoch>start_lin_loss):
            loss_lw = 0.5 * (10**(output_w/10) - 10**(target_w/10)).pow(2).sum()/ (batch_shape*256*batch_size)
            loss_lx = 0.5 * (10**(output_x/10) - 10**(target_x/10)).pow(2).sum()/ (batch_shape*256*batch_size)
            loss=loss+ .05*(loss_lw + loss_lx)
                        
        if constrain_latent:
            loss = loss+(loss_iwc+loss_ar+loss_bms+loss_bas+loss_ams+loss_aas+loss_dm)/7
                        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

    # Some tensorboard stuff
    writer.add_scalar('training_loss',epoch_loss/N, epoch)
    print(epoch, epoch_loss/N,'\n\n')
    writer.add_scalar('training_loss_X',epoch_loss_x/N, epoch)
    writer.add_scalar('training_loss_W',epoch_loss_w/N, epoch)

    # Plot histograms of weights in tensorboard (possibly helps identify vanishing gradients)
    def writer_histogram(m):
        if isinstance(m, nn.Conv2d):
            writer.add_histogram(str(m)+'.weight', m.weight, epoch)
            writer.add_histogram(str(m)+'.weight.grad', m.weight.grad, epoch)
    encoder.apply(writer_histogram)
    
    epoch_loss_previous = epoch_loss/N
    
    # Define the checkpoint
    checkpoint = {
    'nb_epochs_finished': epoch+1,
    'nb_batches_finished':bc+1,
    'model_state': encoder.state_dict(),
    'optimizer_state': optimizer.state_dict()
    }

    # Learning rate scheduler step (optional)
    if use_scheduler:
        scheduler.step()
        checkpoint['scheduler_state']=scheduler.state_dict()
    
    # Save checkpoint (+ every 50 epochs we save a landmark checkpoint in case something goes wrong)
    torch.save(checkpoint, checkpoint_name)
    if (epoch+1)%50==0:
        torch.save(checkpoint, checkpoint_name[:-4]+'_'+str(epoch)+'.pth')

#---------------------------------------------------------
## Plot figures for tensorboard writer

    ex = dataset[1:2][0].to('cuda')
    l = encoder.encode(ex)
    if latent.shape[-1]==1:
        l = l.transpose(1,3)
    l_w = l[:,:,:,[0,1,2,3,4,5,6,7,9,11]]
    l_x = l[:,:,:,[0,1,2,3,4,5,6,8,10,12]]
        
    ow = decoderW.decode(l_w.view((-1,10)))[:,0,:].view(ex[:,0,:,:].shape)[0]
    ox = decoderX.decode(l_x.view((-1,10)))[:,0,:].view(ex[:,1,:,:].shape)[0]        
    
    if force_min_noise:
        ex_w_min = torch.empty(ex[0][0].shape)
        ex_w_min[:] = torch.min(ex[0][0],axis=-1).values[:,None]
        ex_x_min = torch.empty(ex[0][1].shape)
        ex_x_min[:] = torch.min(ex[0][1],axis=-1).values[:,None]
        ow_max = torch.max(ow,ex_w_min.to(device))
        ox_max = torch.max(ox,ex_x_min.to(device))
        ow = ow_max
        ox = ox_max
    
    ex = ex.to('cpu')
    fig, axs = plt.subplots(1,2,figsize=(10,3))
    axs[0].pcolormesh(ex[0][0],vmin= ex[0][0].min()-.05,vmax= ex[0][0].max()+.05)
    axs[1].pcolormesh(ow.to('cpu').detach().numpy(),vmin= ex[0][0].min()-.05,vmax= ex[0][0].max()+.05)
    plt.close()
    
    fig1, axs = plt.subplots(1,2,figsize=(10,3))
    axs[0].pcolormesh(ex[0][1],vmin= ex[0][1].min()-.05,vmax= ex[0][1].max()+.05)
    axs[1].pcolormesh(ox.to('cpu').detach().numpy(),vmin= ex[0][1].min()-.05,vmax= ex[0][1].max()+.05)
    plt.close()
    
    fig2, axs = plt.subplots(1,2,figsize=(10,3))
    axs[0].plot(ex[0][0][15])
    axs[0].plot(ow[15].to('cpu').detach().numpy())
    axs[1].plot(ex[0][0][3])
    axs[1].plot(ow[3].to('cpu').detach().numpy())
    plt.close()
    
    fig3, axs = plt.subplots(1,2,figsize=(8,4))
    axs[0].plot(l[0,0,:,5].to('cpu').detach(),np.arange(l[0,0,:,5].shape[0]))
    axs[1].plot(l[0,0,:,3].to('cpu').detach(),np.arange(l[0,0,:,3].shape[0]))
    plt.close()
    
    writer.add_figure('spectrogram_ex',fig,epoch)
    writer.add_figure('spectrogram_exX',fig1,epoch)
    writer.add_figure('spectra_ex',fig2,epoch)
    writer.add_figure('latent_dmean_and_b', fig3, epoch)
    
    
