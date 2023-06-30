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
sys.path.append('/home/billault/ml_design2/decoder/')
from encoder_models import *
from decoder_models import *
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import argparse
import shutil
import os
import warnings
warnings.filterwarnings("ignore")

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

## Parse input and load config
parser = argparse.ArgumentParser(description='train Encoder model')
parser.add_argument('config_file', type=str)
parser.add_argument('--data_file',type=str,default='')
args = parser.parse_args()

config = json.load(open(args.config_file))

directory = config['directory']
if not(os.path.exists(directory)):
    os.mkdir(directory)
try:
    if os.path.exists( directory+'/config_encoder.json'):
        check_if_replace = input('config_encoder.json already exists in {}. Do you want to replace it? (y/n)'.format(directory))
        if check_if_replace == 'n':
        #     shutil.copy2(args.config_file, directory+'/config_encoder.json')
        # else:
            print('Exiting')
            sys.exit()
    shutil.copy2(args.config_file, directory+'/config_encoder.json')
except shutil.SameFileError:
    pass

#--------------------------------------------------

## Load decoder data and normalization parameters
device = config['device']

config_decX = json.load(open(config['config_X_decoder']))
config_decKa = json.load(open(config['config_Ka_decoder']))
config_decW = json.load(open(config['config_W_decoder']))

model_name_decX=config_decX['model_name'] 
nb_channels_decX, n_hidden_units_decX, n_input_features_decX = config_decX['n_channels'], config_decX['n_hidden_layers'], config_decX['n_input_features']
model_name_decKa=config_decKa['model_name']
nb_channels_decKa, n_hidden_units_decKa, n_input_features_decKa = config_decKa['n_channels'], config_decKa['n_hidden_layers'], config_decKa['n_input_features']
model_name_decW=config_decW['model_name'] 
nb_channels_decW, n_hidden_units_decW, n_input_features_decW = config_decW['n_channels'], config_decW['n_hidden_layers'], config_decW['n_input_features']

# We assume that the normalization parameters are the same for X and W decoders (that's what it should be)
mean_normalization_json = config_decW['normalize_path']+'/means_spec.json'
std_normalization_json = config_decW['normalize_path']+'/stds_spec.json'


# Load model states for W and X decoder
print(config_decX['directory']+config_decX['checkpoint_name'])
checkpoint_X = torch.load(config_decX['directory']+config_decX['checkpoint_name'])
model_X_state_dict = checkpoint_X['model_state']
checkpoint_Ka = torch.load(config_decKa['directory']+config_decKa['checkpoint_name'])
model_Ka_state_dict = checkpoint_Ka['model_state']
checkpoint_W = torch.load(config_decW['directory']+config_decW['checkpoint_name'])
model_W_state_dict = checkpoint_W['model_state']

decoderX = locals()[model_name_decX](nb_channels_decX, n_hidden_units_decX, n_input_features_decX).to(device)
decoderX.load_state_dict(model_X_state_dict)
decoderKa = locals()[model_name_decKa](nb_channels_decKa, n_hidden_units_decKa, n_input_features_decKa).to(device)
decoderKa.load_state_dict(model_Ka_state_dict)
decoderW = locals()[model_name_decW](nb_channels_decW, n_hidden_units_decW, n_input_features_decW).to(device)
decoderW.load_state_dict(model_W_state_dict)

decoderX.eval()
decoderKa.eval()
decoderW.eval()


#--------------------------------------------------
## Prepare dataset and dataloader
data_file = config['data_file']
rgate_start = config['rgate_start']
rgate_end = config['rgate_end']
num_workers_dataloader = config['num_workers_dataloader']
batch_size = config['batch_size']
batch_shape = config['batch_shape']


from dataloader_encoder import encoder_dataset
step_for_data_augmentation = config['step_for_data_augmentation']
dataset = encoder_dataset(data_file, rgate_start=rgate_start, rgate_end=rgate_end, batch_shape = batch_shape, step_for_data_augmentation=step_for_data_augmentation, mean_normalization_json = mean_normalization_json, std_normalization_json = std_normalization_json)


constrain_latent = config['constrain_latent']
epoch_start_secloss = config['epoch_start_secloss']
if constrain_latent:
    mean_normalization_npy = config_decW['normalize_path']+'/means_input.npy',
    std_normalization_npy = config_decW['normalize_path']+'/stds_input.npy',
    mean_latent = torch.from_numpy(np.load(mean_normalization_npy)).to(device)
    std_latent = torch.from_numpy(np.load(std_normalization_npy)).to(device)
N = len(dataset)
dataloader = DataLoader(dataset, num_workers=num_workers_dataloader,batch_size=batch_size, shuffle=True, pin_memory=True) #

inds_latent_vars_X = config['inds_latent_vars_X']
inds_latent_vars_Ka = config['inds_latent_vars_Ka']
inds_latent_vars_W = config['inds_latent_vars_W']

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
sec_loss_method = config['sec_loss_method']

# Remove gradient update for decoder parameters 
# (I think this is not necessary per se because the optimizer only applies to the encoder part anyways, but this is for safety + speeds up)
for p in decoderX.parameters():
    p.requires_grad = False
for p in decoderKa.parameters():
    p.requires_grad = False
for p in decoderW.parameters():
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
    epoch_loss_ka = 0
    epoch_loss_w = 0
    
    
    for sub_input_data in iter(dataloader): # zew_targ, zex_targ, mdvw_targ, mdvx_targ
        sub_input_data = sub_input_data.to(device)
        target_x = sub_input_data[:,0,:,:]
        target_ka = sub_input_data[:,1,:,:]
        target_w = sub_input_data[:,2,:,:]
        latent = encoder.encode(sub_input_data)
        
        # Depending on the model used, the latent variables might be in the features or channel dimension, so we check if axis permutation is needed
        if latent.shape[-1]==1:
            latent = latent.transpose(1,3)

        # We select the indices of the latent variables to be used in the X and W emulators.
        latent_x = latent[:,:,:,inds_latent_vars_X].float()
        latent_ka = latent[:,:,:,inds_latent_vars_Ka].float()
        latent_w = latent[:,:,:,inds_latent_vars_W].float()
        output_x = decoderX.decode(latent_x.view((-1,n_input_features_decX)))[:].view(sub_input_data[:,0,:,:].shape)
        output_ka = decoderKa.decode(latent_ka.view((-1,n_input_features_decKa)))[:,0,:].view(sub_input_data[:,1,:,:].shape)
        output_w = decoderW.decode(latent_w.view((-1,n_input_features_decW)))[:,0,:].view(sub_input_data[:,2,:,:].shape)
        
                
        # Here loss is simply mse
        if loss_method=='mse':
            loss_x = (output_x - target_x).pow(2).sum()/ (batch_shape*256*batch_size)
            loss_ka = (output_ka - target_ka).pow(2).sum()/ (batch_shape*256*batch_size)
            loss_w = (output_w - target_w).pow(2).sum()/ (batch_shape*256*batch_size)
        elif loss_method == 'mae':
            loss_x = (output_x - target_x).abs().sum()/ (batch_shape*256*batch_size)
            loss_ka = (output_ka - target_ka).abs().sum()/ (batch_shape*256*batch_size)
            loss_w = (output_w - target_w).abs().sum()/ (batch_shape*256*batch_size)
        loss = loss_x + loss_ka + loss_w
        
        
        #That's the loss we want to track in the tensorboard instance
        writer.add_scalar('batch_loss',loss, bc)
        epoch_loss+=loss*output_w.size(0)
        epoch_loss_x+= loss_x*output_x.size(0)
        epoch_loss_ka+= loss_ka*output_ka.size(0)
        epoch_loss_w+= loss_w*output_w.size(0)
        bc+=1

        # Exit if loss is too large
        if (epoch>265) & (loss.item()>0.17):
            print('LOSS TOO LARGE, EXITING')
            exit()
        

        # Secondary loss term with various possible methods
        if sec_loss_method == 'avg_mse_total_and_peak':
            peakX = torch.gt(target_x,.1+torch.min(target_x,axis=-1).values[:,:,None])
            peakKa = torch.gt(target_ka,.1+torch.min(target_ka,axis=-1).values[:,:,None])
            peakW = torch.gt(target_w,.1+torch.min(target_w,axis=-1).values[:,:,None])
            loss2 = 1/3*((target_x[peakX] - output_x[peakX]).pow(2).sum() + (target_ka[peakKa] - output_ka[peakKa]).pow(2).sum() + (target_w[peakW] - output_w[peakW]).pow(2).sum())/(batch_shape*batch_size)
            loss = .25*(loss+loss2) #.5*(loss+min(1,en_epochsn_epochs)*loss2)/(1+min(1,epoch/n_epochs))
            
        elif sec_loss_method == 'avg_mse_total_and_finer_peak':
            peakX = torch.gt(target_x,.5+torch.min(target_x,axis=-1).values[:,:,None])
            peakKa = torch.gt(target_ka,.5+torch.min(target_ka,axis=-1).values[:,:,None])
            peakW = torch.gt(target_w,.5+torch.min(target_w,axis=-1).values[:,:,None])
            loss2 = 1/3*((target_x[peakX] - output_x[peakX]).pow(2).sum() + (target_ka[peakKa] - output_ka[peakKa]).pow(2).sum() + (target_w[peakW] - output_w[peakW]).pow(2).sum())/(batch_shape*batch_size)
            loss = .75*loss+.25*loss2
            
        elif (sec_loss_method=='weighted_peak')  & (epoch > epoch_start_secloss):
            peakX = torch.gt(target_x,.1+torch.min(target_x,axis=-1).values[:,:,None])
            peakKa = torch.gt(target_ka,.1+torch.min(target_ka,axis=-1).values[:,:,None])
            peakW = torch.gt(target_w,.1+torch.min(target_w,axis=-1).values[:,:,None])
            loss2_x = .5*((target_x - output_x)*peakX).pow(2).sum()/(batch_shape*batch_size)
            loss2_ka = .5*((target_ka - output_ka)*peakKa).pow(2).sum()/(batch_shape*batch_size)
            loss2_w = .5*((target_w - output_w)*peakW).pow(2).sum()/(batch_shape*batch_size)
            loss = loss+.05*(loss2_x+loss2_ka+loss2_w)
            
        elif (sec_loss_method=='finer_peak') & (epoch > epoch_start_secloss):
            minmax = torch.max(target_w,axis=-1).values[:,:,None]-torch.min(target_w,axis=-1).values[:,:,None]
            peakX = torch.gt(target_x,torch.min(target_x,axis=-1).values[:,:,None]+.75*minmax)
            peakKa = torch.gt(target_ka,torch.min(target_ka,axis=-1).values[:,:,None]+.75*minmax)
            peakW = torch.gt(target_w,torch.min(target_w,axis=-1).values[:,:,None]+.75*minmax)
            loss2_x = .5*((target_x - output_x)*peakX).pow(2).sum()/(batch_shape*256*batch_size)
            loss2_ka = .5*((target_ka - output_ka)*peakKa).pow(2).sum()/(batch_shape*256*batch_size)
            loss2_w = .5*((target_w - output_w)*peakW).pow(2).sum()/(batch_shape*256*batch_size)
            loss = loss+.05*(loss2_x+loss2_ka+loss2_w)
                      

        # Backpropagate                       
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        



    
    writer.add_scalar('training_loss',epoch_loss/N, epoch)
    print(epoch, epoch_loss/N,'\n\n')
    writer.add_scalar('training_loss_X',epoch_loss_x/N, epoch)
    writer.add_scalar('training_loss_Ka',epoch_loss_ka/N, epoch)
    writer.add_scalar('training_loss_W',epoch_loss_w/N, epoch)

    def writer_histogram(m):
        if isinstance(m, nn.Conv2d):
            writer.add_histogram(str(m)+'.weight', m.weight, epoch)
            writer.add_histogram(str(m)+'.weight.grad', m.weight.grad, epoch)
    try:
        encoder.apply(writer_histogram)
    except Exception as e:
        print(e)
        pass

    epoch_loss_previous = epoch_loss/N
    
    checkpoint = {
    'nb_epochs_finished': epoch+1,
    'nb_batches_finished':bc+1,
    'model_state': encoder.state_dict(),
    'optimizer_state': optimizer.state_dict()
    }

    if use_scheduler:
        scheduler.step()
        checkpoint['scheduler_state']=scheduler.state_dict()
    
    torch.save(checkpoint, checkpoint_name)
    
    if (epoch+1)%50==0:
        torch.save(checkpoint, checkpoint_name[:-4]+'_'+str(epoch)+'.pth')
#---------------------------------------------------------
## Plot figures for tensorboard writer

    ex = dataset[5150:5151].to('cuda')

    l = encoder.encode(ex)#.shape


    if l.shape[-1]==1:
        l = l.transpose(1,3)
        # We select the indices of the latent variables to be used in the X and W emulators.
    l_w = l[:,:,:,config['inds_latent_vars_W']]
    l_x = l[:,:,:,config['inds_latent_vars_X']]
    l_ka = l[:,:,:,config['inds_latent_vars_Ka']]
        
    ow = decoderW.decode(l_w.view((-1,11)))[:,0,:].view(ex[:,0,:,:].shape)
    oka = decoderKa.decode(l_ka.view((-1,11)))[:,0,:].view(ex[:,0,:,:].shape)
    ox = decoderX.decode(l_x.view((-1,11)))[:,0,:].view(ex[:,1,:,:].shape)
           
      
    ex = ex.to('cpu')
    fig, axs = plt.subplots(1,2,figsize=(10,3))
    axs[0].pcolormesh(ex[0][0],vmin= ex[0][0].min()-.05,vmax= ex[0][0].max()+.05)
    axs[1].pcolormesh(ox.to('cpu').detach().numpy()[0],vmin= ex[0][0].min()-.05,vmax= ex[0][0].max()+.05)
    plt.close()
    
    fig1, axs = plt.subplots(1,2,figsize=(10,3))
    axs[0].pcolormesh(ex[0][1],vmin= ex[0][1].min()-.05,vmax= ex[0][1].max()+.05)
    axs[1].pcolormesh(oka.to('cpu').detach().numpy()[0],vmin= ex[0][1].min()-.05,vmax= ex[0][1].max()+.05)
    plt.close()

    fig2, axs = plt.subplots(1,2,figsize=(10,3))
    axs[0].pcolormesh(ex[0][2],vmin= ex[0][2].min()-.05,vmax= ex[0][2].max()+.05)
    axs[1].pcolormesh(ow.to('cpu').detach().numpy()[0],vmin= ex[0][2].min()-.05,vmax= ex[0][2].max()+.05)
    plt.close()
    
    fig3, axs = plt.subplots(1,2,figsize=(10,3))
    axs[0].plot(ex[0][1][15])
    axs[0].plot(oka[0,15].to('cpu').detach().numpy())
    axs[1].plot(ex[0][1][3])
    axs[1].plot(oka[0,3].to('cpu').detach().numpy())
    plt.close()
    
    fig4, axs = plt.subplots(1,3,figsize=(12,4))
    axs[0].plot(l[0,0,:,5].to('cpu').detach(),np.arange(l[0,0,:,1].shape[0]))
    axs[1].plot(l[0,0,:,3].to('cpu').detach(),np.arange(l[0,0,:,3].shape[0]))
    axs[2].plot(l[0,0,:,5].to('cpu').detach(),np.arange(l[0,0,:,5].shape[0]))
    plt.close()
    
    writer.add_figure('spectrogram_exX',fig,epoch)
    writer.add_figure('spectrogram_exKa',fig1,epoch)
    writer.add_figure('spectrogram_exW',fig2,epoch)
    writer.add_figure('spectra_ex',fig3,epoch)
    writer.add_figure('latent_vars_1_3_5', fig4, epoch)
    
    