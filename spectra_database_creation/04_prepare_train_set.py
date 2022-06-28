import numpy as np
import pandas as pd
import h5py
import glob
from tqdm import tqdm
import yaml
from turbulence_and_wind import shift_wind

SIMULATIONS_DIR = '/data/acbr_spectra_dbcreation/'

"""
This code does the final steps of the training set pre-processing:
-> it checks for spectra with unrealistic moments and discards them
-> it shifts the spectra with randomly sampled radial wind values
-> each spectrum is shifted with 2 different wind values -> yields 2x more spectra in output (for data augmentation)
-> reduces size of spectra to 256 velocity bins
-> add "min_spec" field to the dataset which contains the minium value of each spectrum
"""

import argparse
parser = argparse.ArgumentParser(description='prepare training set')
parser.add_argument('batch', type=str)
args = parser.parse_args()
batch_name = args.batch


rvel = np.linspace(-6.92, 6.92, 512)
dv = rvel[1]-rvel[0]

f_orig = h5py.File(SIMULATIONS_DIR+batch_name+'_merged.h5','r')

with h5py.File(SIMULATIONS_DIR+batch_name+'_all_wind_ts.h5','w') as f_new:

    ze = f_orig['ZE'][:]
    SW = f_orig['SW'][:]
    MDV = f_orig['MDV'][:]

    inds_to_keep = np.where(((SW[:,0]<2.) & (MDV[:,0]<3.) & (MDV[:,0]>-.1) & (ze[:,1]>-15)))[0]

    N = len(inds_to_keep)
    # Create the wind containers (2x larger than original, because wind shift -> data augmentation)
    f_new.create_dataset('windX',(N*2,))
    f_new.create_dataset('windW',(N*2,))
    # Create the final spectra containers (2x larger than original, because wind shift -> data augmentation, and 1/2 velocity bins, now 256)
    f_new.create_dataset('spectrum_X_256',(N*2,256))
    f_new.create_dataset('spectrum_W_256',(N*2,256))


    for i in tqdm(range(int(N/1000)+1)):
        # We sample randomly wind values in a reasonable interval
        # Note that we sample 2x more wind values because we 
        windX1 = np.random.uniform(-2,3,tempX.shape[0])
        windX2 = np.random.uniform(-2,3,tempX.shape[0])
        windW1 = np.random.uniform(-2,3,tempW.shape[0])
        windW2 = np.random.uniform(-2,3,tempW.shape[0])

        # We use linear units for the part on adding wind
        tempX = np.power(10, 0.1*(f_orig['SPECTRA_TURB'][inds_to_keep[i*1000:(i+1)*1000],0,:]))
        tempW = np.power(10, 0.1*(f_orig['SPECTRA_TURB'][inds_to_keep[i*1000:(i+1)*1000],1,:]))

        # Implement wind shift        
        tempX1 = shift_wind(tempX,windX1,dv)
        tempX2 = shift_wind(tempX,windX2,dv)
        tempW1 = shift_wind(tempW,windW1,dv)
        tempW2 = shift_wind(tempW,windW2,dv)
        
        # Fill the wind containers        
        f_new['windX'][i*1000:min(N,(i+1)*1000)] = windX1
        f_new['windW'][i*1000:min(N,(i+1)*1000)] = windW1
        f_new['windX'][N+i*1000:N+(i+1)*1000] = windX2
        f_new['windW'][N+i*1000:N+(i+1)*1000] = windW2
        
        # Fill the spectra containers
        f_new['spectrum_X_256'][i*1000:min(N,(i+1)*1000),:] = 10*np.log10(tempX1[:,::2] + tempX1[:,1::2])
        f_new['spectrum_W_256'][i*1000:min(N,(i+1)*1000),:] = 10*np.log10(tempW1[:,::2] + tempW1[:,1::2]) 
        f_new['spectrum_X_256'][N+i*1000:N+(i+1)*1000,:] = 10*np.log10(tempX2[:,::2]+tempX2[:,1::2])
        f_new['spectrum_W_256'][N+i*1000:N+(i+1)*1000,:] = 10*np.log10(tempW2[:,::2]+tempW2[:,1::2])
       
    
    # Fill the other containers    
    for key in ['a_mass_size', 'alpha_area_size', 'aspect_ratio', 'b_mass_size', 'b_ssrg', 'beta_area_size', 'dmean', 'k_ssrg', 'lwc', 'noise_level_W', 'noise_level_X', 'ptype', 'temperature']:
        f_new.create_dataset(key, (N*2,))
        print(key)
        for i in tqdm(range(int(N/1000)+1)):
            f_new[key][i*1000:min(N,(i+1)*1000)] = f_orig[key][inds_to_keep[i*1000:(i+1)*1000]]
            f_new[key][N+i*1000:N+(i+1)*1000] = f_orig[key][inds_to_keep[i*1000:(i+1)*1000]]
            
    # Note: we keep the original moments (so careful if use MDV, need to combine with wind value)
    for key in ['SIGMA_TURB','MDV','SW','ZE']:
        f_new.create_dataset(key, (N*2,2))
        print(key)
        for i in tqdm(range(int(N/1000)+1)):
            f_new[key][i*1000:min(N,(i+1)*1000),:] = f_orig[key][inds_to_keep[i*1000:(i+1)*1000],:]
            f_new[key][N+i*1000:N+(i+1)*1000,:] = f_orig[key][inds_to_keep[i*1000:(i+1)*1000],:]

    # Also include the minimum spectra values in the final dataset
    for key in ['min_spec_W', 'min_spec_X']:
        f_new.create_dataset(key, (N*2,))
        print(key)
        for i in tqdm(range(int(N/1000)+1)):
            f_new[key][i*1000:min(N,(i+1)*1000)] = np.min(f_new['spectrum_'+key[-1]+'_256'][i*1000:min(N,(i+1)*1000),:],axis=-1)
            f_new[key][N+i*1000:N+(i+1)*1000] = np.min(f_new['spectrum_'+key[-1]+'_256'][N+i*1000:N+(i+1)*1000,:],axis=-1)

        
