import numpy as np
import pandas as pd
import h5py
import glob
from tqdm import tqdm
import time
from turbulence_and_wind import turb_broadening, add_turbulence_vect
import scipy
import matplotlib.pyplot as plt


"""
This codes adds the turbulent broadening to the spectra.
In-place modification of the hdf5 files created by 02_merge_h5.py
"""

import argparse
parser = argparse.ArgumentParser(description='add turbulence to merged h5')
parser.add_argument('batch', type=str)
args = parser.parse_args()
batch_name = args.batch

SIMULATIONS_DIR = '/data/acbr_spectra_dbcreation/'

#These parameters are those of the radars used for the original study
RADAR_PROPERTIES = 'radar_config.json'
radar_config = json.load(open(RADAR_PROPERTIES))
freqx = radar_config['freq_X']
freqw = radar_config['freq_W']
bmww = radar_config['beamwidth_W']
bmwx = radar_config['beamwidth_W']
WIND = 0
INTERCEPT_T2S = 7.243 #This constant converts turbulent EDR into broadening kernel size



with h5py.File(SIMULATIONS_DIR+batch_name+'_merged.h5','r+') as f: #/data/acbr_spectra_dbcreation/
    if not ('SW' in f.keys()):
        SW = f.create_dataset("SW", (f['iwc'].shape[0],2), maxshape=(None,None))
    if not ('MDV' in f.keys()):
        MDV = f.create_dataset("MDV", (f['iwc'].shape[0],2), maxshape=(None,None))
    if not('ZE' in f.keys()):
        Ze = f.create_dataset("ZE", (f['iwc'].shape[0],2), maxshape=(None,None))
    if not ('SPECTRA_TURB' in f.keys()):
        SPECTURB = f.create_dataset('SPECTRA_TURB',f['spectra'].shape)
        SIGMA_TURB = f.create_dataset('SIGMA_TURB',f['MDV'].shape)


    for i in tqdm(range(int(len(f['iwc'])/1000)+1)):
        S = f['spectra'][i*1000:(i+1)*1000,:,:]
        Smin = np.min(S,axis=-1)
        Smax = np.max(S,axis=-1)
        
        turb = np.zeros((S.shape[0],))+ np.random.exponential(.001,(S.shape[0],))
        sigma_from_turb = np.empty((len(turb),2))
        sigma_from_turb[:,0] = turb_broadening(freqx,bmwx,wind,turb)*np.exp(intercept_t2s)
        sigma_from_turb[:,1] = turb_broadening(freqw,bmww,wind,turb)*np.exp(intercept_t2s)
        
        (sturb,ze,mdv,sw) = add_turbulence_vect(S, sigma_from_turb)
        f['SPECTRA_TURB'][i*1000:(i+1)*1000,:,:] = 10*np.log10(sturb)
        f['SIGMA_TURB'][i*1000:(i+1)*1000,:] = sigma_from_turb
        f['MDV'][i*1000:(i+1)*1000,:] = mdv
        f['SW'][i*1000:(i+1)*1000,:] = sw
        f['ZE'][i*1000:(i+1)*1000,:] = ze

