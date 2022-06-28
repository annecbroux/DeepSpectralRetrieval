
"""
This script concatenates the .feather files generated by the script 00_simulate_spectra.py into hdf5 files.
The .feather files are expected to be stored in a directory structure of the type .../batch0/batch0_0
Should be parsed with the corresponding argument e.g. "batch0_0"
"""

import numpy as np
import pandas as pd
import h5py
import glob
from tqdm import tqdm
import time
import argparse

SIMULATIONS_DIR = '/data/acbr_spectra_dbcreation/'

parser = argparse.ArgumentParser(description='convert feather to h5')
parser.add_argument('batch', type=str)
args = parser.parse_args()
# Note : this browses through the spectra simulations reorganized in a directory structure of the type $main_batch/$secondary_batch (e.g. batch0/batch0_0)
# Each subdirectory contains 500 simulation batches as created by the script 00_simulate_spectra.py (more caused kernel crashes)
# The user might want to adapt this organization depending on their computing power etc.
batch_main = args.batch.split('_')[0]
batch_secondary = args.batch
spec_list = sorted(glob.glob(SIMULATIONS_DIR+batch_main+'/'+batch_secondary+'/spectra_*.feather'))

# Load list of simulations and concatenate to a single pandas DataFrame
slist = []
for i in tqdm(range(len(spec_list))):
    f = spec_list[i]
    slist.append(pd.read_feather(f))
df = pd.concat(slist).drop(columns='index').reset_index()

# Convert particle type to a integer
ptype = np.zeros(len(df))
ptype[df['ptype']=='agg']=1
ptype[df['ptype']=='den']=2
ptype[df['ptype']=='col']=3
ptype[df['ptype']=='grau']=4

ptype=ptype.astype('uint8')

# Save into hdf5 file. 
# float32 is used for storage reasons.
with h5py.File(SIMULATIONS_DIR+batch_secondary+'.hdf5','w') as f:
    lnoiseW = f.create_dataset('noise_level_W',(len(df),),dtype='float32',data=df['noise_level_W'])#,compression='gzip')
    lnoiseX = f.create_dataset('nosie_level_X',(len(df),),dtype='float32',data=df['noise_level_X'])#,compression='gzip')
    kssrg = f.create_dataset('k_ssrg',(len(df),),dtype='float32',data=df['k_ssrg'])#,compression='gzip')
    bssrg = f.create_dataset('b_ssrg',(len(df),),dtype='float32',data=df['b_ssrg'])#,compression='gzip')
    lwc = f.create_dataset('iwc',(len(df),),dtype='float32',data=df['iwc'])#,compression='gzip')
    ptype = f.create_dataset('ptype',(len(df),), dtype='uint8',data =ptype)#,compression='gzip')
    temperature = f.create_dataset('temperature',(len(df),), data = df['temperature'])#,compression='gzip')
    asratio = f.create_dataset('aspect_ratio',(len(df),), dtype='float32',data=df['aspect_ratio'])#,compression='gzip')
    ams = f.create_dataset('a_mass_size',(len(df),),dtype = 'float32',data=df['a_mass_size'])#,compression='gzip')
    bms = f.create_dataset('b_mass_size',(len(df),),dtype = 'float32',data=df['b_mass_size'])#,compression='gzip')
    aas = f.create_dataset('alpha_area_size',(len(df),),dtype='float32',data=df['alpha_area_size'])#,compression='gzip')
    bas = f.create_dataset('beta_area_size',(len(df),),dtype='float32',data=df['beta_area_size'])#,compression='gzip')
    dmean = f.create_dataset('dmean',(len(df),),dtype='float32',data=df['dmean'])#,compression='gzip')
    spectra = f.create_dataset('spectra', (len(df),2,512), dtype = 'float32', data =[np.stack(df['spectra'].to_numpy()[i]) for i in range(len(df))])#,compression='gzip')
    moments = f.create_dataset('moments',(len(df),2,4), dtype = 'float32', data = [np.stack(df['moments'].to_numpy()[i]) for i in range(len(df))])#,compression='gzip')


