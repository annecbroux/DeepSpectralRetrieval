import numpy as np
import pandas as pd
import h5py
import glob
from tqdm import tqdm
import time

import argparse
parser = argparse.ArgumentParser(description='convert feather to h5')
parser.add_argument('in_dir', type=str)
parser.add_argument('out_file', type=str)
parser.add_argument('--nfft', type=int, default=256)
args = parser.parse_args()

in_dir = args.in_dir
out_file = args.out_file
nfft = args.nfft
default_a_ms = 0.02522677
default_b_ms = 2.19978322

spec_list = sorted(glob.glob(in_dir+'*.feather'))

# Concatenate all feather files into one dataframe
slist = []
for i in tqdm(range(len(spec_list))):
    f = spec_list[i]
    slist.append(pd.read_feather(f))
df = pd.concat(slist).reset_index()
# Shuffle
df = df.sample(frac=1).reset_index(drop=True)

# Convert to single h5 file
tic = time.time()
with h5py.File(out_file,'w') as f:
    logN = f.create_dataset('logN',(len(df),),dtype='float32',data=np.log(df['N0']))
    N = f.create_dataset('N',(len(df),),dtype='float64',data=df['N0'])
    lam = f.create_dataset('lam',(len(df),),dtype='float32',data=df['lam'])
    mu = f.create_dataset('mu',(len(df),),dtype='float32',data=df['mu'])
    eps = f.create_dataset('eps',(len(df),),dtype='float32',data=df['eps'])
    uwind = f.create_dataset('uwind',(len(df),),dtype='float32',data=df['uwind'])
    broadX = f.create_dataset('broadX',(len(df),),dtype='float32',data=df['broadeningX'])
    broadKa = f.create_dataset('broadKa',(len(df),),dtype='float32',data=df['broadeningKa'])
    broadW = f.create_dataset('broadW',(len(df),),dtype='float32',data=df['broadeningW'])
    windX = f.create_dataset('windX',(len(df),),dtype='float32',data=df['windX'])
    windKa = f.create_dataset('windKa',(len(df),),dtype='float32',data=df['windKa'])
    windW = f.create_dataset('windW',(len(df),),dtype='float32',data=df['windW'])
    if 'a_ms' not in df.columns:
        a_ms = f.create_dataset('a_ms', (len(df),), dtype = 'float32', data = np.zeros(len(df))+default_a_ms+np.random.normal(0,1,(len(df),)))
        b_ms = f.create_dataset('b_ms', (len(df),), dtype = 'float32', data = np.zeros(len(df))+default_b_ms+np.random.normal(0,1,(len(df),)))
    else:
        a_ms = f.create_dataset('a_ms', (len(df),), dtype = 'float32', data = df['a_ms'])
        b_ms = f.create_dataset('b_ms', (len(df),), dtype = 'float32', data = df['b_ms'])
    noise = f.create_dataset('noise',(len(df),),dtype='float32',data=df['noise'])
    height = f.create_dataset('height',(len(df),),dtype='float32',data=df['height'])
    humidity = f.create_dataset('humidity',(len(df),),dtype='float32',data=df['humidity'])
    temperature = f.create_dataset('temperature',(len(df),),dtype='float32',data=df['temperature'])
    specX = f.create_dataset('specX', (len(df),nfft), dtype = 'float32', data =np.stack(df['specXshifted']))
    specKa = f.create_dataset('specKa', (len(df),nfft), dtype = 'float32', data =np.stack(df['specKashifted']))
    specW = f.create_dataset('specW', (len(df),nfft), dtype = 'float32', data =np.stack(df['specWshifted']))
    
    

print(time.time()-tic)