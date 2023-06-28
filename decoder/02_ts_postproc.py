import numpy as np
import matplotlib.pyplot as plt
import h5py
import json
from tqdm import tqdm

import argparse
parser = argparse.ArgumentParser(description='Postprocessing of training set, create h5 with input features & calculate normalization parameters')
parser.add_argument('config_path', type=str)

args = parser.parse_args()
config_path = args.config_path

with open(config_path,'r') as fp:
    config = json.load(fp)
input_variables = config['input_variables']
features_to_log = config['features_to_log']
features_to_log_plus_one = config['features_to_log_plus_one']
train_size = config['train_size']
i_start = config['i_start_ts']
save_normalize_path = config['normalize_path']


## Log-transform input features and save to h5
f = h5py.File(config['syn_dataset_path'],'r')
input_vector = np.empty((f[input_variables[0]].shape[0],len(input_variables)))
print('Log-transforming input features...')
for i, key in enumerate(input_variables):
    print(key)
    if key in features_to_log:
        input_vector[:,i] = np.log(f[key][:])
    elif key in features_to_log_plus_one:
        input_vector[:,i] = np.log(f[key][:]+1)
    else:
        input_vector[:,i] = f[key][:]
    
with h5py.File(config['input_ds_path'],'w') as f2:
    f2.create_dataset('input_features',input_vector.shape,data=input_vector)
print('Log-transformed input features saved to',config['input_ds_path'])

## Calculate normalization parameters
print('Calculating normalization parameters...')
means_npy = np.mean(input_vector[i_start:i_start+train_size,:],axis=0)
medians_npy = np.median(input_vector[i_start:i_start+train_size,:],axis=0)
stds_npy = np.std(input_vector[i_start:i_start+train_size,:],axis=0)
q1_npy = np.quantile(input_vector[i_start:i_start+train_size,:],0.1,axis=0)
q9_npy = np.quantile(input_vector[i_start:i_start+train_size,:],0.9,axis=0)
print(means_npy)
print(stds_npy)

## Save normalization parameters
np.save(save_normalize_path+'/means_input',means_npy)
np.save(save_normalize_path+'/medians_input',medians_npy)
np.save(save_normalize_path+'/stds_input',stds_npy)
np.save(save_normalize_path+'/iqrs_input',q9_npy-q1_npy)
print('Normalization parameters saved to',save_normalize_path)

## Calculate normalization parameters for output features
# sums = {}
# sums2 = {}
# means = {}
# stds = {}
# print('Calculating normalization parameters for output features...')
# for key in ['specX','specKa', 'specW']:
#     sums[key] = 0
#     sums2[key] = 0
#     currlen = 0
#     for i in tqdm(range(int(i_start/10000)+1,int((i_start+train_size)/10000)+1)):
#         s1 = sums[key]*currlen + np.sum(f[key][i*10000:(i+1)*10000])
#         s2 = sums2[key]*currlen + np.sum(f[key][i*10000:(i+1)*10000]**2)
#         currlen += len(f[key][i*10000:(i+1)*10000])
#         sums[key] = s1/currlen
#         sums2[key] = s2/currlen
#     means[key] = sums[key]/f[key].shape[1]
#     stds[key] = np.sqrt(sums2[key]/f[key].shape[1] - means[key]**2)

means = {}
stds = {}
for key in ['specX','specKa', 'specW']:
    means[key] = float(np.nanmean(f[key][i_start:i_start+train_size,:]))
    stds[key] = float(np.nanstd(f[key][i_start:i_start+train_size,:]))
print(means)
print(stds)

## Save normalization parameters for output features
with open(save_normalize_path+'/means_spec.json', 'w') as fp:
    json.dump(means, fp)
with open(save_normalize_path+'/stds_spec.json', 'w') as fp:
    json.dump(stds, fp)
print('Normalization parameters for output features saved to',save_normalize_path)