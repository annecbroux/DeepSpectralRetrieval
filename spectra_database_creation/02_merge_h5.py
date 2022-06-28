import numpy as np
import pandas as pd
import h5py
import glob
from tqdm import tqdm
import time
import random

SIMULATIONS_DIR = '/data/acbr_spectra_dbcreation/'

"""
This code (not super neat) merges the hdf5 files created by 01_convert_to_h5.py into a single file (can be very large)
"""

import argparse
parser = argparse.ArgumentParser(description='merge h5 datasets')
parser.add_argument('batch', type=str)
args = parser.parse_args()

file_list = sorted(glob.glob(SIMULATIONS_DIR+args.batch+'*.hdf5'))
random.shuffle(file_list)
output_file = h5py.File(SIMULATIONS_DIR+args.batch+'_merged.h5', 'w')

print(file_list)
print(output_file)

total_rows = 0

for n, fname in tqdm(enumerate(file_list)):
    with h5py.File(fname) as f1:
        moments_data = f1['moments'][:]
        lnoiseW_data = f1['noise_level_W'][:]
        lnoiseX_data = f1['nosie_level_X'][:]
        kssrg_data = f1['k_ssrg'][:]
        bssrg_data = f1['b_ssrg'][:]
        iwc_data = f1['iwc'][:]
        ptype_data = f1['ptype'][:]
        temperature_data = f1['temperature'][:]
        asratio_data = f1['aspect_ratio'][:]
        ams_data = f1['a_mass_size'][:]
        bms_data = f1['b_mass_size'][:]
        aas_data = f1['alpha_area_size'][:]
        bas_data = f1['beta_area_size'][:]
        dmean_data = f1['dmean'][:]
        spectra_data = f1['spectra'][:]
    
    
    total_rows = total_rows + moments_data.shape[0]

    if n == 0:
        #first file; create the dummy dataset with no max shape
        moments = output_file.create_dataset("moments", moments_data.shape, maxshape=(None,None,None))
        lnoise_W = output_file.create_dataset("noise_level_W", lnoiseW_data.shape, maxshape=(None,))
        lnoise_X = output_file.create_dataset("noise_level_X", lnoiseX_data.shape, maxshape=(None,))
        k_ssrg = output_file.create_dataset("k_ssrg", kssrg_data.shape, maxshape=(None,))
        b_ssrg = output_file.create_dataset("b_ssrg", bssrg_data.shape, maxshape=(None,))
        iwc = output_file.create_dataset("iwc", iwc_data.shape, maxshape=(None,))
        ptype = output_file.create_dataset("ptype", ptype_data.shape, maxshape=(None,))
        temperature = output_file.create_dataset("temperature", temperature_data.shape, maxshape=(None,))
        aspect_ratio = output_file.create_dataset("aspect_ratio", asratio_data.shape, maxshape=(None,))
        a_mass_size = output_file.create_dataset("a_mass_size", ams_data.shape, maxshape=(None,))
        b_mass_size = output_file.create_dataset("b_mass_size", bms_data.shape, maxshape=(None,))
        alpha_area_size = output_file.create_dataset("alpha_area_size", aas_data.shape, maxshape=(None,))
        beta_area_size = output_file.create_dataset("beta_area_size", bas_data.shape, maxshape=(None, ))
        dmean = output_file.create_dataset("dmean", dmean_data.shape, maxshape=(None,))
        spectra = output_file.create_dataset("spectra", spectra_data.shape, maxshape=(None, None,None))
        
        #fill the first section of the dataset
        moments[:,:,:] = moments_data
        lnoise_W[:] = lnoiseW_data
        lnoise_X[:] = lnoiseX_data
        k_ssrg[:] = kssrg_data
        b_ssrg[:] = bssrg_data
        iwc[:] = iwc_data
        ptype[:] = ptype_data
        temperature[:] = temperature_data
        aspect_ratio[:] = asratio_data
        a_mass_size[:] = ams_data
        b_mass_size[:] = bms_data
        alpha_area_size[:] = aas_data
        beta_area_size[:] = bas_data
        dmean[:] = dmean_data
        spectra[:,:,:]=spectra_data
        
        where_to_start_appending = total_rows

    else:
    #resize the dataset to accomodate the new data
        moments.resize(total_rows, axis=0)
        moments[where_to_start_appending:total_rows, :, :] = moments_data
        lnoise_W.resize(total_rows, axis=0)
        lnoise_W[where_to_start_appending:total_rows] = lnoiseW_data
        lnoise_X.resize(total_rows, axis=0)
        lnoise_X[where_to_start_appending:total_rows] = lnoiseX_data
        k_ssrg.resize(total_rows, axis=0)
        k_ssrg[where_to_start_appending:total_rows] = kssrg_data
        b_ssrg.resize(total_rows, axis=0)
        b_ssrg[where_to_start_appending:total_rows] = bssrg_data
        iwc.resize(total_rows, axis=0)
        iwc[where_to_start_appending:total_rows] = iwc_data
        ptype.resize(total_rows, axis=0)
        ptype[where_to_start_appending:total_rows] = ptype_data
        temperature.resize(total_rows, axis=0)
        temperature[where_to_start_appending:total_rows] = temperature_data
        aspect_ratio.resize(total_rows, axis=0)
        aspect_ratio[where_to_start_appending:total_rows] = asratio_data
        a_mass_size.resize(total_rows, axis=0)
        a_mass_size[where_to_start_appending:total_rows] = ams_data
        b_mass_size.resize(total_rows, axis=0)
        b_mass_size[where_to_start_appending:total_rows] = bms_data
        alpha_area_size.resize(total_rows, axis=0)
        alpha_area_size[where_to_start_appending:total_rows] = aas_data
        beta_area_size.resize(total_rows, axis=0)
        beta_area_size[where_to_start_appending:total_rows] = bas_data
        dmean.resize(total_rows, axis=0)
        dmean[where_to_start_appending:total_rows] = dmean_data
        spectra.resize(total_rows, axis=0)
        spectra[where_to_start_appending:total_rows, :, :] = spectra_data

        where_to_start_appending = total_rows

output_file.close()
