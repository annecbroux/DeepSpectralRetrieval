
"""
This script is the first part of the training set generation.
It samples microphysical parameters according to predefined statistics and runs PAMTRA simulations on them.
It can (should) be parsed with indices controlling how many batches of simulations are run (each batch -> one output file named with the index of the batch)
See below for detail.
"""

import pyPamtra
import numpy as np
import matplotlib.pyplot as plt
import sys
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
import scipy
import random
import pandas as pd
from sample_parameters import sample_from_rv, sample_from_fit, sample_lnoise_X, sample_lnoise_W
from scipy.interpolate import UnivariateSpline
import glob
import time
import copy
import os
import json 


OUTPUT_DATA_DIR = '/data/spectra_simulations/'
# The following parameters are used to simulate spectra in batches
N_HGT_LEV = 9 # number of height levels simulated at once
N_HM_TYPE = 9 # number of hydrometeor types simulated at once
N_SIM = 100 # number of simulated spectra in one batch

RADAR_PROPERTIES = 'radar_config.json'
radar_config = json.load(open(RADAR_PROPERTIES))

import argparse
parser = argparse.ArgumentParser(description='launch pamtra simulations')
# These indices control how many batches of simulations are run
# Each batch of simulations -> one output file named with the index of the batch (i_batch) and containing N_SIM spectra (at X and W)
parser.add_argument('index_start', type=int, default=0)
parser.add_argument('index_end', type=int, default=1000)
args = parser.parse_args()


for i_batch in range(args.index_start,args.index_end):

    # We load the distributions of the parameters / fits from which to sample
    df_massb = pd.read_csv('parameters/mass_paramb_hist_log',sep='\t')
    df_areab = pd.read_csv('parameters/area_paramb_hist_log',sep='\t')
    df_dmean = pd.read_csv('parameters/dmean_hist',sep='\t')
    df_mass_loga_fit = pd.read_csv('parameters/mass_size_log_loga',sep='\t')
    df_area_logalpha_fit = pd.read_csv('parameters/area_size_log_loga',sep='\t')
    df_dmax = pd.read_csv('parameters/dmax',sep='\t')

    ########### DEFINE ENVIRONMENTAL PARAMETERS ###################
    wind_uv_W = 0.
    wind_w_W = 0.
    wind_uv_X = 0.
    wind_w_X = 0.
    temperature = 273.15+np.random.uniform(-20,2) # This allows to account for temperature variability
    turb = 0. 
    iwc = np.random.exponential(.5e-3,(N_SIM,N_HGT_LEV,N_HM_TYPE)) # The iwc is sampled from a negative exponential of mean 0.5e-3 kg/m3
    rho_air = 1.2
    height_level = np.random.uniform(0,5000.,N_SIM) # This is only used for sampling the noise level


    ########### DEFINE RADAR PARAMETERS ###################
    beamwidth_deg_W = radar_config['beamwidth_W']
    beamwidth_deg_X = radar_config['beamwidth_X']
    integration_time_W = 1.
    integration_time_X = 1.

    noise_level_W = sample_lnoise_W(height_level)
    noise_level_X = sample_lnoise_X(height_level)
    fft_len_W = 512 # at this stage we use 512 points in the spectra
    v_nyq_W = 6.92 # for the simulated spectra we use 6.92 as Nyquist velocity (because matched one of the real radars)
    fft_len_X = 512
    v_nyq_X = 6.92

    freq_W = radar_config['freq_W']
    freq_X = radar_config['freq_X']


    ########### DEFINE MICROPHYSICAL PARAMETERS ##################
    ## Sample particle type (40% aggregates, 20% planar crystals, 20% graupel, 20% columnar crystals)
    ptype_x = np.random.randint(0,10)
    if ptype_x < 4:
        ptype = 'agg'
    elif ptype_x < 6:
        ptype = 'den'
    elif ptype_x < 8:
        ptype = 'grau'
    else:
        ptype = 'col'  

    # Load scattering and microphysical properties for this particle type
    spl_beta = np.load('ssrg_coefficients/spl_beta_%s.npy'%ptype,allow_pickle=True)
    spl_kappa = np.load('ssrg_coefficients/spl_kappa_%s.npy'%ptype,allow_pickle=True)
    df_ar = pd.read_csv('parameters/ar_%s'%ptype,sep='\t')

    # Sample D0
    dmean = sample_from_rv(df_dmean[df_dmean['ptype']==ptype],n=N_SIM*N_HM_TYPE)*1e-2

    # Sample Aspect ratio (using D0)
    dmean_for_ar = copy.deepcopy(dmean).reshape(-1,1)
    dmean_for_ar[dmean_for_ar<df_ar['dmin'].iloc()[0]] = df_ar['dmin'].iloc()[0]
    dmean_for_ar[dmean_for_ar>df_ar['dmax'].iloc()[-1]] = df_ar['dmax'].iloc()[-1]
    aspect_ratio = sample_from_rv(df_ar.iloc()[np.where((dmean_for_ar<=np.array(df_ar['dmax']))&(dmean_for_ar>=np.array(df_ar['dmin'])))[1]],n=N_SIM*N_HM_TYPE)

    # Sample b mass-size
    rv_params = df_massb[df_massb['ptype']==ptype]
    rvtype = getattr(stats, rv_params['rv'].iloc()[0])
    rv = rvtype(rv_params['rv_1'],rv_params['rv_2'],rv_params['rv_3'])
    if ((ptype=='agg') | (ptype=='den')):
        b_mass_size = 1.2*rv.ppf(np.random.uniform(0,rv.cdf(3.5/1.2),N_SIM*N_HM_TYPE))
    else:
        b_mass_size = rv.ppf(np.random.uniform(0,rv.cdf(3.5),N_SIM*N_HM_TYPE))

    # Sample beta area-size (bounded with b)
    rv_params = df_areab[df_areab['ptype']==ptype]
    rvtype = getattr(stats, rv_params['rv'].iloc()[0])
    rv = rvtype(rv_params['rv_1'],rv_params['rv_2'],rv_params['rv_3'])
    beta_area_size = 2-rv.ppf(np.random.uniform(rv.cdf(2-b_mass_size),1,N_SIM*N_HM_TYPE))

    # Sample a mass size (from b + fit)
    a_mass_size = sample_from_fit(df_mass_loga_fit[df_mass_loga_fit['ptype']==ptype],b_mass_size)
    # Sample alpha area size (from b + fit)
    alpha_area_size = sample_from_fit(df_area_logalpha_fit[df_area_logalpha_fit['ptype']==ptype],beta_area_size)

    # Sample SSRGA coefficients (from D0)
    dmax_for_spl = copy.deepcopy(dmean)
    if ptype=='grau':
        dmax_for_spl[dmean<8e-4] = 2e-3
    else:
        dmax_for_spl[dmean<8e-4] = 8e-4
    dmax_for_spl[dmean>1.5e-2] = 1.5e-2
    b_ssrg = UnivariateSpline._from_tck(spl_beta)(dmax_for_spl)
    k_ssrg = UnivariateSpline._from_tck(spl_kappa)(dmax_for_spl)
    
    if ptype=='col':
        canting_angle = 90.
    else:
        canting_angle = 0.
    
    ######### RESHAPE PARAMETERS CORRECTLY ##################
    k_ssrg = k_ssrg.reshape(n_sim,-1)
    b_ssrg = b_ssrg.reshape(n_sim,-1)
    aspect_ratio = aspect_ratio.reshape(n_sim,-1)
    a_mass_size = a_mass_size.reshape(n_sim,-1)
    b_mass_size = b_mass_size.reshape(n_sim,-1)
    alpha_area_size = alpha_area_size.reshape(n_sim,-1)
    beta_area_size = beta_area_size.reshape(n_sim,-1)
    dmean = dmean.reshape(n_sim,-1)

    parameters_full = {'wind_uv_W':wind_uv_W,
                  'wind_w_W':wind_w_W,
                  'wind_uv_X':wind_w_W,
                  'wind_w_X':wind_w_W,
                  'temperature':temperature,
                  'turb':turb,
                  'beamwidth_deg_W':beamwidth_deg_W,
                  'beamwidth_deg_X':beamwidth_deg_X,
                  'integration_time_W':integration_time_W,
                  'integration_time_X':integration_time_X,
                  'noise_level_W':noise_level_W,
                  'noise_level_X':noise_level_X,
                  'fft_len_W':fft_len_W,
                  'fft_len_X':fft_len_X,
                  'v_nyq_W':v_nyq_W,
                  'v_nyq_X':v_nyq_X,
                  'freq_W':freq_W,
                  'freq_X':freq_X,
                  'k_ssrg':k_ssrg,
                  'b_ssrg':b_ssrg,
                  'canting_angle':canting_angle,
                  'iwc':iwc, 
                  'rho_air':rho_air,
                  'ptype':ptype, 
                  'aspect_ratio':aspect_ratio, 
                  'a_mass_size':a_mass_size, 
                  'b_mass_size':b_mass_size, 
                  'alpha_area_size':alpha_area_size, 
                  'beta_area_size':beta_area_size,
                  'dmean':dmean
                 }

    ######################## LOOP OVER NUMBER OF SIMULATIONS ###############################

    df_list = []
    for i_sim in range(n_sim):
        p = {}
        for key in parameters_full.keys():
            if isinstance(parameters_full[key],np.ndarray):
                p[key] = parameters_full[key][i_sim]
            else:
                p[key] = parameters_full[key]

        # maximum diameter of PSD is from MASCDB
        dmax = df_dmax[ptype][0]

        # Create PAMTRA instance    
        pam = pyPamtra.pyPamtra()
        # Add hydrometeor descriptors: see pamtra documentation for parameterization (https://pamtra.readthedocs.io/en/latest/)
	# We add several types of hydrometeors in each simulation which allows to speed up the generation of the training set (simulate different sets of parameters at once)
	# i.e. taking advantage of parallel computing in PAMTRA.
        pam.df.addHydrometeor(('ice0', #name
                                 p['aspect_ratio'][0], #aspect ratio, <1 means oblate
                                 -1, #phase: -1=ice, 1=liq
                                 -99.,#200, #density
                                 p['a_mass_size'][0], #a parameter of mass-size
                                 p['b_mass_size'][0], #b parameter of mass-size
                                 p['alpha_area_size'][0], #alpha parameter of cross-section area - size relation
                                 p['beta_area_size'][0], #beta parameter of cross-section area - size relation
                                 23, # moment provided in input file
                                 200, #number of discrete size bins (internally, nbins+1 is used)
                                 'mgamma', #name of psd
                                 -99., #1st parameter of psd
                                 -99.,#2nd parameter of psd
                                 0., #3rd parameter of psd
                                 1., #4th parameter of psd
                                 p['dmean'][0]/100, #min diameter
                                 min(dmax,p['dmean'][0]*50), # max diameter
                                 'ss-rayleigh-gans_%.2f_%.2f'%(p['k_ssrg'][0],p['b_ssrg'][0])
                                 'heymsfield10_particles',
                                 p['canting_angle'])) #canting angle of hydrometeors, only for Tmatrix and SSRG

        pam.df.addHydrometeor(('ice1', p['aspect_ratio'][1], -1, -99., p['a_mass_size'][1], p['b_mass_size'][1], p['alpha_area_size'][1], p['beta_area_size'][1], 23, 200, 'mgamma', -99.,-99.,   0., 1., p['dmean'][1]/100, 
                               min(dmax,p['dmean'][1]*50), 'ss-rayleigh-gans_%.2f_%.2f_%.2f'%(p['k_ssrg'][1],p['b_ssrg'][1]),'heymsfield10_particles', p['canting_angle']))
        pam.df.addHydrometeor(('ice2', p['aspect_ratio'][2], -1, -99., p['a_mass_size'][2], p['b_mass_size'][2], p['alpha_area_size'][2], p['beta_area_size'][2], 23, 200, 'mgamma', -99.,-99.,   0., 1., p['dmean'][2]/100, 
                               min(dmax,p['dmean'][2]*50), 'ss-rayleigh-gans_%.2f_%.2f_%.2f'%(p['k_ssrg'][2],p['b_ssrg'][2]),'heymsfield10_particles', p['canting_angle'])) 
        pam.df.addHydrometeor(('ice3', p['aspect_ratio'][3], -1, -99., p['a_mass_size'][3], p['b_mass_size'][3], p['alpha_area_size'][3], p['beta_area_size'][3], 23, 200, 'mgamma', -99.,-99.,   0., 1., p['dmean'][3]/100, 
                               min(dmax,p['dmean'][3]*50), 'ss-rayleigh-gans_%.2f_%.2f_%.2f'%(p['k_ssrg'][3],p['b_ssrg'][3]),'heymsfield10_particles', p['canting_angle'])) 
        pam.df.addHydrometeor(('ice4', p['aspect_ratio'][4], -1, -99., p['a_mass_size'][4], p['b_mass_size'][4], p['alpha_area_size'][4], p['beta_area_size'][4], 23, 200, 'mgamma', -99.,-99.,   0., 1., p['dmean'][4]/100, 
                               min(dmax,p['dmean'][4]*50), 'ss-rayleigh-gans_%.2f_%.2f_%.2f'%(p['k_ssrg'][4],p['b_ssrg'][4]),'heymsfield10_particles', p['canting_angle'])) 
        pam.df.addHydrometeor(('ice5', p['aspect_ratio'][5], -1, -99., p['a_mass_size'][5], p['b_mass_size'][5], p['alpha_area_size'][5], p['beta_area_size'][5], 23, 200, 'mgamma', -99.,-99.,   0., 1., p['dmean'][5]/100, 
                               min(dmax,p['dmean'][5]*50), 'ss-rayleigh-gans_%.2f_%.2f_%.2f'%(p['k_ssrg'][5],p['b_ssrg'][5]),'heymsfield10_particles', p['canting_angle'])) 
        pam.df.addHydrometeor(('ice6', p['aspect_ratio'][6], -1, -99., p['a_mass_size'][6], p['b_mass_size'][6], p['alpha_area_size'][6], p['beta_area_size'][6], 23, 200, 'mgamma', -99.,-99.,   0., 1., p['dmean'][6]/100, 
                               min(dmax,p['dmean'][6]*50), 'ss-rayleigh-gans_%.2f_%.2f_%.2f'%(p['k_ssrg'][6],p['b_ssrg'][6]),'heymsfield10_particles', p['canting_angle'])) 
        pam.df.addHydrometeor(('ice7', p['aspect_ratio'][7], -1, -99., p['a_mass_size'][7], p['b_mass_size'][7], p['alpha_area_size'][7], p['beta_area_size'][7], 23, 200, 'mgamma', -99.,-99.,   0., 1., p['dmean'][7]/100, 
                               min(dmax,p['dmean'][7]*50), 'ss-rayleigh-gans_%.2f_%.2f_%.2f'%(p['k_ssrg'][7],p['b_ssrg'][7]),'heymsfield10_particles', p['canting_angle'])) 
        pam.df.addHydrometeor(('ice8', p['aspect_ratio'][8], -1, -99., p['a_mass_size'][8], p['b_mass_size'][8], p['alpha_area_size'][8], p['beta_area_size'][8], 23, 200, 'mgamma', -99.,-99.,   0., 1., p['dmean'][8]/100, 
                               min(dmax,p['dmean'][8]*50), 'ss-rayleigh-gans_%.2f_%.2f_%.2f'%(p['k_ssrg'][8],p['b_ssrg'][8]),'heymsfield10_particles', p['canting_angle']))


        # Define the atmospheric conditions in PAMTRA
        # NB the height levels are n_hgt_lev between 900m and 1100m 
        #(if we were to model full column we would have to prescribe more atmospheric conditions, easier to simulate at a constant altitude)
        pam = pyPamtra.importer.createUsStandardProfile(pam,hgt_lev=np.array([np.linspace(900,1100,N_HGT_LEV+1).tolist()]*3).reshape(3,1,-1).repeat(3,axis=1))
        pam.p['temp_lev'][:]=p['temperature']
        pam.p['relhum_lev'][:]=90.
        pam.set["verbose"] = -1

        # prescribe the mixing ratio through IWC
        p['iwc'] = p['iwc'].reshape(3,3,-1)
        pam.p["hydro_q"][0,0,:,0] = p['iwc'][0,0,:]/p['rho_air']
        pam.p["hydro_q"][1,0,:,1] = p['iwc'][1,0,:]/p['rho_air']
        pam.p["hydro_q"][2,0,:,2] = p['iwc'][2,0,:]/p['rho_air']
        pam.p["hydro_q"][0,1,:,3] = p['iwc'][0,1,:]/p['rho_air']
        pam.p["hydro_q"][1,1,:,4] = p['iwc'][1,1,:]/p['rho_air']
        pam.p["hydro_q"][2,1,:,5] = p['iwc'][2,1,:]/p['rho_air']
        pam.p["hydro_q"][0,2,:,6] = p['iwc'][0,2,:]/p['rho_air']
        pam.p["hydro_q"][1,2,:,7] = p['iwc'][1,2,:]/p['rho_air']
        pam.p["hydro_q"][2,2,:,8] = p['iwc'][2,2,:]/p['rho_air']

        # prescribe the effective radius through Do
        pam.p["hydro_reff"][0,0,:,0] = 3/2*p['dmean'][0] 
        pam.p["hydro_reff"][1,0,:,1] = 3/2*p['dmean'][1] 
        pam.p["hydro_reff"][2,0,:,2] = 3/2*p['dmean'][2] 
        pam.p["hydro_reff"][0,1,:,3] = 3/2*p['dmean'][3] 
        pam.p["hydro_reff"][1,1,:,4] = 3/2*p['dmean'][4] 
        pam.p["hydro_reff"][2,1,:,5] = 3/2*p['dmean'][5] 
        pam.p["hydro_reff"][0,2,:,6] = 3/2*p['dmean'][6] 
        pam.p["hydro_reff"][1,2,:,7] = 3/2*p['dmean'][7] 
        pam.p["hydro_reff"][2,2,:,8] = 3/2*p['dmean'][8] 

        # prescribe remaining parameters i.e. radar properties, etc.
        pam.p["wind_w"][:] = p['wind_w_W']
        pam.nmlSet["radar_mode"] = "spectrum"
        pam.nmlSet['save_psd']=True
        pam.nmlSet["radar_noise_distance_factor"] = 2.0
        pam.nmlSet["radar_save_noise_corrected_spectra"]=  False
        pam.nmlSet['passive'] = False
        pam.nmlSet['radar_airmotion'] = False
        pam.nmlSet['radar_aliasing_nyquist_interv'] = 1
        pam.nmlSet['obs_height'] = 0
        pam.nmlSet['hydro_adaptive_grid'] = False
        pam.nmlSet['radar_allow_negative_dD_dU'] = True

        pam.nmlSet['radar_nfft']= p['fft_len_W']
        pam.nmlSet['radar_max_v']= p['v_nyq_W']
        pam.nmlSet['radar_min_v']= -p['v_nyq_W']        
        pam.nmlSet['radar_pnoise0'] = np.array(p['noise_level_W']) #we set the noise level to be that of our W-band radar

        # Run PAMTRA: simulate X and W spectra
        pam.runParallelPamtra([p['freq_X'],p['freq_W']],pp_deltaX=1, pp_deltaY=1, pp_deltaF=1,pp_local_workers=6,timeout=1)

        # set the noise of the X-band spectra to be that of our X-band
        # This is done a posteriori because impossible to simulate 2 different noise levels at once in PAMTRA
        nxs = p['noise_level_X']
        specx = pam.r['radar_spectra'][:,:,:,0,0,:]
        specx = specx - (np.array(nxs)-10*np.log10(512))
        specx[specx<0] = 0
        pam.r['radar_spectra'][:,:,:,0,0,:] = specx+(np.array(nxs)-10*np.log10(512))

        # Wrap up the results in the p container and adjust the dimensions accordingly
        p['spectra']=(pam.r['radar_spectra'][0,0,:,:,0,:].tolist()+
                      pam.r['radar_spectra'][1,0,:,:,0,:].tolist()+
                      pam.r['radar_spectra'][2,0,:,:,0,:].tolist()+
                      pam.r['radar_spectra'][0,1,:,:,0,:].tolist()+
                      pam.r['radar_spectra'][1,1,:,:,0,:].tolist()+
                      pam.r['radar_spectra'][2,1,:,:,0,:].tolist()+
                      pam.r['radar_spectra'][0,2,:,:,0,:].tolist()+
                      pam.r['radar_spectra'][1,2,:,:,0,:].tolist()+
                      pam.r['radar_spectra'][2,2,:,:,0,:].tolist())

        p['moments'] = (pam.r["radar_moments"][0,0,:,:,0,0,:].tolist()+
                        pam.r["radar_moments"][1,0,:,:,0,0,:].tolist()+
                        pam.r["radar_moments"][2,0,:,:,0,0,:].tolist()+
                        pam.r["radar_moments"][0,1,:,:,0,0,:].tolist()+
                        pam.r["radar_moments"][1,1,:,:,0,0,:].tolist()+
                        pam.r["radar_moments"][2,1,:,:,0,0,:].tolist()+
                        pam.r["radar_moments"][0,2,:,:,0,0,:].tolist()+
                        pam.r["radar_moments"][1,2,:,:,0,0,:].tolist()+
                        pam.r["radar_moments"][2,2,:,:,0,0,:].tolist())
        
        p['Ze'] = (pam.r["Ze"][0,0,:,:,0,0].tolist()+
                    pam.r["Ze"][1,0,:,:,0,0].tolist()+
                    pam.r["Ze"][2,0,:,:,0,0].tolist()+
                    pam.r["Ze"][0,1,:,:,0,0].tolist()+
                    pam.r["Ze"][1,1,:,:,0,0].tolist()+
                    pam.r["Ze"][2,1,:,:,0,0].tolist()+
                    pam.r["Ze"][0,2,:,:,0,0].tolist()+
                    pam.r["Ze"][1,2,:,:,0,0].tolist()+
                    pam.r["Ze"][2,2,:,:,0,0].tolist())
        
        
        p['iwc'] = (p['iwc'][0,0,:].tolist()+
                    p['iwc'][1,0,:].tolist()+
                    p['iwc'][2,0,:].tolist()+
                    p['iwc'][0,1,:].tolist()+
                    p['iwc'][1,1,:].tolist()+
                    p['iwc'][2,1,:].tolist()+
                    p['iwc'][0,2,:].tolist()+
                    p['iwc'][1,2,:].tolist()+
                    p['iwc'][2,2,:].tolist())

        p['k_ssrg'] = p['k_ssrg'].repeat(N_HGT_LEV)
        p['b_ssrg'] = p['b_ssrg'].repeat(N_HGT_LEV)
        p['aspect_ratio'] = p['aspect_ratio'].repeat(N_HGT_LEV)
        p['a_mass_size'] = p['a_mass_size'].repeat(N_HGT_LEV)
        p['b_mass_size'] = p['b_mass_size'].repeat(N_HGT_LEV)
        p['alpha_area_size'] = p['alpha_area_size'].repeat(N_HGT_LEV)
        p['beta_area_size'] = p['beta_area_size'].repeat(N_HGT_LEV)
        p['dmean'] = p['dmean'].repeat(N_HGT_LEV)

        for key in ['canting_angle','wind_uv_W','wind_w_W','wind_uv_X','wind_w_X', 'temperature', 'turb', 'beamwidth_deg_W', 'beamwidth_deg_X', 'integration_time_W', 'integration_time_X', 
                    'noise_level_W', 'noise_level_X', 'fft_len_W', 'fft_len_X', 'v_nyq_W', 'v_nyq_X', 'freq_W', 'freq_X', 'rho_air','ptype']:
            p[key] = [p[key]]*N_HGT_LEV*N_HM_TYPE

        df_list.append(pd.DataFrame(p))

    # Concatenate all simulations
    s = pd.concat(df_list,ignore_index=True)

    # Remove spectra when simulation failed
    klist = []
    for i in range(len(s)):
        if ((s['moments'].iloc()[i][0][0]<-100) | (s['moments'].iloc()[i][1][0]<-100)):
            klist.append(i)
    if len(klist)>0:
        s = s.drop(index=klist).reset_index()
    len(klist)
    if 'level_0' in s.columns:
        s.drop(columns=['level_0','index'])
    else:
        s.drop(columns=['index'])

    # Save the dataframe to a .feather file (not good for sharing or storing but fast I/O so adapted here)
    s.to_feather(OUTPUT_DATA_DIR+'/spectra_%d.feather'%i_batch)


