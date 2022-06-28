import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import random
import pandas as pd

rvel = np.linspace(-6.92,6.92,endpoint=False)
dv = rvel[1]-rvel[0]


def sample_lnoise_X(r):
    # This gives the noise level as a function of range for our X-band radar
    slopeX =19.999744739889294
    interceptX = -13.46347477107991
    lnoiseX = np.log10(r/1000)*slopeX+interceptX+np.random.normal(0,5)
    lnoiseX_tot = 10*np.log10(dv*512*10**(lnoiseX/10))
    return lnoiseX_tot


def sample_lnoise_W(r):
    # This gives the noise level as a function of range for our W-band radar (chirp-dependent)
    chirp_rmax = [850, 3800]
    slopeW = [15.85594549857715, 16.80557588203832, 14.58944921200094]
    interceptW = [-30.87714001998062, -32.66563661948545, -33.369303083309845]
    lnoiseW = np.zeros(len(r))
    for ir in range(len(r)):
        rr = r[ir]
        if rr <=chirp_rmax[0]:
            ic = 0
        elif rr <=chirp_rmax[1]:
            ic = 1
        else:
            ic = 2
        lnoiseW[ir] = np.log10(rr/1000)*slopeW[ic]+interceptW[ic]+np.random.normal(0,3)
    lnoiseW_tot = 10*np.log10(dv*512*10**(lnoiseW/10))
    return lnoiseW_tot



def sample_from_rv(rv_params,n=1,plot=False):
    """
    This function samples from a random variable with a known distribution described in rv_params
    rv_params: pandas DataFrame containing the parameters of the distribution (name + parameters: e.g. mean, std for a Gaussian distrib.)
    n: number of samples to draw
    plot: to check the distribution from which we are sampling
    """
    rvtype = getattr(stats, rv_params['rv'].iloc()[0]) # extract the name of the distribution, then turn it into a statistical object
    try:
        rv = rvtype(rv_params['rv_1'],rv_params['rv_2'],rv_params['rv_3']) # 
    except:
        if (rv_params['rv'].iloc()[0]=='norm') :
            rv = rvtype(rv_params['rv_1'],rv_params['rv_2']*1.5)
        else:
            rv = rvtype(rv_params['rv_1'],rv_params['rv_2']) 
    value = rv.rvs(n)
    if plot:
        plt.figure()
        x = np.linspace(0,3.4,100)
        rnd = rv.rvs(1000)
        plt.hist(rnd,25,density=True)
        plt.plot(x,rv.pdf(x))
    return value

def sample_from_fit(fit_params,b):
    """
    This function uses the results of a linear fit to sample randomly a value correlated to another one.
    Relies on the following relation: log(a) = slope * b + intercept + epsilon (Eq1)
    fit_params: dict containing the parameters of the linear fit to use (slope, intercept, rmse)
    b: known value
    returns a (from Eq1)
    """
    loga = (b+np.random.normal(0,5*np.array(fit_params['rmse']),len(b))-np.array(fit_params['intercept']))/np.array(fit_params['slope'])
    a = np.exp(loga)
    return a

