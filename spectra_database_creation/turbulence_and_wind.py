import numpy as np
import scipy


def turb_broadening(freq,bmw,wind,turb_edr):
    """
    Computes turbulent broadening for given radar resulting from turbulent dissipation
    This is largely inspired from the code of PAMTRA (https://pamtra.readthedocs.io/en/latest/)
    Turbulent broadening equations from Shupe 2004
    freq: radar frequency
    bmw: radar beam width (in deg)
    wind: wind speed (m s-1)
    turb_edr: turbulent eddy dissipation rate (SI)
    """
    lamb = 299792458./(freq * 1e9)
    beamWidth=bmw/2./180.*np.pi 
    L_s = (wind * 1) + 2*900*np.sin(beamWidth)
    L_lambda = lamb / 2.
    sig_B2 = wind**2*beamWidth**2/2.76
    sig_T2 = 3*.5/2. * (turb_edr/(2.*np.pi))**(2./3.) * (L_s**(2./3.) - L_lambda**(2./3.))
    return sig_T2+sig_B2


def add_turbulence_vect(S,sigma_from_turb,moments=True):
    """
    Implements convolution of the spectrum with broadening kernel
    S: original spectrum 
    sigma_from_turb: size of convolution kernel
    moments: if we should also compute the moments of the broadened spectrum

    returns spec0_ifft_conv: broadened spectrum (and first moments if specified: Ze, mean Doppler vel, spectral width)
    """
    rvel = np.linspace(-6.92,6.92,S.shape[-1])
    s = np.power(10,(S/10)) # we need to be in linear units
    nfft = s.shape[-1]
    x = np.arange(nfft).reshape(1,-1).repeat(2,axis=0)

    turb1 = 1/np.sqrt(2*sigma_from_turb[:,:,None]*np.pi)*np.exp(-(x[None,:,:]-nfft/2)**2/(2*sigma_from_turb[:,:,None])) # broadening gaussian signal
    turb3 = np.empty(turb1.shape)
    turb3[:,:,:int(nfft/2)] = turb1[:,:,int(nfft/2):]
    turb3[:,:,int(nfft/2):] = turb1[:,:,:int(nfft/2)]
    spec0_fft = scipy.fft(s,axis=-1) # spectrum in frequency domain 
    turb_fft = scipy.fft(turb3,axis=-1) # broadening signal in frequency domain
    spec0_fft_conv = spec0_fft*turb_fft # convolution = product in Fourier space
    spec0_ifft_conv = scipy.ifft(spec0_fft_conv,axis=-1).real # back in real space

    if moments:
        smin = np.min(spec0_ifft_conv,axis=-1)
        spow = np.sum(spec0_ifft_conv-smin[:,:,None],axis=-1)
        mdv = np.sum(rvel*(spec0_ifft_conv-smin[:,:,None]),axis=-1)/spow
        sw = (np.sum((rvel-mdv[:,:,None])**2*(spec0_ifft_conv-smin[:,:,None]),axis=-1)/spow)**.5
        return (spec0_ifft_conv,10*np.log10(spow),mdv,sw)
    else:
        return spec0_ifft_conv


def shift_wind(spectra,wind,dv):
    #Prepare some containers:
    spec_shifted = np.empty(spectra.shape)
    temp = np.empty(spectra.shape)
    temp[:,] = np.arange(0,512)

    n_shift_wind = np.round(wind/dv) # number of indices corresponding to wind shift
    n_shift_wind[n_shift_wind<0]=n_shift_wind[n_shift_wind<0]+512

    # Actually shift the spectra:
    spec_shifted[temp<511-n_shift_wind[:,None]]=spectra[temp>n_shift_wind[:,None]]
    spec_shifted[temp>=511-n_shift_wind[:,None]]=spectra[temp<=n_shift_wind[:,None]]
    return spec_shifted
