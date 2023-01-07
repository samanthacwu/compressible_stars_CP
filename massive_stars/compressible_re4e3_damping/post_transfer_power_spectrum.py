"""
This script computes the wave flux in a d3 spherical simulation

Usage:
    post_transfer_power_spectrum.py
"""
import re
import gc
import os
import time
import sys
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
from docopt import docopt
from configparser import ConfigParser
from scipy import sparse
from scipy.interpolate import interp1d

from plotpal.file_reader import SingleTypeReader as SR
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)

from dedalus.tools.config import config
from scipy.interpolate import interp1d

from d3_stars.defaults import config
from d3_stars.simulations.parser import name_star
out_dir, out_file = name_star()

#args = docopt(__doc__)
with h5py.File(out_file, 'r') as f:
    rB = f['r_B'][()]
    rS1 = f['r_S1'][()]
    rS2 = f['r_S2'][()]
    rhoB = np.exp(f['ln_rho0_B'][()])
    rhoS1 = np.exp(f['ln_rho0_S1'][()])
    rhoS2 = np.exp(f['ln_rho0_S2'][()])
    r = np.concatenate((rB.flatten(), rS1.flatten(), rS2.flatten()))
    rho = np.concatenate((rhoB.flatten(), rhoS1.flatten(), rhoS2.flatten()))
    rho_func = interp1d(r,rho)
    tau_nd = f['tau_nd'][()]
    #Entropy units are erg/K/g
    N2plateau = f['N2plateau'][()] * tau_nd**2
    N2mesa = f['N2_mesa'][()] * tau_nd**2
    r_mesa = f['r_mesa'][()] / f['L_nd'][()]
    N2_func = interp1d(r_mesa, N2mesa)
    R_gas = f['R_gas'][()]
    Cp = f['Cp'][()]

full_out_dir = 'damping_theory_power'
if not os.path.exists(full_out_dir):
    os.makedirs(full_out_dir)


#Fit wave luminosity
#Fit A f ^ alpha ell ^ beta
radius_str = '1.5'
fit_freq_range = (2e-2, 1e-1)
fit_ell_range = (1, 4)
fig = plt.figure()
#possible_alphas = [-6, -13/2, -7]
possible_alphas = [-5, -5.25, -11/2, -5.75, -6, -6.25, -13/2, -6.75, -7, -15/2, -8]
possible_betas = [4]
fit_A = []
fit_alpha = []
fit_beta  = []
with h5py.File('FT_SH_transform_wave_shells/wave_luminosities.h5', 'r') as lum_file:
    freqs = lum_file['freqs'][()]
    good_freqs = (freqs >= fit_freq_range[0])*(freqs <= fit_freq_range[1])
    ells = lum_file['ells'][()].ravel()
    good_ells = (ells >= fit_ell_range[0])*(ells <= fit_ell_range[1])
    for i in range(lum_file['wave_luminosity(r={})'.format(radius_str)][()].shape[0]):
        wave_luminosity = np.abs(lum_file['wave_luminosity(r={})'.format(radius_str)][i,:,:])
        info = []
        error = []
        for j, alpha in enumerate(possible_alphas):
            for k, beta in enumerate(possible_betas):
                A = np.mean((wave_luminosity / freqs[:,None]**(alpha) / ells[None,:]**(beta))[good_freqs[:,None]*good_ells[None,:]])
                fit = A * freqs[:,None]**alpha * ells[None,:]**beta
                error.append(np.mean( np.abs(1 - (np.log10(fit) / np.log10(wave_luminosity))[good_freqs[:,None]*good_ells[None,:]])))
                info.append((A, alpha, beta))
        print(info, error)
        A, alpha, beta = info[np.argmin(error)]

        fit_A.append(A)
        fit_alpha.append(alpha)
        fit_beta.append(beta)
wave_luminosity_power = lambda f, ell: fit_A[-1]*f**(fit_alpha[-1])*ell**(fit_beta[-1])
wave_luminosity_str = r'{:.2e}'.format(fit_A[-1]) + r'$f^{'+'{:.1f}'.format(fit_alpha[-1])+'}\ell^{' + '{:.1f}'.format(fit_beta[-1]) +  ')$'
freqs = np.logspace(-3, 0, 1000)

print('fit_A', fit_A)
print('fit_A frac', np.array(fit_A[1:])/np.array(fit_A[:-1]))
print('fit_alpha', fit_alpha)
print('fit_beta', fit_beta)


with h5py.File('../compressible_re4e3_waves/FT_SH_transform_wave_shells/power_spectra.h5', 'r') as pow_f:
    surface_power = pow_f['shell(s1_S2,r=R)'][-1,:]
    surface_ells = pow_f['ells'][()].squeeze()
    surface_freqs = pow_f['freqs'][()].squeeze()


t_freqs = np.logspace(np.log10(freqs.min()), np.log10(freqs.max()), 100000)

        
powers = []
fig = plt.figure()
for ell in range(64):
    if ell == 0: continue
    try:
        print('plotting ell = {}'.format(ell))
        with h5py.File('eigenvalues/duals_ell{:03d}_eigenvalues.h5'.format(ell), 'r') as f:
            smooth_oms = f['smooth_oms'][()]
            smooth_depths = f['smooth_depths'][()]
            depthfunc = interp1d(smooth_oms/(2*np.pi), smooth_depths, bounds_error=False, fill_value='extrapolate')

        with h5py.File('eigenvalues/transfer_ell{:03d}_eigenvalues.h5'.format(ell), 'r') as ef:
            transfer_func = ef['transfer'][()]
            transfer_freq = ef['om'][()]/(2*np.pi)
            transfer_interp = lambda f: 10**interp1d(np.log10(transfer_freq), np.log10(transfer_func), bounds_error=False, fill_value=0)(np.log10(f))

        wave_flux_rcb = lambda f: wave_luminosity_power(f,ell)/(4*np.pi*1**2*rho_func(1))

        #wave_lum_ell should be wave_flux_ell? - see slack stuff around sept 29 2021
        kr2 = lambda f: (N2plateau/(2*np.pi*f)**2 - 1)*(ell*(ell+1))/1**2 #approximate, r=1
        ur2 = lambda f: (2*np.pi*f) * (R_gas / Cp) * np.sqrt(kr2(f)) * wave_flux_rcb(f) / N2plateau #TODO: fix the factor of f at the beginning.
        fudge = 4
        surface_s1_power = lambda f: fudge * np.conj(transfer_interp(f))*transfer_interp(f) * ur2(f)



        powers.append(surface_s1_power(t_freqs))
        plt.loglog(surface_freqs, surface_power[:,ell], c='orange', label='sim', lw=1.5)
#        plt.loglog(transfer_freq, np.exp(-depthfunc(transfer_freq))*np.conj(transfer_func)*transfer_func*ur2(transfer_freq), c='green', label='transfer', lw=0.5)
        plt.loglog(t_freqs, powers[-1], c='k',  label='transfer', lw=1)
        plt.legend(loc='best')
        plt.title('ell={}'.format(ell))
        plt.xlabel('freqs (sim units)')
        plt.ylabel(r'power')
        plt.ylim(1e-30, 1e-10)
        plt.xlim(3e-3, 1.4)
        fig.savefig('{}/s1_simulated_freq_spectrum_ell{}.png'.format(full_out_dir, ell), dpi=300, bbox_inches='tight')
        plt.clf()
    except:
        print("no eigenvalues for ell = {}".format(ell))
        

powers = np.array(powers)
print(powers.shape)
sum_ells_power = np.sum(powers, axis=0)
plt.loglog(t_freqs, sum_ells_power, c='k')
#    plt.legend(loc='best')
plt.title('summed over ells')
plt.xlabel('freqs (sim units)')
plt.ylabel(r'power')
plt.ylim(1e-30, 1e-10)
plt.xlim(3e-3, 1.4)
fig.savefig('{}/s1_simulated_freq_spectrum_summed_ells.png'.format(full_out_dir), dpi=300, bbox_inches='tight')


with h5py.File('{}/simulated_powers.h5'.format(full_out_dir), 'w') as f:
    f['powers'] = powers
    f['freqs'] = freqs
    f['sum_powers'] = sum_ells_power
