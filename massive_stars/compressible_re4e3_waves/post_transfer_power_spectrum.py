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

from plotpal.file_reader import SingleTypeReader as SR
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)

from dedalus.tools.config import config
from scipy.interpolate import interp1d

from d3_stars.defaults import config
from d3_stars.simulations.parser import name_star
out_dir, out_file = name_star()

#fit_wave_flux = False
fit_wave_flux = True

fudge_factor = 1

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
    chi_rad_min = f['chi_rad_B'][()].min()

full_out_dir = 'damping_theory_power'
if not os.path.exists(full_out_dir):
    os.makedirs(full_out_dir)

if fit_wave_flux:
    #Fit wave luminosity
    #Fit A f ^ alpha ell ^ beta
#    radius_str = '1.1'
    radius_str = '1.25'
    fit_freq_range = (4e-2, 1e-1)
    fit_ell_range = (1, 4)
    fig = plt.figure()
    possible_alphas = [-13/2]
    possible_betas = [4]
#    possible_alphas = [-11/2, -5.75, -6, -6.25, -13/2, -6.75, -7, -7.25, -7.5]
#    possible_betas = [2.5, 2.75, 3, 3.25, 3.5, 3.75, 4]
    fit_A = []
    fit_alpha = []
    fit_beta  = []
    with h5py.File('../twoRcore_re4e3_damping/wave_flux/wave_luminosities.h5', 'r') as lum_file:
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
                    kh = np.sqrt(ells[None,:]*(ells[None,:]+1))
                    A = np.mean((wave_luminosity / freqs[:,None]**(alpha) / kh**(beta))[good_freqs[:,None]*good_ells[None,:]])
                    fit = A * freqs[:,None]**alpha * kh**beta
                    error.append(np.mean( np.abs(1 - (np.log10(fit) / np.log10(wave_luminosity))[good_freqs[:,None]*good_ells[None,:]])))
                    info.append((A, alpha, beta))
            print(info, error)
            A, alpha, beta = info[np.argmin(error)]

            fit_A.append(A)
            fit_alpha.append(alpha)
            fit_beta.append(beta)
    wave_luminosity_power = lambda f, ell: fit_A[-1]*f**(fit_alpha[-1])*np.sqrt(ell*(ell+1))**(fit_beta[-1])
    print('fit_A', fit_A)
    print('fit_A frac', np.array(fit_A[1:])/np.array(fit_A[:-1]))
    print('fit_alpha', fit_alpha)
    print('fit_beta', fit_beta)
#    plt.loglog(freqs, wave_luminosity[:,ells==1])
#    plt.loglog(freqs, wave_luminosity_power(freqs, 1))
#    plt.show()
#    for ell in ells:
#        plt.loglog(freqs, wave_luminosity[:,ell == ells].ravel())
#        plt.loglog(freqs, wave_luminosity_power(freqs, ell))
#        plt.show()

else:
    radius_str = '1.25'
    wave_lums = dict()
    with h5py.File('../twoRcore_re4e3_damping/wave_flux/wave_luminosities.h5', 'r') as lum_file:
        wave_lums['freqs'] = freqs = lum_file['freqs'][()]
        wave_lums['ells'] = lum_file['ells'][()].ravel()
        wave_lums['lum'] = lum_file['wave_luminosity(r={})'.format(radius_str)][0,:]
        radius_str = '1.1'
        wave_lums['lum'][freqs < 2e-2] = lum_file['wave_luminosity(r={})'.format(radius_str)][0,:][freqs < 2e-2,:]

freqs = np.logspace(-3, 0, 1000)
t_freqs = np.logspace(np.log10(freqs.min()), np.log10(freqs.max()), 100000)


with h5py.File('power_spectra/power_spectra.h5', 'r') as pow_f:
    surface_power = pow_f['shell(s1_S2,r=R)'][-1,:]
    surface_ells = pow_f['ells'][()].squeeze()
    surface_freqs = pow_f['freqs'][()].squeeze()
    print('surface power shape', surface_power.shape)


        
powers = []
ell_vals = []
fig = plt.figure()
Lmax = config.eigenvalue['Lmax']
for ell in range(1, Lmax+1):
    try:
        print('plotting ell = {}'.format(ell))
        with h5py.File('eigenvalues/transfer_ell{:03d}_eigenvalues.h5'.format(ell), 'r') as ef:
            transfer_func_root_lum = ef['transfer_root_lum'][()]
            transfer_freq = ef['om'][()]/(2*np.pi)
            transfer_interp = lambda f: 10**interp1d(np.log10(transfer_freq), np.log10(transfer_func_root_lum), bounds_error=False, fill_value=-1e99)(np.log10(f))

        if fit_wave_flux:
            wave_luminosity = lambda f: wave_luminosity_power(f,ell)
        else:
            log_wave_lum = interp1d(np.log10(wave_lums['freqs']), np.log10(wave_lums['lum'][:,ell == wave_lums['ells']].ravel()))
            wave_luminosity = lambda f: 10**(log_wave_lum(np.log10(f)))
#
#        #wave_lum_ell should be wave_flux_ell? - see slack stuff around sept 29 2021
#        kh2 = ell * (ell + 1) / 1**2 #at r = 1
#        kr = lambda f: np.abs(( ((-1)**(3/4) / np.sqrt(2))\
#                          *np.sqrt(-2*1j*kh2 - (2*np.pi*f/chi_rad_min) + np.sqrt((2*np.pi*f)**3 + 4*1j*kh2*chi_rad_min*N2plateau)/(chi_rad_min*np.sqrt(2*np.pi*f)) )).real)
#        ur2 = lambda f: (2*np.pi*f) * (R_gas / Cp) * np.sqrt(kr(f)**2) * wave_flux_rcb(f) / N2plateau 
#        plt.loglog(t_freqs, kr(t_freqs)**2)
#        plt.loglog(t_freqs, kh2*(N2plateau/(2*np.pi*t_freqs)**2 - 1))
#        plt.axvline(N2plateau**(1/2)/(2*np.pi))
#        plt.show()
        fudge =  fudge_factor
        surface_s1_power = lambda f: fudge * np.conj(transfer_interp(f))*transfer_interp(f) * wave_luminosity(f)



        powers.append(surface_s1_power(t_freqs))
        plt.loglog(surface_freqs, surface_power[:,ell], c='orange', label='sim', lw=1.5)
#        plt.loglog(transfer_freq, np.exp(-depthfunc(transfer_freq))*np.conj(transfer_func)*transfer_func*ur2(transfer_freq), c='green', label='transfer', lw=0.5)
        plt.loglog(t_freqs, powers[-1], c='k',  label='transfer', lw=0.5)
        plt.legend(loc='best')
        plt.title('ell={}'.format(ell))
        plt.xlabel('freqs (sim units)')
        plt.ylabel(r'power')
        plt.ylim(1e-30, 1e-10)
        plt.xlim(3e-3, 1.4)
        fig.savefig('{}/s1_simulated_freq_spectrum_ell{}.png'.format(full_out_dir, ell), dpi=300, bbox_inches='tight')
        plt.clf()
        ell_vals.append(ell)
    except:
        raise
        print("no eigenvalues for ell = {}".format(ell))
        
ell_vals = np.array(ell_vals)
powers = np.array(powers)
print(powers.shape)
sim_sum_power = np.sum(surface_power[:,1:ell_vals[-1]+1], axis=1)
sim_sum_hemisphere_power = np.sum((surface_power/surface_ells**2)[:,1:ell_vals[-1]+1], axis=1)
sum_ells_power = np.sum(powers, axis=0)
sum_ells_hemisphere_power = np.sum(powers/ell_vals[:,None]**2, axis=0)
#plt.loglog(surface_freqs, sim_sum_power, c='orange', lw=1)
plt.loglog(surface_freqs, sim_sum_hemisphere_power, c='orange', lw=0.5)
#plt.loglog(t_freqs, sum_ells_power, c='k', label=r'$\sum_{\ell} P_\ell$', lw=1)
plt.loglog(t_freqs, sum_ells_hemisphere_power, c='k', label=r'$\sum_{\ell} \frac{P_\ell}{\ell}$', lw=0.5)
plt.legend(loc='best')
plt.title('summed over ells')
plt.xlabel('freqs (sim units)')
plt.ylabel(r'power')
plt.ylim(1e-30, 1e-10)
plt.xlim(3e-3, 1.4)
fig.savefig('{}/s1_simulated_freq_spectrum_summed_ells.png'.format(full_out_dir), dpi=300, bbox_inches='tight')
fig.savefig('{}/s1_simulated_freq_spectrum_summed_ells.pdf'.format(full_out_dir), dpi=300, bbox_inches='tight')


with h5py.File('{}/simulated_powers.h5'.format(full_out_dir), 'w') as f:
    f['powers'] = powers
    f['freqs'] = freqs
    f['sum_powers'] = sum_ells_power
