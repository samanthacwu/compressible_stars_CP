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
#with h5py.File('SH_wave_flux_spectra/wave_luminosity.h5', 'r') as inf:
#    wave_luminosity = inf['wave_luminosity'][()]
#    freqs = inf['real_freqs'][()]
#    ells = inf['ells'][()]


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

full_out_dir = 'damping_theory_power'
if not os.path.exists(full_out_dir):
    os.makedirs(full_out_dir)


#Fit wave luminosity
#for ell in range(11):
#    if ell == 3:
#        wave_lum_ell = np.abs(wave_luminosity[:,ell])
#        shift_ind = np.argmax(wave_lum_ell)
#        shift_freq = freqs[shift_ind]
#        shift = (wave_lum_ell)[shift_ind]#freqs > 1e-2][0]
#
#        this_ell = 3
#        wave_luminosity_power = lambda f, ell: shift*(f/shift_freq)**(-10)*(ell/this_ell)**4
#        wave_luminosity_str = r'{:.2e}'.format(shift/shift_freq**(-10) / this_ell**4) + r'$f^{-10}\ell^4$'
#        break

freqs = np.logspace(-3, 0, 1000)
wave_luminosity_power = lambda f, ell: 7.5e-37 * f**(-10)*ell**(4)


        
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
            transfer_interp = interp1d(transfer_freq, transfer_func, bounds_error=False, fill_value=0)

        wave_flux_rcb = lambda f: wave_luminosity_power(f,ell)/(4*np.pi*1**2*rho_func(1))

        #wave_lum_ell should be wave_flux_ell? - see slack stuff around sept 29 2021
        kr2 = lambda f: (N2plateau/(2*np.pi*f)**2 - 1)*(ell*(ell+1))/1**2 #approximate, r=1
        ur2 = lambda f: np.sqrt(kr2(f)) * wave_flux_rcb(f) / N2plateau
        surface_s1_power = lambda f: transfer_interp(f)**2 * ur2(f)
#        surface_s1_power = lambda f: np.exp(-depthfunc(f))*transfer_interp(f)**2 * ur2(f) #TODO: fix optical depth?
        print(depthfunc(freqs), smooth_oms, smooth_depths)


        powers.append(surface_s1_power(freqs))
        plt.loglog(freqs, powers[-1], c='k')
    #    plt.legend(loc='best')
        plt.title('ell={}'.format(ell))
        plt.xlabel('freqs (sim units)')
        plt.ylabel(r'power')
        plt.ylim(1e-30, 1e-7)
        plt.xlim(3e-3, 1.4)
        fig.savefig('{}/s1_simulated_freq_spectrum_ell{}.png'.format(full_out_dir, ell), dpi=300, bbox_inches='tight')
        plt.clf()
    except:
        print("no eigenvalues for ell = {}".format(ell))
        

powers = np.array(powers)
print(powers.shape)
sum_ells_power = np.sum(powers, axis=0)
plt.loglog(freqs, sum_ells_power, c='k')
#    plt.legend(loc='best')
plt.title('summed over ells')
plt.xlabel('freqs (sim units)')
plt.ylabel(r'power')
plt.ylim(1e-30, 1e-7)
plt.xlim(3e-3, 1.4)
fig.savefig('{}/s1_simulated_freq_spectrum_summed_ells.png'.format(full_out_dir), dpi=300, bbox_inches='tight')


with h5py.File('{}/simulated_powers.h5'.format(full_out_dir), 'w') as f:
    f['powers'] = powers
    f['freqs'] = freqs
    f['sum_powers'] = sum_ells_power
