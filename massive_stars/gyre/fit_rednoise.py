"""
Calculate transfer function to get surface response of convective forcing.
Outputs a function which, when multiplied by sqrt(wave flux), gives you the surface response.
"""
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import h5py
from scipy import interpolate
from pathlib import Path
from scipy.interpolate import interp1d
from configparser import ConfigParser

from read_mist_models import EEP
import mesa_reader as mr
import d3_stars
from d3_stars.defaults import config
from d3_stars.simulations.parser import name_star
from d3_stars.simulations.star_builder import find_core_cz_radius
from d3_stars.simulations.evp_functions import calculate_refined_transfer
from matplotlib.patches import ConnectionPatch
from palettable.colorbrewer.qualitative import Dark2_5 as cmap
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'


#Calculate transfer functions
output_file = 'magnitude_spectra.h5'

def red_noise(nu, alpha0, nu_char, gamma=2):
    return alpha0/(1 + (nu/nu_char)**gamma)


star_dirs = ['3msol', '15msol', '40msol']
Lmax = [15, 15, 15]
alpha0 = [3.3e-3, 4.5e-2, 6e-1]
nu_char = [3.8e-1, 0.3, 1e-1]
gamma  = [3.7, 3.2, 5]
out_f = h5py.File(output_file, 'r')
freqs = out_f['frequencies'][()]*24*60*60


fig = plt.figure(figsize=(7.5, 2.5))
ax1 = fig.add_axes([0.050, 0.025, 0.275, 0.95])
ax2 = fig.add_axes([0.375, 0.025, 0.275, 0.95])
ax3 = fig.add_axes([0.700, 0.025, 0.275, 0.95])
axs = [ax1, ax2, ax3]

for i, sdir in enumerate(star_dirs):
    plt.axes(axs[i])
    magnitude_sum = out_f['{}_magnitude_sum'.format(sdir)][()]

    alpha_lbl = r'$\alpha_0 = $' + '{:.1e}'.format(alpha0[i])
    nu_lbl    = r'$\nu_{\rm char} = $' + '{:.2f}'.format(nu_char[i])
    gamma_lbl = r'$\gamma = $' + '{:.1f}'.format(gamma[i])
    label     = '{}\n{}\n{}'.format(alpha_lbl, nu_lbl, gamma_lbl)
    plt.loglog(freqs, magnitude_sum, label=sdir, c='k')
    plt.loglog(freqs, red_noise(freqs, alpha0[i], nu_char[i], gamma=gamma[i]), label=label, c='orange')
    if i == 0: #3
        plt.ylim(1e-5, 2e-2)
        plt.xlim(5e-2, 5e0)
    if i == 1: #15
        plt.ylim(1e-4, 3e-1)
        plt.xlim(3e-2, 2e0)
    if i == 2: #40
        plt.ylim(1e-3, 4e0)
        plt.xlim(2e-2, 2e0)
    axs[i].text(0.01, 0.99, label, ha='left', va='top', transform=axs[i].transAxes)

#    plt.legend()
    plt.xlabel('frequency (d$^{-1}$)')
    if i == 0:
        plt.ylabel(r'$\Delta m$ ($\mu$mag)')
fig.savefig('rednoise_fit.png', bbox_inches='tight', dpi=300)
fig.savefig('rednoise_fit.pdf', bbox_inches='tight')

out_f.close()
