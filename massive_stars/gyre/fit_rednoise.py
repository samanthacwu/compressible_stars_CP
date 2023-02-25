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


star_dirs = ['3msol', '40msol', '15msol']
Lmax = [16, 16, 16]
alpha0 = [3.5e-3, 9e-2, 6e-2]
nu_char = [1.8, 0.55, 1.1]
gamma  = [6, 4, 4.5]
out_f = h5py.File(output_file, 'r')
freqs = out_f['frequencies'][()]*24*60*60

for i, sdir in enumerate(star_dirs):
    magnitude_sum = out_f['{}_magnitude_sum'.format(sdir)][()]

    alpha_lbl = r'$\alpha_0 = $' + '{:.1e}'.format(alpha0[i])
    nu_lbl    = r'$\nu_{\rm char} = $' + '{:.2f}'.format(nu_char[i])
    gamma_lbl = r'$\gamma = $' + '{:.1f}'.format(gamma[i])
    label     = '{}, {}, {}'.format(alpha_lbl, nu_lbl, gamma_lbl)
    fig = plt.figure(figsize=(5, 3))
    plt.loglog(freqs, magnitude_sum, label=sdir)
    plt.loglog(freqs, red_noise(freqs, alpha0[i], nu_char[i], gamma=gamma[i]), label=label)
    plt.xlim(1e-1, 1e1)
    plt.ylim(1e-3, 1e-1)
    plt.legend()
    plt.xlabel('freq (1/day)')
    plt.ylabel(r'$\Delta m$ ($\mu$mag)')
    fig.savefig('rednoise_fit_{}.png'.format(sdir))

out_f.close()
