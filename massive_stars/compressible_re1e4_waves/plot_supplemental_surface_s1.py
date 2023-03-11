import os
import numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import h5py
from scipy import interpolate
from pathlib import Path
from scipy.interpolate import interp1d
from configparser import ConfigParser

import mesa_reader as mr
import d3_stars
from d3_stars.defaults import config
from d3_stars.simulations.parser import name_star
from d3_stars.simulations.star_builder import find_core_cz_radius
from d3_stars.simulations.evp_functions import calculate_refined_transfer
from matplotlib.patches import ConnectionPatch
from palettable.colorbrewer.qualitative import Dark2_5
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'


output_file = 'power_spectra/power_spectra.h5'
with h5py.File('star/star_256+192+64_bounds0-0.93R_Re1.00e+04_de1.5_cutoff1.0e-10.h5', 'r') as starf:
    tau_nd = starf['tau_nd'][()]
    s_nd = starf['s_nd'][()]

with h5py.File(output_file, 'r') as out_f:
    freqs = out_f['freqs'][()] / tau_nd * (24 * 60 * 60) #invday
    ells = out_f['ells'][()].ravel()
    s1 = np.sqrt(out_f['shell(s1_S2,r=R)'][0,:,:])*s_nd

    fig = plt.figure(figsize=(7.5, 3)) 
    ax1 = fig.add_axes([0.00 , 0.00, 0.425, 0.80])#fig.add_subplot(1,2,1)
    ax2 = fig.add_axes([0.575, 0.00, 0.425, 0.80])#fig.add_subplot(1,2,2)
    cmap = mpl.cm.plasma
    Lmax = 3
    norm = mpl.colors.Normalize(vmin=1, vmax=Lmax)
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.colors.ListedColormap(Dark2_5.mpl_colors[:Lmax]))

    for j in range(1,Lmax+1):
        ax1.loglog(freqs, s1[:,ells==j], color=sm.to_rgba(j), lw=0.75, zorder=Lmax-j)

    ax2.loglog(freqs, np.sum(s1[:,1:], axis=1), lw=1, c='k', label=r'$L_{\rm max} = 255$')
    ax2.loglog(freqs, np.sum(s1[:,(ells > 0)*(ells <= Lmax)], axis=1), lw=0.5, c='orange', label=r'$L_{\rm max} = 3$')
    ax2.legend()

    ax1.text(0.1, 1, r'$\ell = 1$', c=sm.to_rgba(1), ha='center', va='center')
    ax1.text(0.3, 30, r'$\ell = 2$', c=sm.to_rgba(2), ha='center', va='center')
    ax1.text(1.5, 3, r'$\ell = 3$', c=sm.to_rgba(3), ha='center', va='center')

    for ax in [ax1, ax2]:
        ax.set_xlim(7e-2, 1e1)
        ax.set_ylim(1e-5, 1e2)
        ax.set_xlabel('frequency (d$^{-1}$)')
    ax1.set_ylabel(r'$|s_1|_{\ell}$')
    ax2.set_ylabel(r'$\sum_{\ell=1}^{L_{\rm max}}|s_1|_{\ell}$')
    plt.savefig('wavepropagation_power.png', bbox_inches='tight', dpi=300)
    plt.savefig('wavepropagation_power.pdf', bbox_inches='tight', dpi=300)
    plt.clf()
