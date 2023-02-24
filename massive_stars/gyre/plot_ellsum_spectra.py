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


star_dirs = ['3msol', '40msol', '15msol']
Lmax = [16, 20, 25]
out_f = h5py.File(output_file, 'r')
freqs = out_f['frequencies'][()]

for i, sdir in enumerate(star_dirs):
    magnitude_cube = out_f['{}_magnitude_cube'.format(sdir)][()]
    ell_list = np.arange(1, Lmax[i]+1)
#    if i == 2:
#        plt.figure()
#        plt.loglog(freqs, magnitude_cube[0,:])
#        plt.loglog(freqs, magnitude_cube[-1,:])
#        plt.show()




    fig = plt.figure(figsize=(7.5, 3))
    ax1 = fig.add_axes([0.00 , 0.00, 0.425, 0.80])#fig.add_subplot(1,2,1)
    ax2 = fig.add_axes([0.575, 0.00, 0.425, 0.80])#fig.add_subplot(1,2,2)
    cax = fig.add_axes([0.25, 0.95, 0.50, 0.05])
    cmap = mpl.cm.plasma
    norm = mpl.colors.Normalize(vmin=1, vmax=Lmax[i])
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    for j in range(Lmax[i]):
        sum_mag = np.sum(magnitude_cube[:Lmax[i]-j,:], axis=0)
        ax1.loglog(freqs, magnitude_cube[j,:], color=sm.to_rgba(j+1), lw=0.5)
        ax2.loglog(freqs, sum_mag, color=sm.to_rgba(Lmax[i]-j), lw=0.5)
    ax2.loglog(freqs, np.sum(magnitude_cube[:Lmax[i],:], axis=0), c='k', lw=0.25)

    for ax in [ax1, ax2]:
        ax.set_xlim(1e-7, 4e-4)
        ax.set_ylim(1e-6, 0.3)
        ax.set_xlabel('freq (Hz)')
    ax1.set_ylabel(r'$\Delta m_{\ell}\,(\mu\rm{mag})$')
    ax2.set_ylabel(r'$\sum_{i}^{\ell}\Delta m_{i}\,(\mu\rm{mag})$')
    cb = plt.colorbar(sm, cax=cax, orientation='horizontal')
    cb.set_label(r'$\ell$')
    plt.savefig('ellsums_{}.png'.format(sdir), bbox_inches='tight', dpi=300)
    plt.savefig('ellsums_{}.pdf'.format(sdir), bbox_inches='tight', dpi=300)
    plt.clf()

out_f.close()
