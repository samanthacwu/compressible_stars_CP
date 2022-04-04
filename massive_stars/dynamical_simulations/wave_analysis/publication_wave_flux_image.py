"""
This script plots snapshots of the evolution of 2D slices from a 2D simulation in polar geometry.

The fields specified in 'fig_type' are plotted (temperature and enstrophy by default).
To plot a different set of fields, add a new fig type number, and expand the fig_type if-statement.
"""
import re
import gc
import os
import time
import sys
from collections import OrderedDict
from pathlib import Path
import palettable

import h5py
import numpy as np
from configparser import ConfigParser
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic
from dedalus.tools import logging
from dedalus.tools.parsing import split_equation
from scipy import sparse
from mpi4py import MPI
from scipy.interpolate import interp1d

from plotpal.file_reader import SingleTypeReader as SR
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)

from dedalus.tools.config import config

from power_spectrum_functions import clean_cfft, normalize_cfft_power

task_str = re.compile('(.*)\(r=(.*)\)')
resolution = re.compile('(.*)x(.*)x(.*)')

# Read in master output directory
root_dirs = [
        "../ballShell_AN_sponge_Re1e3_128x64x96+96",
        "../ballShell_AN_sponge_Re1e4_512x256x256+128",
        "../ballShell_AN_sponge_Re2e3_192x96x96+96_safety0.35",
#        "../ballShell_AN_sponge_Re2e4_1024x512x512+256",
        "../ballShell_AN_sponge_Re4e3_256x128x128+96"
        ]

radii = [1.05, 1.15, 1.6]

# Read in additional plot arguments
start_file  = 1
n_files     = np.inf

star_file = '../ncc_creation/nccs_40msol/ballShell_nccs_B96_S96_Re1e3_de1.5.h5'
with h5py.File(star_file, 'r') as f:
    rB = f['rB'][()]
    rS = f['rS'][()]
    ρB = np.exp(f['ln_ρB'][()])
    ρS = np.exp(f['ln_ρS'][()])
    r = np.concatenate((rB.flatten(), rS.flatten()))
    ρ = np.concatenate((ρB.flatten(), ρS.flatten()))
    rho_func = interp1d(r,ρ)
    tau_sec= f['tau'][()]
    tau = tau_sec/(60*60*24)
    r_outer = f['r_outer'][()]
    L_sim = f['L'][()] #cm
    radius = r_outer * L_sim
    #Entropy units are erg/K/g
    s_c = f['s_c'][()]
    N2_plateau_sec = f['N2plateau'][()]
    N2_plateau = N2_plateau_sec * (60*60*24)**2
    N2max_shell = f['N2max_shell'][()]
    r_rcb = 1
    ρ_rcb = rho_func(r_rcb)
    N2_plateau_sim = N2_plateau_sec * tau_sec**2

# Create Plotter object, tell it which fields to plot
out_dir = 'SH_wave_flux_spectra'
full_out_dirs = []
res = []
Re = []
wave_fluxes = []
wave_flux_radii = []
ells_list = []
freqs = []
rotation = []
for root_dir in root_dirs:
    print('reading {}'.format(root_dir))
    for piece in root_dir.split('_'):
        if resolution.match(piece):
            res.append(piece)
            break
        if 'Re' in piece:
            Re.append(float(piece.split('Re')[-1]))
        if 'rotation' in piece:
                rotation.append(piece.split('rotation')[-1])
    if len(rotation) < len(Re):
            rotation.append(None)
    full_out_dir = '{}/{}'.format(root_dir, out_dir)
    full_out_dirs.append(full_out_dir)
    #Get spectrum = rho*(real(ur*conj(p)))
    with h5py.File('{}/transforms.h5'.format(full_out_dir), 'r') as wf:
        freqs.append(wf['real_freqs'][()])
        ells_list.append(wf['ells'][:,:,0,0])
        wave_fluxes.append([])
        for i, radius in enumerate(radii):
            wave_flux_radii.append(radius)
            wave_fluxes[-1].append(wf['wave_flux(r={})'.format(radius)][()])

    
f_norm_pow = -17/2
ell_norm_pow = 4
ell = 5
freq = 5

pub_fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(7.5,4))
for i in range(2):
    axs[i][1].set_prop_cycle('color', palettable.colorbrewer.qualitative.Dark2_8.mpl_colors)
    axs[i][0].set_prop_cycle('color', palettable.colorbrewer.qualitative.Set1_8.mpl_colors)
    axs[0][i].axvline(freq, c='k')
    axs[1][i].axvline(ell, c='k')

    axs[0][i].set_ylim(1e-22, 1e-8)
    axs[1][i].set_ylim(1e-13, 1e-4)

print(len(ells_list), len(full_out_dirs), len(wave_fluxes))

for i, full_out_dir in enumerate(full_out_dirs):
    if Re[i] == np.max(Re):
        ells = ells_list[i]
        these_freqs = freqs[i][:,None]
        flux = wave_fluxes[i][0]
        detrended = flux/(ells**ell_norm_pow * these_freqs**f_norm_pow)
        window = np.abs(detrended[(ells > 0)*(ells <= 10)*(these_freqs > 2)*(these_freqs <= 8)])
        good = np.isfinite(np.log10(window))
        A0 = 10**(np.mean(np.log10(window[good])))
        f0_sec = 1/tau_sec
        A0_ur2_sim = 2*A0/(r_rcb*ρ_rcb*np.sqrt(N2_plateau_sim))
        A0_ur2_sec = A0_ur2_sim * f0_sec**(-f_norm_pow) * L_sim**2 / tau_sec**2 #cm^2/s^2
        for j, radius in enumerate(radii):
            wave_flux = wave_fluxes[i][j][:,ell]
            axs[0][0].loglog(freqs[i], np.abs(wave_flux), label='Re={}, res={}, rot={}'.format(Re[i], res[i], rotation[i]))

            f_ind = np.argmin(np.abs(freqs[i] - freq))
            wave_flux = wave_fluxes[i][j][f_ind, :]
            ells = ells_list[i].flatten()
            axs[1][0].loglog(ells, wave_flux/freq**(f_norm_pow), label='Re={}, res={}, rot={}'.format(Re[i], res[i], rotation[i]))
    wave_flux = wave_fluxes[i][0][:,ell]
    axs[0][1].loglog(freqs[i], np.abs(wave_flux), label='Re={}, res={}, rot={}'.format(Re[i], res[i], rotation[i]))

    f_ind = np.argmin(np.abs(freqs[i] - freq))
    wave_flux = wave_fluxes[i][0][f_ind, :]
    ells = ells_list[i].flatten()
    axs[1][1].loglog(ells, wave_flux/freq**(f_norm_pow), label='Re={}, res={}, rot={}'.format(Re[i], res[i], rotation[i]))

    
pub_fig.savefig('./scratch/pubfig.png', dpi=300, bbox_inches='tight')
