"""
This script plots snapshots of the evolution of 2D slices from a 2D simulation in polar geometry.

The fields specified in 'fig_type' are plotted (temperature and enstrophy by default).
To plot a different set of fields, add a new fig type number, and expand the fig_type if-statement.

Usage:
    compare_wave_fluxes.py <root_dirs>... [options]

Options:
    --data_dir=<dir>                    Name of data handler directory [default: SH_transform_wave_shell_slices]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<num_files>               Total number of files to plot
    --dpi=<dpi>                         Image pixel density [default: 200]

    --col_inch=<in>                     Number of inches / column [default: 3]
    --row_inch=<in>                     Number of inches / row [default: 3]

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
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic
from dedalus.tools import logging
from dedalus.tools.parsing import split_equation
from scipy import sparse
from mpi4py import MPI
from scipy.interpolate import interp1d

from plotpal.file_reader import SingleFiletypePlotter as SFP
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)

from dedalus.tools.config import config

from power_spectrum_functions import clean_cfft, normalize_cfft_power

args = docopt(__doc__)
res = re.compile('(.*)\(r=(.*)\)')
resolution = re.compile('(.*)x(.*)')

# Read in master output directory
root_dirs    = args['<root_dirs>']
data_dir    = args['--data_dir']

# Read in additional plot arguments
start_file  = int(args['--start_file'])
n_files     = args['--n_files']
if n_files is not None: 
    n_files = int(n_files)

star_file = '../mesa_stars/nccs_40msol/ballShell_nccs_B63_S63_Re1e4.h5'
with h5py.File(star_file, 'r') as f:
    rB = f['rB'][()]
    rS = f['rS'][()]
    ρB = np.exp(f['ln_ρB'][()])
    ρS = np.exp(f['ln_ρS'][()])
    r = np.concatenate((rB.flatten(), rS.flatten()))
    ρ = np.concatenate((ρB.flatten(), ρS.flatten()))
    rho_func = interp1d(r,ρ)
    tau = f['tau'][()]/(60*60*24)
    r_outer = f['r_outer'][()]
    radius = r_outer * f['L'][()]
    #Entropy units are erg/K/g
    s_c = f['s_c'][()]
    N2plateau = f['N2plateau'][()] * (60*60*24)**2

# Create Plotter object, tell it which fields to plot
out_dir = 'SH_wave_flux_spectra'
plotters = []
full_out_dirs = []
L_res = []
for root_dir in root_dirs:
    for piece in root_dir.split('_'):
        if resolution.match(piece):
            L_res.append(int(piece.split('x')[0]))
            break
    full_out_dir = '{}/{}'.format(root_dir, out_dir)
    plotter = SFP(root_dir, file_dir=data_dir, fig_name=out_dir, start_file=start_file, n_files=n_files, distribution='single')
    with h5py.File(plotter.files[0], 'r') as f:
        fields = list(f.keys())
    fields.remove('time')
    fields.remove('ells')
    fields.remove('ms')
    radii = []
    for f in fields:
        if res.match(f):
            radius = float(f.split('r=')[-1].split(')')[0])
            if radius not in radii:
                radii.append(radius)
    plotters.append(plotter)
    full_out_dirs.append(full_out_dir)

fig = plt.figure()
freqs_for_dfdell = [0.2, 0.5, 1]
for f in freqs_for_dfdell:
    for i, full_out_dir in enumerate(full_out_dirs):
        with h5py.File('{}/transforms.h5'.format(full_out_dir), 'r') as rf:
            freqs = rf['real_freqs_inv_day'][()]
            ells = rf['ells'][()].flatten()
            print('plotting f = {}'.format(f))
            f_ind = np.argmin(np.abs(freqs - f))
            for radius in radii:
                if radius != 1.15: continue
                wave_flux = rf['wave_flux(r={})'.format(radius)][f_ind, :]
                plt.loglog(ells, ells*wave_flux, label='r={}, Lmax={}'.format(radius, L_res[i]))
                shift = (ells*wave_flux)[ells == 2]
            if i == 0:
                plt.loglog(ells, shift*(ells/2)**4, c='k', label=r'$\ell^4$')
            plt.legend(loc='best')
            plt.title('f = {} 1/day'.format(f))
            plt.xlabel(r'$\ell$')
            plt.ylabel(r'$\frac{\partial^2 F}{\partial\ln\ell}$')
            plt.ylim(1e-25, 1e-9)
            plt.xlim(1, ells.max())
    fig.savefig('./scratch/comparison_ell_spectrum_freq{}_invday.png'.format(f), dpi=300, bbox_inches='tight')
    plt.clf()
    
 
for ell in range(11):
    for i, full_out_dir in enumerate(full_out_dirs):
        if ell == 0: continue
        print('plotting ell = {}'.format(ell))
        with h5py.File('{}/transforms.h5'.format(full_out_dir), 'r') as rf:
            freqs = rf['real_freqs_inv_day'][()]
            for radius in radii:
                if radius != 1.15: continue
                wave_flux = rf['wave_flux(r={})'.format(radius)][:,ell]
                plt.loglog(freqs, freqs*wave_flux, label='r={}, Lmax={}'.format(radius, L_res[i]))
                shift = (freqs*wave_flux)[freqs > 0.1][0]
        if i == 0:
            plt.loglog(freqs, shift*10*(freqs/freqs[freqs > 0.1][0])**(-13/2), c='k', label=r'$f^{-13/2}$')
            plt.loglog(freqs, shift*10*(freqs/freqs[freqs > 0.1][0])**(-2), c='grey', label=r'$f^{-2}$')
        plt.legend(loc='best')
        plt.title('ell={}'.format(ell))
        plt.xlabel('freqs (1/day)')
        plt.ylabel(r'$\frac{\partial^2 F}{\partial \ln f}$')
        plt.ylim(1e-25, 1e-9)
    fig.savefig('./scratch/freq_spectrum_ell{}.png'.format(ell), dpi=300, bbox_inches='tight')
    plt.clf()
    
    

