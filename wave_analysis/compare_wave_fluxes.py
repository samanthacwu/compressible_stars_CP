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

from plotpal.file_reader import SingleTypeReader as SR
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)

from dedalus.tools.config import config

from power_spectrum_functions import clean_cfft, normalize_cfft_power

args = docopt(__doc__)
task_str = re.compile('(.*)\(r=(.*)\)')
resolution = re.compile('(.*)x(.*)x(.*)')

# Read in master output directory
root_dirs    = args['<root_dirs>']
data_dir    = args['--data_dir']

# Read in additional plot arguments
start_file  = int(args['--start_file'])
n_files     = args['--n_files']
if n_files is not None: 
    n_files = int(n_files)

star_file = '../mesa_stars/nccs_40msol/ballShell_nccs_B96_S96_Re1e3_de1.5.h5'
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
    with h5py.File('{}/wave_flux.h5'.format(full_out_dir), 'r') as rf:
        freqs.append(rf['real_freqs'][()])
#        freqs_inv_day.append(rf['real_freqs_inv_day'][()])
        ells_list.append(rf['ells'][:,:,0])
        wave_fluxes.append(rf['wave_flux'][()])

u_r = []
for i in range(len(ells_list)):
    u_r.append(np.sqrt(2*np.sqrt(ells_list[i]*(ells_list[i]+1)) * wave_fluxes[i] / (1 * ρ_rcb * np.sqrt(N2_plateau))))
print('rho_rcb: {}, N_plateau: {}'.format(ρ_rcb, np.sqrt(N2_plateau)))


f_norm_pow = -17/2
ell_norm_pow = 4

for i, full_out_dir in enumerate(full_out_dirs):
    if Re[i] == np.max(Re):
        ells = ells_list[i]
        freq = freqs[i][:,None]
        flux = wave_fluxes[i]
        detrended = flux/(ells**ell_norm_pow * freq**f_norm_pow)
        window = np.abs(detrended[(ells > 0)*(ells <= 10)*(freq > 2)*(freq <= 8)])
        good = np.isfinite(np.log10(window))
        A0 = 10**(np.mean(np.log10(window[good])))
        f0_sec = 1/tau_sec
        A0_ur2_sim = 2*A0/(r_rcb*ρ_rcb*np.sqrt(N2_plateau_sim))
        A0_ur2_sec = A0_ur2_sim * f0_sec**(-f_norm_pow) * L_sim**2 / tau_sec**2 #cm^2/s^2
print('best fit: {:.3e} f^{:.1f} ell^{:.1f}'.format(A0, f_norm_pow, ell_norm_pow))
print('best fit ur^2 (cm^2/s^2; f in 1/s): {:.3e} f^{:.1f} ell^{:.1f} sqrt(1 + 1/ell)'.format(A0_ur2_sec, f_norm_pow, ell_norm_pow+1))


fig = plt.figure()
freqs_for_dfdell = [2, 5, 8]
for f in freqs_for_dfdell:
    print('plotting f = {}'.format(f))
    for i, full_out_dir in enumerate(full_out_dirs):
        f_ind = np.argmin(np.abs(freqs[i] - f))
        wave_flux = wave_fluxes[i][f_ind, :]
        ells = ells_list[i].flatten()
#        plt.loglog(ells, ells*wave_flux, label='Re={}, res={}, rot={}'.format(Re[i], res[i], rotation[i]))
        plt.loglog(ells, wave_flux/ells**4/f**(f_norm_pow), label='Re={}, res={}, rot={}'.format(Re[i], res[i], rotation[i]))
        shift = (wave_flux)[ells == 2]
        plt.title('f = {} sim units'.format(f))
        plt.xlabel(r'$\ell$')
#        plt.ylabel(r'$\frac{\partial^2 F}{\partial\ln\ell}$')
        plt.ylabel(r'$f^{{{:.1f}}}$'.format(-f_norm_pow) + r'$\ell^{-4}\,F(\omega,\ell)|_\omega$')
        plt.ylim(1e-13, 1e-10)
#        plt.ylim(1e-25, 1e-9)
        plt.xlim(1, ells.max())
    fit_label = r'${{{:.3e}}} f^{{{:.1f}}} \ell^{{{:.1f}}}$'.format(A0, f_norm_pow, ell_norm_pow)
    plt.axhline(A0, label=fit_label, c='k')
    plt.legend(loc='best')
    fig.savefig('./scratch/comparison_ell_spectrum_freq{}.png'.format(f), dpi=300, bbox_inches='tight')
    plt.clf()
    
 
for ell in range(11):
    print('plotting ell = {}'.format(ell))
    for i, full_out_dir in enumerate(full_out_dirs):
        if ell == 0: continue
        wave_flux = wave_fluxes[i][:,ell]
        freq = freqs[i]
        plot = plt.loglog(freq, np.abs(wave_flux), label='Re={}, res={}, rot={}'.format(Re[i], res[i], rotation[i]))
#        if Re[i] == np.max(Re):
##            for p in [-15/2]:
##                shift = (wave_flux)[freq > 2][0] / 2**(p)
##                plt.loglog(freq, shift*freq**(p), c='k', label='({:.3e})'.format(shift) + r'$f^{{{:.1f}}}$'.format(p))
#
#            for fmin, fmax in zip((0.8, 2), (2, 10)):
#                try:
#                    good = (freq > fmin)*(freq <= fmax)
#                    x = np.log10(freq[good])
#                    y = np.log10(np.abs(wave_flux)[good])
#                    slope, intercept = np.polyfit(x, y, 1)
#                    print('best fit {} < f <= {}: {:.3e} f ^ ( {:.1f} )'.format(fmin, fmax, 10**intercept, slope))
#                    if fmin == 2 and fmax == 10:
#                        line = 10**(intercept)*freq**slope
#                        plt.loglog(freq, line, c='k', label='({:.3e})'.format(10**intercept) + r'$f^{{{:.1f}}}$'.format(slope))
#                except:
#                    print('no convergence for {} < f <= {}'.format(fmin, fmax))
#        plt.loglog(freq, freq*wave_flux/(freq**(-13/2)), label='Re={}, res={}, rot={}'.format(Re[i], res[i], rotation[i]))
        if rotation[i] is not None:
                Ω = tau * 2*np.pi / float(rotation[i])
                plt.axvline(Ω / (2*np.pi), color=plot[-1]._color, ls='--')
                plt.axvline(tau_sec*np.sqrt(N2max_shell) / (2*np.pi), color=plot[-1]._color)
        plt.title('ell={}'.format(ell))
        plt.xlabel('freq (sim units)')
        plt.ylabel(r'$F(\omega,\ell)|_\ell$')
#        plt.ylabel(r'$f^{-13/2}\frac{\partial^2 F}{\partial \ln f}$')
        plt.ylim(1e-20, 1e-7)
    fit_label = r'${{{:.3e}}} f^{{{:.1f}}} \ell^{{{:.1f}}}$'.format(A0, f_norm_pow, ell_norm_pow)
    plt.loglog(freq, A0*freq**(f_norm_pow)*ell**(ell_norm_pow), c='k', label=fit_label)
    plt.legend(loc='lower left', fontsize=8)
    fig.savefig('./scratch/freq_spectrum_ell{}.png'.format(ell), dpi=300, bbox_inches='tight')
    plt.clf()
    
    

