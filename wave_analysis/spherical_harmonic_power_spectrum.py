"""
This script plots snapshots of the evolution of 2D slices from a 2D simulation in polar geometry.

The fields specified in 'fig_type' are plotted (temperature and enstrophy by default).
To plot a different set of fields, add a new fig type number, and expand the fig_type if-statement.

Usage:
    spherical_harmonic_power_spectrum.py <root_dir> [options]

Options:
    --data_dir=<dir>                    Name of data handler directory [default: SH_transform_surface_shell_slices]
    --start_fig=<fig_start_num>         Number of first figure file [default: 1]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<num_files>               Total number of files to plot
    --dpi=<dpi>                         Image pixel density [default: 200]

    --col_inch=<in>                     Number of inches / column [default: 3]
    --row_inch=<in>                     Number of inches / row [default: 3]

    --radius=<r>                        Radius at which the SWSH basis lives [default: 2.59]

    --plot_only
    --field=<f>                         If specified, only transform this field
"""
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

from plotpal.file_reader import SingleFiletypePlotter as SFP
import matplotlib.pyplot as plt

import logging
logger = logging.getLogger(__name__)

from dedalus.tools.config import config

from power_spectrum_functions import clean_cfft, normalize_cfft_power

args = docopt(__doc__)

# Read in master output directory
root_dir    = args['<root_dir>']
data_dir    = args['--data_dir']
if root_dir is None:
    print('No dedalus output dir specified, exiting')
    import sys
    sys.exit()

# Read in additional plot arguments
start_fig   = int(args['--start_fig'])
start_file  = int(args['--start_file'])
n_files     = args['--n_files']
if n_files is not None: 
    n_files = int(n_files)

star_file = '../mesa_stars/nccs_40msol/ballShell_nccs_B63_S63_Re1e4.h5'
with h5py.File(star_file, 'r') as f:
    tau = f['tau'][()]/(60*60*24)
    r_outer = f['r_outer'][()]
    radius = r_outer * f['L'][()]
    #Entropy units are erg/K/g
    s_c = f['s_c'][()]
    N2plateau = f['N2plateau'][()] * (60*60*24)**2

# Create Plotter object, tell it which fields to plot
if 'SH_transform' in data_dir:
    out_dir = data_dir.replace('SH_transform', 'SH_power')
else:
    out_dir = 'SH_power_spectra'
full_out_dir = '{}/{}'.format(root_dir, out_dir)
plotter = SFP(root_dir, file_dir=data_dir, fig_name=out_dir, start_file=start_file, n_files=n_files, distribution='single')
if args['--field'] is None:
    with h5py.File(plotter.files[0], 'r') as f:
        fields = list(f['tasks'].keys())
else:
    fields = [args['--field'],]


if not args['--plot_only']:
    times = []
    print('getting times...')
    while plotter.files_remain([], fields):
        print('reading file {}...'.format(plotter.current_filenum+1))
        file_name = plotter.files[plotter.current_filenum]
        with h5py.File('{}'.format(file_name), 'r') as f:
            if plotter.current_filenum == 0:
                ells = f['ells'][()]
                ms = f['ms'][()]
            times.append(f['time'][()])
        plotter.current_filenum += 1
    times = np.concatenate(times)

    with h5py.File('{}/power_spectra.h5'.format(full_out_dir), 'w') as out_f:
        out_f['ells']  = ells
        for f in fields:
            print('filling datacube...')
            writes = 0
            while plotter.files_remain([], fields):
                first = plotter.current_filenum == 0
                print('reading file {}...'.format(plotter.current_filenum+1))
                file_name = plotter.files[plotter.current_filenum]
                with h5py.File('{}'.format(file_name), 'r') as in_f:
                    task = in_f['tasks/'+f][()]
                    shape = list(task.shape)
                    if first:
                        data_cube = np.zeros([times.shape[0],]+shape[1:], dtype=np.complex128)
                    this_file_writes = len(in_f['time'][()])
                    data_cube[writes:writes+this_file_writes,:] = in_f['tasks/'+f][:,:,:]
                    writes += this_file_writes
                plotter.current_filenum += 1

            if ells.shape[1] == data_cube.shape[1]:
                vector = False
            else:
                vector = True

            print('taking transform')
            transform = np.zeros(data_cube.shape, dtype=np.complex128)
            for ell in ells.flatten():
                print('taking transforms {}/{}'.format(ell+1, ells.flatten().shape[0]))
                for m in ms.flatten():
                    if m > ell: continue
                    if not vector:
                        freqs, transform[:,ell,m] = clean_cfft(times, data_cube[:,ell,m])
                    else:
                        freqs, transform[:,:,ell,m] = clean_cfft(times, data_cube[:,:,ell,m])
            transform -= np.mean(transform, axis=0) #remove the temporal average; don't care about it.
            del data_cube
            gc.collect()
            freqs0 = freqs
            freqs, full_power = normalize_cfft_power(freqs0, transform)
            del transform
            gc.collect()

            power_per_ell = np.sum(full_power, axis=2) #sum over m's
            del full_power
            gc.collect()

            out_f['tasks/' + f + '_power_per_ell'] = power_per_ell
            if f == fields[0]:
                out_f['freqs'] = freqs 
                out_f['freqs_inv_day'] = freqs/tau

powers_per_ell = OrderedDict()
with h5py.File('{}/power_spectra.h5'.format(full_out_dir), 'r') as out_f:
    for f in fields:
        powers_per_ell[f] = out_f['tasks/'+f+'_power_per_ell'][()]
    ells = out_f['ells'][()]
    freqs = out_f['freqs_inv_day'][()]


good = freqs >= 0
min_freq = 1e-1
max_freq = freqs.max()
for k, powspec in powers_per_ell.items():
    good_axis = np.arange(len(powspec.shape))[np.array(powspec.shape) == len(ells.flatten())][0]
    print(good_axis, powspec.shape, len(ells.flatten()))
    sum_power = np.sum(powspec, axis=good_axis).squeeze()
        
    ymin = sum_power[(freqs > 5e-2)*(freqs < max_freq)][-1].min()/2
    ymax = sum_power[(freqs > 5e-2)*(freqs <= max_freq)].max()*2

    plt.figure()
    if len(sum_power.shape) > 1:
        for i in range(sum_power.shape[1]):
            plt.plot(freqs[good], sum_power[good, i], c = 'k', label=r'axis {}, sum over $\ell$ values'.format(i))
    else:
        plt.plot(freqs[good], sum_power[good], c = 'k', label=r'sum over $\ell$ values')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'Power ({})'.format(k))
    plt.xlabel(r'Frequency (day$^{-1}$)')
    plt.axvline(np.sqrt(N2plateau)/(2*np.pi), c='k')
    plt.xlim(min_freq, max_freq)
    plt.ylim(ymin, ymax)
    plt.legend(loc='best')
    k_out = k.replace('(', '_').replace(')', '_').replace('=', '')

    plt.savefig('{}/{}_summed_power.png'.format(full_out_dir, k_out), dpi=600)

    if k == 'uB(r=0.5)':
        scalar_plotter = SFP(root_dir, file_dir='scalars', fig_name=out_dir, start_file=1, n_files=np.inf, distribution='single')
        with h5py.File(scalar_plotter.files[0], 'r') as f:
            re_ball = f['tasks']['Re_avg_ball'][()]
            re_ball_avg = np.mean(re_ball.flatten()[int(len(re_ball.flatten())/2):])
            re_ball_avg *= (1.1/1.0)**3 #r_ball/r_cz cubed
            Re_input = float(root_dir.split('Re')[-1].split('_')[0])
            u_ball_avg = re_ball_avg / Re_input

        grid_dir = data_dir.replace('SH_transform_', '')
        start_file = int(plotter.files[0].split('.h5')[0].split('_s')[-1])
        n_files = len(plotter.files)
        grid_plotter = SFP(root_dir, file_dir=grid_dir, fig_name=out_dir, start_file=start_file, n_files=n_files, distribution='single')
        dealias = 1
        Lmax = powspec.squeeze().shape[-1] - 1
        c = coords.SphericalCoordinates('φ', 'θ', 'r')
        d = distributor.Distributor((c,), mesh=None, comm=MPI.COMM_SELF)
        b = basis.SphericalShellBasis(c, (2*(Lmax+2), Lmax+1, 1), radii=(0.5-1e-6, 0.5), dtype=np.float64)
        φ, θ, r = b.local_grids((dealias, dealias, dealias))
        weight_φ = np.gradient(φ.flatten()).reshape(φ.shape)
        weight_θ = b.local_colatitude_weights(dealias)
        weight = weight_θ * weight_φ
        volume = np.sum(weight)

        time_weight = np.expand_dims(weight, axis=0)

        uB_vals = []
        while grid_plotter.files_remain([], [k,]):
            bases, tasks, write_num, sim_time = grid_plotter.read_next_file()
            uB = tasks['uB(r=0.5)']
            uB_mag = np.sqrt(np.sum(uB**2, axis=1))
            uB_val = np.sum(np.sum(uB_mag*time_weight, axis=2), axis=1).squeeze()/volume
            for v in uB_val:
                uB_vals.append(v)

        avg_u_ball = np.mean(uB_vals) 
        avg_u_ball_perday1 = avg_u_ball / tau
        avg_u_ball_perday2 = u_ball_avg / tau

            
        plt.figure()
        KE_v_f = np.sum(np.sum(powspec.squeeze(), axis=2), axis=1)
#        plt.loglog(freqs[freqs > 0], (KE_v_f)[freqs > 0])
        plt.loglog(freqs[freqs > 0], (freqs*KE_v_f)[freqs > 0])
#        plt.loglog(freqs[freqs > 0], freqs[freqs > 0]**(-5/3)/1.2e5, c='k')
        plt.axvline(avg_u_ball_perday1, c='k')
        plt.axvline(avg_u_ball_perday2, c='grey')
#        plt.ylabel(r'$\frac{\partial (KE)}{\partial f}$ (cz)')
        plt.ylabel(r'$f\,\, \frac{\partial (KE)}{\partial f}$ (cz)')
        plt.xlabel(r'Frequency (day$^{-1}$)')
        plt.ylim(1e-6, 1e-4)
        plt.savefig('{}/fke_spec.png'.format(full_out_dir), dpi=600)

        plt.figure()
        df = np.gradient(freqs)[:,None]
        KE_v_ell = np.sum(np.sum(powspec.squeeze(), axis=1), axis=0)
        plt.loglog(ells.flatten()[ells.flatten() > 0], (ells.flatten()*KE_v_ell)[ells.flatten() > 0])
        plt.loglog(ells.flatten()[ells.flatten() > 0], ells.flatten()[ells.flatten() > 0]**(-2/3)/3, c='k')
        plt.ylabel(r'$\ell\,\, \frac{\partial (KE)}{\partial ell}$ (cz)')
        plt.xlabel(r'$\ell$')
        plt.ylim(1e-3, 1e0)
        plt.savefig('{}/ellke_spec.png'.format(full_out_dir), dpi=600)
        

        

