"""
This script plots snapshots of the evolution of 2D slices from a 2D simulation in polar geometry.

The fields specified in 'fig_type' are plotted (temperature and enstrophy by default).
To plot a different set of fields, add a new fig type number, and expand the fig_type if-statement.

Usage:
    post_ivp_SH_power_spectrum.py [options]

Options:
    --data_dir=<dir>                    Name of data handler directory [default: SH_transform_shells]
    --start_fig=<fig_start_num>         Number of first figure file [default: 1]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<num_files>               Total number of files to plot
    --dpi=<dpi>                         Image pixel density [default: 200]

    --col_inch=<in>                     Number of inches / column [default: 3]
    --row_inch=<in>                     Number of inches / row [default: 3]

    --radius=<r>                        Radius at which the SWSH basis lives [default: 2.59]

    --no_ft
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
from scipy.interpolate import interp1d

from plotpal.file_reader import SingleTypeReader as SR

import logging
logger = logging.getLogger(__name__)

from dedalus.tools.config import config

from d3_stars.simulations.parser import parse_std_config
from d3_stars.post.power_spectrum_functions import clean_cfft, normalize_cfft_power

args = docopt(__doc__)

# Read in master output directory
root_dir    = './'
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

config, raw_config, star_dir, star_file = parse_std_config('controls.cfg')
with h5py.File(star_file, 'r') as f:
    rB = f['r_B'][()]
    rS1 = f['r_S1'][()]
    rS2 = f['r_S2'][()]
    rhoB = np.exp(f['ln_rho_B'][()])
    rhoS1 = np.exp(f['ln_rho_S1'][()])
    rhoS2 = np.exp(f['ln_rho_S2'][()])
    r = np.concatenate((rB.flatten(), rS1.flatten(), rS2.flatten()))
    rho = np.concatenate((rhoB.flatten(), rhoS1.flatten(), rhoS2.flatten()))
    rho_func = interp1d(r,rho)
    tau = f['tau_nd'][()]/(60*60*24)
    r_outer = f['r_outer'][()]
    radius = r_outer * f['L_nd'][()]
    #Entropy units are erg/K/g
    s_c = f['s_nd'][()]
    N2plateau = f['N2plateau'][()] * (60*60*24)**2

    N2plateau_sim = f['N2plateau'][()] * tau**2
    N2plateau = f['N2plateau'][()] * (60*60*24)**2

# Create Plotter object, tell it which fields to plot
if 'SH_transform' in data_dir:
    out_dir = data_dir.replace('SH_transform', 'SH_temporal_power')
else:
    out_dir = 'SH_temporal_power_spectra'
full_out_dir = '{}/{}'.format(root_dir, out_dir)
reader = SR(root_dir, data_dir, out_dir, start_file=start_file, n_files=n_files, distribution='single')
if args['--field'] is None:
    with h5py.File(reader.files[0], 'r') as f:
        fields = list(f['tasks'].keys())
else:
    fields = [args['--field'],]



temporal_chunks = 10


if not args['--no_ft']:
    times = []
    print('getting times...')
    ells = ms = None
    while reader.writes_remain():
        dsets, ni = reader.get_dsets([])
        times.append(reader.current_file_handle['time'][ni])
        if ells is None and ms is None:
            ells = reader.current_file_handle['ells'][()]
            ms = reader.current_file_handle['ms'][()]

    times = np.array(times)
    N_time = len(times) // temporal_chunks
    time_chunks = [times[N_time*i:N_time*(i+1)] for i in range(temporal_chunks)]

    with h5py.File('{}/power_spectra.h5'.format(full_out_dir), 'w') as wf:
        wf['ells']  = ells
        wf['times']  = times
        wf['time_chunks'] = np.array(time_chunks)
    with h5py.File('{}/transforms.h5'.format(full_out_dir), 'w') as wf:
        wf['ells']  = ells
        wf['time_chunks'] = np.array(time_chunks)

    #TODO: only load in one ell and m at a time, that'll save memory.
    data_cubes = dict()
    for i, f in enumerate(fields):
        print('reading field {}'.format(f))

        print('filling datacube...')
        writes = 0
        first = True
        while reader.writes_remain():
            print('reading file {}...'.format(reader.current_file_number+1))
            dsets, ni = reader.get_dsets([f,])
            if first:
                shape = list(dsets[f][()].shape)[:-1]
                shape[0] = times.shape[0]
                shape[-1] = ms.shape[2]
                shape[-2] = ells.shape[1]
                data_cube = np.zeros(shape, dtype=np.complex128)
                first = False
            data_cube[writes,:] = dsets[f][ni,:].squeeze()
            writes += 1
        print(data_cube.shape)
        data_cubes[f] = data_cube

    transform_arrs = dict()
    freq_arrs = dict()
    power_transform_arrs = dict()
    power_freq_arrs = dict()
    power_per_ell_arrs = dict()
    for i, f in enumerate(fields):
        for dt in [transform_arrs, freq_arrs, power_transform_arrs, power_freq_arrs, power_per_ell_arrs]:
            dt[f] = []
        data_cube = data_cubes[f]
        for tchunk in time_chunks:
            d_cube = data_cube[(times >= tchunk.min())*(times <= tchunk.max()),:]
            print('taking transform of time chunk starting {:.3e} / ending {:.3e}'.format(tchunk.min(), tchunk.max()))
            transform = np.zeros(d_cube.shape, dtype=np.complex128)
            for ell in range(d_cube.shape[1]):
                print('taking transforms on {}: {}/{}'.format(f, ell+1, d_cube.shape[1]))
                for m in range(d_cube.shape[2]):
                    if m > ell: continue
                    if len(d_cube.shape) == 4:
                        for v in range(d_cube.shape[1]):
                            freqs, transform[:,v,ell,m] = clean_cfft(tchunk, d_cube[:,v,ell,m])
                    else:
                        freqs, transform[:,ell,m] = clean_cfft(tchunk, d_cube[:,ell,m])
    #        transform -= np.mean(transform, axis=0) #remove the temporal average; don't care about it.
            transform_arrs[f].append(transform)
            freq_arrs[f].append(np.copy(freqs))
            gc.collect()
            if len(d_cube.shape) == 4:
                full_power = []
                for v in range(d_cube.shape[1]):
                    norm_freqs, this_power = normalize_cfft_power(freqs, transform[:,v,:])
                    full_power.append(this_power)
                full_power = np.array(full_power)
            else:
                norm_freqs, full_power = normalize_cfft_power(freqs, transform)
            power_per_ell = np.sum(full_power, axis=-1) #sum over m's
            power_transform_arrs[f].append(full_power)
            power_freq_arrs[f].append(norm_freqs)
            power_per_ell_arrs[f].append(power_per_ell)
        with h5py.File('{}/power_spectra.h5'.format(full_out_dir), 'r+') as wf:
            wf['{}_power_per_ell'.format(f)] = np.array(power_per_ell_arrs[f])
            wf['freqs'] = np.array(power_freq_arrs[f])
            wf['freqs_inv_day'] = np.array(power_freq_arrs[f])/tau
        with h5py.File('{}/transforms.h5'.format(full_out_dir), 'r+') as wf:
            wf['{}_cft'.format(f)] = np.array(transform_arrs[f])
            if i == 0:
                wf['freqs'] = np.array(freq_arrs[f])
                wf['freqs_inv_day'] = np.array(freq_arrs[f])/tau


powers_per_ell = OrderedDict()
with h5py.File('{}/power_spectra.h5'.format(full_out_dir), 'r') as out_f:
    for f in fields:
        powers_per_ell[f] = out_f['{}_power_per_ell'.format(f)][()]
    ells = out_f['ells'][()]
    freqs_chunks = out_f['freqs'][()]
    time_chunks = out_f['time_chunks'][()]

import matplotlib as mpl
import matplotlib.pyplot as plt

norm = mpl.colors.Normalize(vmin=time_chunks.min(), vmax=time_chunks.max())
sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm)
sm.set_array([])

fig = plt.figure()
ax = fig.add_axes([0, 0, 1, 0.9])
cax = fig.add_axes([0.05, 0.925, 0.9, 0.05])

cbar = mpl.colorbar.ColorbarBase(cax, cmap=mpl.cm.get_cmap('plasma'), norm=norm, orientation='horizontal', ticklocation='top')
cbar.set_label('sim time')

min_freq = 3e-3
max_freq = freqs_chunks.max()
for k, powspec_chunks in powers_per_ell.items():
    k_out = k.replace('(', '_').replace(')', '').replace('=', '').replace(',','_')
    for i, tchunk in enumerate(time_chunks):
        powspec = powspec_chunks[i,:]
        if len(powspec.shape) != 3:
            powspec = np.expand_dims(powspec, axis=0)
        full_powspec = np.copy(powspec)
        
        freqs = freqs_chunks[i,:]
        times = time_chunks[i,:]
        color = sm.to_rgba(np.median(times))
        good = freqs >= 0

        v = 0
        powspec = full_powspec[v,:]
        good_axis = np.arange(len(powspec.shape))[np.array(powspec.shape) == len(ells.flatten())][0]
                
        sum_power = np.sum(powspec, axis=good_axis).squeeze()
        sum_power_ell2 = np.sum((powspec/ells[:,:,0,0]**2).reshape(powspec.shape)[:,ells[0,:,0,0] > 0], axis=good_axis).squeeze()
            
        ymin = sum_power[(freqs > min_freq)*(freqs < max_freq)][-1].min()/2
        ymax = sum_power[(freqs > min_freq)*(freqs <= max_freq)].max()*2

        if len(sum_power.shape) > 1:
            for j in range(sum_power.shape[1]):
                ax.plot(freqs[good], sum_power[good, j], c = color)#, label=r'axis {}, sum over $\ell$ values'.format(j))
#                    ax.plot(freqs[good], sum_power_ell2[good, j], c = color, ls='--', label=r'axis {}, sum over $\ell$ values with $\ell^{-2}$'.format(j))
        else:
            ax.plot(freqs[good], sum_power[good], c = color)#, label=r'sum over $\ell$ values')
#                ax.plot(freqs[good], sum_power_ell2[good], c = color, ls='--', label=r'sum over $\ell$ values with $\ell^{-2}$')
        ax.axvline(np.sqrt(N2plateau_sim)/(2*np.pi), c='k')
#        ax.legend(loc='best')

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel(r'summed Power ({})'.format(k))
    ax.set_xlabel(r'Frequency (sim units)')
    ax.set_xlim(min_freq, max_freq)
    ax.set_ylim(ymin, ymax)

    fig.savefig('{}/{}_summed_power.png'.format(full_out_dir, k_out), dpi=600)

    ax.clear()

    for ell in range(1, 21):
        print('plotting ell = {}'.format(ell))
        for i, tchunk in enumerate(time_chunks):
            powspec = powspec_chunks[i,:]
            if len(powspec.shape) != 3:
                powspec = np.expand_dims(powspec, axis=0)
            full_powspec = np.copy(powspec)
            
            freqs = freqs_chunks[i,:]
            times = time_chunks[i,:]
            color = sm.to_rgba(np.median(times))
            good = freqs >= 0

            ax.loglog(freqs[good], powspec[0,good,ells.flatten()==ell], c=color)
        ax.text(0.02, 0.95, 'ell={}'.format(ell), ha='left', transform=ax.transAxes)
        ax.set_xlim(min_freq, max_freq)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel('frequency (sim units)')
        ax.set_ylabel('Power')
        fig.savefig('{}/{}_ell_{:03d}.png'.format(full_out_dir, k_out, ell), dpi=200, bbox_inches='tight')
        ax.clear()

