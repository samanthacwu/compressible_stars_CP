"""
This script computes the wave flux in a d3 spherical simulation

Usage:
    post_ivp_SH_wave_flux.py [options]

Options:
    --data_dir=<dir>                    Name of data handler directory [default: SH_transform_wave_shells]
    --start_fig=<fig_start_num>         Number of first figure file [default: 1]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 40]
    --n_files=<num_files>               Total number of files to plot
    --dpi=<dpi>                         Image pixel density [default: 200]

    --col_inch=<in>                     Number of inches / column [default: 3]
    --row_inch=<in>                     Number of inches / row [default: 3]

    --radius=<r>                        Radius at which the SWSH basis lives [default: 2.59]

    --no_ft                             Do the base fourier transforms
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

from d3_stars.defaults import config
from d3_stars.post.power_spectrum_functions import clean_cfft, normalize_cfft_power
from d3_stars.simulations.parser import name_star

args = docopt(__doc__)
res = re.compile('(.*),r=(.*)')

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

out_dir, star_file = name_star()
with h5py.File(star_file, 'r') as f:
    rB = f['r_B'][()]
    rS1 = f['r_S1'][()]
    rS2 = f['r_S2'][()]
    rhoB = np.exp(f['ln_rho0_B'][()])
    rhoS1 = np.exp(f['ln_rho0_S1'][()])
    rhoS2 = np.exp(f['ln_rho0_S2'][()])
    r = np.concatenate((rB.flatten(), rS1.flatten(), rS2.flatten()))
    rho = np.concatenate((rhoB.flatten(), rhoS1.flatten(), rhoS2.flatten()))
    rho_func = interp1d(r,rho)
    tau = f['tau_nd'][()]/(60*60*24)
    r_outer = f['r_outer'][()]
    radius = r_outer * f['L_nd'][()]
    #Entropy units are erg/K/g
    s_c = f['s_nd'][()]
    N2plateau = f['N2plateau'][()] * (60*60*24)**2
    N2plateau_simunit = N2plateau * tau**2

# Create Plotter object, tell it which fields to plot
out_dir = 'SH_wave_flux_spectra'.format(data_dir)
full_out_dir = '{}/{}'.format(root_dir, out_dir)
reader = SR(root_dir, data_dir, out_dir, start_file=start_file, n_files=n_files, distribution='single')
with h5py.File(reader.files[0], 'r') as f:
    fields = list(f['tasks'].keys())
radii = []
for f in fields:
    if res.match(f):
        radius_str = f.split('r=')[-1].split(')')[0]
        if radius_str not in radii:
            radii.append(radius_str)
if not args['--no_ft']:
    times = []
    print('getting times...')
    first = True
    while reader.writes_remain():
        dsets, ni = reader.get_dsets([], verbose=False)
        times.append(reader.current_file_handle['time'][ni])
        if first:
            ells = reader.current_file_handle['ells'][()]
            ms = reader.current_file_handle['ms'][()]
            first = False

    times = np.array(times)

    with h5py.File('{}/transforms.h5'.format(full_out_dir), 'w') as wf:
        wf['ells']  = ells
    print(ells.shape, ms.shape)

    #TODO: only load in one ell and m at a time, that'll save memory.
    for i, f in enumerate(fields):
        print('reading field {}'.format(f))

        print('filling datacube...')
        writes = 0
        while reader.writes_remain():
            dsets, ni = reader.get_dsets([], verbose=False)
            rf = reader.current_file_handle
            this_task = rf['tasks'][f][ni,:].squeeze()
            if writes == 0: 
                if this_task.shape[0] == 3 and len(this_task.shape) == 3:
                    #vector
                    data_cube = np.zeros((times.shape[0], 3, ells.shape[1], ms.shape[2]), dtype=np.complex128)
                else:
                    data_cube = np.zeros((times.shape[0], ells.shape[1], ms.shape[2]), dtype=np.complex128)
            data_cube[writes,:] = this_task
            writes += 1

        print('taking transform')
        transform = np.zeros(data_cube.shape, dtype=np.complex128)
        for ell in range(data_cube.shape[-2]):
            print('taking transforms {}/{}'.format(ell+1, data_cube.shape[-2]))
            for m in range(data_cube.shape[-1]):
                if m > ell: continue
                if len(data_cube.shape) == 3:
                    input_data = data_cube[:,ell,m]
                    freqs, transform[:,ell,m] = clean_cfft(times, input_data)
                else:
                    input_data = data_cube[:,:,ell,m]
                    freqs, transform[:,:,ell,m] = clean_cfft(times, input_data)
        del data_cube
        gc.collect()

        with h5py.File('{}/transforms.h5'.format(full_out_dir), 'r+') as wf:
            wf['{}_cft'.format(f)] = transform
            if i == 0:
                wf['freqs'] = freqs 
                wf['freqs_inv_day'] = freqs/tau

with h5py.File('{}/transforms.h5'.format(full_out_dir), 'r') as wf:
    with h5py.File('{}/power_spectra.h5'.format(full_out_dir), 'w') as pf:
        raw_freqs = wf['freqs'][()]
        raw_freqs_invDay = wf['freqs_inv_day'][()]
        pf['real_freqs'] = wf['freqs'][raw_freqs >= 0]
        pf['real_freqs_inv_day'] = wf['freqs_inv_day'][raw_freqs >= 0]
        pf['ells'] = wf['ells'][()]
        for i, f in enumerate(fields):
            transform = wf['{}_cft'.format(f)]
            power = (np.conj(transform)*transform).real
            for fq in raw_freqs:
                if fq < 0:
                    power[raw_freqs == -fq] += power[raw_freqs == fq]
            power = power[raw_freqs >= 0,:]
            power = np.sum(power, axis=-1) #sum over m
            pf['{}'.format(f)] = power
       

fig = plt.figure()
for ell in range(11):
    if ell == 0: continue
    print('plotting ell = {}'.format(ell))
    with h5py.File('{}/power_spectra.h5'.format(full_out_dir), 'r') as rf:

        freqs = rf['real_freqs'][()]
        power = rf['shell(enthalpy_fluc_S2,r=R)'][:,ell]
        plt.loglog(freqs, power)
    plt.title('ell={}'.format(ell))
    plt.xlabel('freqs (sim units)')
    plt.ylabel(r'|enthalpy surface power|')

    plt.axvline(np.sqrt(N2plateau_simunit)/(2*np.pi))
    plt.ylim(1e-40, 1e-25)
    fig.savefig('{}/freq_spectrum_ell{}.png'.format(full_out_dir, ell), dpi=300, bbox_inches='tight')
    plt.clf()
    
    



freqs_for_dfdell = [8e-2, 2e-1, 5e-1]
with h5py.File('{}/power_spectra.h5'.format(full_out_dir), 'r') as rf:
    freqs = rf['real_freqs'][()]
    ells = rf['ells'][()].flatten()
    for f in freqs_for_dfdell:
        print('plotting f = {}'.format(f))
        f_ind = np.argmin(np.abs(freqs - f))
        power = rf['shell(enthalpy_fluc_S2,r=R)'][f_ind,:]
        plt.loglog(ells, power)
        plt.title('f = {} 1/day'.format(f))
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'|enthalpy surface power|')
        plt.ylim(1e-40, 1e-25)
        plt.xlim(1, ells.max())
        fig.savefig('{}/ell_spectrum_freq{}.png'.format(full_out_dir, f), dpi=300, bbox_inches='tight')
        plt.clf()
    
 

