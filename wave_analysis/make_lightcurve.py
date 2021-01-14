"""
This script plots snapshots of the evolution of 2D slices from a 2D simulation in polar geometry.

The fields specified in 'fig_type' are plotted (temperature and enstrophy by default).
To plot a different set of fields, add a new fig type number, and expand the fig_type if-statement.

Usage:
    make_lightcurve.py <root_dir> [options]

Options:
    --data_dir=<dir>                    Name of data handler directory [default: surface_shell_slices]
    --start_fig=<fig_start_num>         Number of first figure file [default: 1]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<num_files>               Total number of files to plot [default: 100000]
    --dpi=<dpi>                         Image pixel density [default: 200]

    --col_inch=<in>                     Number of inches / column [default: 3]
    --row_inch=<in>                     Number of inches / row [default: 3]

    --radius=<r>                        Radius at which the SWSH basis lives [default: 2.59]

    --plot_only
    --writes_per_spectrum=<w>           Max number of writes per power spectrum [default: 1e4]
    --dwrite=<dw>                       Number of writes to move before next power spectrum [default: 2e2]

    --mesa_file=<mf>                    MESA file
    --analyze

"""

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
n_files     = int(args['--n_files'])

# Create Plotter object, tell it which fields to plot
out_dir = 'lightcurve'
plotter = SFP(root_dir, file_dir=data_dir, fig_name=out_dir, start_file=start_file, n_files=n_files, distribution='even')
fields = ['s1_surf',]#, 'u_theta_surf',]
bases  = []

Lmax = int(root_dir.split('Re')[-1].split('_')[1].split('x')[0])

# Parameters
dtype = np.float64

# Bases
dealias = 1
c = coords.S2Coordinates('φ', 'θ')
d = distributor.Distributor((c,), mesh=None, comm=MPI.COMM_SELF)
b = basis.SWSH(c, (2*(Lmax+2), Lmax+1), radius=float(args['--radius']), dtype=dtype)
φ, θ = b.local_grids((dealias, dealias))
φg, θg = b.global_grids((dealias, dealias))

hemisphere = (φg.flatten() >= 0)*(φg.flatten() <= np.pi)
global_weight_φ = (np.ones_like(φg)*np.pi/((b.Lmax+1)*dealias))
hemisphere_weight_φ = global_weight_φ[hemisphere,:]
volume_φ = np.sum(hemisphere_weight_φ)

global_weight_θ = b.global_colatitude_weights(dealias)
theta_vol = np.sum(global_weight_θ)

phi_avg = lambda A: np.sum(hemisphere_weight_φ*A, axis=0)/volume_φ
theta_avg = lambda A: np.sum(global_weight_θ*A, axis=1)/theta_vol
phi_theta_avg = lambda A: np.sum(hemisphere_weight_φ*global_weight_θ*A)/volume_φ/theta_vol


out_time   = []
out_lum_fluc  = []
if args['--analyze']:
    with h5py.File(args['--mesa_file'], 'r') as f:
        cp_surf = f['cp_surf'][()]
        tau = f['tau'][()]
        time_day = tau/60/60/24
    while plotter.files_remain(bases, fields):
        file_name = plotter.files[plotter.current_filenum]
        file_num  = int(file_name.split('_s')[-1].split('.h5')[0])
        bs, tsk, write, time = plotter.read_next_file()
        outputs = OrderedDict()
        for i, t in enumerate(time):
            out_time.append(t)
            s1cp_surf = tsk['s1_surf'][i,hemisphere,:,0]/cp_surf
            lum_fluc = (1 + s1cp_surf)**4
            out_lum_fluc.append(phi_theta_avg(lum_fluc))
    out_time = np.array(out_time)
    out_lum_fluc = np.array(out_lum_fluc)
    with h5py.File('{}/{}/{}.h5'.format(root_dir, out_dir, out_dir), 'w') as f:
        f['time'] = out_time*time_day
        f['lum_fluc']   = out_lum_fluc

with h5py.File('{}/{}/{}.h5'.format(root_dir, out_dir, out_dir), 'r') as f:
    out_time = f['time'][()]
    out_lum_fluc = f['lum_fluc'][()]

fig = plt.figure(figsize=(8,3))
plt.plot(out_time, out_lum_fluc - 1, c='k')
plt.xlabel('sim time (days)')
plt.ylabel('fractional luminosity change')
fig.savefig('{}/{}/plot_{}.png'.format(root_dir, out_dir, out_dir), dpi=300, bbox_inches='tight')

writes_per = int(float(args['--writes_per_spectrum']))
dwrite = int(float(args['--dwrite']))
n_points = out_time.squeeze().shape[0]
slices = []
if writes_per > n_points:
    slices.append(slice(0,n_points,1))
else:
    for i in range(int(np.floor((n_points-writes_per)/dwrite))):
        slices.append(slice(i*dwrite, i*dwrite+writes_per,1))

for i, sl in enumerate(slices):
    print('plotting slice {}/{}'.format(i, len(slices)))
    N = out_lum_fluc[sl].shape[0]
    window = np.hanning(N)
    fft_lum = np.fft.fft(window*(out_lum_fluc[sl] - 1)) / (N/2)
    power = (fft_lum*np.conj(fft_lum)).real
    fft_freq = np.fft.fftfreq(len(out_time[sl]), np.mean(np.gradient(out_time[sl])))
    true_freqs = fft_freq[fft_freq >= 0]
    true_power = np.zeros_like(true_freqs)
    for j,f in enumerate(true_freqs):
        true_power[j] += power[fft_freq == f]
        if f != 0:
            true_power[j] += power[fft_freq == -f]
    fig = plt.figure(figsize=(8,3))
    plt.loglog(true_freqs, np.sqrt(true_power))
    plt.xlim(5e-2, 1e1)
    plt.xlabel('frequency (1/day)')
    plt.ylabel('amplitude')
    plt.title('t={:.2e}-{:.2e}'.format(out_time[sl].min(), out_time[sl].max()))
    fig.savefig('{}/{}/amplitude_{}_{:06d}.png'.format(root_dir, out_dir, out_dir, i+1), dpi=300, bbox_inches='tight')
    plt.clf()
