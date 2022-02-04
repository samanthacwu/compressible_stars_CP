"""
This script plots snapshots of the evolution of 2D slices from a 2D simulation in polar geometry.

The fields specified in 'fig_type' are plotted (temperature and enstrophy by default).
To plot a different set of fields, add a new fig type number, and expand the fig_type if-statement.

Usage:
    get_wave_spectrum.py <root_dir> [options]

Options:
    --data_dir=<dir>                    Name of data handler directory [default: surface_shell_slices]
    --start_fig=<fig_start_num>         Number of first figure file [default: 1]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<num_files>               Total number of files to plot
    --dpi=<dpi>                         Image pixel density [default: 200]

    --col_inch=<in>                     Number of inches / column [default: 3]
    --row_inch=<in>                     Number of inches / row [default: 3]

    --plot_only
    --writes_per_spectrum=<w>           Max number of writes per power spectrum

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
n_files     = args['--n_files']
if n_files is not None: 
    n_files = int(n_files)

# Create Plotter object, tell it which fields to plot
plotter = SFP(root_dir, file_dir=data_dir, fig_name='wave_analysis', start_file=start_file, n_files=n_files)
fields = ['s1_surf',]#, 'u_theta_surf',]
bases  = ['φ', 'θ']

star_file = '../mesa_stars/MESA_Models_Dedalus_Full_Sphere_6_ballShell/ballShell_nccs_B63_S63.h5'

with h5py.File(star_file, 'r') as f:
    tau = f['tau'][()]/(60*60*24)
    r_outer = f['r_outer'][()]
    radius = r_outer * f['L'][()]
    #Entropy units are erg/K/g
    s_c = f['s_c'][()]
    N2plateau = f['N2plateau'][()] * (60*60*24)**2

if not args['--plot_only']:
    Lmax = int(root_dir.split('Re')[-1].split('_')[1].split('x')[0])

    # Parameters
    dtype = np.float64

    # Bases
    dealias = 1
    c = coords.SphericalCoordinates('φ', 'θ', 'r')
    d = distributor.Distributor((c,), mesh=None)
    b = basis.BallBasis(c, (2*(Lmax+2), Lmax+1, 1), radius=r_outer, dtype=dtype)
    b_S2 = b.S2_basis()
    φ, θ, r = b.local_grids((dealias, dealias, dealias))
    φg, θg, rg = b.global_grids((dealias, dealias, dealias))

    ells = b.local_ell
    ell_values = np.unique(ells)

    field_s1 = field.Field(dist=d, bases=(b_S2,), dtype=dtype)
    #field_utheta = field.Field(dist=d, bases=(b_S2,), dtype=dtype)

    out_bs = None
    out_tsk = OrderedDict()
    for f in fields: out_tsk[f] = []
    out_write  = []
    out_time   = []
    while plotter.files_remain(bases, fields):
        bs, tsk, write, time = plotter.read_next_file()
        out_bs = bs
        for f in fields:
            out_tsk[f].append(np.array(tsk[f]))
        out_write.append(np.array(write))
        out_time.append(np.array(time))

    for b in out_bs.keys(): out_bs[b] = out_bs[b].squeeze()
    for f in fields: out_tsk[f] = np.concatenate(out_tsk[f], axis=0).squeeze()
    out_write = np.concatenate(out_write, axis=None)
    out_time  = np.concatenate(out_time, axis=None)
    out_time *= tau

    max_writes = args['--writes_per_spectrum']
    total_writes = len(out_time)
    if max_writes is not None:
        max_writes = int(max_writes)
        num_spectra = int(np.floor(total_writes/max_writes))
    else:
        num_spectra = 1
        max_writes = total_writes

    #Shape is m, ell, r
    data_cube_s1 = np.zeros((num_spectra, max_writes, Lmax+2, Lmax+1), dtype=np.complex128)
    spectrum_each_ell = np.zeros((num_spectra, len(ell_values), max_writes))
    spectrum_times = np.zeros((num_spectra, max_writes))
    for i in range(len(out_time)):
        spectrum_number = int(np.floor(i/max_writes))
        time_index = int(i % max_writes)
        if spectrum_number >= num_spectra:
            break
        field_s1['g'][:,:,0] = s_c*out_tsk[fields[0]][i,:]
        data_cube_s1[spectrum_number, time_index,:] = radius**2*field_s1['c'].squeeze()
        spectrum_times[spectrum_number, time_index] = out_time[i]

    print('done loading data, ffting...')
    dt = np.mean(np.gradient(out_time))
    freqs = np.fft.fftfreq(data_cube_s1.shape[1], d=dt)

    window = np.hanning(freqs.shape[0]).reshape((freqs.shape[0], 1, 1))
    fft_s1 = np.zeros_like(data_cube_s1, dtype=np.complex128)
    power = np.zeros_like(data_cube_s1)
    power_spectrum = np.zeros((num_spectra, max_writes))
    for i in range(num_spectra):
        fft_s1[i,:] = np.fft.fft(data_cube_s1[i,:]*window, axis=0)
        power[i,:] = fft_s1[i,:]*np.conj(fft_s1[i,:])
        for j, ell in enumerate(ell_values):
            spectrum_each_ell[i,j,:] = np.sum(power[i,:][:,ells == ell], axis=-1) / (freqs.shape[0]/2)**2
        power_spectrum[i,:] = np.sum(spectrum_each_ell[i,:], axis=0)

    with h5py.File('frequency_post.h5', 'w') as f:
        f['ells'] = ells
        f['power_each_ell'] = spectrum_each_ell 
        f['full_power_spectrum'] = power_spectrum
        f['fft_freqs'] = freqs
        f['spectrum_times'] = spectrum_times
else:
    with h5py.File('frequency_post.h5', 'r') as f:
        ells = f['ells'][()]
        ell_values = np.unique(ells)
        spectrum_each_ell = f['power_each_ell'][()]
        power_spectrum = f['full_power_spectrum'][()]
        freqs = f['fft_freqs'][()]
        spectrum_times = f['spectrum_times'][()]
fig = plt.figure()
good = freqs >= 0
min_freq = 1e-1
max_freq = freqs.max()
ymin = power_spectrum[:, (freqs > 5e-2)*(freqs < max_freq)][:,-1].min()/2
ymax = power_spectrum[:, (freqs > 5e-2)*(freqs <= max_freq)].max()*2

import matplotlib
cmap = matplotlib.cm.ScalarMappable(
      norm = matplotlib.colors.Normalize(0, power_spectrum.shape[0]-1),
            cmap = 'viridis')
for i in range(power_spectrum.shape[0]):
    #for i, ell in enumerate(ell_values):
    #    if ell == 0 or ell > 5:
    #        continue
    #    plt.figure()
    #    spectrum = spectrum_each_ell[i,:]
    #    plt.plot(freqs[good], spectrum[good])
    #    plt.yscale('log')
    #    plt.xscale('log')
    #    plt.ylabel(r'Power (erg cm$^2$ / g / K)$^2$ (?)')
    #    plt.xlabel(r'Frequency (day$^{-1}$)')
    #    plt.axvline(np.sqrt(N2plateau)/(2*np.pi), c='k')
    #    plt.xlim(min_freq, max_freq)
    #    plt.ylim(ymin, ymax)
    #    plt.title('$\ell = {{{}}}$'.format(ell))

    plt.plot(freqs[good], power_spectrum[i, good], c = cmap.to_rgba(i))
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'Power (erg cm$^2$ / g / K)$^2$ (?)')
    plt.xlabel(r'Frequency (day$^{-1}$)')
    plt.axvline(np.sqrt(N2plateau)/(2*np.pi), c='k')
    plt.xlim(min_freq, max_freq)
    plt.ylim(ymin, ymax)
plt.savefig('power.png', dpi=600)

