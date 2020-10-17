"""
This script plots snapshots of the evolution of 2D slices from a 2D simulation in polar geometry.

The fields specified in 'fig_type' are plotted (temperature and enstrophy by default).
To plot a different set of fields, add a new fig type number, and expand the fig_type if-statement.

Usage:
    plot_mollweide_snapshots.py <root_dir> [options]

Options:
    --data_dir=<dir>                    Name of data handler directory [default: shell_slice]
    --start_fig=<fig_start_num>         Number of first figure file [default: 1]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<num_files>               Total number of files to plot
    --dpi=<dpi>                         Image pixel density [default: 200]

    --col_inch=<in>                     Number of inches / column [default: 3]
    --row_inch=<in>                     Number of inches / row [default: 3]

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
plotter = SFP(root_dir, file_dir=data_dir, fig_name='wave_analysis', start_file=1, n_files=np.inf)
fields = ['s1_r0.95', 'ur_r0.95',]
bases  = ['φ', 'θ']

sim_radius = 0.95

Lmax = int(root_dir.split('Re')[-1].split('_')[1].split('x')[0])

# Parameters
dtype = np.float64

# Bases
dealias = 1
c = coords.SphericalCoordinates('φ', 'θ', 'r')
d = distributor.Distributor((c,), mesh=None)
b = basis.BallBasis(c, (2*(Lmax+2), Lmax+1, 1), radius=sim_radius, dtype=dtype)
b_S2 = b.S2_basis()
φ, θ, r = b.local_grids((dealias, dealias, dealias))
φg, θg, rg = b.global_grids((dealias, dealias, dealias))


field_s1 = field.Field(dist=d, bases=(b_S2,), dtype=dtype)
field_ur = field.Field(dist=d, bases=(b_S2,), dtype=dtype)


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

#Shape is m, ell, r
print(field_s1['c'].shape)
data_cube_ur = np.zeros((len(out_time), Lmax+2, Lmax+1), dtype=np.complex128)
data_cube_s1 = np.zeros((len(out_time), Lmax+2, Lmax+1), dtype=np.complex128)
for i in range(len(out_time)):
    field_s1['g'][:,:,0] = out_tsk['s1_r0.95'][i,:]
    field_ur['g'][:,:,0] = out_tsk['ur_r0.95'][i,:]
    data_cube_ur[i,:,:] = field_ur['c'].squeeze() 
    data_cube_s1[i,:,:] = field_s1['c'].squeeze() 

fft_ur = np.fft.fft(data_cube_ur, axis=0)
fft_s1 = np.fft.fft(data_cube_s1, axis=0)

power_ur = fft_ur**2
power_s1 = fft_s1**2

power_ell_f_ur = np.sum(power_ur, axis=1)
power_ell_f_s1 = np.sum(power_s1, axis=1)
