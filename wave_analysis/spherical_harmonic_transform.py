"""
This script plots snapshots of the evolution of 2D slices from a 2D simulation in polar geometry.

The fields specified in 'fig_type' are plotted (temperature and enstrophy by default).
To plot a different set of fields, add a new fig type number, and expand the fig_type if-statement.

Usage:
    plot_mollweide_snapshots.py <root_dir> [options]

Options:
    --data_dir=<dir>                    Name of data handler directory [default: surface_shells]
    --start_fig=<fig_start_num>         Number of first figure file [default: 1]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<num_files>               Total number of files to plot
    --dpi=<dpi>                         Image pixel density [default: 200]

    --col_inch=<in>                     Number of inches / column [default: 3]
    --row_inch=<in>                     Number of inches / row [default: 3]

    --radius=<r>                        Radius at which the SWSH basis lives [default: 2.59]

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
n_files     = args['--n_files']
if n_files is not None: 
    n_files = int(n_files)

# Create Plotter object, tell it which fields to plot
out_dir = 'SH_transform_{}'.format(data_dir)
plotter = SFP(root_dir, file_dir=data_dir, fig_name=out_dir, start_file=start_file, n_files=n_files, distribution='even')
fields = ['s1_surf',]#, 'u_theta_surf',]
bases  = ['φ', 'θ']

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

ells = b.local_ell
ms = b.local_m
ell_values = np.unique(ells)
m_values = np.unique(ms)
ell_ms = np.zeros_like(ell_values)
for i, ell in enumerate(ell_values):
    ell_ms[i] = int(np.sum(ells == ell)/2)

field = field.Field(dist=d, bases=(b,), dtype=dtype)

out_bs = None
out_tsk = OrderedDict()
for f in fields: out_tsk[f] = []
out_write  = []
out_time   = []
while plotter.files_remain(bases, fields):
    file_name = plotter.files[plotter.current_filenum]
    file_num  = int(file_name.split('_s')[-1].split('.h5')[0])
    bs, tsk, write, time = plotter.read_next_file()
    out_bs = bs
    outputs = OrderedDict()
    for f in fields:
        out_field = np.zeros((time.shape[0], ell_values.shape[0], m_values.shape[0]), dtype=np.complex128)
        for i in range(time.shape[0]):
            field['g'] = tsk[f][i,:,:,0]
            for j, ell in enumerate(ell_values):
                for k, m in enumerate(m_values):
                    bool_map = (ell == ells)*(m == ms)
                    if np.sum(bool_map) > 0:
                        values = field['c'][bool_map]
                        out_field[i,j,k] = values[0] + 1j*values[1]
        outputs[f] = out_field
    with h5py.File('{}/{}/{}_s{}.h5'.format(root_dir, out_dir, out_dir, file_num), 'w') as f:
        f['ells'] = np.expand_dims(ell_values, axis=(0,2))
        f['ms']   = np.expand_dims(m_values, axis=(0,1))
        f['time'] = time
        for fd in fields:
            f[fd] = outputs[fd]
