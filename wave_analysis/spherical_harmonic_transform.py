"""
This script plots snapshots of the evolution of 2D slices from a 2D simulation in polar geometry.

The fields specified in 'fig_type' are plotted (temperature and enstrophy by default).
To plot a different set of fields, add a new fig type number, and expand the fig_type if-statement.

Usage:
    spherical_harmonic_transform.py <root_dir> [options]

Options:
    --data_dir=<dir>                    Name of data handler directory [default: surface_shell_slices]
    --start_fig=<fig_start_num>         Number of first figure file [default: 1]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<num_files>               Total number of files to plot
    --dpi=<dpi>                         Image pixel density [default: 200]

    --col_inch=<in>                     Number of inches / column [default: 3]
    --row_inch=<in>                     Number of inches / row [default: 3]

    --radius=<r>                        Radius at which the SWSH basis lives [default: 2.59]

    --plot_only
    --writes_per_spectrum=<w>           Max number of writes per power spectrum

    --field=<f>                         If specified, only transform this field
    --shell_basis                       If flagged, use SphericalShellBasis not SWSH

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

bases = []
if not plotter.idle:
    if args['--field'] is None:
        with h5py.File(plotter.files[0], 'r') as f:
            fields = list(f['tasks'].keys())
    else:
        fields = [args['--field'],]

    Lmax = int(root_dir.split('Re')[-1].split('_')[1].split('x')[0])

    # Parameters
    dtype = np.float64

    res = re.compile('(.*)\(r=(.*)\)')

    # Bases
    dealias = 1.5
    if args['--shell_basis']:
        c = coords.SphericalCoordinates('φ', 'θ', 'r')
        dealias_tuple = (dealias, dealias, dealias)
        d = distributor.Distributor((c,), mesh=None, comm=MPI.COMM_SELF)
        global_b = basis.SphericalShellBasis(c, (2*(Lmax+2), Lmax+1, 1), radii=((1-1e-6)*float(args['--radius']), float(args['--radius'])), dtype=dtype, dealias=dealias_tuple)
        φ, θ, r= global_b.local_grids(dealias_tuple)
        φg, θg, rg = global_b.global_grids(dealias_tuple)
    else:
        c = coords.S2Coordinates('φ', 'θ')
        dealias_tuple = (dealias, dealias)
        d = distributor.Distributor((c,), mesh=None, comm=MPI.COMM_SELF)
        global_b = basis.SWSH(c, (2*(Lmax+2), Lmax+1), radius=float(args['--radius']), dtype=dtype, dealias=dealias_tuple)
        φ, θ = global_b.local_grids(dealias_tuple)
        φg, θg = global_b.global_grids(dealias_tuple)

    ells = global_b.local_ell
    ms = global_b.local_m
    ell_values = np.unique(ells)
    m_values = np.unique(ms)
    ell_ms = np.zeros_like(ell_values)
    for i, ell in enumerate(ell_values):
        ell_ms[i] = int(np.sum(ells == ell)/2)

    global_s_field = field.Field(dist=d, bases=(global_b,), dtype=dtype)
    global_v_field = field.Field(dist=d, bases=(global_b,), tensorsig=(c,), dtype=dtype)
    global_s_field.require_scales(dealias_tuple)
    global_v_field.require_scales(dealias_tuple)

    out_tsk = OrderedDict()
    for f in fields: out_tsk[f] = []
    out_write  = []
    out_time   = []
    S2_bases = OrderedDict()
    while plotter.files_remain(bases, fields):
        file_name = plotter.files[plotter.current_filenum]
        plotter.current_filenum += 1
        file_num  = int(file_name.split('_s')[-1].split('.h5')[0])

        with h5py.File('{}/{}/{}_s{}.h5'.format(root_dir, out_dir, out_dir, file_num), 'w') as of:
            of['ells'] = np.expand_dims(ell_values, axis=(0,2))
            of['ms']   = np.expand_dims(m_values, axis=(0,1))

            with h5py.File(file_name, 'r') as in_f:
                sim_times = in_f['scales/sim_time'][()]
                of['time'] = sim_times
                outputs = OrderedDict()
                for f in fields:
                    if res.match(f) and not args['--shell_basis']:
                        radius = float(f.split('r=')[-1].split(')')[0])
                        k = 'r={}'.format(radius)
                        if k not in S2_bases.keys():
                            b = basis.SWSH(c, (2*(Lmax+2), Lmax+1), radius=radius, dtype=dtype, dealias=dealias_tuple)
                            s_field = field.Field(dist=d, bases=(b,), dtype=dtype)
                            v_field = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
                            s_field.require_scales(dealias_tuple)
                            v_field.require_scales(dealias_tuple)
                            S2_bases[k] = (b, s_field, v_field)
                        else:
                            b, s_field, v_field = S2_bases[k]
                    else:
                        s_field = global_s_field
                        v_field = global_v_field
                
                    for i in range(len(sim_times)):
                        task_data = in_f['tasks/{}'.format(f)][i,:]
                        shape = list(task_data.shape)
                        if len(shape) == len(s_field['g'].shape):
                            shape[0] = ell_values.shape[0]
                            shape[1] = m_values.shape[0]
                        else:
                            shape[1] = ell_values.shape[0]
                            shape[2] = m_values.shape[0]
                        out_field = np.zeros(shape, dtype=np.complex128)
                        if i == 0:
                            of.create_dataset(name='tasks/'+f, shape=[len(sim_times),] + shape, dtype=np.complex128)
                        logger.info('file {}, transforming {}, {}/{}'.format(file_num, f, i+1, len(sim_times)))
                        if len(shape) == 3:
                            s_field['g'] = task_data.reshape(s_field['g'].shape)
                        else:
                            v_field['g'] = task_data.reshape(v_field['g'].shape)
                        for j, ell in enumerate(ell_values):
                            for k, m in enumerate(m_values):
                                bool_map = (ell == ells)*(m == ms)
                                if np.sum(bool_map) > 0:
                                    if len(shape) == len(s_field['g'].shape):
                                        values = s_field['c'][bool_map]
                                        out_field[j,k] = values[0] + 1j*values[1]
                                    else:
                                        for v in range(shape[0]):
                                            values = v_field['c'][v, bool_map]
                                            out_field[v,j,k] = values[0] + 1j*values[1]

                        if len(shape) == 3:
                            out_field[:,0]  /= np.sqrt(2) #m == 0 normalization
                            out_field[:,1:] /= 2          #m != 0 normalization
                        else:
                            out_field[:,:,0]  /= np.sqrt(2) #m == 0 normalization
                            out_field[:,:,1:] /= 2          #m != 0 normalization
                        of['tasks/'+f][i, :] = out_field
                        gc.collect()
                        first = False

