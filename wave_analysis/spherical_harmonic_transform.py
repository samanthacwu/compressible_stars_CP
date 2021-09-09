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

from plotpal.file_reader import SingleTypeReader as SR
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
reader = SR(root_dir, file_dir=data_dir, fig_name=out_dir, start_file=start_file, n_files=n_files, distribution='even-file')

bases = []
if not reader.idle:
    if args['--field'] is None:
        with h5py.File(reader.files[0], 'r') as f:
            fields = list(f['tasks'].keys())
    else:
        fields = [args['--field'],]

    resolution_regex = re.compile('(.*)x(.*)x(.*)')
    field_regex = re.compile('(.*)\(r=(.*)\)')
    for str_bit in root_dir.split('_'):
        if resolution_regex.match(str_bit):
            res_strs = str_bit.split('x')
            resolution = [int(res_strs[0]), int(res_strs[1]), 1]

    # Parameters
    dtype = np.float64
    dealias = 1
    radius = float(args['--radius'])
    c = coords.SphericalCoordinates('φ', 'θ', 'r')
    dealias_tuple = (dealias, dealias, dealias)
    dist = distributor.Distributor((c,), mesh=None, comm=MPI.COMM_SELF, dtype=dtype)
    basis = basis.ShellBasis(c, resolution, radii=((1-1e-6)*radius, radius), dtype=dtype, dealias=dealias_tuple)
    φg, θg, rg = basis.global_grids(basis.dealias)

    ells = basis.local_ell
    ms = basis.local_m
    ell_values = np.unique(ells)
    m_values = np.unique(ms)
    ell_ms = np.zeros_like(ell_values)
    for i, ell in enumerate(ell_values):
        ell_ms[i] = int(np.sum(ells == ell)/2)

    scalar_field = dist.Field(bases=basis)
    vector_field = dist.VectorField(bases=basis, coordsys=c)

    out_tsk = OrderedDict()
    for f in fields: out_tsk[f] = []
    out_write  = []
    out_time   = []
    while reader.writes_remain():
        dsets, ni = reader.get_dsets(fields)
        file_name = reader.current_file_name
        file_num  = int(file_name.split('_s')[-1].split('.h5')[0])
        if ni == 0:
            file_mode = 'w'
        else:
            file_mode = 'a'
        output_file_name = '{}/{}/{}_s{}.h5'.format(root_dir, out_dir, out_dir, file_num)

        with h5py.File(output_file_name, file_mode) as of:
            sim_times = dsets[fields[0]].dims[0]['sim_time']
            if ni == 0:
                of['ells'] = np.expand_dims(ell_values, axis=(0,2))
                of['ms']   = np.expand_dims(m_values, axis=(0,1))
                of['time'] = sim_times[()]
                for attr in ['writes', 'set_number', 'handler_name']:
                    of.attrs[attr] = reader.current_file_handle.attrs[attr]

            outputs = OrderedDict()
            for f in fields:
                scalar = vector = False
                task_data = dsets[f][ni,:]
                shape = list(task_data.shape)
                if len(shape) == len(scalar_field['g'].shape):
                    shape[0] = ell_values.shape[0]
                    shape[1] = m_values.shape[0]
                    scalar = True
                else:
                    shape[1] = ell_values.shape[0]
                    shape[2] = m_values.shape[0]
                    vector = True
                out_field = np.zeros(shape, dtype=np.complex128)
                if ni == 0:
                    of.create_dataset(name='tasks/'+f, shape=[len(sim_times),] + shape, dtype=np.complex128)
                logger.info('file {}, transforming {}, {}/{}'.format(file_num, f, ni+1, len(sim_times)))
                if scalar:
                    scalar_field['g'] = task_data.reshape(scalar_field['g'].shape)
                elif vector:
                    vector_field['g'] = task_data.reshape(vector_field['g'].shape)
                for j, ell in enumerate(ell_values):
                    for k, m in enumerate(m_values):
                        bool_map = (ell == ells)*(m == ms)
                        if np.sum(bool_map) > 0:
                            if scalar:
                                values = scalar_field['c'][bool_map]
                                out_field[j,k] = values[0] + 1j*values[1]
                            elif vector:
                                for v in range(shape[0]):
                                    values = vector_field['c'][v, bool_map]
                                    out_field[v,j,k] = values[0] + 1j*values[1]

                if len(shape) == 3:
                    out_field[:,0]  /= np.sqrt(2) #m == 0 normalization
                    out_field[:,1:] /= 2          #m != 0 normalization
                else:
                    out_field[:,:,0]  /= np.sqrt(2) #m == 0 normalization
                    out_field[:,:,1:] /= 2          #m != 0 normalization
                of['tasks/'+f][ni, :] = out_field
                gc.collect()
                first = False
