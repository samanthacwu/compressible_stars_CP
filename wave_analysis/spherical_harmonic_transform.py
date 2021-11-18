"""
This script plots snapshots of the evolution of 2D slices from a 2D simulation in polar geometry.

The fields specified in 'fig_type' are plotted (temperature and enstrophy by default).
To plot a different set of fields, add a new fig type number, and expand the fig_type if-statement.

Usage:
    spherical_harmonic_transform.py <root_dir> [options]

Options:
    --data_dir=<dir>                    Name of data handler directory [default: wave_shell_slices]
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
reader = SR(root_dir, data_dir, out_dir, start_file=start_file, n_files=n_files, distribution='even-file')

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
            resolution = (int(res_strs[0]), int(res_strs[1]), 1)

    # Parameters
    dtype = np.float64
    dealias = 1
    radius = float(args['--radius'])
    c = coords.SphericalCoordinates('φ', 'θ', 'r')
    dealias_tuple = (dealias, dealias, dealias)
    dist = distributor.Distributor((c,), mesh=None, comm=MPI.COMM_SELF, dtype=dtype)
    basis = basis.ShellBasis(c, resolution, radii=((1-1e-6)*radius, radius), dtype=dtype, dealias=dealias_tuple)
    φg, θg, rg = basis.global_grids(basis.dealias)

    ell_maps = basis.ell_maps
    m_maps   = basis.sphere_basis.m_maps

    max_ell = 0
    max_m = 0
    for m, mg_slice, mc_slice, n_slice in m_maps:
        if m > max_m:
            max_m = m
    for ell, m_ind, ell_ind in ell_maps:
        if ell > max_ell:
            max_ell = ell

    ell_values = np.arange(max_ell+1).reshape((max_ell+1,1))
    m_values = np.arange(max_m+1).reshape((1,max_m+1))

    slices = dict()
    domain = basis.domain
    coeff_layout = dist.coeff_layout
    group_coupling = [True] * domain.dist.dim
    group_coupling[0] = False
    group_coupling[1] = False
    group_coupling = tuple(group_coupling)
    groupsets = coeff_layout.local_groupsets(group_coupling, domain, scales=domain.dealias, broadcast=True)
    for ell_v in ell_values.squeeze():
        if MPI.COMM_WORLD.rank == 0:
            print('getting slices for ell = {}'.format(ell_v))
        for m_v in m_values.squeeze():
            if (m_v, ell_v, None) in groupsets:
                groupset_slices = coeff_layout.local_groupset_slices((m_v, ell_v, None), domain, scales=domain.dealias, broadcast=True)
                slices['{},{}'.format(ell_v,m_v)] = groupset_slices

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
                    shape[1] = m_values.shape[1]
                    scalar = True
                else:
                    shape[1] = ell_values.shape[0]
                    shape[2] = m_values.shape[1]
                    vector = True
                out_field = np.zeros(shape, dtype=np.complex128)
                if ni == 0:
                    of.create_dataset(name='tasks/'+f, shape=[len(sim_times),] + shape, dtype=np.complex128)
                logger.info('file {}, transforming {}, {}/{}'.format(file_num, f, ni+1, len(sim_times)))
                if scalar:
                    scalar_field['g'] = task_data.reshape(scalar_field['g'].shape)
                elif vector:
                    vector_field['g'] = task_data.reshape(vector_field['g'].shape)
                for j, ell in enumerate(ell_values.squeeze()):
                    for k, m in enumerate(m_values.squeeze()):
                        if '{},{}'.format(ell,m) in slices.keys():
                            sl = slices['{},{}'.format(ell,m)][0]
                            if scalar:
                                values = scalar_field['c'][sl].squeeze()
                                out_field[j,k] = values[0] + 1j*values[1]
                            elif vector:
                                for v in range(shape[0]):
                                    v_sl = tuple([v] + list(sl))
                                    values = vector_field['c'][v_sl].squeeze()
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
