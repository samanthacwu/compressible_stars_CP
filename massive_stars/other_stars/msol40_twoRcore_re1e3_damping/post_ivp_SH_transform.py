"""
This script plots snapshots of the evolution of 2D slices from a 2D simulation in polar geometry.

The fields specified in 'fig_type' are plotted (temperature and enstrophy by default).
To plot a different set of fields, add a new fig type number, and expand the fig_type if-statement.

Usage:
    post_ivp_SH_transform.py [options]

Options:
    --data_dir=<dir>                    Name of data handler directory [default: wave_shells]
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
import dedalus.public as d3
from dedalus.tools import logging
from scipy import sparse
from mpi4py import MPI
import matplotlib.pyplot as plt

from d3_stars.defaults import config

from plotpal.file_reader import SingleTypeReader as SR

import logging
logger = logging.getLogger(__name__)


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

    ntheta = config.dynamics['ntheta']
    nphi = 2*ntheta

    resolution = (nphi, ntheta, 1)

    # Parameters
    dtype = np.float64
    dealias = 1.5
    radius = float(args['--radius'])
    c = d3.SphericalCoordinates('φ', 'θ', 'r')
    dealias_tuple = (dealias, dealias, dealias)
    Lmax = resolution[1]-1
    dr = 1e-4
    shell_vol = (4/3)*np.pi*(radius**3 - (radius-dr)**3)
    dist = d3.Distributor((c,), mesh=None, comm=MPI.COMM_SELF, dtype=dtype)
    basis = d3.ShellBasis(c, resolution, radii=(radius-dr, radius), dtype=dtype, dealias=dealias_tuple)
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

    scalar_field = dist.Field(bases=basis)
    vector_field = dist.VectorField(bases=basis, coordsys=c)
    scalar_field.change_scales(dealias)
    vector_field.change_scales(dealias)
    power_scalar_op = d3.integ(scalar_field**2) / shell_vol
    power_vector_op = d3.integ(vector_field@vector_field) / shell_vol

    slices = dict()
    for i in range(scalar_field['c'].shape[0]):
        for j in range(scalar_field['c'].shape[1]):
            groups = basis.elements_to_groups((False, False, False), (np.array((i,)),np.array((j,)), np.array((0,))))
            m = groups[0][0]
            ell = groups[1][0]
            key = '{},{}'.format(ell, m)
            this_slice = (slice(i, i+1, 1), slice(j, j+1, 1), slice(None))
            if key not in slices.keys():
                slices[key] = [this_slice]
            else:
                slices[key].append(this_slice)

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
            sim_times = reader.current_file_handle['scales/sim_time'][()]
            if ni == 0:
                of['ells'] = ell_values[None,:,:,None]
                of['ms']   = m_values[None,:,:,None]
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
                if scalar:
                    scalar_field['g'] = task_data.reshape(scalar_field['g'].shape)
                elif vector:
                    vector_field['g'] = task_data.reshape(vector_field['g'].shape)
                for j, ell in enumerate(ell_values.squeeze()):
                    for k, m in enumerate(m_values.squeeze()):
                        if '{},{}'.format(ell,m) in slices.keys():
                            sl = slices['{},{}'.format(ell,m)]
                            if scalar:
                                value1 = scalar_field['c'][sl[0]].squeeze()
                                value2 = scalar_field['c'][sl[1]].squeeze()
                                out_field[j,k] = value1 + 1j*value2
                            elif vector:
                                for v in range(shape[0]):
                                    v_sl1 = tuple([v] + list(sl[0]))
                                    v_sl2 = tuple([v] + list(sl[1]))
                                    value1 = vector_field['c'][v_sl1].squeeze()
                                    value2 = vector_field['c'][v_sl2].squeeze()
                                    out_field[v,j,k] = value1 + 1j*value2

                if len(shape) == 3:
                    out_field[:,0]  /= np.sqrt(2*np.pi) #m == 0 normalization
                    out_field[:,1:] /= 2*np.sqrt(np.pi)          #m != 0 normalization
                else:
                    out_field[:,:,0]  /= np.sqrt(2*np.pi) #m == 0 normalization
                    out_field[:,:,1:] /= 2*np.sqrt(np.pi)         #m != 0 normalization
                of['tasks/'+f][ni, :] = out_field
                #Check power conservation
                power_transform = np.sum(out_field * np.conj(out_field)).real
                power_scalar = power_scalar_op.evaluate()['g'].ravel()[0]
                power_vector = power_vector_op.evaluate()['g'].ravel()[0]
                logger.info('finishing file {}, transforming {}, {}/{}'.format(file_num, f, ni+1, len(sim_times)) + ', power factors: scalar {:.3e}, vector {:.3e}'.format(power_scalar/power_transform, power_vector/power_transform))
                gc.collect()
                first = False
