"""
This script plots snapshots of the evolution of 2D slices from a 2D simulation in polar geometry.

The fields specified in 'fig_type' are plotted (temperature and enstrophy by default).
To plot a different set of fields, add a new fig type number, and expand the fig_type if-statement.

Usage:
    post_ivp_SH_transform.py [options]

Options:
    --data_dir=<dir>                    Name of data handler directory [default: shells]
    --start_fig=<fig_start_num>         Number of first figure file [default: 1]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<num_files>               Total number of files to plot
    --dpi=<dpi>                         Image pixel density [default: 200]

    --col_inch=<in>                     Number of inches / column [default: 3]
    --row_inch=<in>                     Number of inches / row [default: 3]


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
import dedalus.public as d3
from dedalus.tools import logging
from scipy import sparse
from mpi4py import MPI
import matplotlib.pyplot as plt

from plotpal.file_reader import SingleTypeReader as SR

from d3_stars.post.transforms import SHTransformer

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


    ntheta = 32
    nphi = 4
    dtype = np.float64
    dealias = 1
    radius = 1
    transformer = SHTransformer(nphi, ntheta, dtype=dtype, dealias=dealias, radius=radius)

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
                of['ells'] = transformer.ell_values[None,:,:]
                of['ms']   = transformer.m_values[None,:,:]
                of['time'] = sim_times[()]
                for attr in ['writes', 'set_number', 'handler_name']:
                    of.attrs[attr] = reader.current_file_handle.attrs[attr]

            outputs = OrderedDict()
            for f in fields:
                scalar = vector = False
                task_data = dsets[f][ni,:]
                shape = list(task_data.shape)
                true_shape = list(task_data.squeeze().shape)
                if len(shape) == len(transformer.scalar_field['g'].shape):
                    scalar = True
                    true_shape[0] = transformer.ell_values.shape[0]
                    true_shape[1] = transformer.m_values.shape[1]
                else:
                    vector = True
                    true_shape[1] = transformer.ell_values.shape[0]
                    true_shape[2] = transformer.m_values.shape[1]
                if ni == 0:
                    of.create_dataset(name='tasks/'+f, shape=[len(sim_times),] + true_shape, dtype=np.complex128)


                if scalar:
                    out_field = transformer.transform_scalar_field(task_data)
                else:
                    out_field = transformer.transform_vector_field(task_data)
                of['tasks/'+f][ni, :] = out_field
                gc.collect()
