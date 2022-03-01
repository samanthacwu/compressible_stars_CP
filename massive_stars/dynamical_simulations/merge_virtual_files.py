"""
This script merges a virtual file so that it can be scp'd off a supercomputer and used in visualization locally.

Usage:
    merge_virtual_files.py <root_dir>
"""
import os
import glob
import shutil

import h5py
import numpy as np
from mpi4py import MPI

from dedalus.tools.general import natural_sort

from docopt import docopt
args = docopt(__doc__)

root_dir = args['<root_dir>']
handlers = ['slices', 'profiles', 'scalars', 'checkpoint', 'final_checkpoint']
cleanup = True

for data_dir in handlers:
    files = glob.glob('{}/{}/{}_s*.h5'.format(root_dir, data_dir, data_dir))
    sorted_files = natural_sort(files)
    do_file = np.zeros_like(sorted_files, dtype=bool)
    for i in range(len(sorted_files)):
        if i % MPI.COMM_WORLD.size == MPI.COMM_WORLD.rank:
            do_file[i] = True

    for i in range(len(sorted_files)):
        if do_file[i]:
            permanent_file = sorted_files[i]
            tmp_file = permanent_file.replace('.h5', '_tmp.h5')
            print('merging {}'.format(permanent_file))

            #Copy virtual -> permanent
            with h5py.File(permanent_file, 'r') as vf:
                with h5py.File(tmp_file, 'w') as mf:
                    for k, attr in vf.attrs.items():
                        #adds handler_name, set_number, writes
                        mf.attrs[k] = attr

                    vf.copy('scales', mf)
                    scale_group = mf['scales']

                    task_group = mf.create_group('tasks')
                    for k in vf['tasks'].keys():
                        dset = task_group.create_dataset(k, data=vf['tasks'][k])
                        vf_dset = vf['tasks/{}'.format(k)] 
                        for attr in ['global_shape', 'start', 'count', 'task_number', 'constant', 'grid_space', 'scales']:
                            dset.attrs[attr] = vf_dset.attrs[attr]
                        for i, d in enumerate(vf_dset.dims):
                            dset.dims[i].label = d.label
                            for scalename in d:
                                if scalename == '':
                                    if d.label == 'constant':
                                        continue
                                    else:
                                        scalename = '{}_{}'.format(d.label, k)
                                        scale_group.create_dataset(scalename, data=d[0])
                                scale = mf['scales'][scalename]
                                dset.dims.create_scale(scale, scalename)
                                dset.dims[i].attach_scale(scale)
            os.remove(permanent_file)
            os.rename(tmp_file, permanent_file)

            if cleanup:
                folder = permanent_file.replace('.h5', '/')
                if os.path.isdir(folder):
                    partial_files = glob.glob('{}/*.h5'.format(folder))
                    for pf in partial_files:
                        os.remove(pf)
                    os.rmdir(folder)
