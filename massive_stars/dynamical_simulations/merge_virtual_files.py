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
handlers = ['slices',]

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

            #Copy virtual -> permanent
            with h5py.File(permanent_file, 'r') as vf:
                with h5py.File(tmp_file, 'w') as mf:

                    for k, attr in vf.attrs.items():
                        mf.attrs[k] = attr

                    scale_group = mf.create_group('scales')
                    for k in vf['scales'].keys():
                        scale_group.create_dataset(k, data=vf['scales'][k])
                        print(dir(vf['scales'][k]))
                        print(vf['scales'][k].write_direct)

                    task_group = mf.create_group('tasks')
                    for k in vf['tasks'].keys():
                        dset = task_group.create_dataset(k, data=vf['tasks'][k])
                        vf_dset = vf['tasks/{}'.format(k)] 
                        for attr in ['global_shape', 'start', 'count', 'task_number', 'constant', 'grid_space', 'scales']:
                            dset.attrs[attr] = vf_dset.attrs[attr]
                        for i, d in enumerate(vf_dset.dims):
                            dset.dims[i].label = d.label
                            if len(d.keys()) > 0:
                                for k in d.keys():
                                    if len(k) > 0:
                                        scale = scale_group[k]
                                        scale.make_scale(k)
                                        dset.dims[i].attach_scale(scale)
                            else:
                                pass
                                #TODO: figure this out!! - maybe can get it from scales?
    



#            #Replace virtual with permanent
            os.remove(permanent_file)
            os.rename(tmp_file, permanent_file)


            with h5py.File(permanent_file, 'r') as mf:
                print(mf['tasks/s1_eq'])
                for d in mf['tasks/s1_eq'].dims:
                    print(d[0][:])


