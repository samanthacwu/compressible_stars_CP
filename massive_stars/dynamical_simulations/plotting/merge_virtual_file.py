"""
This script merges a virtual file so that it can be scp'd off a supercomputer and used in visualization locally.
"""
import os
import glob
import shutil

import h5py
import numpy as np
from mpi4py import MPI

file_path='../ballShell_AN_sponge_Re1e3_128x64x96+96/slices/slices_s1.h5'
out_file_path='../ballShell_AN_sponge_Re1e3_128x64x96+96/merged_slices_s1.h5'

if MPI.COMM_WORLD.rank == 0:

    with h5py.File(file_path, 'r') as vf:
        with h5py.File(out_file_path, 'w') as mf:

            scale_group = mf.create_group('scales')
            for k in vf['scales'].keys():
                scale_group.create_dataset(k, data=vf['scales'][k])

            task_group = mf.create_group('tasks')
            for k in vf['tasks'].keys():
                task_group.create_dataset(k, data=vf['tasks'][k])

