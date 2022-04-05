#!/usr/bin/python3
import os
import shutil

dirs = ['star', 'eigenvalues', 'evp_matrices',]
ivp_dirs = ['profiles', 'scalars', 'slices', 'checkpoint', 'final_checkpoint', 'wave_shell_slices']

for d in dirs + ivp_dirs:
    path = './{:s}/'.format(d)
    if os.path.exists(path):
        print('removing {}'.format(path))
        shutil.rmtree(path)
