#!/usr/bin/python3
import os
import shutil

dirs = ['star', 'eigenvalues', 'evp_matrices']

for d in dirs:
    path = './{:s}/'.format(d)
    if os.path.exists(path):
        print('removing {}'.format(path))
        shutil.rmtree(path)
