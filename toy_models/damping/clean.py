#!/usr/bin/python3
import os
import shutil

dirs = ['star', 'eigenvalues', 'evp_matrices',]
ivp_dirs = ['profiles', 'scalars', 'slices', 'checkpoints', 'final_checkpoint', 'shells']
post_dirs = ['SH_transform_shells', 'SH_wave_flux_spectra', 'snapshots_equatorial']

for d in dirs + ivp_dirs + post_dirs:
    path = './{:s}/'.format(d)
    if os.path.exists(path):
        print('removing {}'.format(path))
        shutil.rmtree(path)
