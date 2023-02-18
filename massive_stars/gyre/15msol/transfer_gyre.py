"""
This file reads in gyre eigenfunctions, calculates the velocity and velocity dual basis, and outputs in a clean format so that it's ready to be fed into the transfer function calculation.
"""
import os
import numpy as np
import pygyre as pg

from d3_stars.simulations.star_builder import find_core_cz_radius
from d3_stars.gyre.clean_eig import GyreMSGPostProcessor

plot = False
use_delta_L = True
Lmax = 40
ell_list = np.arange(1, Lmax+1)
for ell in ell_list:
    om_list = np.logspace(-8, -2, 1000) #Hz * 2pi

    pulse_file = 'LOGS/profile47.data.GYRE'
    mesa_LOG = 'LOGS/profile47.data'
    pos_mode_base = './gyre_output/mode_ell{:03d}_m+00_n{:06d}.txt'
    neg_mode_base = pos_mode_base.replace('pos', 'neg')
    pos_files = []
    neg_files = []

    max_n_pg = 100
    if ell <= 5:
        pos_summary_file='gyre_output/summary_ell01-05.txt'.format(ell)
    elif ell <= 10:
        pos_summary_file='gyre_output/summary_ell06-10.txt'.format(ell)
    elif ell <= 20:
        pos_summary_file='gyre_output/summary_ell11-20.txt'.format(ell)
    elif ell <= 40:
        pos_summary_file='gyre_output/summary_ell21-40.txt'.format(ell)
    pos_summary = pg.read_output(pos_summary_file)
    neg_summary_file = None

    #sort eigenvalues by 1/freq
    sorting = np.argsort(pos_summary['freq'].real**(-1))
    pos_summary = pos_summary[sorting]

    good_freqs = []
    counted_n_pgs = []
    for row in pos_summary:
        this_ell = row['l']
        if this_ell != ell: continue
        n_pg = row['n_pg']
        #Check consistency...
        if n_pg >= 0: continue
        if np.abs(n_pg) > max_n_pg: continue
        if n_pg in counted_n_pgs: continue
        counted_n_pgs.append(n_pg)
        pos_files.append(pos_mode_base.format(ell, n_pg))
        good_freqs.append(complex(row['freq']))

    post = GyreMSGPostProcessor(ell, pos_summary_file, pos_files, pulse_file, mesa_LOG,
                  specgrid='OSTAR2002', filters=['Red',],
                  MSG_DIR = os.environ['MSG_DIR'],
                  GRID_DIR=os.path.join('..','gyre-phot','specgrid'),
                  PASS_DIR=os.path.join('..','gyre-phot','passbands'))
    post.sort_eigenfunctions()
    data_dicts = post.evaluate_magnitudes()
    data_dict = post.calculate_duals()
    post.calculate_transfer(plot=plot, use_delta_L=use_delta_L)
