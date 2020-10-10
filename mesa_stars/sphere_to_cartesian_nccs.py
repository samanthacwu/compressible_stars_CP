"""
Turns a MESA .data file of a massive star into a .h5 file of NCCs for a d3 run.

Usage:
    make_coreCZ_nccs.py <in_file> [options]

Options:
    --nz=<N>        Maximum radial coefficients [default: 64]
"""
import os
import time

import numpy as np
import h5py
from mpi4py import MPI
import mesa_reader as mr
import matplotlib.pyplot as plt

from dedalus import public as de
from docopt import docopt

from astropy import units as u
from astropy import constants
from scipy.signal import savgol_filter
from numpy.polynomial import Chebyshev as Pfit

args = docopt(__doc__)

def plot_ncc_figure(r, mesa_y, dedalus_y, N, ylabel="", fig_name="", out_dir='.', zero_line=False):
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    if zero_line:
        ax1.axhline(0, c='k', lw=0.5)

    ax1.plot(r, mesa_y, label='mesa', c='k', lw=3)
    ax1.plot(r, dedalus_y, label='dedalus', c='red')
    plt.legend(loc='best')
    ax1.set_xlabel('radius/L', labelpad=-3)
    ax1.set_ylabel('{}'.format(ylabel))
    ax1.xaxis.set_ticks_position('top')
    ax1.xaxis.set_label_position('top')

    ax2 = fig.add_subplot(2,1,2)
    difference = np.abs(1 - dedalus_y/mesa_y)
    ax2.plot(r, np.abs(difference).flatten())
    ax2.set_ylabel('abs(1 - dedalus/mesa)')
    ax2.set_xlabel('radius/L')
    ax2.set_yscale('log')
    fig.suptitle('coeff bandwidth = {}'.format(N))
    fig.savefig('{:s}/{}.png'.format(out_dir, fig_name), bbox_inches='tight', dpi=200)


#def load_data(nr1, nr2, r_int, get_dimensions=False):
nz = int(args['--nz'])
read_file = args['<in_file>']
out_dir  = read_file.split('/nccs')[0]
out_file = read_file.split('nccs')[0]+'cartesian_nccs_{}.h5'.format(nz)
if not os.path.exists('{:s}'.format(out_dir)):
    os.mkdir('{:s}'.format(out_dir))

print('moving {} to {}'.format(read_file, out_file))

z_basis = de.Chebyshev('z', nz, interval = [0, 1], dealias=1)
domain = de.Domain([z_basis,], grid_dtype=np.float64, mesh=None)
rg = domain.grid(-1)

with h5py.File('{:s}'.format(read_file), 'r') as f_in:
    with h5py.File('{:s}'.format(out_file), 'w') as f:
        r = f_in['r'][()].flatten()
        for k in f_in.keys():
            
            data_in = np.array(f_in[k][()])
            if data_in.shape == ():
                f[k] = data_in
            else:
                new_field = domain.new_field()
                new_field['g'] = np.interp(rg, r, data_in.flatten())
                f[k] = new_field['g']
