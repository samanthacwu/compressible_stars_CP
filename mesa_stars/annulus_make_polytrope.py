"""
Turns a MESA .data file of a massive star into a .h5 file of NCCs for a d3 run.

Usage:
    make_coreCZ_nccs.py [options]

Options:
    --nz=<N>        Maximum radial coefficients [default: 64]
    --file=<f>      Path to MESA log file [default: MESA_Models_Dedalus_Full_Sphere/LOGS/h1_0.6.data]
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
read_file = args['--file']
out_dir  = 'annulus_polytrope'
out_file = '{:s}/annulus_nccs_{}.h5'.format(out_dir, nz)
if not os.path.exists('{:s}'.format(out_dir)):
    os.mkdir('{:s}'.format(out_dir))


L = 1
rbot = 0.5
z_basis = de.Chebyshev('r', nz, interval = [rbot, L + rbot], dealias=1)
domain = de.Domain([z_basis,], grid_dtype=np.float64, mesh=None)
rg = domain.grid(-1)


n_rho = 1
gamma = 5/3
gradT = np.exp(n_rho*(1 - gamma)) - 1

T0   = 1 + gradT*(rg-rbot)
rho0 = T0**(1/(gamma - 1))


### Log Density
N = 10
ln_rho_field  = domain.new_field()
ln_rho = np.log(rho0)
ln_rho_field['g'] = ln_rho
ln_rho_field['c'][N:] = 0
plot_ncc_figure(rg.flatten(), (-1)+ln_rho.flatten(), (-1)+ln_rho_field['g'].flatten(), N, ylabel=r"$\ln\rho - 1$", fig_name="ln_rho", out_dir=out_dir)

grad_ln_rho_field  = domain.new_field()
ln_rho_field.differentiate('r', out=grad_ln_rho_field)
grad_ln_rho = np.gradient(ln_rho,rg)
plot_ncc_figure(rg.flatten(), grad_ln_rho.flatten(), grad_ln_rho_field['g'].flatten(), N, ylabel=r"$\nabla\ln\rho$", fig_name="grad_ln_rho", out_dir=out_dir)

### log (temp)
N = 10
ln_T_field  = domain.new_field()
ln_T = np.log(T0)
ln_T_field['g'] = ln_T
ln_T_field['c'][N:] = 0
plot_ncc_figure(rg.flatten(), (-1)+ln_T.flatten(), (-1)+ln_T_field['g'].flatten(), N, ylabel=r"$\ln(T) - 1$", fig_name="ln_T", out_dir=out_dir)

grad_ln_T_field  = domain.new_field()
ln_T_field.differentiate('r', out=grad_ln_T_field)
grad_ln_T = np.gradient(ln_T,rg)
plot_ncc_figure(rg.flatten(), grad_ln_T.flatten(), grad_ln_T_field['g'].flatten(), N, ylabel=r"$\nabla\ln(T)$", fig_name="grad_ln_T", out_dir=out_dir)

### Temperature
N = 5
T_field = domain.new_field()
T_field['g'] = T0
T_field['c'][N:] = 0
plot_ncc_figure(rg.flatten(), T0.flatten(), T_field['g'].flatten(), N, ylabel=r"$T$", fig_name="T", out_dir=out_dir)

### effective heating / (rho * T)
N = 10
L = 1 - 4*(rg-(0.5+rbot))**2
H_eff = -8*(rg-(0.5+rbot))/rho0/T0/(2*np.pi*rg)

H_field = domain.new_field()
H_field['g'] = H_eff
H_field['c'][N:] = 0
plot_ncc_figure(rg.flatten(), H_eff.flatten(), H_field['g'].flatten(), N, ylabel=r"$(H_{eff}/(\rho T))$ (nondimensional)", fig_name="H_eff", out_dir=out_dir, zero_line=True)


with h5py.File('{:s}'.format(out_file), 'w') as f:
    f['r']     = rg
    f['T']     = T_field['g']
    f['H_eff'] = H_field['g']
    f['ln_ρ']  = ln_rho_field['g'] 
    f['ln_T']  = ln_T_field['g']
