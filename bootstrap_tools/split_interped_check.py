"""
Dedalus script for Boussinesq convection in a sphere (spherical, rotating, rayleigh-benard convection).

This script is functional with and dependent on commit 7eda513 of the d3_eval_backup branch of dedalus (https://github.com/DedalusProject/dedalus).
This script is also functional with and dependent on commit 7eac019f of the d3 branch of dedalus_sphere (https://github.com/DedalusProject/dedalus_sphere/tree/d3).
While the inputs are Ra, Ek, and Pr, the control parameters are the modified Rayleigh number (Ram), the Prandtl number (Pr) and the Ekman number (Ek).
Ram = Ra * Ek / Pr, where Ra is the traditional Rayleigh number.

Usage:
    split_interped_check.py <root_dir> --res_frac=<f> --mesh=<m> [options]
    split_interped_check.py <root_dir> --L_frac=<f> --N_frac=<f> --mesh=<m> [options]

Options:
    --L=<L>                    initial resolution
    --N=<N>                    initial resolution
    --node_size=<n>            Size of node; use to reduce memory constraints for large files
"""
import h5py
from fractions import Fraction
import time
from collections import OrderedDict

import numpy as np
from docopt import docopt
from mpi4py import MPI

from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic
from dedalus.tools import logging
from dedalus.tools.parsing import split_equation
from dedalus.extras.flow_tools import GlobalArrayReducer
from dedalus.tools.config import config
config['linear algebra']['MATRIX_FACTORIZER'] = 'SuperLUNaturalFactorizedTranspose'
import dedalus_sphere

from output.averaging    import VolumeAverager, EquatorSlicer, PhiAverager
from output.writing      import ScalarWriter,  MeridionalSliceWriter, EquatorialSliceWriter, SphericalShellWriter

import logging
logger = logging.getLogger(__name__)
matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.WARNING) 

args   = docopt(__doc__)
root_dir = args['<root_dir>']
if args['--res_frac'] is not None:
    L_frac = N_frac = float(Fraction(args['--res_frac']))
else:
    L_frac = float(Fraction(args['--L_frac']))
    N_frac = float(Fraction(args['--N_frac']))

res_str = root_dir.split('Re')[-1].split('_')[1]
resolutions = []
if args['--L'] is None or args['--N'] is None:
    for r in res_str.split('x'):
        if r[-1] == '/':
            resolutions.append(int(r[:-1]))
        else:
            resolutions.append(int(r))
        
    Lmax, Nmax = resolutions
else:
    Lmax = int(args['--L'])
    Nmax = int(args['--N'])
new_Lmax = int((Lmax+2)*L_frac) - 2
new_Nmax = int((Nmax+1)*N_frac) - 1
print(new_Nmax, new_Lmax)
dealias = 1

dtype     = np.float64
radius = 1
mesh      = args['--mesh']
if mesh is not None:
    mesh = [int(n) for n in mesh.split(',')]

# Bases
c2       = coords.SphericalCoordinates('φ', 'θ', 'r')
d2       = distributor.Distributor((c2,), mesh=mesh)
b2       = basis.BallBasis(c2, (2*(new_Lmax+2), new_Lmax+1, new_Nmax+1), radius=radius, dtype=dtype, dealias=(dealias, dealias, dealias))
φ2, θ2, r2    = b2.local_grids((dealias, dealias, dealias))
φg2, θg2, rg2 = b2.global_grids((dealias, dealias, dealias))

u = field.Field(dist=d2, bases=(b2,), tensorsig=(c2,), dtype=dtype)
s1 = field.Field(dist=d2, bases=(b2,), dtype=dtype)

out_dir='{:s}/checkpoint_L{:.2f}_N{:.2f}/'.format(root_dir, L_frac, N_frac)
import os
if not os.path.exists('{:s}/'.format(out_dir)):
    os.makedirs('{:s}/'.format(out_dir))

node_size = args['--node_size']
if node_size is not None: 
    node_size = int(node_size)
else:
    node_size = 1

import sys
for i in range(node_size):
    if d2.comm_cart.rank % node_size == i:
        print('reading on node rank {}'.format(i))
        sys.stdout.flush()
        with h5py.File('{:s}/checkpoint_L{:.2f}_N{:.2f}_s1.h5'.format(out_dir[:-1], L_frac, N_frac), 'r') as f:
            rg = f['rg'][()]
            φg = f['φg'][()]
            θg = f['θg'][()]

            rgood = np.zeros_like(rg, dtype=bool)
            φgood = np.zeros_like(φg, dtype=bool)
            θgood = np.zeros_like(θg, dtype=bool)
            for rv in r2.flatten():
                if rv in rg:
                    rgood[0, 0, rg.flatten() == rv] = True
            for φv in φ2.flatten():
                if φv in φg:
                    φgood[φg.flatten() == φv, 0, 0] = True
            for θv in θ2.flatten():
                if θv in θg:
                    θgood[0, θg.flatten() == θv, 0] = True
            global_good = rgood*φgood*θgood
            for i in range(3):
                u['g'][i,:] = f['tasks']['u'][()][i][global_good].reshape(u['g'][i,:].shape)
            s1['g'] = f['tasks']['s1'][()][global_good].reshape(s1['g'].shape)
            del global_good
    else:
        for i in range(3):
            u['g'][i,:] = u['g'][i,:]
        s1['g'] = s1['g']
    d2.comm_cart.barrier()

split_out_dir = '{:s}/checkpoint_L{:.2f}_N{:.2f}_s1/'.format(out_dir[:-1], L_frac, N_frac)
import os
if d2.comm_cart.rank == 0:
    if not os.path.exists('{:s}/'.format(split_out_dir)):
        os.makedirs('{:s}/'.format(split_out_dir))
d2.comm_cart.Barrier()

with h5py.File('{:s}/checkpoint_L{:.2f}_N{:.2f}_s1_p{}.h5'.format(split_out_dir, L_frac, N_frac, int(d2.comm_cart.rank)), 'w') as f:
    task_group = f.create_group('tasks')
    f['tasks']['u'] = np.expand_dims(u['c'], axis=0)
    f['tasks']['s1'] = np.expand_dims(s1['c'], axis=0)
