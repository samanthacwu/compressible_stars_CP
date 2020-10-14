"""
Dedalus script for Boussinesq convection in a sphere (spherical, rotating, rayleigh-benard convection).

This script is functional with and dependent on commit 7eda513 of the d3_eval_backup branch of dedalus (https://github.com/DedalusProject/dedalus).
This script is also functional with and dependent on commit 7eac019f of the d3 branch of dedalus_sphere (https://github.com/DedalusProject/dedalus_sphere/tree/d3).
While the inputs are Ra, Ek, and Pr, the control parameters are the modified Rayleigh number (Ram), the Prandtl number (Pr) and the Ekman number (Ek).
Ram = Ra * Ek / Pr, where Ra is the traditional Rayleigh number.

Usage:
    interp_check_size.py <root_dir> --res_frac=<f> [options]
    interp_check_size.py <root_dir> --L_frac=<f> --N_frac=<f> [options]

Options:
    --start_check_folder=<f>   Path to the checkpoint folder to start with; use final_checkpoint otherwise.
    --L=<L>                    initial resolution
    --N=<N>                    initial resolution
    --node_size=<n>            Size of node; use to reduce memory constraints for large files
"""
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
mesh = None

# Bases
c       = coords.SphericalCoordinates('φ', 'θ', 'r')
d       = distributor.Distributor((c,), mesh=mesh)
b2       = basis.BallBasis(c, (2*(new_Lmax+2), new_Lmax+1, new_Nmax+1), radius=radius, dtype=dtype, dealias=(dealias, dealias, dealias))
φ2, θ2, r2    = b2.local_grids((dealias, dealias, dealias))
φg2, θg2, rg2 = b2.global_grids((dealias, dealias, dealias))

b1       = basis.BallBasis(c, (2*(Lmax+2), Lmax+1, Nmax+1), radius=radius, dtype=dtype, dealias=(dealias, dealias, dealias))
φ1, θ1, r1    = b1.local_grids( (L_frac, L_frac, N_frac))
φg1, θg1, rg1 = b1.global_grids((L_frac, L_frac, N_frac))

if args['--start_check_folder'] is None:
    check_folder = '{:s}/final_checkpoint/final_checkpoint_s1/'.format(root_dir)
else:
    check_folder = '{:s}'.format(args['--start_check_folder'])
from dedalus.tools import post
logger.info('merging old checkpoint')
post.merge_distributed_set(check_folder)
check_file = check_folder[:-1] + '.h5'

u1 = field.Field(dist=d, bases=(b1,), tensorsig=(c,), dtype=dtype)
s1 = field.Field(dist=d, bases=(b1,), dtype=dtype)
u2 = field.Field(dist=d, bases=(b2,), tensorsig=(c,), dtype=dtype)
s2 = field.Field(dist=d, bases=(b2,), dtype=dtype)

import h5py

with h5py.File(check_file, 'r') as f:
    if L_frac >= 1:
        u2['c'][:, :(Lmax+2), :Lmax+1, :Nmax+1] = f['tasks']['u'][-1,:]
        s2['c'][:(Lmax+2), :Lmax+1, :Nmax+1] = f['tasks']['s1'][-1,:]
    else:
        u1['c'] = f['tasks']['u'][-1,:]
        s1['c'] = f['tasks']['s1'][-1,:]
        for f in [u1, s1]:
            f.require_scales((L_frac, L_frac, N_frac))
            f['g']
            f['c']
        u2['c'] = u1['c'][:, :(new_Lmax+2), :new_Lmax+1, :new_Nmax+1]
        s2['c'] = s1['c'][:(new_Lmax+2), :new_Lmax+1, :new_Nmax+1]


out_dir='{:s}/checkpoint_L{:.2f}_N{:.2f}/'.format(root_dir, L_frac, N_frac)
import os
if not os.path.exists('{:s}/'.format(out_dir)):
    os.makedirs('{:s}/'.format(out_dir))

with h5py.File('{:s}/checkpoint_L{:.2f}_N{:.2f}_s1.h5'.format(out_dir[:-1], L_frac, N_frac), 'w') as f:
    print('{:s}/checkpoint_L{:.2f}_N{:.2f}_s1.h5'.format(out_dir[:-1], L_frac, N_frac))
    task_group = f.create_group('tasks')
    f['tasks']['u']  = u2['g']
    f['tasks']['s1'] = s2['g']

    f['rg'] = rg2
    f['φg'] = φg2
    f['θg'] = θg2
     
