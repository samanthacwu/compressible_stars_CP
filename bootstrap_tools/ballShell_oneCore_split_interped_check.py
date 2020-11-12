"""
Dedalus script for Boussinesq convection in a sphere (spherical, rotating, rayleigh-benard convection).

This script is functional with and dependent on commit 7eda513 of the d3_eval_backup branch of dedalus (https://github.com/DedalusProject/dedalus).
This script is also functional with and dependent on commit 7eac019f of the d3 branch of dedalus_sphere (https://github.com/DedalusProject/dedalus_sphere/tree/d3).
While the inputs are Ra, Ek, and Pr, the control parameters are the modified Rayleigh number (Ram), the Prandtl number (Pr) and the Ekman number (Ek).
Ram = Ra * Ek / Pr, where Ra is the traditional Rayleigh number.

Usage:
    ballShell_split_interped_check.py <root_dir> --res_fracB=<f> --res_fracS_<f> --mesh=<m> --mesa_file=<f> [options]
    ballShell_split_interped_check.py <root_dir> --L_frac=<f> --N_fracB=<f> --N_fracS=<f> --mesh=<m> --mesa_file=<f> [options]

Options:
    --L=<L>                    initial resolution
    --NB=<N>                    initial resolution
    --NS=<N>                    initial resolution
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
if args['--res_fracB'] is not None:
    L_fracB = N_fracB = float(Fraction(args['--res_fracB']))
    L_fracS = N_fracS = float(Fraction(args['--res_fracS']))
else:
    L_fracB = L_fracS = float(Fraction(args['--L_frac']))
    N_fracB = float(Fraction(args['--N_fracB']))
    N_fracS = float(Fraction(args['--N_fracS']))

res_strings = root_dir.split('Re')[-1].split('_')[1:3]
res_strB, res_strS = res_strings
resolutions = [[], []]
if args['--L'] is None or args['--NS'] is None or args['--NB'] is None:
    for r in res_strB.split('x'):
        if r[-1] == '/':
            resolutions[0].append(int(r[:-1]))
        else:
            resolutions[0].append(int(r))
    for r in res_strS.split('x'):
        if r[-1] == '/':
            resolutions[1].append(int(r[:-1]))
        else:
            resolutions[1].append(int(r))
        
else:
    resolutions[0] = [int(args['--L']), int(args['--NB'])]
    resolutions[1] = [int(args['--L']), int(args['--NS'])]
LmaxB, NmaxB = resolutions[0]
LmaxS, NmaxS = resolutions[1]


new_LmaxB = int((LmaxB+2)*L_fracB) - 2
new_LmaxS = int((LmaxS+2)*L_fracS) - 2
new_NmaxB = int((NmaxB+1)*N_fracB) - 1
new_NmaxS = int((NmaxS+1)*N_fracS) - 1
dealias = 1
dtype   = np.float64
with h5py.File(args['--mesa_file'], 'r') as f:
    r_inner = f['r_inner'][()]
    r_outer = f['r_outer'][()]
mesh    = args['--mesh'].split(',')
mesh = [int(m) for m in mesh]


# Bases
c       = coords.SphericalCoordinates('φ', 'θ', 'r')
d       = distributor.Distributor((c,), mesh=None)
b2B       = basis.BallBasis(c, (2*(new_LmaxB+2), new_LmaxB+1, new_NmaxB+1), radius=r_inner, dtype=dtype, dealias=(dealias, dealias, dealias))
b2S       = basis.SphericalShellBasis(c, (2*(new_LmaxS+2), new_LmaxS+1, new_NmaxS+1), radii=(r_inner, r_outer), dtype=dtype, dealias=(dealias, dealias, dealias))
φB2, θB2, rB2    = b2B.local_grids((dealias, dealias, dealias))
φBg2, θBg2, rBg2 = b2B.global_grids((dealias, dealias, dealias))
φS2,  θS2,  rS2  = b2S.local_grids((dealias, dealias, dealias))
φSg2, θSg2, rSg2 = b2S.global_grids((dealias, dealias, dealias))

uB2 = field.Field(dist=d, bases=(b2B,), tensorsig=(c,), dtype=dtype)
sB2 = field.Field(dist=d, bases=(b2B,), dtype=dtype)
pB2 = field.Field(dist=d, bases=(b2B,), dtype=dtype)
uS2 = field.Field(dist=d, bases=(b2S,), tensorsig=(c,), dtype=dtype)
sS2 = field.Field(dist=d, bases=(b2S,), dtype=dtype)
pS2 = field.Field(dist=d, bases=(b2S,), dtype=dtype)


check_str = 'checkpoint_LB{:.2f}_NB{:.2f}_LS{:.2f}_NS{:.2f}'.format(L_fracB, N_fracB, L_fracS, N_fracS)
out_dir = '{:s}/{:s}'.format(root_dir, check_str)

node_size = args['--node_size']
if node_size is not None: 
    node_size = int(node_size)
else:
    node_size = 1

import sys

sys.stdout.flush()
with h5py.File('{:s}/{:s}_s1.h5'.format(out_dir, check_str), 'r') as f:
    sB2['c'] = f['tasks']['s1B'][()]
    pB2['c'] = f['tasks']['pB'][()]
    uB2['c'] = f['tasks']['uB'][()]
    sS2['c'] = f['tasks']['s1S'][()]
    pS2['c'] = f['tasks']['pS'][()]
    uS2['c'] = f['tasks']['uS'][()]

split_out_dir = '{:s}/{:s}_s1/'.format(out_dir, check_str)
import os
if d.comm_cart.rank == 0:
    print('root node checking dir')
    sys.stdout.flush()
    if not os.path.exists('{:s}/'.format(split_out_dir)):
        os.makedirs('{:s}/'.format(split_out_dir))

num_procs = np.prod(mesh)
nφB_per = int((new_LmaxB+2)/mesh[0])
nφS_per = int((new_LmaxS+2)/mesh[0])
nθB_per = int((new_LmaxB+2)/mesh[1])
nθS_per = int((new_LmaxS+2)/mesh[1])

for i in range(num_procs):
    print('creating split file {}'.format(i))
    sys.stdout.flush()
    if i + 1 % mesh[-1] == 0:
        my_nθB_per = nθB_per - 1
        my_nθS_per = nθS_per - 1
    else:
        my_nθB_per = nθB_per
        my_nθS_per = nθS_per

    φ_ind = int(np.floor(i/mesh[-1]))
    θ_ind = i % mesh[-1]

    sliceB = (slice(φ_ind*nφB_per, (φ_ind+1)*nφB_per, 1), slice(θ_ind*nθB_per, (θ_ind)*nθB_per+my_nθB_per, 1), slice(0, new_NmaxB+1, 1))
    sliceS = (slice(φ_ind*nφS_per, (φ_ind+1)*nφS_per, 1), slice(θ_ind*nθS_per, (θ_ind)*nθS_per+my_nθS_per, 1), slice(0, new_NmaxS+1, 1))
    vec_sliceB = (slice(0,3,1),) + sliceB
    vec_sliceS = (slice(0,3,1),) + sliceS
    with h5py.File('{:s}/{:s}_s1_p{}.h5'.format(split_out_dir, check_str, i), 'w') as f:
        task_group = f.create_group('tasks')
        f['tasks']['uB']  = np.expand_dims(uB2['c'][vec_sliceB], axis=0)
        f['tasks']['s1B'] = np.expand_dims(sB2['c'][sliceB], axis=0)
        f['tasks']['pB']  = np.expand_dims(pB2['c'][sliceB], axis=0)
        f['tasks']['uS']  = np.expand_dims(uS2['c'][vec_sliceS], axis=0)
        f['tasks']['s1S'] = np.expand_dims(sS2['c'][sliceS], axis=0)
        f['tasks']['pS']  = np.expand_dims(pS2['c'][sliceS], axis=0)
