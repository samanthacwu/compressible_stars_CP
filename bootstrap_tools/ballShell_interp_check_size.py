"""
Dedalus script for Boussinesq convection in a sphere (spherical, rotating, rayleigh-benard convection).

This script is functional with and dependent on commit 7eda513 of the d3_eval_backup branch of dedalus (https://github.com/DedalusProject/dedalus).
This script is also functional with and dependent on commit 7eac019f of the d3 branch of dedalus_sphere (https://github.com/DedalusProject/dedalus_sphere/tree/d3).
While the inputs are Ra, Ek, and Pr, the control parameters are the modified Rayleigh number (Ram), the Prandtl number (Pr) and the Ekman number (Ek).
Ram = Ra * Ek / Pr, where Ra is the traditional Rayleigh number.

Usage:
    ballShell_interp_check_size.py <root_dir> --res_fracB=<f> --res_fracS=<f> --mesa_file=<f> [options]
    ballShell_interp_check_size.py <root_dir> --L_frac=<f> --N_fracB=<f> --N_fracS=<f> --mesa_file=<f> [options]

Options:
    --start_check_folder=<f>   Path to the checkpoint folder to start with; use final_checkpoint otherwise.
    --L=<L>                    initial resolution
    --NB=<N>                    initial resolution
    --NS=<N>                    initial resolution
    --node_size=<n>            Size of node; use to reduce memory constraints for large files
"""
import pathlib
from fractions import Fraction
import time
from collections import OrderedDict

import numpy as np
import h5py
from docopt import docopt
from mpi4py import MPI

from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic
from dedalus.tools import logging
from dedalus.tools.parsing import split_equation
from dedalus.extras.flow_tools import GlobalArrayReducer
from dedalus.tools.config import config
from dedalus.tools.general import natural_sort
config['linear algebra']['MATRIX_FACTORIZER'] = 'SuperLUNaturalFactorizedTranspose'
import dedalus_sphere

from output.averaging    import VolumeAverager, EquatorSlicer, PhiAverager
from output.writing      import ScalarWriter,  MeridionalSliceWriter, EquatorialSliceWriter, SphericalShellWriter

import logging
logger = logging.getLogger(__name__)
matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.WARNING) 




def merge_distributed_set(set_path, cleanup=False):
    """
    Merge a distributed analysis set from a FileHandler.

    Parameters
    ----------
    set_path : str of pathlib.Path
        Path to distributed analysis set folder
    cleanup : bool, optional
        Delete distributed files after merging (default: False)

    """
    set_path = pathlib.Path(set_path)
    logger.info("Merging set {}".format(set_path))

    set_stem = set_path.stem
    proc_paths = set_path.glob("{}_p*.h5".format(set_stem))
    proc_paths = natural_sort(proc_paths)
    joint_path = set_path.parent.joinpath("{}.h5".format(set_stem))

    # Create joint file, overwriting if it already exists
    with h5py.File(str(joint_path), mode='w') as joint_file:
        # Setup joint file based on first process file (arbitrary)
        merge_setup(joint_file, proc_paths[0])
        # Merge data from all process files
        for proc_path in proc_paths:
            merge_data(joint_file, proc_path)
    # Cleanup after completed merge, if directed
    if cleanup:
        for proc_path in proc_paths:
            proc_path.unlink()
        set_path.rmdir()


def merge_setup(joint_file, proc_path):
    """
    Merge HDF5 setup from part of a distributed analysis set into a joint file.

    Parameters
    ----------
    joint_file : HDF5 file
        Joint file
    proc_path : str or pathlib.Path
        Path to part of a distributed analysis set

    """
    proc_path = pathlib.Path(proc_path)
    logger.info("Merging setup from {}".format(proc_path))

    with h5py.File(str(proc_path), mode='r') as proc_file:
        # File metadata
        try:
            joint_file.attrs['set_number'] = proc_file.attrs['set_number']
        except KeyError:
            joint_file.attrs['set_number'] = proc_file.attrs['file_number']
        joint_file.attrs['handler_name'] = proc_file.attrs['handler_name']
        try:
            joint_file.attrs['writes'] = writes = proc_file.attrs['writes']
        except KeyError:
            joint_file.attrs['writes'] = writes = len(proc_file['scales']['write_number'])
        # Copy scales (distributed files all have global scales)
        proc_file.copy('scales', joint_file)
        # Tasks
        joint_tasks = joint_file.create_group('tasks')
        proc_tasks = proc_file['tasks']
        for taskname in proc_tasks:
            # Setup dataset with automatic chunking
            proc_dset = proc_tasks[taskname]
            spatial_shape = proc_dset.attrs['global_shape']
            joint_shape = (writes,) + tuple(spatial_shape)
            joint_dset = joint_tasks.create_dataset(name=proc_dset.name,
                                                    shape=joint_shape,
                                                    dtype=proc_dset.dtype,
                                                    chunks=True)
            # Dataset metadata
            joint_dset.attrs['task_number'] = proc_dset.attrs['task_number']
            joint_dset.attrs['constant'] = proc_dset.attrs['constant']
            joint_dset.attrs['grid_space'] = proc_dset.attrs['grid_space']
            joint_dset.attrs['scales'] = proc_dset.attrs['scales']
            # Dimension scales
            for i, proc_dim in enumerate(proc_dset.dims):
                joint_dset.dims[i].label = proc_dim.label
#                for scalename in proc_dim:
#                    scale = joint_file['scales'][scalename]
#                    joint_dset.dims.create_scale(scale, scalename)
#                    joint_dset.dims[i].attach_scale(scale)


def merge_data(joint_file, proc_path):
    """
    Merge data from part of a distributed analysis set into a joint file.

    Parameters
    ----------
    joint_file : HDF5 file
        Joint file
    proc_path : str or pathlib.Path
        Path to part of a distributed analysis set

    """
    proc_path = pathlib.Path(proc_path)
    logger.info("Merging data from {}".format(proc_path))

    with h5py.File(str(proc_path), mode='r') as proc_file:
        for taskname in proc_file['tasks']:
            joint_dset = joint_file['tasks'][taskname]
            proc_dset = proc_file['tasks'][taskname]
            # Merge across spatial distribution
            start = proc_dset.attrs['start']
            count = proc_dset.attrs['count']
            spatial_slices = tuple(slice(s, s+c) for (s,c) in zip(start, count))
            # Merge maintains same set of writes
            slices = (slice(None),) + spatial_slices
            joint_dset[slices] = proc_dset[:]



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
mesh    = None

# Bases
c       = coords.SphericalCoordinates('φ', 'θ', 'r')
d       = distributor.Distributor((c,), mesh=mesh)
b2B       = basis.BallBasis(c, (2*(new_LmaxB+2), new_LmaxB+1, new_NmaxB+1), radius=r_inner, dtype=dtype, dealias=(dealias, dealias, dealias))
b2S       = basis.SphericalShellBasis(c, (2*(new_LmaxS+2), new_LmaxS+1, new_NmaxS+1), radii=(r_inner, r_outer), dtype=dtype, dealias=(dealias, dealias, dealias))
φB2, θB2, rB2    = b2B.local_grids((dealias, dealias, dealias))
φBg2, θBg2, rBg2 = b2B.global_grids((dealias, dealias, dealias))
φS2,  θS2,  rS2  = b2S.local_grids((dealias, dealias, dealias))
φSg2, θSg2, rSg2 = b2S.global_grids((dealias, dealias, dealias))

b1B            = basis.BallBasis(c, (2*(LmaxB+2), LmaxB+1, NmaxB+1), radius=r_inner, dtype=dtype, dealias=(dealias, dealias, dealias))
b1S             = basis.SphericalShellBasis(c, (2*(LmaxS+2), LmaxS+1, NmaxS+1), radii=(r_inner, r_outer), dtype=dtype, dealias=(dealias, dealias, dealias))
φB1, θB1, rB1    = b1B.local_grids( (L_fracB, L_fracB, N_fracB))
φBg1, θBg1, rBg1 = b1B.global_grids((L_fracB, L_fracB, N_fracB))
φS1,  θS1,  rS1  = b1S.local_grids( (L_fracS, L_fracS, N_fracS))
φSg1, θSg1, rSg1 = b1S.global_grids((L_fracS, L_fracS, N_fracS))

if args['--start_check_folder'] is None:
    check_folder = '{:s}/final_checkpoint/final_checkpoint_s1/'.format(root_dir)
else:
    check_folder = '{:s}'.format(args['--start_check_folder'])
logger.info('merging old checkpoint')
merge_distributed_set(check_folder)
check_file = check_folder[:-1] + '.h5'

uB1 = field.Field(dist=d, bases=(b1B,), tensorsig=(c,), dtype=dtype)
sB1 = field.Field(dist=d, bases=(b1B,), dtype=dtype)
uB2 = field.Field(dist=d, bases=(b2B,), tensorsig=(c,), dtype=dtype)
sB2 = field.Field(dist=d, bases=(b2B,), dtype=dtype)
uS1 = field.Field(dist=d, bases=(b1S,), tensorsig=(c,), dtype=dtype)
sS1 = field.Field(dist=d, bases=(b1S,), dtype=dtype)
uS2 = field.Field(dist=d, bases=(b2S,), tensorsig=(c,), dtype=dtype)
sS2 = field.Field(dist=d, bases=(b2S,), dtype=dtype)

import h5py

with h5py.File(check_file, 'r') as f:
    [print(k) for k in f.keys()]
    if L_fracB >= 1 or N_fracB >= 1:
        uB2['c'][:, :(LmaxB+2), :LmaxB+1, :NmaxB+1] = f['tasks']['uB'][-1,:]
        sB2['c'][   :(LmaxB+2), :LmaxB+1, :NmaxB+1] = f['tasks']['s1B'][-1,:]
        uS2['c'][:, :(LmaxS+2), :LmaxS+1, :NmaxS+1] = f['tasks']['uS'][-1,:]
        sS2['c'][   :(LmaxS+2), :LmaxS+1, :NmaxS+1] = f['tasks']['s1S'][-1,:]
    else:
        uB1['c'] = f['tasks']['uB'][-1,:]
        sB1['c'] = f['tasks']['s1B'][-1,:]
        uS1['c'] = f['tasks']['uS'][-1,:]
        sS1['c'] = f['tasks']['s1S'][-1,:]
        for f in [uB1, sB1]:
            f.require_scales((L_fracB, L_fracB, N_fracB))
            f['g']
            f['c']
        for f in [uS1, sS1]:
            f.require_scales((L_fracS, L_fracS, N_fracS))
            f['g']
            f['c']
        uB2['c'] = uB1['c'][:, :(new_LmaxB+2), :new_LmaxB+1, :new_NmaxB+1]
        sB2['c'] = sB1['c'][   :(new_LmaxB+2), :new_LmaxB+1, :new_NmaxB+1]
        uS2['c'] = uS1['c'][:, :(new_LmaxS+2), :new_LmaxS+1, :new_NmaxS+1]
        sS2['c'] = sS1['c'][   :(new_LmaxS+2), :new_LmaxS+1, :new_NmaxS+1]


check_str = 'checkpoint_LB{:.2f}_NB{:.2f}_LS{:.2f}_NS{:.2f}'.format(L_fracB, N_fracB, L_fracS, N_fracS)
out_dir='{:s}/{:s}/'.format(root_dir, check_str)
import os
if not os.path.exists('{:s}/'.format(out_dir)):
    os.makedirs('{:s}/'.format(out_dir))

with h5py.File('{:s}/{:s}_s1.h5'.format(out_dir[:-1], check_str,), 'w') as f:
    print('{:s}/{:s}_s1.h5'.format(out_dir[:-1], check_str))
    task_group = f.create_group('tasks')
    f['tasks']['uB']  = np.expand_dims(uB2['g'], axis=0)
    f['tasks']['s1B'] = np.expand_dims(sB2['g'], axis=0)
    f['tasks']['uS']  = np.expand_dims(uS2['g'], axis=0)
    f['tasks']['s1S'] = np.expand_dims(sS2['g'], axis=0)

    f['rBg'] = rBg2
    f['φBg'] = φBg2
    f['θBg'] = θBg2
    f['rSg'] = rSg2
    f['φSg'] = φSg2
    f['θSg'] = θSg2
     
