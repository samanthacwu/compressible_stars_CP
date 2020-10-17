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
logger.info('merging old checkpoint')
merge_distributed_set(check_folder)
check_file = check_folder[:-1] + '.h5'

u1 = field.Field(dist=d, bases=(b1,), tensorsig=(c,), dtype=dtype)
s1 = field.Field(dist=d, bases=(b1,), dtype=dtype)
u2 = field.Field(dist=d, bases=(b2,), tensorsig=(c,), dtype=dtype)
s2 = field.Field(dist=d, bases=(b2,), dtype=dtype)

import h5py

with h5py.File(check_file, 'r') as f:
    [print(k) for k in f.keys()]
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
     
