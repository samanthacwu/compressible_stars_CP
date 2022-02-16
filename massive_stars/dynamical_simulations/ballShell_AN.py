"""
d3 script for anelastic convection in a stitched BallBasis and ShellBasis domain.

A config file can be provided which overwrites any number of the options below, but command line flags can also be used if preferred.
Note that options specified in a cnofig file override command line arguments.

Usage:
    ballShell_AN.py [options]
    ballShell_AN.py <config> [options]

Options:
    --Re=<Re>            The Reynolds number of the numerical diffusivities [default: 2e2]
    --Pr=<Prandtl>       The Prandtl number  of the numerical diffusivities [default: 1]
    --ntheta=<res>       Number of theta grid points (Lmax+1)   [default: 16]
    --nrB=<res>          Number of radial grid points in ball (Nmax+1)   [default: 24]
    --nrS=<res>          Number of radial grid points in shell (Nmax+1)   [default: 8]
    --sponge             If flagged, add a damping layer in the shell that damps out waves.

    --wall_hours=<t>     Max number of wall hours to run simulation for [default: 24]
    --buoy_end_time=<t>  Max number of buoyancy time units to simulate [default: 1e5]

    --mesh=<n,m>         The processor mesh over which to distribute the cores

    --RK222              Use RK222 (default is SBDF2)
    --SBDF4              Use SBDF4 (default is SBDF2)
    --safety=<s>         Timestep CFL safety factor [default: 0.2]
    --CFL_max_r=<r>      zero out velocities above this radius value for CFL

    --ncc_file=<f>      path to a .h5 file of ICCs, curated from a MESA model
    --restart=<chk_f>    path to a checkpoint file to restart from
    --A0=<A>             Amplitude of random noise initial conditions [default: 1e-6]

    --label=<label>      A label to add to the end of the output directory

    --rotation_time=<t>  Rotation timescale, in days (if ncc_file is not None) or sim units (for polytrope)
"""
import os
import time
import sys
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
from docopt import docopt
from configparser import ConfigParser
import dedalus.public as d3
from mpi4py import MPI

import logging
logger = logging.getLogger(__name__)

# Define smooth Heaviside functions
from scipy.special import erf
def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
    return -(one_to_zero(*args, **kwargs) - 1)

# Read options
args   = docopt(__doc__)
if args['<config>'] is not None: 
    config_file = Path(args['<config>'])
    config = ConfigParser()
    config.read(str(config_file))
    for n, v in config.items('parameters'):
        for k in args.keys():
            if k.split('--')[-1].lower() == n:
                if v == 'true': v = True
                args[k] = v

# Parameters
nθ = int(args['--ntheta'])
nφ = int(2*nθ)
nrB = int(args['--nrB'])
nrS = int(args['--nrS'])
resolutionB = (nφ, nθ, nrB)
resolutionS = (nφ, nθ, nrS)
L_dealias = N_dealias = dealias = 1.5
dtype = np.float64
Re  = float(args['--Re'])
Pr  = 1
Pe  = Pr*Re
ncc_file = args['--ncc_file']
wall_hours = float(args['--wall_hours'])
buoy_end_time = float(args['--buoy_end_time'])
sponge = args['--sponge']

# rotation
do_rotation = False
rotation_time = args['--rotation_time']
if rotation_time is not None:
    do_rotation = True
    rotation_time = float(rotation_time)
    dimensional_Ω = 2*np.pi / rotation_time  #radians / day [in MESA units]

# Initial conditions
restart = args['--restart']
A0 = float(args['--A0'])

# Timestepper
if args['--SBDF4']:
    ts = d3.SBDF4
    timestepper_history = [0, 1, 2, 3]
elif args['--RK222']:
    ts = d3.RK222
    timestepper_history = [0, ]
else:
    ts = d3.SBDF2
    timestepper_history = [0, 1,]
hermitian_cadence = 100
safety = float(args['--safety'])
CFL_max_r = args['--CFL_max_r']
if CFL_max_r is not None:
    CFL_max_r = float(CFL_max_r)
else:
    CFL_max_r = np.inf

# Processor mesh
mesh = args['--mesh']
ncpu = MPI.COMM_WORLD.size
if mesh is not None:
    mesh = mesh.split(',')
    mesh = [int(mesh[0]), int(mesh[1])]
else:
    log2 = np.log2(ncpu)
    if log2 == int(log2):
        mesh = [int(2**np.ceil(log2/2)),int(2**np.floor(log2/2))]
    logger.info("running on processor mesh={}".format(mesh))

# Output directory
out_dir = './' + sys.argv[0].split('.py')[0]
if sponge:
    out_dir += '_sponge'
if ncc_file is None:
    out_dir += '_polytrope'
if do_rotation:
    out_dir += '_rotation{}'.format(rotation_time)

out_dir += '_Re{}_{}x{}x{}+{}'.format(args['--Re'], nφ, nθ, nrB, nrS)
if args['--label'] is not None:
    out_dir += '_{:s}'.format(args['--label'])
logger.info('saving data to {:s}'.format(out_dir))
if MPI.COMM_WORLD.rank == 0:
    if not os.path.exists('{:s}/'.format(out_dir)):
        os.makedirs('{:s}/'.format(out_dir))

# Read in domain bound values
if ncc_file is not None:
    with h5py.File(args['--ncc_file'], 'r') as f:
        r_inner = f['r_inner'][()]
        r_outer = f['r_outer'][()]
else:
    r_inner = 1.1
    r_outer = 1.5
logger.info('r_inner: {:.2f} / r_outer: {:.2f}'.format(r_inner, r_outer))

# Bases
coords  = d3.SphericalCoordinates('φ', 'θ', 'r')
dist    = d3.Distributor((coords,), mesh=mesh, dtype=dtype)
ball_basis  = d3.BallBasis(coords, resolutionB, radius=r_inner, dtype=dtype, dealias=(L_dealias, L_dealias, N_dealias))
shell_basis = d3.ShellBasis(coords, resolutionS, radii=(r_inner, r_outer), dtype=dtype, dealias=(L_dealias, L_dealias, N_dealias))
top_ball_S2_basis  = ball_basis.S2_basis(radius=r_inner)
bot_shell_S2_basis = shell_basis.S2_basis(radius=r_inner)
top_shell_S2_basis = shell_basis.S2_basis(radius=r_outer)
φB,  θB,  rB     = ball_basis.local_grids(ball_basis.dealias)
φ1B,  θ1B,  r1B  = ball_basis.local_grids((1,1,1))
φS,  θS,  rS     = shell_basis.local_grids(shell_basis.dealias)
φ1S,  θ1S,  r1S  = shell_basis.local_grids((1,1,1))

# Fields
field_dict = OrderedDict()
vec_fields = ['u', 'eφ', 'eθ', 'er', 'ex', 'ey', 'ez']
scalar_fields = ['p', 's1', 'inv_T', 'H', 'ρ', 'T']
vec_taus = ['tau_u']
scalar_taus = ['tau_s']

#Tau fields
for S2_basis, name in zip((top_ball_S2_basis, bot_shell_S2_basis, top_shell_S2_basis),('B', 'Sbot', 'Stop')):
    for fn in vec_taus:
        key = '{}_{}'.format(fn, name)
        field_dict[key] = dist.VectorField(coords, name=key, bases=S2_basis)
    for fn in scalar_taus:
        key = '{}_{}'.format(fn, name)
        field_dict[key] = dist.Field(name=key, bases=S2_basis)
field_dict['tau_p'] = dist.Field(name='tau_p')

#Other fields
for basis, name in zip((ball_basis, shell_basis), ('B', 'S')):
    for fn in vec_fields:
        key = '{}_{}'.format(fn, name)
        field_dict[key] = dist.VectorField(coords, name=key, bases=basis)
    for fn in scalar_fields:
        key = '{}_{}'.format(fn, name)
        field_dict[key] = dist.Field(name=key, bases=basis)

for k in field_dict.keys():
    field_dict[k]['g'][:] = 0

#Define unit vectors
for basis_name, grid_points in zip(['B', 'S'], [(r1B, θ1B, φ1B), (r1S, θ1S, φ1S)]):
    field_dict['eφ_{}'.format(basis_name)]['g'][0,:] = 1 
    field_dict['eθ_{}'.format(basis_name)]['g'][1,:] = 1 
    field_dict['er_{}'.format(basis_name)]['g'][2,:] = 1 

    r, θ, φ =  grid_points
    field_dict['ex_{}'.format(basis_name)]['g'][0] = -np.sin(φ)
    field_dict['ex_{}'.format(basis_name)]['g'][1] = np.cos(θ)*np.cos(φ)
    field_dict['ex_{}'.format(basis_name)]['g'][2] = np.sin(θ)*np.cos(φ)

    field_dict['ey_{}'.format(basis_name)]['g'][0] = np.cos(φ)
    field_dict['ey_{}'.format(basis_name)]['g'][1] = np.cos(θ)*np.sin(φ)
    field_dict['ey_{}'.format(basis_name)]['g'][2] = np.sin(θ)*np.sin(φ)

    field_dict['ez_{}'.format(basis_name)]['g'][0] = 0
    field_dict['ez_{}'.format(basis_name)]['g'][1] = -np.sin(θ)
    field_dict['ez_{}'.format(basis_name)]['g'][2] =  np.cos(θ)
    
    for k in ['ex', 'ey', 'ez']:
        field_dict['{}_{}'.format(k, basis_name)] = d3.Grid(field_dict['{}_{}'.format(k, basis_name)]).evaluate()

# NCCs
ncc_dict = OrderedDict()
tensor_nccs = ['I_matrix']
vec_nccs = ['grad_ln_ρ', 'grad_ln_T', 'grad_s0', 'grad_T', 'grad_inv_Pe_rad']
scalar_nccs = ['ln_ρ', 'ln_T', 'inv_Pe_rad', 'sponge']

for basis, name in zip((ball_basis, shell_basis), ('B', 'S')):
    for fn in tensor_nccs:
        key = '{}_{}'.format(fn, name)
        ncc_dict[key] = dist.TensorField(coords, name=key, bases=basis.radial_basis)
    for fn in vec_nccs:
        key = '{}_{}'.format(fn, name)
        ncc_dict[key] = dist.VectorField(coords, name=key, bases=basis.radial_basis)
    for fn in scalar_nccs:
        key = '{}_{}'.format(fn, name)
        ncc_dict[key] = dist.Field(name=key, bases=basis.radial_basis)

for k in ncc_dict.keys():
    ncc_dict[k]['g'][:] = 0

for basis_name in ['B', 'S']:
    for i in range(3):
        ncc_dict['I_matrix_{}'.format(basis_name)]['g'][i,i,:] = 1

if sponge:
    L_shell = r_outer - r_inner
    ncc_dict['sponge_S']['g'] = zero_to_one(r1S, r_inner + 2*L_shell/3, 0.1*L_shell)

# Cartesian unit vectors for post
# Load MESA NCC file or setup NCCs using polytrope
grid_slices_B  = dist.layouts[-1].slices(field_dict['{}_{}'.format(vec_fields[0], 'B')].domain, N_dealias)
grid_slices_S  = dist.layouts[-1].slices(field_dict['{}_{}'.format(vec_fields[0], 'S')].domain, N_dealias)
ncc_dict['{}_{}'.format(vec_nccs[0], 'B')].change_scales(ball_basis.dealias)
ncc_dict['{}_{}'.format(vec_nccs[0], 'S')].change_scales(shell_basis.dealias)
local_vncc_shape_B = ncc_dict['{}_{}'.format(vec_nccs[0], 'B')]['g'].shape
local_vncc_shape_S = ncc_dict['{}_{}'.format(vec_nccs[0], 'S')]['g'].shape
if ncc_file is not None:
    for basis, basis_name in zip((ball_basis, shell_basis), ['B', 'S']):
        for k in vec_nccs + scalar_nccs:
            ncc_dict['{}_{}'.format(k, basis_name)].change_scales(basis.dealias)
        for k in ['H', 'ρ', 'T', 'inv_T']:
            field_dict['{}_{}'.format(k, basis_name)].change_scales(basis.dealias)
    with h5py.File(ncc_file, 'r') as f:
        for k in vec_nccs:
            if np.prod(local_vncc_shape_B) > 0:
                ncc_dict['{}_B'.format(k)]['g'] = f['{}B'.format(k)][:,0,0,grid_slices_B[-1]][:,None,None,:]
            if np.prod(local_vncc_shape_S) > 0:
                ncc_dict['{}_S'.format(k)]['g'] = f['{}S'.format(k)][:,0,0,grid_slices_S[-1]][:,None,None,:]
        for k in scalar_nccs:
            if '{}{}'.format(k, basis_name) not in f.keys():
                continue
            for basis_name, grid_slices in zip(['B', 'S'], [grid_slices_B, grid_slices_S]):
                ncc_dict['{}_{}'.format(k, basis_name)]['g'] = f['{}{}'.format(k, basis_name)][:,:,grid_slices[-1]]
        field_dict['H_B']['g']         = f['H_effB'][:,:,grid_slices_B[-1]]
        field_dict['ρ_B']['g']         = np.exp(f['ln_ρB'][:,:,grid_slices_B[-1]])[None,None,:]
        field_dict['T_B']['g']         = f['TB'][:,:,grid_slices_B[-1]][None,None,:]
        field_dict['inv_T_B']['g']     = 1/field_dict['T_B']['g']

        field_dict['H_S']['g']          = f['H_effS'][:,:,grid_slices_S[-1]]
        field_dict['ρ_S']['g']         = np.exp(f['ln_ρS'][:,:,grid_slices_S[-1]])[None,None,:]
        field_dict['T_S']['g']         = f['TS'][:,:,grid_slices_S[-1]][None,None,:]
        field_dict['inv_T_S']['g']     = 1/field_dict['T_S']['g']

        max_dt = f['max_dt'][()]
        t_buoy = 1 #assume nondimensionalization on heating ~ buoyancy time

        if do_rotation:
            sim_tau_sec = f['tau'][()]
            sim_tau_day = sim_tau_sec / (60*60*24)
            Ω = sim_tau_day * dimensional_Ω 
            t_rot = 1/(2*Ω)
        else:
            t_rot = np.inf

        if sponge:
            f_brunt = f['tau'][()]*np.sqrt(f['N2max_shell'][()])/(2*np.pi)
            ncc_dict['sponge_S']['g'] *= f_brunt

else:
    logger.info("Using polytropic initial conditions")
    from scipy.interpolate import interp1d
    with h5py.File('benchmark/poly_nOuter1.6.h5', 'r') as f:
        T_func = interp1d(f['r'][()], f['T'][()])
        ρ_func = interp1d(f['r'][()], f['ρ'][()])
        grad_s0_func = interp1d(f['r'][()], f['grad_s0'][()])
        H_func   = interp1d(f['r'][()], f['H_eff'][()])
    max_grad_s0 = grad_s0_func(r_outer)
    max_dt = 2/np.sqrt(max_grad_s0)
    t_buoy      = 1
    if do_rotation:
        Ω = dimensional_Ω 
        t_rot = 1/(2*Ω)
    else:
        t_rot = np.inf

    for r1, basis, basis_name, local_vncc_shape in zip((r1B, r1S), (ball_basis, shell_basis), ('B', 'S'), (local_vncc_shape_B, local_vncc_shape_S)):
        field_dict['T_{}'.format(basis_name)]['g'] = T_func(r1)
        field_dict['ρ_{}'.format(basis_name)]['g'] = ρ_func(r1)
        field_dict['H_{}'.format(basis_name)]['g'] = H_func(r1)
        field_dict['inv_T_{}'.format(basis_name)]['g'] = 1/T_func(r1)
        
        grad_ln_ρ_full = (d3.grad(field_dict['ρ_{}'.format(basis_name)])/field_dict['ρ_{}'.format(basis_name)]).evaluate()
        grad_T_full = d3.grad(field_dict['T_{}'.format(basis_name)]).evaluate()
        grad_ln_T_full = (grad_T_full/field_dict['T_{}'.format(basis_name)]).evaluate()
        if np.prod(local_vncc_shape) > 0:
            ncc_dict['grad_s0_{}'.format(basis_name)].change_scales(1)
            print(ncc_dict['grad_s0_{}'.format(basis_name)]['g'].shape, 'grad_s0_{}'.format(basis_name))
            ncc_dict['grad_s0_{}'.format(basis_name)]['g'][2]  = grad_s0_func(r1)
            for f in ['grad_ln_ρ', 'grad_ln_T', 'grad_T']: ncc_dict['{}_{}'.format(f, basis_name)].change_scales(basis.dealias)
            ncc_dict['grad_ln_ρ_{}'.format(basis_name)]['g']   = grad_ln_ρ_full['g'][:,0,0,None,None,:]
            ncc_dict['grad_ln_T_{}'.format(basis_name)]['g']   = grad_ln_T_full['g'][:,0,0,None,None,:]
            ncc_dict['grad_T_{}'.format(basis_name)]['g']      = grad_T_full['g'][:,0,0,None,None,:]
            ncc_dict['grad_inv_Pe_rad_{}'.format(basis_name)]['g'] = 0
        ncc_dict['ln_T_{}'.format(basis_name)]['g']   = np.log(T_func(r1))
        ncc_dict['ln_ρ_{}'.format(basis_name)]['g']   = np.log(ρ_func(r1))
        ncc_dict['inv_Pe_rad_{}'.format(basis_name)]['g'] = 1/Pe

if do_rotation:
    logger.info("Running with Coriolis Omega = {:.3e}".format(Ω))

# Put nccs and fields into locals()
locals().update(ncc_dict)
locals().update(field_dict)

#Stress matrices & viscous terms (assumes uniform kinematic viscosity; so dynamic viscosity mu = const * rho)
divU_B = d3.div(u_B)
E_B = 0.5*(d3.grad(u_B) + d3.transpose(d3.grad(u_B)))
σ_B = 2*(E_B - (1/3)*divU_B*I_matrix_B)
visc_div_stress_B = d3.div(σ_B) + d3.dot(σ_B, grad_ln_ρ_B)
VH_B  = 2*(d3.trace(d3.dot(E_B, E_B)) - (1/3)*divU_B*divU_B)

divU_S = d3.div(u_S)
E_S = 0.5*(d3.grad(u_S) + d3.transpose(d3.grad(u_S)))
σ_S = 2*(E_S - (1/3)*divU_S*I_matrix_S)
visc_div_stress_S = d3.div(σ_S) + d3.dot(σ_S, grad_ln_ρ_S)
VH_S  = 2*(d3.trace(d3.dot(E_S, E_S)) - (1/3)*divU_S*divU_S)

# Grid-lock some operators / define grad's
H_B = d3.Grid(H_B).evaluate()
H_S = d3.Grid(H_S).evaluate()
inv_T_B = d3.Grid(inv_T_B).evaluate()
inv_T_S = d3.Grid(inv_T_S).evaluate()
grad_s1_B = d3.grad(s1_B)
grad_s1_S = d3.grad(s1_S)

# Rotation and damping terms
if do_rotation:
    rotation_term_B = -2*Ω*d3.cross(ez_B, u_B)
    rotation_term_S = -2*Ω*d3.cross(ez_S, u_S)
else:
    rotation_term_B = 0
    rotation_term_S = 0

if args['--sponge']:
    sponge_term_S = sponge_S*u_S
else:
    sponge_term_S = 0
sponge_term_B = 0

# Lift operators for boundary conditions
lift_ball_basis = ball_basis.clone_with(k=0)
lift_shell_basis = shell_basis.clone_with(k=2)
liftB   = lambda A: d3.Lift(A, lift_ball_basis, -1)
liftS   = lambda A, n: d3.Lift(A, lift_shell_basis, n)
integ     = lambda A: d3.Integrate(A, coords)
BC_u_B = liftB(tau_u_B)
BC_u_S = liftS(tau_u_Sbot, -1) + liftS(tau_u_Stop, -2)
BC_s1_B = liftB(tau_s_B)
BC_s1_S = liftS(tau_s_Sbot, -1) + liftS(tau_s_Stop, -2)

# Problem
problem = d3.IVP([p_B, u_B, p_S, u_S, s1_B, s1_S, tau_u_B, tau_u_Sbot, tau_u_Stop, tau_s_B, tau_s_Sbot, tau_s_Stop], namespace=locals())

# Equations
problem.add_equation("div(u_B) + dot(u_B, grad_ln_ρ_B) = 0", condition="nθ != 0")
problem.add_equation("dt(u_B) + grad(p_B) + grad_T_B*s1_B - (1/Re)*visc_div_stress_B + sponge_term_B + BC_u_B = cross(u_B, curl(u_B)) + rotation_term_B", condition="nθ != 0")
problem.add_equation("div(u_S) + dot(u_S, grad_ln_ρ_S) = 0", condition="nθ != 0")
problem.add_equation("dt(u_S) + grad(p_S) + grad_T_S*s1_S - (1/Re)*visc_div_stress_S + sponge_term_S + BC_u_S = cross(u_S, curl(u_S)) + rotation_term_S", condition="nθ != 0")
problem.add_equation("p_B = 0", condition="nθ == 0")
problem.add_equation("u_B = 0", condition="nθ == 0")
problem.add_equation("p_S = 0", condition="nθ == 0")
problem.add_equation("u_S = 0", condition="nθ == 0")
problem.add_equation("dt(s1_B) + dot(u_B, grad_s0_B) - (inv_Pe_rad_B)*(lap(s1_B) + dot(grad_s1_B, (grad_ln_ρ_B + grad_ln_T_B))) - dot(grad_s1_B, grad_inv_Pe_rad_B) + BC_s1_B = - dot(u_B, grad_s1_B) + H_B + (1/Re)*inv_T_B*VH_B ")
problem.add_equation("dt(s1_S) + dot(u_S, grad_s0_S) - (inv_Pe_rad_S)*(lap(s1_S) + dot(grad_s1_S, (grad_ln_ρ_S + grad_ln_T_S))) - dot(grad_s1_S, grad_inv_Pe_rad_S) + BC_s1_S = - dot(u_S, grad_s1_S) + H_S + (1/Re)*inv_T_S*VH_S ")

# Boundary Conditions
problem.add_equation("u_B(r=r_inner) - u_S(r=r_inner) = 0", condition="nθ != 0")
problem.add_equation("p_B(r=r_inner) - p_S(r=r_inner) = 0", condition="nθ != 0")
problem.add_equation("angular(radial(σ_B(r=r_inner) - σ_S(r=r_inner)), index=0) = 0", condition="nθ != 0")
problem.add_equation("radial(u_S(r=r_outer)) = 0", condition="nθ != 0")
problem.add_equation("angular(radial(E_S(r=r_outer))) = 0", condition="nθ != 0")
problem.add_equation("tau_u_B = 0", condition="nθ == 0")
problem.add_equation("tau_u_Sbot = 0", condition="nθ == 0")
problem.add_equation("tau_u_Stop = 0", condition="nθ == 0")

# Entropy BCs
problem.add_equation("s1_B(r=r_inner) - s1_S(r=r_inner) = 0")
problem.add_equation("radial(grad_s1_B(r=r_inner) - grad_s1_S(r=r_inner)) = 0")
problem.add_equation("radial(grad_s1_S(r=r_outer)) = 0")

## Problem
#problem = d3.IVP([p_B, u_B, p_S, u_S, s1_B, s1_S, tau_p, tau_u_B, tau_u_Sbot, tau_u_Stop, tau_s_B, tau_s_Sbot, tau_s_Stop], namespace=locals())
#
## Equations
#problem.add_equation("div(u_B) + dot(u_B, grad_ln_ρ_B) + tau_p = 0")
#problem.add_equation("dt(u_B) + grad(p_B) + grad_T_B*s1_B - (1/Re)*visc_div_stress_B + sponge_term_B + BC_u_B = cross(u_B, curl(u_B)) + rotation_term_B")
#problem.add_equation("div(u_S) + dot(u_S, grad_ln_ρ_S) + tau_p = 0")
#problem.add_equation("dt(u_S) + grad(p_S) + grad_T_S*s1_S - (1/Re)*visc_div_stress_S + sponge_term_S + BC_u_S = cross(u_S, curl(u_S)) + rotation_term_S")
#problem.add_equation("dt(s1_B) + dot(u_B, grad_s0_B) - (inv_Pe_rad_B)*(lap(s1_B) + dot(grad_s1_B, (grad_ln_ρ_B + grad_ln_T_B))) - dot(grad_s1_B, grad_inv_Pe_rad_B) + BC_s1_B = - dot(u_B, grad_s1_B) + H_B + (1/Re)*inv_T_B*VH_B ")
#problem.add_equation("dt(s1_S) + dot(u_S, grad_s0_S) - (inv_Pe_rad_S)*(lap(s1_S) + dot(grad_s1_S, (grad_ln_ρ_S + grad_ln_T_S))) - dot(grad_s1_S, grad_inv_Pe_rad_S) + BC_s1_S = - dot(u_S, grad_s1_S) + H_S + (1/Re)*inv_T_S*VH_S ")
#
## Boundary Conditions
#problem.add_equation("u_B(r=r_inner) - u_S(r=r_inner) = 0")
#problem.add_equation("p_B(r=r_inner) - p_S(r=r_inner) = 0")
#problem.add_equation("angular(radial(σ_B(r=r_inner) - σ_S(r=r_inner)), index=0) = 0")
#problem.add_equation("radial(u_S(r=r_outer)) = 0")
#problem.add_equation("angular(radial(E_S(r=r_outer))) = 0")
#
## Entropy BCs
#problem.add_equation("s1_B(r=r_inner) - s1_S(r=r_inner) = 0")
#problem.add_equation("radial(grad_s1_B(r=r_inner) - grad_s1_S(r=r_inner)) = 0")
#problem.add_equation("radial(grad_s1_S(r=r_outer)) = 0")
#problem.add_equation("integ(p_B) + integ(p_S) = 0")

logger.info("Problem built")
# Solver
solver = problem.build_solver(ts)
solver.stop_sim_time = buoy_end_time*t_buoy
solver.stop_wall_time = wall_hours * 60 * 60
logger.info("solver built")

# Initial conditions / Checkpoint
write_mode = 'overwrite'
timestep = None
if restart is not None:
    write, timestep = solver.load_state(restart)
    write_mode = 'append'
else:
    # Initial conditions
    s1_B.fill_random(layout='g', seed=42, distribution='normal', scale=A0)
    s1_B.low_pass_filter(scales=0.25)
    s1_B['g'] *= one_to_zero(r1B, 0.9*r_inner, width=0.04*r_inner)

## Analysis Setup
# Cadence
scalar_dt = 0.25*t_buoy
lum_dt   = 0.5*t_buoy
visual_dt = 0.05*t_buoy
outer_shell_dt = max_dt

#TODO: revisit outputs
# Operators, extra fields
r_vals_B = dist.Field(name='r_vals_B', bases=ball_basis)
r_vals_S = dist.Field(name='r_vals_S', bases=shell_basis)
r_vals_B['g'] = r1B
r_vals_S['g'] = r1S
er_B = d3.Grid(er_B).evaluate()
er_S = d3.Grid(er_S).evaluate()
r_vals_B = d3.Grid(r_vals_B).evaluate()
r_vals_S = d3.Grid(r_vals_S).evaluate()
u_B_squared = d3.dot(u_B, u_B)
u_S_squared = d3.dot(u_S, u_S)
ur_B = d3.dot(er_B, u_B)
ur_S = d3.dot(er_S, u_S)
h_B = p_B - 0.5*u_B_squared + T_B*s1_B
h_S = p_S - 0.5*u_S_squared + T_S*s1_S
pomega_hat_B = p_B - 0.5*u_B_squared
pomega_hat_S = p_S - 0.5*u_S_squared

visc_flux_B_r = 2*d3.dot(er_B, d3.dot(u_B, E_B) - (1/3) * u_B * divU_B)
visc_flux_S_r = 2*d3.dot(er_S, d3.dot(u_S, E_S) - (1/3) * u_S * divU_S)

# Angular momentum
r_B_vec_post = dist.VectorField(coords, name='rB_vec_post', bases=ball_basis)
r_B_vec_post['g'][2] = r1B
L_AM_B = d3.cross(r_B_vec_post, ρ_B*u_B)
Lx_AM_B = d3.dot(ex_B, L_AM_B)
Ly_AM_B = d3.dot(ey_B, L_AM_B)
Lz_AM_B = d3.dot(ez_B, L_AM_B)

r_S_vec_post = dist.VectorField(coords, name='rS_vec_post', bases=shell_basis)
r_S_vec_post['g'][2] = r1S
L_AM_S = d3.cross(r_S_vec_post, ρ_S*u_S)
Lx_AM_S = d3.dot(ex_S, L_AM_S)
Ly_AM_S = d3.dot(ey_S, L_AM_S)
Lz_AM_S = d3.dot(ez_S, L_AM_S)

# Averaging Operations
volume  = (4/3)*np.pi*r_outer**3
volume_B = (4/3)*np.pi*r_inner**3
volume_S = volume - volume_B

az_avg = lambda A: d3.Average(A, coords.coords[0])
s2_avg = lambda A: d3.Average(A, coords.S2coordsys)
vol_avg_B = lambda A: d3.Integrate(A/volume_B, coords)
vol_avg_S = lambda A: d3.Integrate(A/volume_S, coords)
luminosity_B = lambda A: s2_avg((4*np.pi*r_vals_B**2) * A)
luminosity_S = lambda A: s2_avg((4*np.pi*r_vals_S**2) * A)

analysis_tasks = []

slices = solver.evaluator.add_file_handler('{:s}/slices'.format(out_dir), sim_dt=visual_dt, max_writes=40)
slices.add_task(u_B(θ=np.pi/2), name='u_B_eq', layout='g')
slices.add_task(u_S(θ=np.pi/2), name='u_S_eq', layout='g')
slices.add_task(s1_B(θ=np.pi/2), name='s1_B_eq', layout='g')
slices.add_task(s1_S(θ=np.pi/2), name='s1_S_eq', layout='g')
for fd, name in zip((u_B, s1_B), ('u_B', 's1_B')):
    for radius, r_str in zip((0.5, 1), ('0.5', '1')):
        operation = fd(r=radius)
        slices.add_task(operation, name=name+'(r={})'.format(r_str), layout='g')
for fd, name in zip((u_S, s1_S), ('u_S', 's1_S')):
    for radius, r_str in zip((0.95*r_outer,), ('0.95R',)):
        operation = fd(r=radius)
        slices.add_task(operation, name=name+'(r={})'.format(r_str), layout='g')
for az_val, name in zip([0, np.pi/2, np.pi, 3*np.pi/2], ['(phi=0)', '(phi=0.5*pi)', '(phi=pi)', '(phi=1.5*pi)',]):
    slices.add_task(u_B(φ=az_val),  name='u_B' +name, layout='g')
    slices.add_task(s1_B(φ=az_val), name='s1_B'+name, layout='g')
for az_val, name in zip([0, np.pi/2, np.pi, 3*np.pi/2], ['(phi=0)', '(phi=0.5*pi)', '(phi=pi)', '(phi=1.5*pi)',]):
    slices.add_task(u_S(φ=az_val),  name='u_S' +name, layout='g')
    slices.add_task(s1_S(φ=az_val), name='s1_S'+name, layout='g')
analysis_tasks.append(slices)

scalars = solver.evaluator.add_file_handler('{:s}/scalars'.format(out_dir), sim_dt=scalar_dt, max_writes=np.inf)
scalars.add_task(vol_avg_B(Re*(u_B_squared)**(1/2)), name='Re_avg_ball',  layout='g')
scalars.add_task(vol_avg_S(Re*(u_S_squared)**(1/2)), name='Re_avg_shell', layout='g')
scalars.add_task(vol_avg_B(ρ_B*u_B_squared/2), name='KE_ball',   layout='g')
scalars.add_task(vol_avg_S(ρ_S*u_S_squared/2), name='KE_shell',  layout='g')
scalars.add_task(vol_avg_B(ρ_B*T_B*s1_B), name='TE_ball',  layout='g')
scalars.add_task(vol_avg_S(ρ_S*T_S*s1_S), name='TE_shell', layout='g')
scalars.add_task(vol_avg_B(Lx_AM_B), name='Lx_AM_ball', layout='g')
scalars.add_task(vol_avg_B(Ly_AM_B), name='Ly_AM_ball', layout='g')
scalars.add_task(vol_avg_B(Lz_AM_B), name='Lz_AM_ball', layout='g')
scalars.add_task(vol_avg_S(Lx_AM_S), name='Lx_AM_shell', layout='g')
scalars.add_task(vol_avg_S(Ly_AM_S), name='Ly_AM_shell', layout='g')
scalars.add_task(vol_avg_S(Lz_AM_S), name='Lz_AM_shell', layout='g')
analysis_tasks.append(scalars)

profiles = solver.evaluator.add_file_handler('{:s}/profiles'.format(out_dir), sim_dt=visual_dt, max_writes=100)
profiles.add_task(luminosity_B(ρ_B*ur_B*pomega_hat_B),         name='wave_lumB', layout='g')
profiles.add_task(luminosity_S(ρ_S*ur_S*pomega_hat_S),         name='wave_lumS', layout='g')
profiles.add_task(luminosity_B(ρ_B*ur_B*h_B),                   name='enth_lumB', layout='g')
profiles.add_task(luminosity_S(ρ_S*ur_S*h_S),                   name='enth_lumS', layout='g')
profiles.add_task(luminosity_B(-ρ_B*visc_flux_B_r/Re),         name='visc_lumB', layout='g')
profiles.add_task(luminosity_S(-ρ_S*visc_flux_S_r/Re),         name='visc_lumS', layout='g')
profiles.add_task(luminosity_B(-ρ_B*T_B*d3.dot(er_B, grad_s1_B)/Pe), name='cond_lumB', layout='g')
profiles.add_task(luminosity_S(-ρ_S*T_S*d3.dot(er_S, grad_s1_S)/Pe), name='cond_lumS', layout='g')
profiles.add_task(luminosity_B(0.5*ρ_B*ur_B*u_B_squared),       name='KE_lumB',   layout='g')
profiles.add_task(luminosity_S(0.5*ρ_S*ur_S*u_S_squared),       name='KE_lumS',   layout='g')
analysis_tasks.append(profiles)

if args['--sponge']:
    surface_shell_slices = solver.evaluator.add_file_handler('{:s}/wave_shell_slices'.format(out_dir), sim_dt=100000*max_dt, max_writes=20)
    for rval in [0.90, 1.05]:
        surface_shell_slices.add_task(d3.radial(u_B(r=rval)), name='u(r={})'.format(rval), layout='g')
        surface_shell_slices.add_task(pomega_hat_B(r=rval), name='pomega(r={})'.format(rval),    layout='g')
    for rval in [1.15, 1.60]:
        surface_shell_slices.add_task(d3.radial(u_S(r=rval)), name='u(r={})'.format(rval), layout='g')
        surface_shell_slices.add_task(pomega_hat_S(r=rval), name='pomega(r={})'.format(rval),    layout='g')
    analysis_tasks.append(surface_shell_slices)
else:
    surface_shell_slices = solver.evaluator.add_file_handler('{:s}/wave_shell_slices'.format(out_dir), sim_dt=100000*max_dt, max_writes=20)
    surface_shell_slices.add_task(s1_S(r=r_outer),         name='s1_surf',    layout='g')
    analysis_tasks.append(surface_shell_slices)

if Re > 1e4:
    chk_time = 2*t_buoy
else:
    chk_time = 10*t_buoy
checkpoint = solver.evaluator.add_file_handler('{:s}/checkpoint'.format(out_dir), max_writes=1, sim_dt=chk_time)
checkpoint.add_tasks(solver.state, layout='g')

re_ball = solver.evaluator.add_dictionary_handler(iter=10)
re_ball.add_task(vol_avg_B(Re*(u_B_squared)**(1/2)), name='Re_avg_ball', layout='g')

#CFL setup
heaviside_cfl = dist.Field(name='heaviside_cfl', bases=ball_basis)
heaviside_cfl['g'] = 1
if np.sum(rB > CFL_max_r) > 0:
    heaviside_cfl['g'][:,:, r1B.flatten() > CFL_max_r] = 0
heaviside_cfl = d3.Grid(heaviside_cfl).evaluate()

initial_max_dt = np.min((visual_dt, t_rot*0.5))
while initial_max_dt < max_dt:
    max_dt /= 2
if timestep is None:
    timestep = initial_max_dt
my_cfl = d3.CFL(solver, timestep, safety=safety, cadence=1, max_dt=initial_max_dt, min_change=0.1, max_change=1.5, threshold=0.1)
my_cfl.add_velocity(heaviside_cfl*u_B)

#startup iterations
for i in range(10):
    solver.step(timestep)
    logger.info("startup iteration %d, t = %f, timestep = %f" %(i, solver.sim_time, timestep))
    timestep = my_cfl.compute_timestep()

# Main loop
start_time = time.time()
start_iter = solver.iteration
max_dt_check = True
current_max_dt = my_cfl.max_dt
slice_process = False
just_wrote    = False
slice_time = np.inf
Re0 = 0
try:
    while solver.proceed:
        timestep = my_cfl.compute_timestep()

        if just_wrote:
            just_wrote = False
            num_steps = np.ceil(outer_shell_dt / timestep)
            timestep = current_max_dt = my_cfl.stored_dt = outer_shell_dt/num_steps
        elif max_dt_check:
            timestep = np.min((timestep, current_max_dt))
        else:
            my_cfl.stored_dt = timestep = current_max_dt

        t_future = solver.sim_time + timestep
        if t_future >= slice_time*(1-1e-8):
           slice_process = True

        if solver.iteration % hermitian_cadence in timestepper_history:
            for f in solver.state:
                f.require_grid_space()

        solver.step(timestep)

        if solver.iteration % 10 == 0:
            Re_avg = re_ball.fields['Re_avg_ball']
            if dist.comm_cart.rank == 0:
                Re0 = Re_avg['g'].min()
            else:
                Re0 = None
            Re0 = dist.comm_cart.bcast(Re0, root=0)
            logger.info("t = %f, timestep = %f, Re = %e" %(solver.sim_time, timestep, Re0))
        if max_dt_check and timestep < outer_shell_dt:
            my_cfl.max_dt = max_dt
            max_dt_check = False
            just_wrote = True
            slice_time = solver.sim_time + outer_shell_dt

        if slice_process:
            slice_process = False
            wall_time = time.time() - solver.start_time
            solver.evaluator.evaluate_handlers([surface_shell_slices],wall_time=wall_time, sim_time=solver.sim_time, iteration=solver.iteration,world_time = time.time(),timestep=timestep)
            slice_time = solver.sim_time + outer_shell_dt
            just_wrote = True

        if np.isnan(Re0):
            logger.info('exiting with NaN')
            break

except:
    logger.info('something went wrong in main loop, making final checkpoint')
    raise
finally:
    end_time = time.time()
    end_iter = solver.iteration
    cpu_sec  = end_time - start_time
    n_iter   = end_iter - start_iter


    fcheckpoint = solver.evaluator.add_file_handler('{:s}/final_checkpoint'.format(out_dir), max_writes=1, sim_dt=10*t_buoy)
    fcheckpoint.add_tasks(solver.state, layout='g')
    solver.step(timestep)

    if dist.comm_cart.rank == 0:
        for handler in analysis_tasks:
            handler.process_virtual_file()

    #TODO: Make the end-of-sim report better
    n_coeffs = np.prod(resolutionB) + np.prod(resolutionS)
    n_cpu    = dist.comm_cart.size
    dof_cycles_per_cpusec = n_coeffs*n_iter/(cpu_sec*n_cpu)
    logger.info('DOF-cycles/cpu-sec : {:e}'.format(dof_cycles_per_cpusec))
    logger.info('Run iterations: {:e}'.format(n_iter))
    logger.info('Sim end time: {:e}'.format(solver.sim_time))
    logger.info('Run time: {}'.format(cpu_sec))
