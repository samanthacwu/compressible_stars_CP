"""
d3 script for anelastic convection in a stitched BallBasis and ShellBasis domain.

A config file can be provided which overwrites any number of the options below, but command line flags can also be used if preferred.
Note that options specified in a cnofig file override command line arguments.

Usage:
    minimal_shell_AN.py [options]
    minimal_shell_AN.py <config> [options]

Options:
    --Re=<Re>            The Reynolds number of the numerical diffusivities [default: 2e2]
    --Pr=<Prandtl>       The Prandtl number  of the numerical diffusivities [default: 1]
    --ntheta=<res>       Number of theta grid points (Lmax+1)   [default: 4]
    --nr=<res>          Number of radial grid points in shell (Nmax+1)   [default: 8]

    --wall_hours=<t>     Max number of wall hours to run simulation for [default: 24]
    --buoy_end_time=<t>  Max number of buoyancy time units to simulate [default: 1e5]

    --ncc_file=<f>      path to a .h5 file of ICCs, curated from a MESA model
    --restart=<chk_f>    path to a checkpoint file to restart from
    --A0=<A>             Amplitude of random noise initial conditions [default: 1e-6]

    --label=<label>      A label to add to the end of the output directory

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
nθ  = int(args['--ntheta'])
nφ  = int(2*nθ)
nrS = int(args['--nr'])
resolutionS = (nφ, nθ, nrS)
L_dealias = N_dealias = dealias = 1.5
dtype = np.float64
Re  = float(args['--Re'])
Pr  = 1
Pe  = Pr*Re
ncc_file = args['--ncc_file']
wall_hours = float(args['--wall_hours'])
buoy_end_time = float(args['--buoy_end_time'])

# Initial conditions
restart = args['--restart']
A0 = float(args['--A0'])

# Timestepper
ts = d3.SBDF2
timestepper_history = [0, 1,]
hermitian_cadence = 100
safety = 0.2

# Processor mesh
ncpu = MPI.COMM_WORLD.size
log2 = np.log2(ncpu)
if log2 == int(log2):
    mesh = [int(2**np.ceil(log2/2)),int(2**np.floor(log2/2))]
logger.info("running on processor mesh={}".format(mesh))

# Read in domain bound values
if ncc_file is not None:
    with h5py.File(args['--ncc_file'], 'r') as f:
        r_inner = f['r_inner'][()]
        r_outer = f['r_outer'][()]
else:
    r_inner = 1.1
    r_outer = 1.5
r_inner = 1.0
r_outer = 3.0
logger.info('r_inner: {:.2f} / r_outer: {:.2f}'.format(r_inner, r_outer))

# Bases
coords  = d3.SphericalCoordinates('φ', 'θ', 'r')
dist    = d3.Distributor((coords,), mesh=mesh, dtype=dtype)
shell_basis = d3.ShellBasis(coords, resolutionS, radii=(r_inner, r_outer), dtype=dtype, dealias=(L_dealias, L_dealias, N_dealias))
bot_shell_S2_basis = shell_basis.S2_basis(radius=r_inner)
top_shell_S2_basis = shell_basis.S2_basis(radius=r_outer)
φS,  θS,  rS     = shell_basis.local_grids(shell_basis.dealias)
φ1S,  θ1S,  r1S  = shell_basis.local_grids((1,1,1))

# Fields
field_dict = OrderedDict()
vec_fields = ['u', 'eφ', 'eθ', 'er']
scalar_fields = ['p', 's1', 'inv_T', 'H', 'ρ', 'T']
vec_taus = ['tau_u']
scalar_taus = ['tau_s']
single_taus = ['tau_p']

#Tau fields
for S2_basis, name in zip((bot_shell_S2_basis, top_shell_S2_basis),('Sbot', 'Stop')):
    for fn in vec_taus:
        key = '{}_{}'.format(fn, name)
        field_dict[key] = dist.VectorField(coords, name=key, bases=S2_basis)
    for fn in scalar_taus:
        key = '{}_{}'.format(fn, name)
        field_dict[key] = dist.Field(name=key, bases=S2_basis)

#Other fields
for basis, name in zip((shell_basis,), ('S',)):
    for fn in vec_fields:
        key = '{}_{}'.format(fn, name)
        field_dict[key] = dist.VectorField(coords, name=key, bases=basis)
    for fn in scalar_fields:
        key = '{}_{}'.format(fn, name)
        field_dict[key] = dist.Field(name=key, bases=basis)
    for fn in single_taus:
        key = '{}_{}'.format(fn, name)
        field_dict[key] = dist.Field(name=key)

for k in field_dict.keys():
    field_dict[k]['g'][:] = 0

#Define unit vectors
for basis_name, grid_points in zip(['S',], [(r1S, θ1S, φ1S),]):
    field_dict['eφ_{}'.format(basis_name)]['g'][0,:] = 1 
    field_dict['eθ_{}'.format(basis_name)]['g'][1,:] = 1 
    field_dict['er_{}'.format(basis_name)]['g'][2,:] = 1 

# NCCs
ncc_dict = OrderedDict()
tensor_nccs = ['I_matrix']
vec_nccs = ['grad_ln_ρ', 'grad_ln_T', 'grad_s0', 'grad_T', 'grad_inv_Pe_rad']
scalar_nccs = ['ln_ρ', 'ln_T', 'inv_Pe_rad']

for basis, name in zip((shell_basis,), ('S')):
    for fn in tensor_nccs:
        key = '{}_{}'.format(fn, name)
        ncc_dict[key] = dist.TensorField(coords, name=key, bases=basis.radial_basis)
    for fn in vec_nccs:
        key = '{}_{}'.format(fn, name)
        ncc_dict[key] = dist.VectorField(coords, name=key, bases=basis.radial_basis)
    for fn in scalar_nccs:
        key = '{}_{}'.format(fn, name)
        ncc_dict[key] = dist.Field(name=key, bases=basis.radial_basis)
ncc_dict['T_NCC'] = dist.Field(name='T_NCC', bases=shell_basis.radial_basis)

for k in ncc_dict.keys():
    ncc_dict[k]['g'][:] = 0

ncc_dict['rvec'] = dist.VectorField(coords, name='T_NCC', bases=shell_basis.radial_basis)
ncc_dict['rvec']['g'][2] = r1S

for basis_name in ['S']:
    for i in range(3):
        ncc_dict['I_matrix_{}'.format(basis_name)]['g'][i,i,:] = 1

# Cartesian unit vectors for post
# Load MESA NCC file or setup NCCs using polytrope
grid_slices_S  = dist.layouts[-1].slices(field_dict['{}_{}'.format(vec_fields[0], 'S')].domain, N_dealias)
ncc_dict['{}_{}'.format(vec_nccs[0], 'S')].change_scales(shell_basis.dealias)
local_vncc_shape_S = ncc_dict['{}_{}'.format(vec_nccs[0], 'S')]['g'].shape
if ncc_file is not None:
    logger.info('reading NCCs from {}'.format(ncc_file))
    for basis, basis_name in zip((shell_basis,), ['S',]):
        for k in vec_nccs + scalar_nccs:
            ncc_dict['{}_{}'.format(k, basis_name)].change_scales(basis.dealias)
        for k in ['H', 'ρ', 'T', 'inv_T']:
            field_dict['{}_{}'.format(k, basis_name)].change_scales(basis.dealias)
    with h5py.File(ncc_file, 'r') as f:
        for k in vec_nccs:
            if np.prod(local_vncc_shape_S) > 0:
                ncc_dict['{}_S'.format(k)]['g'] = f['{}S'.format(k)][:,0,0,grid_slices_S[-1]][:,None,None,:]
        for k in scalar_nccs:
            if '{}{}'.format(k, basis_name) not in f.keys():
                continue
            for basis_name, grid_slices in zip(['S',], [grid_slices_S,]):
                ncc_dict['{}_{}'.format(k, basis_name)]['g'] = f['{}{}'.format(k, basis_name)][:,:,grid_slices[-1]]
        field_dict['H_S']['g']          = f['H_effS'][:,:,grid_slices_S[-1]]
        field_dict['ρ_S']['g']         = np.exp(f['ln_ρS'][:,:,grid_slices_S[-1]])[None,None,:]
        field_dict['T_S']['g']         = f['TS'][:,:,grid_slices_S[-1]][None,None,:]
        field_dict['inv_T_S']['g']     = 1/field_dict['T_S']['g']

        ncc_dict['T_NCC'].change_scales(basis.dealias)
        ncc_dict['T_NCC']['g'] = f['TS'][:,:,grid_slices_S[-1]]

        max_dt = f['max_dt'][()]
        t_buoy = 1 #assume nondimensionalization on heating ~ buoyancy time

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
            ncc_dict['grad_s0_{}'.format(basis_name)]['g'][2]  = grad_s0_func(r1)
            for f in ['grad_ln_ρ', 'grad_ln_T', 'grad_T']: ncc_dict['{}_{}'.format(f, basis_name)].change_scales(basis.dealias)
            ncc_dict['grad_ln_ρ_{}'.format(basis_name)]['g']   = grad_ln_ρ_full['g'][:,0,0,None,None,:]
            ncc_dict['grad_ln_T_{}'.format(basis_name)]['g']   = grad_ln_T_full['g'][:,0,0,None,None,:]
            ncc_dict['grad_T_{}'.format(basis_name)]['g']      = grad_T_full['g'][:,0,0,None,None,:]
            ncc_dict['grad_inv_Pe_rad_{}'.format(basis_name)]['g'] = 0
        ncc_dict['ln_T_{}'.format(basis_name)]['g']   = np.log(T_func(r1))
        ncc_dict['ln_ρ_{}'.format(basis_name)]['g']   = np.log(ρ_func(r1))
        ncc_dict['inv_Pe_rad_{}'.format(basis_name)]['g'] = 1/Pe

# Put nccs and fields into locals()
locals().update(ncc_dict)
locals().update(field_dict)
max_dt /= 1

ncc_mag = 1e4
ncc_pow = 10
#grad_s0_S['g'] = ncc_mag*(((rS - r_inner)/(r_outer - r_inner))**ncc_pow + 1)
logger.info("NCC expansions:")
ncc_cutoff=1e-6
for ncc in [rvec, grad_s0_S]:
        logger.info("{}: {}".format(ncc, np.where(np.abs(ncc['c']) >= ncc_cutoff)[0].shape))

# Lift operators for boundary conditions
lift_k2_basis = shell_basis.clone_with(k=2)
lift_k2   = lambda A, n: d3.Lift(A, lift_k2_basis, n)
lift_shell_basis = shell_basis.clone_with(k=1)
liftS   = lambda A, n: d3.Lift(A, lift_shell_basis, n)
integ     = lambda A: d3.Integrate(A, coords)
BC_u_S = lift_k2(tau_u_Stop, -1) + lift_k2(tau_u_Sbot, -2)
BC_s1_S = lift_k2(tau_s_Stop, -1) + lift_k2(tau_s_Sbot, -2)

#Stress matrices & viscous terms (assumes uniform kinematic viscosity; so dynamic viscosity mu = const * rho)
divU_S = d3.div(u_S)
grad_u_S = d3.grad(u_S) + rvec*liftS(tau_u_Sbot, -1)
E_S = 0.5*(d3.grad(u_S) + d3.transpose(d3.grad(u_S)))
σ_S = 2*(E_S - (1/3)*divU_S*I_matrix_S)
visc_div_stress_S = d3.div(σ_S) + d3.dot(σ_S, grad_ln_ρ_S)
VH_S  = 2*(d3.trace(d3.dot(E_S, E_S)) - (1/3)*divU_S*divU_S)

# Grid-lock some operators / define grad's
H_S = d3.Grid(H_S).evaluate()
inv_T_S = d3.Grid(inv_T_S).evaluate()
grad_s1_S = d3.grad(s1_S) + rvec*liftS(tau_s_Sbot, -1)

#div_rad_flux_S = (inv_Pe_rad_S)*(d3.div(grad_s1_S) + d3.dot(grad_s1_S, (grad_ln_ρ_S + grad_ln_T_S))) + d3.dot(grad_s1_S, d3.grad(inv_Pe_rad_S))
#div_rad_flux_S = (inv_Pe_rad_S)*(d3.div(grad_s1_S + rvec*liftS(tau_s_Sbot, -1)))
div_rad_flux_S = (inv_Pe_rad_S)*(d3.div(grad_s1_S))
div_rad_flux_S += d3.dot(grad_s1_S, d3.grad(inv_Pe_rad_S))
div_rad_flux_S += inv_Pe_rad_S*d3.dot(grad_s1_S, (grad_ln_ρ_S+ grad_ln_T_S)) 

# Problem
problem = d3.IVP([p_S, s1_S, u_S, tau_p_S, tau_s_Sbot, tau_s_Stop, tau_u_Sbot, tau_u_Stop, ], namespace=locals())

#FOF
problem.add_equation("trace(grad_u_S) + dot(u_S, grad_ln_ρ_S) + tau_p_S = 0")
problem.add_equation("dt(s1_S) + dot(u_S, grad_s0_S) - (1/Re)*div(grad_s1_S) + liftS(tau_s_Stop, -1) = 0 ")
problem.add_equation("dt(u_S) + grad(p_S) + grad_T_S*s1_S - (1/Re)*div(grad_u_S) + liftS(tau_u_Stop, -1) = 0")

##k2 formalism
#problem.add_equation("div(u_S) + dot(u_S, grad_ln_ρ_S) + tau_p_S = 0")
#problem.add_equation("dt(s1_S) + dot(u_S, grad_s0_S) - (1/Re)*lap(s1_S) + BC_s1_S = 0 ")
#problem.add_equation("dt(u_S) + grad(p_S) + grad_T_S*s1_S - (1/Re)*lap(u_S) + BC_u_S = 0")


problem.add_equation("s1_S(r=r_inner) = 0")
problem.add_equation("u_S(r=r_inner) = 0")
problem.add_equation("s1_S(r=r_outer) = 0")
problem.add_equation("u_S(r=r_outer) = 0")

#problem.add_equation("radial(u_S(r=r_inner)) = 0")
#problem.add_equation("angular(radial(E_S(r=r_inner))) = 0")
#problem.add_equation("radial(u_S(r=r_outer)) = 0")
#problem.add_equation("angular(radial(E_S(r=r_outer))) = 0")

# Entropy BCs
#problem.add_equation("radial(grad_s1_S(r=r_inner)) = 0")
#problem.add_equation("radial(grad_s1_S(r=r_outer)) = 0")

problem.add_equation("integ(p_S) = 0")

logger.info("Problem built")
# Solver
solver = problem.build_solver(ts, ncc_cutoff=ncc_cutoff)
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
    s1_S.fill_random(layout='g', seed=42, distribution='normal', scale=1)
    s1_S.low_pass_filter(scales=0.25)
    s1_S['g'] *= (r1S - r_inner) * (r_outer - r1S) # Damp noise at walls
#    s1_S['g'] *= zero_to_one(r1S, r_inner+0.3, width=0.1)*one_to_zero(r1S, r_outer-0.3, width=0.1)
u_S['g'] = 0

# Averaging Operations
volume_S = (4/3)*np.pi*(r_outer**3 - r_inner**3)
vol_avg_S = lambda A: d3.Integrate(A/volume_S, coords)
u_S_squared = d3.dot(u_S, u_S)

re_shell = solver.evaluator.add_dictionary_handler(iter=1)
re_shell.add_task(vol_avg_S(Re*(u_S_squared)**(1/2)), name='Re_avg_shell', layout='g')

#CFL setup
initial_max_dt = max_dt
if timestep is None:
    timestep = initial_max_dt
my_cfl = d3.CFL(solver, timestep, safety=safety, cadence=1, max_dt=initial_max_dt, min_change=0.1, max_change=1.5, threshold=0.1)
my_cfl.add_velocity(u_S)

# Main loop
start_time = time.time()
start_iter = solver.iteration
Re0 = 0
try:
    while solver.proceed:
        timestep = my_cfl.compute_timestep()

        if solver.iteration % hermitian_cadence in timestepper_history:
            for f in solver.state:
                f.require_grid_space()

        solver.step(timestep)

        if solver.iteration % 1 == 0:
            Re_avg = re_shell.fields['Re_avg_shell']
            if dist.comm_cart.rank == 0:
                Re0 = Re_avg['g'].min()
                tau_p = tau_p_S['g'].squeeze()
            else:
                Re0 = None
                tau_p = -1
            Re0 = dist.comm_cart.bcast(Re0, root=0)
            this_str = "t = {:f}, timestep = {:f}, Re = {:.4e}".format(solver.sim_time, timestep, Re0)
            this_str += ", tau_ps = {:.4e}".format(tau_p)
            logger.info(this_str)

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


    #TODO: Make the end-of-sim report better
    n_coeffs = np.prod(resolutionS)
    n_cpu    = dist.comm_cart.size
    dof_cycles_per_cpusec = n_coeffs*n_iter/(cpu_sec*n_cpu)
    logger.info('DOF-cycles/cpu-sec : {:e}'.format(dof_cycles_per_cpusec))
    logger.info('Run iterations: {:e}'.format(n_iter))
    logger.info('Sim end time: {:e}'.format(solver.sim_time))
    logger.info('Run time: {}'.format(cpu_sec))
