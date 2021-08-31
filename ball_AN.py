"""
d3 script for anelastic convection in a massive star's core CZ.

A config file can be provided which overwrites any number of the options below, but command line flags can also be used if preferred.

Usage:
    coreCZ_AN.py [options]
    coreCZ_AN.py <config> [options]

Options:
    --Re=<Re>            The Reynolds number of the numerical diffusivities [default: 1e3]
    --Pr=<Prandtl>       The Prandtl number  of the numerical diffusivities [default: 1]
    --L=<Lmax>           The value of Lmax   [default: 16]
    --N=<Nmax>           The value of Nmax   [default: 64]

    --wall_hours=<t>     The number of hours to run for [default: 24]
    --buoy_end_time=<t>  Number of buoyancy times to run [default: 1e5]
    --safety=<s>         Timestep CFL safety factor [default: 0.4]

    --mesh=<n,m>         The processor mesh over which to distribute the cores
    --A0=<A>             Amplitude of initial noise [default: 1e-6]

    --label=<label>      A label to add to the end of the output directory

    --SBDF2              Use SBDF2 (default)
    --SBDF4              Use SBDF4

    --mesa_file=<f>      path to a .h5 file of ICCs, curated from a MESA model
    --restart=<chk_f>    path to a checkpoint file to restart from

    --benchmark          If flagged, do a simple benchmark problem for comparison with the ball-shell
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
Lmax      = nθ = int(args['--L'])
Nmax      = nr = int(args['--N'])
nφ = int(2*nθ)
resolution = (nφ, nθ, nr)
L_dealias = N_dealias = dealias = 1.5

out_dir = './' + sys.argv[0].split('.py')[0]
if args['--mesa_file'] is None:
    out_dir += '_polytrope'
if args['--benchmark']:
    out_dir += '_benchmark'
out_dir += '_Re{}_{}x{}x{}'.format(args['--Re'], *resolution)
if args['--label'] is not None:
    out_dir += '_{:s}'.format(args['--label'])
logger.info('saving data to {:s}'.format(out_dir))
if MPI.COMM_WORLD.rank == 0:
    if not os.path.exists('{:s}/'.format(out_dir)):
        os.makedirs('{:s}/'.format(out_dir))



if args['--SBDF4']:
    ts = d3.SBDF4
    timestepper_history = [0, 1, 2, 3]
else:
    ts = d3.SBDF2
    timestepper_history = [0, 1,]
dtype = np.float64
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


Re  = float(args['--Re'])
Pr  = 1
Pe  = Pr*Re


if args['--mesa_file'] is not None:
    with h5py.File(args['--mesa_file'], 'r') as f:
        radius = f['r_inner'][()]
else:
    raise ValueError("Must provide a path to a MESA NCC file with --mesa_file flag")

# Bases
coords = d3.SphericalCoordinates('φ', 'θ', 'r')
dist   = d3.Distributor((coords,), mesh=mesh, dtype=dtype)
basis  = d3.BallBasis(coords, shape=resolution, radius=radius, dtype=dtype, dealias=(L_dealias, L_dealias, N_dealias))
s2_basis = basis.S2_basis()
radial_basis = basis.radial_basis
φ, θ, r = basis.local_grids(basis.dealias)
φg, θg, rg = basis.global_grids(basis.dealias)

#Operators
ddt  = d3.TimeDerivative
curl = d3.Curl
div  = d3.Divergence
trace = d3.Trace
transpose = d3.TransposeComponents
radComp = d3.RadialComponent
lap       = lambda A: d3.Laplacian(A, coords)
grad      = lambda A: d3.Gradient(A, coords)
dot       = lambda A, B: d3.DotProduct(A, B)
cross     = lambda A, B: d3.CrossProduct(A, B)
angComp   = lambda A, index=1: d3.AngularComponent(A, index=index)
lift_basis = basis.clone_with(k=0)
lift      = lambda A: d3.LiftTau(A, lift_basis, -1)

# Problem variables
u = dist.VectorField(coords, name='u', bases=basis)
p, s1 = [dist.Field(name=n, bases=basis) for n in ['p', 's1']]
tau_u = dist.VectorField(coords, name='tau_u', bases=s2_basis)
tau_T = dist.Field(name='tau_T', bases=s2_basis)

#nccs
grad_ln_ρ, grad_ln_T, grad_s0, grad_T, grad_inv_Pe \
            = [dist.VectorField(coords, name=n, bases=radial_basis) for n in ['grad_ln_ρ', 'grad_ln_T', 'grad_s0', 'grad_T', 'grad_inv_Pe']]
ln_ρ, ln_T, inv_Pe = [dist.Field(name=n, bases=radial_basis) for n in ['ln_ρ', 'ln_T', 'inv_Pe']]
inv_T, H, ρ, T = [dist.Field(name=n, bases=basis) for n in ['inv_T', 'H', 'ρ', 'T']]

#unit vectors
eφ, eθ, er = [dist.VectorField(coords, name=n, bases=basis) for n in ['eφ', 'eθ', 'er']]
eφ['g'][0,:] = 1
eθ['g'][1,:] = 1
er['g'][2,:] = 1

grid_slices = dist.layouts[-1].slices(u.domain, 1)
local_vncc_shape = grad_s0['g'].shape
with h5py.File(args['--mesa_file'], 'r') as f:
    if np.prod(local_vncc_shape) > 0:
        grad_s0['g']         = f['grad_s0B'][()][:,:,:,  grid_slices[2]].reshape(local_vncc_shape)
        grad_ln_ρ['g']       = f['grad_ln_ρB'][()][:,:,:,grid_slices[2]].reshape(local_vncc_shape)
        grad_ln_T['g']       = f['grad_ln_TB'][()][:,:,:,grid_slices[2]].reshape(local_vncc_shape)
        grad_T['g']          = f['grad_TB'][()][:,:,:,grid_slices[2]].reshape(local_vncc_shape)
        grad_inv_Pe['g']     = f['grad_inv_Pe_radB'][()][:,:,:,grid_slices[2]].reshape(local_vncc_shape)
    H['g']         = f['H_effB'][()][:,:,grid_slices[2]]
    ln_ρ['g']      = f['ln_ρB'][()][:,:, grid_slices[2]]
    ln_T['g']      = f['ln_TB'][()][:,:, grid_slices[2]]
    inv_Pe['g']    = f['inv_Pe_radB'][()][:,:, grid_slices[2]]
    ρ['g']         = np.exp(f['ln_ρB'][()][:,:,grid_slices[2]])
    T['g']         = f['TB'][()][:,:,grid_slices[2]]
    inv_T['g']     = 1/T['g']

    max_dt = f['max_dt'][()]
    t_buoy = 1 #Assume nondimensionalization on heating ~ buoyancy time

logger.info('buoyancy time is {}'.format(t_buoy))
t_end = float(args['--buoy_end_time'])*t_buoy

# Stress matrices & viscous terms
I_matrix = dist.TensorField(coords, name='I_matrix', bases=radial_basis)
I_matrix['g'] = 0
for i in range(3):
    I_matrix['g'][i,i,:] = 1

divU = div(u)
E = 0.5*(grad(u) + transpose(grad(u)))
σ = 2*(E - (1/3)*divU*I_matrix)
momentum_viscous_terms = div(σ) + dot(σ, grad_ln_ρ)
VH  = 2*(trace(dot(E, E)) - (1/3)*divU**2)

#Impenetrable, stress-free boundary conditions
u_r_bc    = radComp(u(r=radius))
u_perp_bc = radComp(angComp(E(r=radius), index=1))
therm_bc  = s1(r=radius)

#Extra operators
H = d3.Grid(H).evaluate()
inv_T = d3.Grid(inv_T).evaluate()
grad_s1 = grad(s1)

# Problem
def eq_eval(eq_str):
    return [eval(expr) for expr in d3.split_equation(eq_str)]
problem = d3.IVP([p, u, s1, tau_u, tau_T])

problem.add_equation(eq_eval("div(u) + dot(u, grad_ln_ρ) = 0"), condition="nθ != 0")
problem.add_equation(eq_eval("p = 0"), condition="nθ == 0")
problem.add_equation(eq_eval("ddt(u) + grad(p) + grad_T*s1 - (1/Re)*momentum_viscous_terms + lift(tau_u) = cross(u, curl(u))"), condition = "nθ != 0")
problem.add_equation(eq_eval("u = 0"), condition="nθ == 0")
problem.add_equation(eq_eval("ddt(s1) + dot(u, grad_s0) - inv_Pe*(lap(s1) + dot(grad_s1, (grad_ln_ρ + grad_ln_T))) - dot(grad_s1, grad_inv_Pe) + lift(tau_T) = - dot(u, grad_s1) + H + (1/Re)*inv_T*VH "))
problem.add_equation(eq_eval("u_r_bc    = 0"), condition="nθ != 0")
problem.add_equation(eq_eval("u_perp_bc = 0"), condition="nθ != 0")
problem.add_equation(eq_eval("tau_u     = 0"), condition="nθ == 0")
problem.add_equation(eq_eval("therm_bc  = 0"))

logger.info("Problem built")
# Solver
solver = problem.build_solver(ts)
solver.stop_sim_time = t_end
logger.info("solver built")

## Analysis Setup
scalar_dt = 0.25*t_buoy
flux_dt   = 0.5*t_buoy
visual_dt = 0.05*t_buoy
logger.info("output times... scalars: {:2e} / profiles: {:2e} / slices: {:.2e}".format(scalar_dt, flux_dt, visual_dt))
ur = dot(er, u)
u_squared = dot(u,u)
h = p - 0.5*u_squared + T*s1
pomega_hat = p - 0.5*u_squared
visc_flux_r = 2*dot(er, dot(u, E) - (1/3) * u * divU)

r_vals = dist.Field(name='r_vals', bases=basis)
r_vals.set_scales(basis.dealias)
r_vals['g'] = r
r_vals = d3.Grid(r_vals).evaluate()

volume = (4/3)*np.pi*radius**3
logger.info('volume: {}'.format(volume))

#Analysis operations
az_avg = lambda A: d3.Average(A, coords.coords[0])
s2_avg = lambda A: d3.Average(A, coords.S2coordsys)
vol_avg = lambda A: d3.Integrate(A/volume, coords)
luminosity = lambda A: (4*np.pi*r_vals**2) * s2_avg(A)

#Setup outputs
analysis_tasks = []
scalars = solver.evaluator.add_file_handler('{:s}/scalars'.format(out_dir), max_writes=np.inf, sim_dt=scalar_dt)
scalars.add_task(vol_avg(Re*(u_squared)**(1/2)), name='Re_avg', layout='g')
scalars.add_task(vol_avg(ρ*u_squared/2),         name='KE',     layout='g')
scalars.add_task(vol_avg(ρ*T*s1),                name='TE',     layout='g')
analysis_tasks.append(scalars)

profiles = solver.evaluator.add_file_handler('{:s}/profiles'.format(out_dir), max_writes=100, sim_dt=flux_dt)
profiles.add_task(luminosity(ρ*ur*h),                      name='enth_lum', layout='g')
profiles.add_task(luminosity(-ρ*visc_flux_r/Re),           name='visc_lum', layout='g')
profiles.add_task(luminosity(-ρ*T*dot(er, grad_s1)/Pe),    name='cond_lum', layout='g')
profiles.add_task(luminosity(0.5*ρ*ur*u_squared),          name='KE_lum',   layout='g')
profiles.add_task(luminosity(ρ*ur*pomega_hat),             name='wave_lum', layout='g')
analysis_tasks.append(profiles)

slices = solver.evaluator.add_file_handler('{:s}/slices'.format(out_dir), max_writes=40, sim_dt=visual_dt)
slices.add_task(u(r=0.5), name='u_r0.5', layout='g')
slices.add_task(s1(r=0.5), name='s1_r0.5',  layout='g')
slices.add_task(u(r=0.95), name='u_r0.95', layout='g')
slices.add_task(s1(r=0.95), name='s1_r0.95',  layout='g')
slices.add_task(u(θ=np.pi/2),  name='u_eq', layout='g')
slices.add_task(s1(θ=np.pi/2), name='s1_eq', layout='g')
analysis_tasks.append(slices)

checkpoint = solver.evaluator.add_file_handler('{:s}/checkpoint'.format(out_dir), max_writes=1, sim_dt=10*t_buoy)
checkpoint.add_task(s1, name='s1', scales=1, layout='g')
checkpoint.add_task(u, name='u', scales=1, layout='g')

imaginary_cadence = 100

#CFL setup
dt = max_dt
my_cfl = d3.CFL(solver, max_dt, safety=float(args['--safety']), cadence=1, max_dt=max_dt, min_change=0.1, max_change=1.5, threshold=0.1)
my_cfl.add_velocity(u)

#Loop Re output setup
re_ball = solver.evaluator.add_dictionary_handler(iter=10)
re_ball.add_task(vol_avg(Re*(u_squared)**(1/2)), name='Re_avg', layout='g')



if args['--restart'] is not None:
    fname = args['--restart']
    fdir = fname.split('.h5')[0]
    check_name = fdir.split('/')[-1]
    #Try to just load the loal piece file

    import h5py
    with h5py.File('{}/{}_p{}.h5'.format(fdir, check_name, dist.comm_cart.rank), 'r') as f:
        s1.set_scales(1)
        u.set_scales(1)
        s1['c'] = f['tasks/s1'][()][-1,:]
        u['c'] = f['tasks/u'][()][-1,:]
        s1.require_scales(dealias)
        u.require_scales(dealias)
else:
    if args['--benchmark']:
        #Marti benchmark-like ICs
        A0 = 1e-3
        s1['g'] = A0*np.sqrt(35/np.pi)*(r/radius)**3*(1-(r/radius)**2)*(np.cos(φ)+np.sin(φ))*np.sin(θ)**3
    else:
        # Initial conditions
        A0   = float(1e-6)
        seed = 42 + dist.comm_cart.rank
        rand = np.random.RandomState(seed=seed)
        filter_scale = 0.25

        # Generate noise & filter it
        s1['g'] = A0*rand.standard_normal(s1['g'].shape)
        s1.require_scales(filter_scale)
        s1['c']
        s1['g']
        s1.require_scales(dealias)

# Main loop
start_time = time.time()
start_iter = solver.iteration
try:
    while solver.ok:
        solver.step(dt)
        dt = my_cfl.compute_dt()

        if solver.iteration % 10 == 0:
            Re0 = re_ball.fields['Re_avg']['g']
            if dist.comm_cart.rank == 0:
                logger.info("t = {:f}, dt = {:f}, Re = {:e}".format(solver.sim_time, dt, Re0.min()))

        if solver.iteration % imaginary_cadence in timestepper_history:
            for f in solver.state:
                f.require_grid_space()
except:
    logger.info('something went wrong in main loop, making final checkpoint')
    raise
finally:
    fcheckpoint = solver.evaluator.add_file_handler('{:s}/final_checkpoint'.format(out_dir), max_writes=1, iter=1)
    fcheckpoint.add_task(s1, name='s1', scales=1, layout='c')
    fcheckpoint.add_task(u, name='u', scales=1, layout='c')
    solver.step(1e-5*dt)

    end_time = time.time()
    end_iter = solver.iteration
    cpu_sec  = end_time - start_time
    n_iter   = end_iter - start_iter

    #TODO: Make the end-of-sim report better
    n_coeffs = 2*(Nmax+1)*(Lmax+1)*(Lmax+2)
    n_cpu    = dist.comm_cart.size
    dof_cycles_per_cpusec = n_coeffs*n_iter/(cpu_sec*n_cpu)
    logger.info('DOF-cycles/cpu-sec : {:e}'.format(dof_cycles_per_cpusec))
    logger.info('Run iterations: {:e}'.format(n_iter))
    logger.info('Sim end time: {:e}'.format(solver.sim_time))
    logger.info('Run time: {}'.format(cpu_sec))
