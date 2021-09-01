"""
d3 script for anelastic convection in a massive star's core CZ.

A config file can be provided which overwrites any number of the options below, but command line flags can also be used if preferred.

Usage:
    ball_AN.py [options]
    ball_AN.py <config> [options]

Options:
    --Re=<Re>            The Reynolds number of the numerical diffusivities [default: 2e2]
    --Pr=<Prandtl>       The Prandtl number  of the numerical diffusivities [default: 1]
    --L=<Lmax>           Spherical harmonic degrees of freedom (Lmax+1)   [default: 16]
    --N=<Nmax>           Radial degrees of freedom (Nmax+1)   [default: 32]

    --wall_hours=<t>     Max number of wall hours to run simulation for [default: 24]
    --buoy_end_time=<t>  Max number of buoyancy time units to simulate [default: 1e5]

    --mesh=<n,m>         The processor mesh over which to distribute the cores
    --A0=<A>             Amplitude of initial noise [default: 1e-6]

    --label=<label>      A label to add to the end of the output directory

    --SBDF4              Use SBDF4 (default: SBDF2)
    --safety=<s>         CFL safety factor for determining timestep size [default: 0.2]

    --mesa_file=<f>      path to a .h5 file of ICCs, curated from a MESA model; if unspecified, uses a polytropic stratification.
    --restart=<chk_f>    path to a checkpoint file to restart from
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
nθ = int(args['--L'])
nr = int(args['--N'])
nφ = int(2*nθ)
resolution = (nφ, nθ, nr)
L_dealias = N_dealias = dealias = 1.5
dtype = np.float64
Re  = float(args['--Re'])
Pr  = 1
Pe  = Pr*Re
mesa_file = args['--mesa_file']
wall_hours = float(args['--wall_hours'])
buoy_end_time = float(args['--buoy_end_time'])

# Initial conditions
restart = args['--restart']
A0      = float(args['--A0'])

# timestepper
if args['--SBDF4']:
    ts = d3.SBDF4
    timestepper_history = [0, 1, 2, 3]
else:
    ts = d3.SBDF2
    timestepper_history = [0, 1,]
safety = float(args['--safety'])
hermitian_cadence = 100


# processor mesh
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

#output directory
out_dir = './' + sys.argv[0].split('.py')[0]
if mesa_file is None:
    out_dir += '_polytrope'
out_dir += '_Re{}_{}x{}x{}'.format(args['--Re'], *resolution)
if args['--label'] is not None:
    out_dir += '_{:s}'.format(args['--label'])
logger.info('saving data to {:s}'.format(out_dir))
if MPI.COMM_WORLD.rank == 0:
    if not os.path.exists('{:s}/'.format(out_dir)):
        os.makedirs('{:s}/'.format(out_dir))


if mesa_file is not None:
    with h5py.File(mesa_file, 'r') as f:
        radius = f['r_inner'][()]
else:
    radius = 1.5

# Bases
coords = d3.SphericalCoordinates('φ', 'θ', 'r')
dist   = d3.Distributor((coords,), mesh=mesh, dtype=dtype)
basis  = d3.BallBasis(coords, shape=resolution, radius=radius, dtype=dtype, dealias=(L_dealias, L_dealias, N_dealias))
s2_basis = basis.S2_basis()
radial_basis = basis.radial_basis
φ, θ, r = basis.local_grids(basis.dealias)
φ1, θ1, r1 = basis.local_grids((1,1,1))
φg, θg, rg = basis.global_grids(basis.dealias)

#Operators
ddt  = d3.TimeDerivative
curl = d3.Curl
div  = d3.Divergence
trace = d3.Trace
transpose = d3.TransposeComponents
radComp = d3.RadialComponent
cross = d3.CrossProduct
dot = d3.DotProduct
lap       = lambda A: d3.Laplacian(A, coords)
grad      = lambda A: d3.Gradient(A, coords)
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

#unit vectors & (NCC) identity matrix
eφ, eθ, er = [dist.VectorField(coords, name=n, bases=basis) for n in ['eφ', 'eθ', 'er']]
I_matrix = dist.TensorField(coords, name='I_matrix', bases=radial_basis)
for f in [eφ, eθ, er, I_matrix]: f['g'] = 0
eφ['g'][0,:] = 1
eθ['g'][1,:] = 1
er['g'][2,:] = 1
for i in range(3):
    I_matrix['g'][i,i,:] = 1

# Stress matrices & viscous terms (assumes uniform kinematic viscosity, so dynamic viscosity mu = const * rho)
divU = div(u)
E = 0.5*(grad(u) + transpose(grad(u)))
σ = 2*(E - (1/3)*divU*I_matrix)
visc_div_stress = div(σ) + dot(σ, grad_ln_ρ)
VH  = 2*(trace(dot(E, E)) - (1/3)*divU**2)

# Impenetrable, stress-free boundary conditions
u_r_bc    = radComp(u(r=radius))
u_perp_bc = radComp(angComp(E(r=radius), index=1))
therm_bc  = s1(r=radius)

# Load MESA NCC file or setup NCCs using polytrope.
grid_slices = dist.layouts[-1].slices(u.domain, 1)
local_vncc_shape = grad_s0['g'].shape
if mesa_file is not None:
    with h5py.File(mesa_file, 'r') as f:
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
else:
    logger.info("Using polytropic initial conditions")
    from scipy.interpolate import interp1d
    with h5py.File('polytropes/poly_nOuter1.6.h5', 'r') as f:
        T_func = interp1d(f['r'][()], f['T'][()])
        ρ_func = interp1d(f['r'][()], f['ρ'][()])
        grad_s0_func = interp1d(f['r'][()], f['grad_s0'][()])
        H_func   = interp1d(f['r'][()], f['H_eff'][()])

    T['g']       = T_func(r1)
    ρ['g']       = ρ_func(r1)
    H['g']       = H_func(r1)
    inv_T['g']   = 1/T_func(r1)

    grad_ln_ρ_full = (grad(ρ)/ρ).evaluate()
    grad_T_full = grad(T).evaluate()
    grad_ln_T_full = (grad_T_full/T).evaluate()
    if np.prod(local_vncc_shape) > 0:
        grad_s0['g'][2]  = grad_s0_func(r1)
        for f in [grad_ln_ρ, grad_ln_T, grad_T]: f.require_scales(basis.dealias)
        grad_ln_ρ['g']   = grad_ln_ρ_full['g'][:,0,0,None,None,:]
        grad_ln_T['g']   = grad_ln_T_full['g'][:,0,0,None,None,:]
        grad_T['g']      = grad_T_full['g'][:,0,0,None,None,:]
        grad_inv_Pe['g'] = 0
    ln_T['g']        = np.log(T_func(r1))
    ln_ρ['g']        = np.log(ρ_func(r1))
    inv_Pe['g']      = 1/Pe

    t_buoy      = 1

# Problem
def eq_eval(eq_str):
    return [eval(expr) for expr in d3.split_equation(eq_str)]
problem = d3.IVP([p, u, s1, tau_u, tau_T])

# Grid-lock / define extra operators
H = d3.Grid(H).evaluate()
inv_T = d3.Grid(inv_T).evaluate()
grad_s1 = grad(s1)

# Equations / Problem
problem.add_equation(eq_eval("div(u) + dot(u, grad_ln_ρ) = 0"), condition="nθ != 0")
problem.add_equation(eq_eval("p = 0"), condition="nθ == 0")
problem.add_equation(eq_eval("ddt(u) + grad(p) + grad_T*s1 - (1/Re)*visc_div_stress + lift(tau_u) = cross(u, curl(u))"), condition = "nθ != 0")
problem.add_equation(eq_eval("u = 0"), condition="nθ == 0")
problem.add_equation(eq_eval("ddt(s1) + dot(u, grad_s0) - inv_Pe*(lap(s1) + dot(grad_s1, (grad_ln_ρ + grad_ln_T))) - dot(grad_s1, grad_inv_Pe) + lift(tau_T) = - dot(u, grad_s1) + H + (1/Re)*inv_T*VH "))
problem.add_equation(eq_eval("u_r_bc    = 0"), condition="nθ != 0")
problem.add_equation(eq_eval("u_perp_bc = 0"), condition="nθ != 0")
problem.add_equation(eq_eval("tau_u     = 0"), condition="nθ == 0")
problem.add_equation(eq_eval("therm_bc  = 0"))
logger.info("Problem built")

# Solver
solver = problem.build_solver(ts)
solver.stop_sim_time = buoy_end_time*t_buoy
solver.stop_wall_time = wall_hours * 60 * 60
logger.info("solver built")

# Initial conditions / Checkpoint
write_mode = 'overwrite'
dt = None
if restart is not None:
    write, dt = solver.load_state(restart)
    write_mode = 'append'
else:
    # Initial conditions
    A0   = float(args['--A0'])
    seed = 42 + dist.comm_cart.rank
    rand = np.random.RandomState(seed=seed)
    filter_scale = 0.25

    # Generate noise & filter it
    s1['g'] = A0*rand.standard_normal(s1['g'].shape)*(1 - (r1/radius)**2)
    s1.require_scales(filter_scale)
    s1['c']
    s1['g']
    s1.require_scales(dealias)

## Analysis Setup
# Cadence
scalar_dt = 0.25*t_buoy
flux_dt   = 0.5*t_buoy
visual_dt = 0.05*t_buoy
logger.info("output times... scalars: {:2e} / profiles: {:2e} / slices: {:.2e}".format(scalar_dt, flux_dt, visual_dt))

# Operators, extra fields
ur = dot(er, u)
u_squared = dot(u,u)
h = p - 0.5*u_squared + T*s1
pomega_hat = p - 0.5*u_squared
visc_flux_r = 2*dot(er, dot(u, E) - (1/3) * u * divU)

r_vals = dist.Field(name='r_vals', bases=basis)
r_vals['g'] = r1
r_vals = d3.Grid(r_vals).evaluate()

volume = (4/3)*np.pi*radius**3
logger.info('volume: {}'.format(volume))

# Averaging operations
az_avg = lambda A: d3.Average(A, coords.coords[0])
s2_avg = lambda A: d3.Average(A, coords.S2coordsys)
vol_avg = lambda A: d3.Integrate(A/volume, coords)
luminosity = lambda A: (4*np.pi*r_vals**2) * s2_avg(A)

# Specify output tasks
analysis_tasks = []
scalars = solver.evaluator.add_file_handler('{:s}/scalars'.format(out_dir), max_writes=np.inf, sim_dt=scalar_dt, mode=write_mode)
scalars.add_task(vol_avg(Re*(u_squared)**(1/2)), name='Re_avg', layout='g')
scalars.add_task(vol_avg(ρ*u_squared/2),         name='KE',     layout='g')
scalars.add_task(vol_avg(ρ*T*s1),                name='TE',     layout='g')
analysis_tasks.append(scalars)

profiles = solver.evaluator.add_file_handler('{:s}/profiles'.format(out_dir), max_writes=100, sim_dt=flux_dt, mode=write_mode)
profiles.add_task(luminosity(ρ*ur*h),                      name='enth_lum', layout='g')
profiles.add_task(luminosity(-ρ*visc_flux_r/Re),           name='visc_lum', layout='g')
profiles.add_task(luminosity(-ρ*T*dot(er, grad_s1)/Pe),    name='cond_lum', layout='g')
profiles.add_task(luminosity(0.5*ρ*ur*u_squared),          name='KE_lum',   layout='g')
profiles.add_task(luminosity(ρ*ur*pomega_hat),             name='wave_lum', layout='g')
analysis_tasks.append(profiles)

slices = solver.evaluator.add_file_handler('{:s}/slices'.format(out_dir), max_writes=40, sim_dt=visual_dt, mode=write_mode)
slices.add_task(u(r=0.5), name='u_r0.5', layout='g')
slices.add_task(s1(r=0.5), name='s1_r0.5',  layout='g')
slices.add_task(u(r=0.95), name='u_r0.95', layout='g')
slices.add_task(s1(r=0.95), name='s1_r0.95',  layout='g')
slices.add_task(u(θ=np.pi/2),  name='u_eq', layout='g')
slices.add_task(s1(θ=np.pi/2), name='s1_eq', layout='g')
analysis_tasks.append(slices)

# Checkpoint 
checkpoint_sim_dt_cadence = 10*t_buoy
checkpoint = solver.evaluator.add_file_handler('{:s}/checkpoint'.format(out_dir), max_writes=1, sim_dt=checkpoint_sim_dt_cadence, mode=write_mode)
checkpoint.add_tasks(solver.state, layout='g')

#CFL setup
max_dt = 0.5*t_buoy
if dt is None:
    dt = max_dt
my_cfl = d3.CFL(solver, dt, safety=safety, cadence=1, max_dt=max_dt, min_change=0.1, max_change=1.5, threshold=0.1)
my_cfl.add_velocity(u)

#Loop Re output setup
re_ball = solver.evaluator.add_dictionary_handler(iter=10)
re_ball.add_task(vol_avg(Re*(u_squared)**(1/2)), name='Re_avg', layout='g')

# Main loop
start_time = time.time()
start_iter = solver.iteration
try:
    while solver.proceed:
        solver.step(dt)
        dt = my_cfl.compute_timestep()

        if solver.iteration % 10 == 0:
            Re0 = re_ball.fields['Re_avg']['g']
            logger.info("t = {:f}, dt = {:f}, Re = {:e}".format(solver.sim_time, dt, Re0.min()))

        if solver.iteration % hermitian_cadence in timestepper_history:
            for f in solver.state:
                f.require_grid_space()
except:
    logger.info('something went wrong in main loop, making final checkpoint')
    raise
finally:
    fcheckpoint = solver.evaluator.add_file_handler('{:s}/final_checkpoint'.format(out_dir), max_writes=1, iter=1, mode=write_mode)
    fcheckpoint.add_tasks(solver.state, layout='g')
    solver.step(dt)

    end_time = time.time()
    end_iter = solver.iteration
    cpu_sec  = end_time - start_time
    n_iter   = end_iter - start_iter

    #TODO: Make the end-of-sim report better
    n_coeffs = np.prod(resolution)
    n_cpu    = dist.comm_cart.size
    dof_cycles_per_cpusec = n_coeffs*n_iter/(cpu_sec*n_cpu)
    logger.info('DOF-cycles/cpu-sec : {:e}'.format(dof_cycles_per_cpusec))
    logger.info('Run iterations: {:e}'.format(n_iter))
    logger.info('Sim end time: {:e}'.format(solver.sim_time))
    logger.info('Run time: {}'.format(cpu_sec))
