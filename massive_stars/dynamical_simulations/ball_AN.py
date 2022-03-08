"""
d3 script for anelastic convection in a fully-convective ball domain.

A config file can be provided which overwrites any number of the options below, but command line flags can also be used if preferred.
Note that options specified in a config file override command line arguments.

Usage:
    ball_AN.py [options]
    ball_AN.py <config> [options]

Options:
    --Re=<Re>            The Reynolds number of the numerical diffusivities [default: 2e2]
    --Pr=<Prandtl>       The Prandtl number  of the numerical diffusivities [default: 1]
    --ntheta=<res>       Number of theta grid points (Lmax+1)   [default: 16]
    --nr=<res>           Number of radial grid points (Nmax+1)   [default: 32]

    --wall_hours=<t>     Max number of wall hours to run simulation for [default: 24]
    --buoy_end_time=<t>  Max number of buoyancy time units to simulate [default: 1e5]

    --mesh=<n,m>         The processor mesh over which to distribute the cores
    --A0=<A>             Amplitude of initial noise [default: 1e-6]

    --label=<label>      A label to add to the end of the output directory

    --SBDF4              Use SBDF4 (default: SBDF2)
    --safety=<s>         CFL safety factor for determining timestep size [default: 0.2]

    --ncc_file=<f>      path to a .h5 file of NCCs, curated from a MESA model; if None, uses a polytropic stratification.
    --restart=<chk_f>    path to a checkpoint file to restart from
"""
import os
import time
import sys
from operator import itemgetter
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

from anelastic_functions import make_bases, make_fields, fill_structure, get_anelastic_variables, set_anelastic_problem

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
ntheta = int(args['--ntheta'])
nr = int(args['--nr'])
nphi = int(2*ntheta)
resolution = (nphi, ntheta, nr)
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
if ncc_file is None:
    out_dir += '_polytrope'
out_dir += '_Re{}_{}x{}x{}'.format(args['--Re'], *resolution)
if args['--label'] is not None:
    out_dir += '_{:s}'.format(args['--label'])
logger.info('saving data to {:s}'.format(out_dir))
if MPI.COMM_WORLD.rank == 0:
    if not os.path.exists('{:s}/'.format(out_dir)):
        os.makedirs('{:s}/'.format(out_dir))

# Bases
if ncc_file is not None:
    with h5py.File(ncc_file, 'r') as f:
        radius = f['r_inner'][()]
else:
    radius = 1.5

resolutions = (resolution,)
stitch_radii = ()
radius = radius
coords, dist, bases, bases_keys = make_bases(resolutions, stitch_radii, radius, dealias=(L_dealias, L_dealias, N_dealias), dtype=dtype, mesh=mesh)
print(bases, bases_keys)

vec_fields = ['u',]
scalar_fields = ['p', 's1', 'inv_T', 'H', 'ρ', 'T']
vec_taus = ['tau_u']
scalar_taus = ['tau_s']
vec_nccs = ['grad_ln_ρ', 'grad_ln_T', 'grad_s0', 'grad_T', 'grad_inv_Pe_rad']
scalar_nccs = ['ln_ρ', 'ln_T', 'inv_Pe_rad', 'sponge']
variables = make_fields(bases, coords, dist, 
                        vec_fields=vec_fields, scalar_fields=scalar_fields, 
                        vec_taus=vec_taus, scalar_taus=scalar_taus, 
                        vec_nccs=vec_nccs, scalar_nccs=scalar_nccs,
                        sponge=False, do_rotation=False)


variables, timescales = fill_structure(bases, dist, variables, ncc_file, radius, Pe, 
                                vec_fields=vec_fields, vec_nccs=vec_nccs, scalar_nccs=scalar_nccs,
                                sponge=False, do_rotation=False)
max_dt, t_buoy, t_rot = timescales

# Put nccs and fields into locals()
locals().update(variables)


# Problem
prob_variables = get_anelastic_variables(bases, bases_keys, variables)
problem = d3.IVP(prob_variables, namespace=locals())

problem = set_anelastic_problem(problem, bases, bases_keys)

# Solver
solver = problem.build_solver(ts)
solver.stop_sim_time = buoy_end_time*t_buoy
solver.stop_wall_time = wall_hours * 60 * 60
logger.info("solver built")


bn = 'B'
basis = bases[bn]
phi, theta, r = itemgetter('phi_'+bn, 'theta_'+bn, 'r_'+bn)(variables)
phi1, theta1, r1 = itemgetter('phi1_'+bn, 'theta1_'+bn, 'r1_'+bn)(variables)
ex, ey, ez = itemgetter('ex_'+bn, 'ey_'+bn, 'ez_'+bn)(variables)
T, ρ = itemgetter('T_{}'.format(bn), 'ρ_{}'.format(bn))(variables)
div_u, E = itemgetter('div_u_RHS_{}'.format(bn), 'E_RHS_{}'.format(bn))(variables)
u = variables['u_{}'.format(bn)]
p = variables['p_{}'.format(bn)]
s1 = variables['s1_{}'.format(bn)]
er = variables['er_{}'.format(bn)]

# Initial conditions / Checkpoint
write_mode = 'overwrite'
dt = None
if restart is not None:
    write, dt = solver.load_state(restart)
    write_mode = 'append'
else:
    s1.fill_random(layout='g', seed=42, distribution='normal', scale=A0)
    s1.low_pass_filter(scales=0.25)

## Analysis Setup
# Cadence
scalar_dt = 0.25*t_buoy
flux_dt   = 0.5*t_buoy
visual_dt = 0.05*t_buoy
logger.info("output times... scalars: {:2e} / profiles: {:2e} / slices: {:.2e}".format(scalar_dt, flux_dt, visual_dt))

# Operators, extra fields
ur = d3.dot(er, u)
u_squared = d3.dot(u,u)
h = p - 0.5*u_squared + T*s1
pomega_hat = p - 0.5*u_squared
visc_flux_r = 2*d3.dot(er, d3.dot(u, E) - (1/3) * u * div_u)

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
profiles.add_task(luminosity(-ρ*T*d3.dot(er, d3.grad(s1))/Pe),    name='cond_lum', layout='g')
profiles.add_task(luminosity(0.5*ρ*ur*u_squared),          name='KE_lum',   layout='g')
profiles.add_task(luminosity(ρ*ur*pomega_hat),             name='wave_lum', layout='g')
analysis_tasks.append(profiles)

slices = solver.evaluator.add_file_handler('{:s}/slices'.format(out_dir), max_writes=40, sim_dt=visual_dt, mode=write_mode)
slices.add_task(u(r=0.5), name='u_r0.5', layout='g')
slices.add_task(s1(r=0.5), name='s1_r0.5',  layout='g')
slices.add_task(u(r=0.95), name='u_r0.95', layout='g')
slices.add_task(s1(r=0.95), name='s1_r0.95',  layout='g')
slices.add_task(u(theta=np.pi/2),  name='u_eq', layout='g')
slices.add_task(s1(theta=np.pi/2), name='s1_eq', layout='g')
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
            Re_avg = re_ball.fields['Re_avg']
            if dist.comm_cart.rank == 0:
                Re0 = Re_avg['g'].min()
            else:
                Re0 = None
            Re0 = dist.comm_cart.bcast(Re0, root=0)
            logger.info("t = %f, dt = %f, Re = %e" %(solver.sim_time, dt, Re0))

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
