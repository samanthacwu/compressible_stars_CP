"""
d3 script for anelastic convection in a massive star's core CZ.

A config file can be provided which overwrites any number of the options below, but command line flags can also be used if preferred.

Usage:
    coreCZ_AN.py [options]
    coreCZ_AN.py <config> [options]

Options:
    --Re=<Re>            The Reynolds number of the numerical diffusivities [default: 5e1]
    --Pr=<Prandtl>       The Prandtl number  of the numerical diffusivities [default: 1]
    --L=<Lmax>           The value of Lmax   [default: 6]
    --N=<Nmax>           The value of Nmax   [default: 31]

    --wall_hours=<t>     The number of hours to run for [default: 24]
    --safety=<s>         Timestep CFL safety factor [default: 0.4]
    --niter=<n>          Number of iterations to run [default: 110]

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
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic
from dedalus.tools import logging
from dedalus.tools.parsing import split_equation
from dedalus.extras.flow_tools import GlobalArrayReducer
from scipy import sparse
import dedalus_sphere
from mpi4py import MPI

from d3_outputs.extra_ops    import BallVolumeAverager, EquatorSlicer, PhiAverager, PhiThetaAverager, OutputRadialInterpolate, GridSlicer
from d3_outputs.writing      import d3FileHandler

import logging
logger = logging.getLogger(__name__)

from dedalus.tools.config import config
config['linear algebra']['MATRIX_FACTORIZER'] = 'SuperLUNaturalFactorizedTranspose'

from scipy.special import erf
def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
    return -(one_to_zero(*args, **kwargs) - 1)




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
Lmax      = int(args['--L'])
Nmax      = int(args['--N'])
L_dealias = N_dealias = dealias = 1

out_dir = './' + sys.argv[0].split('.py')[0]
if args['--mesa_file'] is None:
    out_dir += '_polytrope'
if args['--benchmark']:
    out_dir += '_benchmark'
out_dir += '_Re{}_{}x{}'.format(args['--Re'], args['--L'], args['--N'])
if args['--label'] is not None:
    out_dir += '_{:s}'.format(args['--label'])
logger.info('saving data to {:s}'.format(out_dir))
if MPI.COMM_WORLD.rank == 0:
    if not os.path.exists('{:s}/'.format(out_dir)):
        os.makedirs('{:s}/'.format(out_dir))



if args['--SBDF4']:
    ts = timesteppers.SBDF4
    timestepper_history = [0, 1, 2, 3]
else:
    ts = timesteppers.SBDF2
    timestepper_history = [0, 1,]
dtype = np.float64

comm = MPI.COMM_WORLD
ncpu = comm.size
rank = comm.rank
mesh = args['--mesh']
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
    radius = 1.5

# Bases
c = coords.SphericalCoordinates('φ', 'θ', 'r')
d = distributor.Distributor((c,), mesh=mesh)
b = basis.BallBasis(c, (2*(Lmax+2), Lmax+1, Nmax+1), radius=radius, dtype=dtype)
b_S2 = b.S2_basis()
φ, θ, r = b.local_grids((dealias, dealias, dealias))
φg, θg, rg = b.global_grids((dealias, dealias, dealias))

#Operators
div       = lambda A: operators.Divergence(A, index=0)
lap       = lambda A: operators.Laplacian(A, c)
grad      = lambda A: operators.Gradient(A, c)
dot       = lambda A, B: arithmetic.DotProduct(A, B)
curl      = lambda A: operators.Curl(A)
cross     = lambda A, B: arithmetic.CrossProduct(A, B)
trace     = lambda A: operators.Trace(A)
ddt       = lambda A: operators.TimeDerivative(A)
transpose = lambda A: operators.TransposeComponents(A)
radComp   = lambda A: operators.RadialComponent(A)
angComp   = lambda A, index=1: operators.AngularComponent(A, index=index)
LiftTau   = lambda A: operators.LiftTau(A, b, -1)

# Fields
u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
p = field.Field(dist=d, bases=(b,), dtype=dtype)
s1 = field.Field(dist=d, bases=(b,), dtype=dtype)
tau_u = field.Field(dist=d, bases=(b_S2,), tensorsig=(c,), dtype=dtype)
tau_T = field.Field(dist=d, bases=(b_S2,), dtype=dtype)

ρ   = field.Field(dist=d, bases=(b,), dtype=dtype)
T   = field.Field(dist=d, bases=(b,), dtype=dtype)


er = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
er.set_scales(dealias)
er['g'][2,:] = 1

#nccs
grad_ln_ρ    = field.Field(dist=d, bases=(b.radial_basis,), tensorsig=(c,), dtype=dtype)
grad_s0      = field.Field(dist=d, bases=(b.radial_basis,), tensorsig=(c,), dtype=dtype)
grad_ln_T    = field.Field(dist=d, bases=(b.radial_basis,), tensorsig=(c,), dtype=dtype)
ln_ρ    = field.Field(dist=d, bases=(b.radial_basis,), dtype=dtype)
ln_T    = field.Field(dist=d, bases=(b.radial_basis,), dtype=dtype)
T_NCC   = field.Field(dist=d, bases=(b.radial_basis,), dtype=dtype)
inv_T   = field.Field(dist=d, bases=(b,), dtype=dtype) #only on RHS, multiplies other terms
H_eff   = field.Field(dist=d, bases=(b,), dtype=dtype)
grad_s0_RHS      = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)


slicer = GridSlicer(p)
if args['--mesa_file'] is not None:
    with h5py.File(args['--mesa_file'], 'r') as f:
        if np.prod(grad_s0['g'].shape) > 0:
            grad_s0['g']        = f['grad_s0B'][()][:,:,:,  slicer[2]].reshape(grad_s0['g'].shape)
            grad_ln_ρ['g']      = f['grad_ln_ρB'][()][:,:,:,slicer[2]].reshape(grad_s0['g'].shape)
            grad_ln_T['g']      = f['grad_ln_TB'][()][:,:,:,slicer[2]].reshape(grad_s0['g'].shape)
        ln_ρ['g']      = f['ln_ρB'][()][:,:, slicer[2]]
        ln_T['g']      = f['ln_TB'][()][:,:, slicer[2]]
        H_eff['g']     = f['H_effB'][()][:,:,slicer[2]]
        T_NCC['g']     = f['TB'][()][:,:,slicer[2]]
        ρ['g']         = np.exp(f['ln_ρB'][()][:,:,slicer[2]].reshape(r.shape))
        T['g']         = f['TB'][()][:,:,slicer[2]].reshape(r.shape)
        inv_T['g']     = 1/T['g']
        grad_s0_RHS['g'][2]        = f['grad_s0B'][()][2,:,:,slicer[2]].reshape(r.shape)

        t_buoy = 1
        max_grad_s0 = f['grad_s0B'][()][2,0,0,-1]

else:
    logger.info("Using polytropic initial conditions")
    from scipy.interpolate import interp1d
    with h5py.File('../polytropes/poly_nOuter1.6.h5', 'r') as f:
        T_func = interp1d(f['r'][()], f['T'][()])
        ρ_func = interp1d(f['r'][()], f['ρ'][()])
        grad_s0_func = interp1d(f['r'][()], f['grad_s0'][()])
        H_eff_func   = interp1d(f['r'][()], f['H_eff'][()])
    grad_s0['g'][2]     = grad_s0_func(r)#*zero_to_one(r, 0.5, width=0.1)
    grad_s0_RHS['g'][2] = grad_s0_func(r)#*zero_to_one(r, 0.5, width=0.1)
    T['g']           = T_func(r)
    T_NCC['g']       = T_func(r)
    ρ['g']           = ρ_func(r)
    inv_T['g']       = T_func(r)
    H_eff['g']       = H_eff_func(r)
    ln_T['g']        = np.log(T_func(r))
    ln_ρ['g']        = np.log(ρ_func(r))
    grad_ln_ρ['g']        = grad(ln_ρ).evaluate()['g']
    grad_ln_T['g']        = grad(ln_T).evaluate()['g']

    max_grad_s0 = grad_s0_func(radius)
    t_buoy      = 1

max_dt = 3/np.sqrt(max_grad_s0)
logger.info('buoyancy time is {}'.format(t_buoy))

for f in [u, s1, p, ln_ρ, ln_T, inv_T, H_eff, ρ]:
    f.require_scales(dealias)

# Stress matrices & viscous terms
I_matrix_post = field.Field(dist=d, bases=(b,), tensorsig=(c,c,), dtype=dtype)
I_matrix = field.Field(dist=d, bases=(b.radial_basis,), tensorsig=(c,c,), dtype=dtype)
I_matrix_post['g'] = 0
I_matrix['g'] = 0
for i in range(3):
    I_matrix_post['g'][i,i,:] = 1
    I_matrix['g'][i,i,:] = 1

E = 0.5*(grad(u) + transpose(grad(u)))
E.store_last = True
divU = div(u)
divU.store_last = True
σ = 2*(E - (1/3)*divU*I_matrix)
σ_post = 2*(E - (1/3)*divU*I_matrix_post)
momentum_viscous_terms = div(σ) + dot(σ, grad_ln_ρ)

#trace_E = trace(E)
#trace_E.store_last = True
VH  = 2*(trace(dot(E, E)) - (1/3)*divU*divU)

#Impenetrable, stress-free boundary conditions
u_r_bc    = radComp(u(r=radius))
u_perp_bc = radComp(angComp(E(r=radius), index=1))
therm_bc  = s1(r=radius)

H_eff = operators.Grid(H_eff).evaluate()
inv_T = operators.Grid(inv_T).evaluate()
grads1 = grad(s1)
grads1.store_last = True

# Problem
def eq_eval(eq_str):
    return [eval(expr) for expr in split_equation(eq_str)]
problem = problems.IVP([p, u, s1, tau_u, tau_T])

problem.add_equation(eq_eval("div(u) + dot(u, grad_ln_ρ) = 0"), condition="nθ != 0")
problem.add_equation(eq_eval("p = 0"), condition="nθ == 0")
problem.add_equation(eq_eval("ddt(u) + grad(p) + grad(T_NCC)*s1 - (1/Re)*momentum_viscous_terms + LiftTau(tau_u) = cross(u, curl(u))"), condition = "nθ != 0")
problem.add_equation(eq_eval("u = 0"), condition="nθ == 0")
problem.add_equation(eq_eval("ddt(s1) + dot(u, grad_s0) - (1/Pe)*(lap(s1) + dot(grads1, (grad_ln_ρ + grad_ln_T))) + LiftTau(tau_T) = - dot(u, grads1) + H_eff + (1/Re)*inv_T*VH "))
problem.add_equation(eq_eval("u_r_bc    = 0"), condition="nθ != 0")
problem.add_equation(eq_eval("u_perp_bc = 0"), condition="nθ != 0")
problem.add_equation(eq_eval("tau_u     = 0"), condition="nθ == 0")
problem.add_equation(eq_eval("therm_bc  = 0"))

logger.info("Problem built")
# Solver
solver = solvers.InitialValueSolver(problem, ts)
solver.stop_iteration = int(args['--niter'])
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
visc_flux_r = 2*(dot(er, dot(u, E)) - (1/3) * ur * divU)

r_vals = field.Field(dist=d, bases=(b,), dtype=dtype)
r_vals['g'] = r
r_vals = operators.Grid(r_vals).evaluate()

vol_averager      = BallVolumeAverager(p)
radial_averager   = PhiThetaAverager(p)
equator_slicer    = EquatorSlicer(p)
analysis_tasks = []

re_ball = solver.evaluator.add_dictionary_handler(iter=10)
re_ball.add_task(Re*(u_squared)**(1/2), name='Re_avg', layout='g')

scalars = d3FileHandler(solver, '{:s}/scalars'.format(out_dir), max_writes=np.inf, sim_dt=scalar_dt)
scalars.add_task(Re*(u_squared)**(1/2), name='Re_avg', layout='g', extra_op=vol_averager, extra_op_comm=True)
scalars.add_task(ρ*u_squared/2,         name='KE',     layout='g', extra_op=vol_averager, extra_op_comm=True)
scalars.add_task(ρ*T*s1,                name='TE',     layout='g', extra_op=vol_averager, extra_op_comm=True)
analysis_tasks.append(scalars)

profiles =d3FileHandler(solver, '{:s}/profiles'.format(out_dir), max_writes=100, sim_dt=flux_dt)
profiles.add_task((4*np.pi*r_vals**2)*(ρ*ur*h),                        name='enth_lum', layout='g', extra_op=radial_averager, extra_op_comm=True)
profiles.add_task((4*np.pi*r_vals**2)*(-ρ*visc_flux_r/Re),             name='visc_lum', layout='g', extra_op=radial_averager, extra_op_comm=True)
profiles.add_task((4*np.pi*r_vals**2)*(-ρ*T*dot(er, grads1)/Pe),     name='cond_lum', layout='g', extra_op=radial_averager, extra_op_comm=True)
profiles.add_task((4*np.pi*r_vals**2)*(0.5*ρ*ur*u_squared),            name='KE_lum',   layout='g', extra_op=radial_averager, extra_op_comm=True)
profiles.add_task((4*np.pi*r_vals**2)*(ρ*ur*pomega_hat),              name='wave_lum', layout='g', extra_op=radial_averager, extra_op_comm=True)
analysis_tasks.append(profiles)

ORI = OutputRadialInterpolate
slices = d3FileHandler(solver, '{:s}/slices'.format(out_dir), max_writes=40, sim_dt=visual_dt)
slices.add_task(u(r=0.5), extra_op=ORI(s1, u(r=0.5)), name='u_r0.5', layout='g')
slices.add_task(s1(r=0.5), extra_op=ORI(s1, s1(r=0.5)), name='s1_r0.5',  layout='g')
slices.add_task(u(r=1.0), extra_op=ORI(s1, u(r=1.0)), name='u_r1', layout='g')
slices.add_task(s1(r=1.0), extra_op=ORI(s1, s1(r=1.0)), name='s1_r1',  layout='g')
slices.add_task(u, name='u', extra_op=equator_slicer, layout='g')
slices.add_task(s1, name='s1', extra_op=equator_slicer, layout='g')
analysis_tasks.append(slices)


checkpoint = solver.evaluator.add_file_handler('{:s}/checkpoint'.format(out_dir), max_writes=1, sim_dt=10*t_buoy)
checkpoint.add_task(s1, name='s1', scales=1, layout='c')
checkpoint.add_task(p, name='p', scales=1, layout='c')
checkpoint.add_task(u, name='u', scales=1, layout='c')


imaginary_cadence = 100

#CFL setup
from dedalus.extras.flow_tools import CFL

dt = max_dt
my_cfl = CFL(solver, max_dt, safety=float(args['--safety']), cadence=1, max_dt=max_dt, min_change=0.1, max_change=1.5, threshold=0.1)
my_cfl.add_velocity(u)


if args['--restart'] is not None:
    fname = args['--restart']
    fdir = fname.split('.h5')[0]
    check_name = fdir.split('/')[-1]
    #Try to just load the loal piece file

    import h5py
    with h5py.File('{}/{}_p{}.h5'.format(fdir, check_name, d.comm_cart.rank), 'r') as f:
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
        seed = 42 + d.comm_cart.rank
        rand = np.random.RandomState(seed=seed)
        filter_scale = 0.25

        # Generate noise & filter it
        s1['g'] = A0*rand.standard_normal(s1['g'].shape)
        s1.require_scales(filter_scale)
        s1['c']
        s1['g']
        s1.require_scales(dealias)

start_time = time.time()
for i in range(10):
    logger.info('initial timestep {}'.format(i))
    solver.step(dt)
    dt = my_cfl.compute_dt()

# Main loop
start_iter = solver.iteration
def main_loop():
    global dt
    while solver.ok:
        solver.step(dt)
        d.comm_cart.Barrier()
        dt = my_cfl.compute_dt()

        if solver.iteration % 10 == 0:
            Re0 = vol_averager(re_ball.fields['Re_avg'], comm=True)
            logger.info("t = %f, dt = %f, Re = %e" %(solver.sim_time, dt, Re0))

        if solver.iteration % imaginary_cadence in timestepper_history:
            for f in solver.state:
                f.require_grid_space()

import cProfile
main_start = time.time()
cProfile.run('main_loop()', filename='{}/prof.{:d}'.format(out_dir, rank))
end_time = time.time()

niter = solver.iteration - start_iter

startup_time = main_start - start_time
main_loop_time = end_time - main_start
DOF = (2*(Lmax+2))*(Lmax+1)*(Nmax+1)
if rank==0:
    print('performance metrics:')
    print('    startup time   : {:}'.format(startup_time))
    print('    main loop time : {:}'.format(main_loop_time))
    print('    main loop iter : {:d}'.format(niter))
    print('    wall time/iter : {:f}'.format(main_loop_time/niter))
    print('          iter/sec : {:f}'.format(niter/main_loop_time))
    print('DOF-cycles/cpu-sec : {:}'.format(DOF*niter/(ncpu*main_loop_time)))

from d3_outputs import post
for t in analysis_tasks:
    post.merge_analysis(t.base_path)
