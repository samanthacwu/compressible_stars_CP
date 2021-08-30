"""
d3 script for anelastic convection in a massive star's core CZ.

A config file can be provided which overwrites any number of the options below, but command line flags can also be used if preferred.

Usage:
    coreCZ_AN.py [options]
    coreCZ_AN.py <config> [options]

Options:
    --Re=<Re>            The Reynolds number of the numerical diffusivities [default: 5e1]
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
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic
from dedalus.tools import logging
from dedalus.tools.parsing import split_equation
from dedalus.extras.flow_tools import GlobalArrayReducer
from scipy import sparse
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
L_dealias = N_dealias = dealias = 1.5
dealias_tuple = (L_dealias, L_dealias, N_dealias)

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
mesh = args['--mesh']
if mesh is not None:
    mesh = [int(m) for m in mesh.split(',')]

Re  = float(args['--Re'])
Pr  = 1
Pe  = Pr*Re


if args['--mesa_file'] is not None:
    with h5py.File(args['--mesa_file'], 'r') as f:
        radius = f['radius'][()]
else:
    raise ValueError("Must provide a path to a MESA NCC file with --mesa_file flag")

# Bases
c = coords.SphericalCoordinates('φ', 'θ', 'r')
d = distributor.Distributor((c,), mesh=mesh, dtype=dtype)
b = basis.BallBasis(c, (2*(Lmax), Lmax, Nmax), radius=radius, dtype=dtype, dealias=dealias_tuple)
b_S2 = b.S2_basis()
φ, θ, r = b.local_grids(dealias_tuple)
φg, θg, rg = b.global_grids(dealias_tuple)

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

# Problem variables
u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
p = field.Field(dist=d, bases=(b,), dtype=dtype)
s1 = field.Field(dist=d, bases=(b,), dtype=dtype)
tau_u = field.Field(dist=d, bases=(b_S2,), tensorsig=(c,), dtype=dtype)
tau_T = field.Field(dist=d, bases=(b_S2,), dtype=dtype)


#nccs
grad_ln_ρ       = field.Field(dist=d, bases=(b.radial_basis,), tensorsig=(c,), dtype=dtype)
grad_s0         = field.Field(dist=d, bases=(b.radial_basis,), tensorsig=(c,), dtype=dtype)
grad_ln_T       = field.Field(dist=d, bases=(b.radial_basis,), tensorsig=(c,), dtype=dtype)
grad_T0         = field.Field(dist=d, bases=(b.radial_basis,), tensorsig=(c,), dtype=dtype)
grad_inv_Pe_rad = field.Field(dist=d, bases=(b.radial_basis,), tensorsig=(c,), dtype=dtype)
ln_ρ            = field.Field(dist=d, bases=(b.radial_basis,), dtype=dtype)
ln_T            = field.Field(dist=d, bases=(b.radial_basis,), dtype=dtype)
T0              = field.Field(dist=d, bases=(b.radial_basis,), dtype=dtype)
inv_Pe_rad      = field.Field(dist=d, bases=(b.radial_basis,), dtype=dtype)
inv_T           = field.Field(dist=d, bases=(b,), dtype=dtype) #only on RHS, multiplies other terms
H_eff           = field.Field(dist=d, bases=(b,), dtype=dtype)
ρ               = field.Field(dist=d, bases=(b,), dtype=dtype)
T               = field.Field(dist=d, bases=(b,), dtype=dtype)

#Radial unit vector
er = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
er.set_scales(dealias)
er['g'][2,:] = 1


slicer = GridSlicer(p)
with h5py.File(args['--mesa_file'], 'r') as f:
    if np.prod(grad_s0['g'].shape) > 0:
        grad_s0['g']         = f['grad_s0B'][()][:,:,:,  slicer[2]].reshape(grad_s0['g'].shape)
        grad_ln_ρ['g']       = f['grad_ln_ρB'][()][:,:,:,slicer[2]].reshape(grad_s0['g'].shape)
        grad_ln_T['g']       = f['grad_ln_TB'][()][:,:,:,slicer[2]].reshape(grad_s0['g'].shape)
        grad_T0['g']         = f['grad_TB'][()][:,:,:,slicer[2]].reshape(grad_s0['g'].shape)
        grad_inv_Pe_rad['g'] = f['grad_inv_Pe_radB'][()][:,:,:,slicer[2]].reshape(grad_s0['g'].shape)
    T0['g']        = f['TB'][()][:,:,slicer[2]]
    H_eff['g']     = f['H_effB'][()][:,:,slicer[2]]
    ln_ρ['g']      = f['ln_ρB'][()][:,:, slicer[2]]
    ln_T['g']      = f['ln_TB'][()][:,:, slicer[2]]
    inv_Pe_rad['g']= f['inv_Pe_radB'][()][:,:, slicer[2]]
    ρ['g']         = np.exp(f['ln_ρB'][()][:,:,slicer[2]])
    T['g']         = f['TB'][()][:,:,slicer[2]]
    inv_T['g']     = 1/T['g']

    max_dt = f['max_dt'][()]
    t_buoy = 1 #Assume nondimensionalization on heating ~ buoyancy time

logger.info('buoyancy time is {}'.format(t_buoy))
t_end = float(args['--buoy_end_time'])*t_buoy

for f in [u, s1, p, ln_ρ, ln_T, inv_T, H_eff, ρ]:
    f.require_scales(dealias)

# Stress matrices & viscous terms
I_matrix = field.Field(dist=d, bases=(b.radial_basis,), tensorsig=(c,c,), dtype=dtype)
I_matrix['g'] = 0
for i in range(3):
    I_matrix['g'][i,i,:] = 1

E = 0.5*(grad(u) + transpose(grad(u)))
E.store_last = True
divU = div(u)
divU.store_last = True
σ = 2*(E - (1/3)*divU*I_matrix)
momentum_viscous_terms = div(σ) + dot(σ, grad_ln_ρ)

VH  = 2*(trace(dot(E, E)) - (1/3)*divU*divU)

#Impenetrable, stress-free boundary conditions
u_r_bc    = radComp(u(r=radius))
u_perp_bc = radComp(angComp(E(r=radius), index=1))
therm_bc  = s1(r=radius)

H_eff = operators.Grid(H_eff).evaluate()
inv_T = operators.Grid(inv_T).evaluate()
grads1 = grad(s1)

# Problem
def eq_eval(eq_str):
    return [eval(expr) for expr in split_equation(eq_str)]
problem = problems.IVP([p, u, s1, tau_u, tau_T])

problem.add_equation(eq_eval("div(u) + dot(u, grad_ln_ρ) = 0"), condition="nθ != 0")
problem.add_equation(eq_eval("p = 0"), condition="nθ == 0")
problem.add_equation(eq_eval("ddt(u) + grad(p) + grad_T0*s1 - (1/Re)*momentum_viscous_terms + LiftTau(tau_u) = cross(u, curl(u))"), condition = "nθ != 0")
problem.add_equation(eq_eval("u = 0"), condition="nθ == 0")
problem.add_equation(eq_eval("ddt(s1) + dot(u, grad_s0) - inv_Pe_rad*(lap(s1) + dot(grads1, (grad_ln_ρ + grad_ln_T))) - dot(grads1, grad_inv_Pe_rad) + LiftTau(tau_T) = - dot(u, grads1) + H_eff + (1/Re)*inv_T*VH "))
problem.add_equation(eq_eval("u_r_bc    = 0"), condition="nθ != 0")
problem.add_equation(eq_eval("u_perp_bc = 0"), condition="nθ != 0")
problem.add_equation(eq_eval("tau_u     = 0"), condition="nθ == 0")
problem.add_equation(eq_eval("therm_bc  = 0"))

logger.info("Problem built")
# Solver
solver = solvers.InitialValueSolver(problem, ts)
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
visc_flux_r = 2*(dot(er, dot(u, E)) - (1/3) * ur * divU)

r_vals = field.Field(dist=d, bases=(b,), dtype=dtype)
r_vals.set_scales(dealias_tuple)
r_vals['g'] = r
r_vals = operators.Grid(r_vals).evaluate()

vol_averager      = BallVolumeAverager(p)
radial_averager   = PhiThetaAverager(p)
equator_slicer    = EquatorSlicer(p)
analysis_tasks = []

re_ball = solver.evaluator.add_dictionary_handler(iter=10)
re_ball.add_task(Re*(u_squared)**(1/2), name='Re_avg', layout='g', scales=dealias_tuple)

scalars = d3FileHandler(solver, '{:s}/scalars'.format(out_dir), max_writes=np.inf, sim_dt=scalar_dt)
scalars.add_task(Re*(u_squared)**(1/2), name='Re_avg', layout='g', extra_op=vol_averager, extra_op_comm=True)
scalars.add_task(ρ*u_squared/2,         name='KE',     layout='g', extra_op=vol_averager, extra_op_comm=True)
scalars.add_task(ρ*T*s1,                name='TE',     layout='g', extra_op=vol_averager, extra_op_comm=True)
analysis_tasks.append(scalars)

profiles =d3FileHandler(solver, '{:s}/profiles'.format(out_dir), max_writes=100, sim_dt=flux_dt)
profiles.add_task((4*np.pi*r_vals**2)*(ρ*ur*h),                      name='enth_lum', layout='g', extra_op=radial_averager, extra_op_comm=True)
profiles.add_task((4*np.pi*r_vals**2)*(-ρ*visc_flux_r/Re),           name='visc_lum', layout='g', extra_op=radial_averager, extra_op_comm=True)
profiles.add_task((4*np.pi*r_vals**2)*(-ρ*T*dot(er, grads1)/Pe),     name='cond_lum', layout='g', extra_op=radial_averager, extra_op_comm=True)
profiles.add_task((4*np.pi*r_vals**2)*(0.5*ρ*ur*u_squared),          name='KE_lum',   layout='g', extra_op=radial_averager, extra_op_comm=True)
profiles.add_task((4*np.pi*r_vals**2)*(ρ*ur*pomega_hat),             name='wave_lum', layout='g', extra_op=radial_averager, extra_op_comm=True)
analysis_tasks.append(profiles)

ORI = OutputRadialInterpolate
slices = d3FileHandler(solver, '{:s}/slices'.format(out_dir), max_writes=40, sim_dt=visual_dt)
slices.add_task(u(r=0.5), extra_op=ORI(s1, u(r=0.5*radius)), name='u_r0.5', layout='g')
slices.add_task(s1(r=0.5), extra_op=ORI(s1, s1(r=0.5*radius)), name='s1_r0.5',  layout='g')
slices.add_task(u(r=0.95), extra_op=ORI(s1, u(r=0.95*radius)), name='u_r0.95', layout='g')
slices.add_task(s1(r=0.95), extra_op=ORI(s1, s1(r=0.95*radius)), name='s1_r0.95',  layout='g')
slices.add_task(u,  name='u_eq', extra_op=equator_slicer, layout='g')
slices.add_task(s1, name='s1_eq', extra_op=equator_slicer, layout='g')
analysis_tasks.append(slices)

checkpoint = solver.evaluator.add_file_handler('{:s}/checkpoint'.format(out_dir), max_writes=1, sim_dt=10*t_buoy)
checkpoint.add_task(s1, name='s1', scales=1, layout='c')
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

# Main loop
start_time = time.time()
start_iter = solver.iteration
try:
    while solver.ok:
        solver.step(dt)
        dt = my_cfl.compute_dt()

        if solver.iteration % 10 == 0:
            Re0 = vol_averager(re_ball.fields['Re_avg'], comm=True)
            logger.info("t = %f, dt = %f, Re = %e" %(solver.sim_time, dt, Re0))

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
    n_cpu    = d.comm_cart.size
    dof_cycles_per_cpusec = n_coeffs*n_iter/(cpu_sec*n_cpu)
    logger.info('DOF-cycles/cpu-sec : {:e}'.format(dof_cycles_per_cpusec))
    logger.info('Run iterations: {:e}'.format(n_iter))
    logger.info('Sim end time: {:e}'.format(solver.sim_time))
    logger.info('Run time: {}'.format(cpu_sec))

from d3_outputs import post
for t in analysis_tasks:
    post.merge_analysis(t.base_path)
