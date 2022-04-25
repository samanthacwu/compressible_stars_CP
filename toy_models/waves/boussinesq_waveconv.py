"""
Dedalus script simulating internally-heated Boussinesq convection in the ball.
This script demonstrates soving an initial value problem in the ball. It can be
ran serially or in parallel, and uses the built-in analysis framework to save
data snapshots to HDF5 files. The `plot_ball.py` script can be used to produce
plots from the saved data. The simulation should take roughly 15 cpu-minutes to run.

The strength of gravity is proportional to radius, as for a constant density ball.
The problem is non-dimensionalized using the ball radius and freefall time, so
the resulting thermal diffusivity and viscosity are related to the Prandtl
and Rayleigh numbers as:

    kappa = (Rayleigh * Prandtl)**(-1/2)
    nu = (Rayleigh / Prandtl)**(-1/2)

We use stress-free boundary conditions, and maintain a constant flux on the outer
boundary. The convection is driven by the internal heating term with a conductive
equilibrium of T(r) = 1 - r**2.

For incompressible hydro in the ball, we need one tau term each for the velocity
and temperature. Here we choose to lift them to the original (k=0) basis.

The simulation will run to t=10, about the time for the first convective plumes
to hit the top boundary. After running this initial simulation, you can restart
the simulation with the command line option '--restart'.

To run, restart, and plot using e.g. 4 processes:
    $ mpiexec -n 4 python3 internally_heated_convection.py
    $ mpiexec -n 4 python3 internally_heated_convection.py --restart
    $ mpiexec -n 4 python3 plot_ball.py slices/*.h5
"""
import traceback
import sys
import numpy as np
import dedalus.public as d3
import logging
from mpi4py import MPI
logger = logging.getLogger(__name__)


from scipy.special import erf
def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
    return -(one_to_zero(*args, **kwargs) - 1)

# Allow restarting via command line
restart = (len(sys.argv) > 1 and sys.argv[1] == '--restart')

# Parameters
Nphi, Ntheta, Nr = 4, 64, 128
Rayleigh = 1e6
Prandtl = 1
dealias = 1
S=100
timestepper = d3.SBDF2
dtype = np.float64
mesh = (1, 32)

r_transition=1
radius=2

resolution=(Nphi, Ntheta, Nr)

# Bases
coords = d3.SphericalCoordinates('phi', 'theta', 'r')
dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)
basis = d3.BallBasis(coords, shape=resolution, radius=radius, dealias=dealias, dtype=dtype)
S2_basis = basis.S2_basis()

# Fields
u = dist.VectorField(coords, name='u',bases=basis)
p = dist.Field(name='p', bases=basis)
T = dist.Field(name='T', bases=basis)
tau_p = dist.Field(name='tau_p')
tau_u = dist.VectorField(coords, name='tau u', bases=S2_basis)
tau_T = dist.Field(name='tau T', bases=S2_basis)

# Substitutions
phi, theta, r = dist.local_grids(basis)
r_vec = dist.VectorField(coords, bases=basis.radial_basis)
r_vec['g'][2] = r

grad_T0_source = dist.VectorField(coords, bases=basis.radial_basis)
grad_T0_source['g'][2] = S * r * zero_to_one(r, r_transition, width=0.1)
kappa = (Rayleigh * Prandtl)**(-1/2)
nu = (Rayleigh / Prandtl)**(-1/2)
lift = lambda A: d3.Lift(A, basis, -1)
strain_rate = d3.grad(u) + d3.trans(d3.grad(u))
shear_stress = d3.angular(d3.radial(strain_rate(r=radius), index=1))

N2 = (r_vec@grad_T0_source).evaluate()
if N2['g'].size > 0:
    f_bv_max = np.array((np.sqrt(N2['g'].max())/(2*np.pi),), dtype=np.float64)
else:
    f_bv_max = np.array((0,), dtype=np.float64)
dist.comm_cart.Allreduce(MPI.IN_PLACE, f_bv_max, op=MPI.MAX)
f_bv_max = float(f_bv_max[0])

min_freq = 5e-2
max_freq = 2*f_bv_max
df = 0.5*min_freq
max_timestep = 1/max_freq
stop_sim_time = 1000*(2/min_freq) + 200 * max_timestep
force_freqs = np.arange(min_freq, max_freq, step=df)[None,None,None,:,None]#phi,theta,r,f,ell
logger.info('forcing from {} to {} at df = {} / dt = {}; freq_steps = {}; stop time = {}'.format(min_freq, max_freq, df, max_timestep, force_freqs.size, stop_sim_time))
#force_freqs = np.logspace(-2, 2, 100)[None,None,None,:,None]#phi,theta,r,f,ell
force_ells = np.arange(1, 10)[None,None,None,None,:]
force_norm = force_freqs.size*force_ells.size
powf = -4
powl = 4
scaling = ((force_freqs/force_freqs.min())**(powf)*(force_ells/force_ells.max())**(powl))

de_phi, de_theta, de_r = dist.local_grids(basis, scales=basis.dealias)
theta_force = de_theta[:,:,:,None,None]
force_spatial = np.exp(-(de_r - r_transition)**2/0.1**2)[:,:,:,None,None] * np.cos(force_ells*theta_force)

def F_func(time):
    warmup = zero_to_one(time, 100*max_timestep, width=10*max_timestep)
    return warmup*np.sum(np.sum(scaling*force_spatial*np.sin(2*np.pi*force_freqs*time),axis=-1),axis=-1) / force_norm


F = dist.VectorField(coords, bases=basis)
F.change_scales(basis.dealias)
F['g'][0] = F_func(0)

# Problem
problem = d3.IVP([p, u, T, tau_p, tau_u, tau_T], namespace=locals())
problem.add_equation("div(u) + tau_p = 0")
problem.add_equation("dt(u) - nu*lap(u) - r_vec*T + grad(p) + lift(tau_u) = F")
problem.add_equation("dt(T) + u@grad_T0_source - kappa*lap(T) + lift(tau_T) = 0")
problem.add_equation("shear_stress = 0")  # Stress free
problem.add_equation("radial(u(r=radius)) = 0")  # Impermeable
problem.add_equation("radial(grad(T)(r=radius)) = 0")
problem.add_equation("integ(p) = 0")  # Pressure gauge

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
if not restart:
    file_handler_mode = 'overwrite'
    initial_timestep = max_timestep
else:
    write, initial_timestep = solver.load_state('checkpoints/checkpoints_s20.h5')
    initial_timestep = 2e-2
    file_handler_mode = 'append'

# Analysis
slices = solver.evaluator.add_file_handler('slices', sim_dt=0.1, max_writes=10, mode=file_handler_mode)
slices.add_task(T(phi=0), scales=dealias, name='T(phi=0)')
slices.add_task(T(phi=np.pi), scales=dealias, name='T(phi=pi)')
slices.add_task(T(phi=3/2*np.pi), scales=dealias, name='T(phi=3/2*pi)')
slices.add_task(T(r=radius), scales=dealias, name='T(r=radius)')
slices.add_task(T(theta=0), scales=dealias, name='T(theta=0)')
slices.add_task(F(phi=0), scales=dealias, name='F')

shells = solver.evaluator.add_file_handler('shells', sim_dt=max_timestep, max_writes=1000)
shells.add_task(p(r=radius), scales=dealias, name='T(r=radius)')
shells.add_task(p(r=radius), scales=dealias, name='p(r=radius)')
shells.add_task(u(r=radius), scales=dealias, name='u(r=radius)')

checkpoints = solver.evaluator.add_file_handler('checkpoints', wall_dt=1*60*60, max_writes=1)
checkpoints.add_tasks(solver.state)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(u@u, name='u2')
flow.add_property(F@F, name='F2')

timestep = max_timestep
# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
#        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_u = np.sqrt(flow.max('u2'))
            max_f = np.sqrt(flow.max('F2'))
            logger.info("Iteration=%i, Time=%e, dt=%e, max(u)=%e, max(f) = %e" %(solver.iteration, solver.sim_time, timestep, max_u, max_f))
        F['g'] = F_func(solver.sim_time)
except Exception:
    logger.error('Exception raised, triggering end of main loop.')
    print(traceback.format_exc()) 
    sys.stdout.flush()
finally:
    solver.log_stats()

