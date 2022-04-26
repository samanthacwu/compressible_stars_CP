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
from scipy.special import sph_harm


from scipy.special import erf
def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
    return -(one_to_zero(*args, **kwargs) - 1)

# Allow restarting via command line
restart = (len(sys.argv) > 1 and sys.argv[1] == '--restart')

# Parameters
Nphi, Ntheta, Nr = 4, 64, 128
Rayleigh = 1e16
Prandtl = 1
dealias = 1
S=100
timestepper = d3.SBDF2
dtype = np.float64
mesh = (1, MPI.COMM_WORLD.size)

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

er = dist.VectorField(coords, bases=basis)
er['g'][2] = 1


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

warmup_iter = 200
sample_iter = 1000
stop_iter = sample_iter + warmup_iter

sample_freq = 2*f_bv_max
max_timestep = 1/sample_freq
df = sample_freq/sample_iter
min_freq = sample_freq - df*sample_iter #should be 0
#force_freqs = np.arange(min_freq+df, f_bv_max, step=df)[None,None,None,None,:]#phi,theta,r,ell, f
force_freqs = 0.1 * np.ones((1,))[None,None,None,None,None,:]
logger.info('forcing from {} to {} at df = {} / dt = {}; freq_steps = {}; stop iter = {}'.format(min_freq, sample_freq, df, max_timestep, force_freqs.size, stop_iter))
#force_ells = np.arange(1, 2,dtype=np.float64)[None,None,None,:]#phi,theta,r,ell
#force_ells = np.arange(1, Ntheta-4,dtype=np.float64)[None,None,None,:]#phi,theta,r,ell
force_ells = np.arange(4, 5,dtype=np.float64)[None,None,None,None,:]
leading_normalization = 1e-2
powf = 1
powl = 1
f_scaling = (force_freqs)**(powf) 
ell_scaling = (force_ells)**(powl)

scalar_F = dist.Field(bases=basis)
F = dist.VectorField(coords, bases=basis)
grad_F = d3.grad(scalar_F)

de_phi, de_theta, de_r = dist.local_grids(basis, scales=basis.dealias)
theta_force = de_theta[None,:,:,:]
phi_force = de_phi[None,:,:,:]
force_radial = np.exp(-(de_r - r_transition)**2/0.1**2)[None,:,:,:,None]
force_angular = np.zeros((*tuple(F['g'].shape), force_ells.size))
for i, ell in enumerate(force_ells.ravel()): 
    scalar_F['g'] = sph_harm(0, ell, phi_force, theta_force).real
    force_angular[:,:,:,:,i] = grad_F.evaluate()['g']

force_spatial = force_angular * force_radial

def F_func(time):
    warmup = zero_to_one(time, 100*max_timestep, width=10*max_timestep)
    #sum over freqs, then ells
    return leading_normalization * warmup * np.sum(ell_scaling*force_spatial*np.sum(f_scaling*np.sin(2*np.pi*force_freqs*time),axis=-1),axis=-1)


damper = dist.Field(bases=basis.radial_basis)
damper.change_scales(basis.dealias)
damper['g'] = zero_to_one(r, radius*0.925, width=radius*0.025)

F['g'] = F_func(0)

# Problem
problem = d3.IVP([p, u, T, tau_p, tau_u, tau_T], namespace=locals())
problem.add_equation("div(u) + tau_p = 0")
problem.add_equation("dt(u) + u*damper*f_bv_max - nu*lap(u) - r_vec*T + grad(p) + lift(tau_u) = F")
problem.add_equation("dt(T) + u@grad_T0_source - kappa*lap(T) + lift(tau_T) = 0")
problem.add_equation("shear_stress = 0")  # Stress free
problem.add_equation("radial(u(r=radius)) = 0")  # Impermeable
problem.add_equation("radial(grad(T)(r=radius)) = 0")
problem.add_equation("integ(p) = 0")  # Pressure gauge

# Solver
solver = problem.build_solver(timestepper)
solver.stop_iteration = stop_iter

# Initial conditions
if not restart:
#    T.fill_random('g', seed=42, distribution='normal', scale=0.01) # Random noise
#    T.low_pass_filter(scales=0.5)
    file_handler_mode = 'overwrite'
    initial_timestep = max_timestep
else:
    write, initial_timestep = solver.load_state('checkpoints/checkpoints_s20.h5')
    initial_timestep = 2e-2
    file_handler_mode = 'append'

# Analysis
#slices = solver.evaluator.add_file_handler('slices', sim_dt=0.1, max_writes=10, mode=file_handler_mode)
#slices.add_task(T(phi=0), scales=dealias, name='T(phi=0)')
#slices.add_task(T(phi=np.pi), scales=dealias, name='T(phi=pi)')
#slices.add_task(T(phi=3/2*np.pi), scales=dealias, name='T(phi=3/2*pi)')
#slices.add_task(T(r=radius), scales=dealias, name='T(r=radius)')
#slices.add_task(T(theta=0), scales=dealias, name='T(theta=0)')
#slices.add_task(F(phi=0), scales=dealias, name='F')


s2_avg = lambda A: d3.Average(A, coords.S2coordsys)
profiles = solver.evaluator.add_file_handler('profiles', iter=1, max_writes=100, mode=file_handler_mode)
profiles.add_task(s2_avg(4*np.pi*(er@r_vec)*er@(u*p)), name='wave_flux')
profiles.add_task(s2_avg(4*np.pi*(er@r_vec)*er@(nu*d3.cross(u,d3.curl(u)))), name='visc_flux')

## CFL
#CFL = d3.CFL(solver, initial_timestep, cadence=10, safety=0.5, threshold=0.1, max_dt=max_timestep)
#CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(u@u, name='u2')
flow.add_property((er@u)**2, name='ur2')
flow.add_property(F@F, name='F2')

timestep = max_timestep
start_iter = solver.iteration
# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
#        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_u = np.sqrt(flow.max('u2'))
            max_ur = np.sqrt(flow.max('ur2'))
            max_f = np.sqrt(flow.max('F2'))
            logger.info("Iteration=%i, Time=%e, dt=%e, max(u)=%e / r: %e, max(f) = %e" %(solver.iteration, solver.sim_time, timestep, max_u, max_ur, max_f))

        F['g'] = F_func(solver.sim_time)
        
        if solver.iteration - start_iter == warmup_iter - 1:
            shells = solver.evaluator.add_file_handler('shells', sim_dt=max_timestep, max_writes=100)
            shells.add_task(p(r=1), scales=dealias, name='p(r=1)')
            shells.add_task(p(r=1.25), scales=dealias, name='p(r=1.25)')
            shells.add_task(p(r=1.4), scales=dealias, name='p(r=1.4)')
            shells.add_task(p(r=1.5), scales=dealias, name='p(r=1.5)')
            shells.add_task(p(r=1.6), scales=dealias, name='p(r=1.6)')
            shells.add_task(p(r=1.75), scales=dealias, name='p(r=1.75)')
            shells.add_task(p(r=radius), scales=dealias, name='p(r=radius)')
            shells.add_task(u(r=1), scales=dealias, name='u(r=1)')
            shells.add_task(u(r=1.25), scales=dealias, name='u(r=1.25)')
            shells.add_task(u(r=1.4), scales=dealias, name='u(r=1.4)')
            shells.add_task(u(r=1.5), scales=dealias, name='u(r=1.5)')
            shells.add_task(u(r=1.6), scales=dealias, name='u(r=1.6)')
            shells.add_task(u(r=1.75), scales=dealias, name='u(r=1.75)')
            shells.add_task(u(r=radius), scales=dealias, name='u(r=radius)')


except Exception:
    logger.error('Exception raised, triggering end of main loop. Error message: {}'.format(traceback.format_exc()))
finally:
    solver.log_stats()
    if not shells.check_file_limits():
        file = shells.get_file()
        file.close()
        if dist.comm_cart.rank == 0:
            shells.process_virtual_file()

