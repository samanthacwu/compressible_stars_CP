"""
d3 script for anelastic convection in a polytrope.

Usage:
    polytrope_ball_AN.py [options]

Options:
    --L=<Lmax>          The value of Lmax   [default: 15]
    --N=<Nmax>          The value of Nmax   [default: 15]
    --ktau=<k>          The value of ktau   [default: 0]
    --krhs=<k>          The value of krhs   [default: 0]



"""
import os
import time
import sys

import h5py
import numpy as np
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic
from dedalus.tools import logging
from dedalus.tools.parsing import split_equation
from dedalus.extras.flow_tools import GlobalArrayReducer
from mpi4py import MPI

import logging
logger = logging.getLogger(__name__)

from dedalus.tools.config import config
config['linear algebra']['MATRIX_FACTORIZER'] = 'SuperLUNaturalFactorizedTranspose'

from docopt import docopt
args   = docopt(__doc__)

# Parameters
Nmax      = int(args['--N'])
Lmax      = int(args['--L'])
ktau      = int(args['--ktau'])
krhs      = int(args['--krhs'])
L_dealias = N_dealias = dealias = 1.5
dealias_tuple = (L_dealias, L_dealias, N_dealias)


ts = timesteppers.SBDF2
timestepper_history = [0, 1,]
dtype = np.float64
mesh = None
ncpu = MPI.COMM_WORLD.size
if mesh is not None:
    mesh = mesh.split(',')
    mesh = [int(mesh[0]), int(mesh[1])]
else:
    log2 = np.log2(ncpu)
    if log2 == int(log2):
        mesh = [int(2**np.ceil(log2/2)),int(2**np.floor(log2/2))]
    logger.info("running on processor mesh={}".format(mesh))



Re  = 2e2
Pr  = 1
Pe  = Pr*Re

t_buoy = 1
dt = 1e-1* t_buoy
t_end = 2e3*t_buoy

alpha = 1
rhoc  = 1
k_d_R = 1
R     = 1
Tc    = k_d_R*rhoc
G     = (rhoc/Tc)*2*R/(4*np.pi*alpha**2)
radius = np.pi/(2*alpha)

# Bases
c = coords.SphericalCoordinates('φ', 'θ', 'r')
d = distributor.Distributor((c,), mesh=mesh)
b = basis.BallBasis(c, (2*(Lmax+1), Lmax+1, Nmax+1), radius=radius, dtype=dtype, dealias=dealias_tuple)
b_S2 = b.S2_basis()
φ, θ, r = b.local_grids(dealias_tuple)
φg, θg, rg = b.global_grids(dealias_tuple)

b_tau = b._new_k(ktau)
b_rhs = b._new_k(krhs)

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
LiftTau   = lambda A: operators.LiftTau(A, b_tau, -1)
Grid = operators.Coeff
Coeff = operators.Coeff
Conv = operators.Convert
RHS = lambda A: Coeff(Conv(A, b_rhs))


# Problem variables
u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
p = field.Field(dist=d, bases=(b,), dtype=dtype)
s1 = field.Field(dist=d, bases=(b,), dtype=dtype)
tau_u = field.Field(dist=d, bases=(b_S2,), tensorsig=(c,), dtype=dtype)
tau_T = field.Field(dist=d, bases=(b_S2,), dtype=dtype)


#nccs
grad_ln_ρ       = field.Field(dist=d, bases=(b.radial_basis,), tensorsig=(c,), dtype=dtype)
grad_ln_T       = field.Field(dist=d, bases=(b.radial_basis,), tensorsig=(c,), dtype=dtype)
grad_T0         = field.Field(dist=d, bases=(b.radial_basis,), tensorsig=(c,), dtype=dtype)
grad_ρ0         = field.Field(dist=d, bases=(b.radial_basis,), tensorsig=(c,), dtype=dtype)
grav            = field.Field(dist=d, bases=(b.radial_basis,), tensorsig=(c,), dtype=dtype)
Mr              = field.Field(dist=d, bases=(b.radial_basis,), dtype=dtype)
HSE             = field.Field(dist=d, bases=(b.radial_basis,), dtype=dtype)
T0              = field.Field(dist=d, bases=(b.radial_basis,), dtype=dtype)
inv_T           = field.Field(dist=d, bases=(b,), dtype=dtype) #only on RHS, multiplies other terms
Heat            = field.Field(dist=d, bases=(b,), dtype=dtype)
ρ               = field.Field(dist=d, bases=(b,), dtype=dtype)
T               = field.Field(dist=d, bases=(b,), dtype=dtype)

for f in [u, p, s1, grad_ln_ρ, grad_ln_T, grad_T0, grad_ρ0, T0, inv_T, Heat, ρ, T, grav, Mr, HSE]:
    f.require_scales(dealias_tuple)

ρ['g'] = rhoc * np.sin(alpha*r) / (alpha * r)
T['g'] = k_d_R * ρ['g']
inv_T['g'] = 1/T['g']
Heat['g'] = 1 / Pe


gradT = grad(T).evaluate()
gradρ = grad(ρ).evaluate()


if np.prod(grad_T0['g'].shape) > 0:
    scalar_ncc_shape = T0['g'].shape
    vector_ncc_shape = grad_T0['g'].shape
    #not all procs have ncc data
    grad_T0['g'] = gradT['g'][:,0,0,:].reshape(vector_ncc_shape)
    grad_ρ0['g'] = gradρ['g'][:,0,0,:].reshape(vector_ncc_shape)
    sys.stdout.flush()
    grad_ln_T['g'][:,0,0,:] = gradT['g'][:,0,0,:]/T['g'][0,0,:]
    grad_ln_ρ['g'][:,0,0,:] = gradρ['g'][:,0,0,:]/ρ['g'][0,0,:]
    T0['g'][0,0,:] = T['g'][0,0,:]
    Mr['g']   = 4 * np.pi * rhoc * (np.sin(alpha*r) - alpha*r*np.cos(alpha*r)) / alpha**3
    grav['g'][2,:] = -(G/r**2) * Mr['g']
    HSE['g'][0,0,:] = R*(ρ['g'][0,0,:]*grad_T0['g'][2,0,0,:] + T['g'][0,0,:]*grad_ρ0['g'][2,0,0,:]) - ρ['g'][0,0,:]*grav['g'][2,0,0,:]
    print('max hse error: {:.3e} on proc {}'.format(np.abs(HSE['g']).max(), d.comm_cart.rank))
else:
    #need to reference 'g' in full fields on other procs or we get a parallel hangup
    gradT['g']
    gradρ['g']
    gradT['g']
    gradρ['g']
    T['g']
    ρ['g']
    T['g']
    ρ['g']



    


#Radial unit vector
er = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
er.set_scales(dealias_tuple)
er['g'][2,:] = 1


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

Heat = operators.Grid(Heat).evaluate()
inv_T = operators.Grid(inv_T).evaluate()
grads1 = grad(s1)



# Problem
def eq_eval(eq_str):
    return [eval(expr) for expr in split_equation(eq_str)]
problem = problems.IVP([p, u, s1, tau_u, tau_T])

problem.add_equation(eq_eval("div(u) + dot(u, grad_ln_ρ) = 0"), condition="nθ != 0")
problem.add_equation(eq_eval("p = 0"), condition="nθ == 0")
problem.add_equation(eq_eval("ddt(u) + grad(p) + grad_T0*s1 - (1/Re)*momentum_viscous_terms + LiftTau(tau_u) = RHS(cross(u, curl(u)))"), condition = "nθ != 0")
problem.add_equation(eq_eval("u = 0"), condition="nθ == 0")
problem.add_equation(eq_eval("ddt(s1) - (1/Pe)*(lap(s1) + dot(grads1, (grad_ln_ρ + grad_ln_T))) + LiftTau(tau_T) = RHS(- dot(u, grads1) + Heat + (1/Re)*inv_T*VH)"))
problem.add_equation(eq_eval("u_r_bc    = 0"), condition="nθ != 0")
problem.add_equation(eq_eval("u_perp_bc = 0"), condition="nθ != 0")
problem.add_equation(eq_eval("tau_u     = 0"), condition="nθ == 0")
problem.add_equation(eq_eval("therm_bc  = 0"))



logger.info("Problem built")
# Solver
solver = solvers.InitialValueSolver(problem, ts)
solver.stop_sim_time = t_end
logger.info("solver built")

# Analysis
weight_theta = b.local_colatitude_weights(L_dealias)
weight_r = b.local_radial_weights(N_dealias)
reducer = GlobalArrayReducer(d.comm_cart)
vol_test = np.sum(weight_r*weight_theta+0*p['g'])*np.pi/(Lmax+1)/L_dealias
vol_test = reducer.reduce_scalar(vol_test, MPI.SUM)
vol_correction = 4*np.pi/3/vol_test

#ICs similar but not identical to marti benchmark
A0 = 1e-2
s1['g'] = (Pe/6)*Heat['g']*(1-r**2) +  A0*(r/radius)**3*(1-(r/radius)**2)*np.cos(φ)*np.sin(θ)**3


from d3_outputs.extra_ops    import BallVolumeAverager, ShellVolumeAverager, EquatorSlicer, PhiAverager, PhiThetaAverager, OutputRadialInterpolate, GridSlicer, MeridionSlicer
from d3_outputs.writing      import d3FileHandler

out_dir = './' + sys.argv[0].split('.py')[0]
out_dir += '_Re{}_{}x{}_ktau{}_krhs{}'.format(Re, Lmax, Nmax, ktau, krhs)
logger.info('saving data to {:s}'.format(out_dir))
if MPI.COMM_WORLD.rank == 0:
    if not os.path.exists('{:s}/'.format(out_dir)):
        os.makedirs('{:s}/'.format(out_dir))

vol_averager      = BallVolumeAverager(p)

u_squared = dot(u,u)
analysis_tasks = []
scalars = d3FileHandler(solver, '{:s}/scalars'.format(out_dir), iter=1, max_writes=np.inf)
scalars.add_task(Re**2 * u_squared / 2, name='KE', layout='g', extra_op = vol_averager, extra_op_comm=True)
analysis_tasks.append(scalars)


ke_dict = solver.evaluator.add_dictionary_handler(iter=10)
ke_dict.add_task(Re**2 * u_squared / 2, name='KE', layout='g', scales=dealias_tuple)



# Main loop
start_time = time.time()
start_iter = solver.iteration
hermitian_cadence = 100
try:
    while solver.ok:
        solver.step(dt)

        if solver.iteration % 10 == 0:
            KE = vol_averager(ke_dict.fields['KE'], comm=True)
#            KE = np.sum(vol_correction*weight_r*weight_theta*(Re*u['g'].real)**2)
#            KE = 0.5*KE*(np.pi)/(Lmax+1)/L_dealias
#            KE = reducer.reduce_scalar(KE, MPI.SUM)
            logger.info("t = %f, dt = %f, KE = %.10e" %(solver.sim_time, dt, KE))

            if not np.isfinite(KE) or np.isnan(KE):
                logger.info('exiting with NaN')
                break

        if solver.iteration % hermitian_cadence in timestepper_history:
            for f in solver.state:
                f.require_grid_space()
except:
    logger.info('something went wrong in main loop')
    raise
finally:
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
