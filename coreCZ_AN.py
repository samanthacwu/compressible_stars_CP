"""
d3 script for anelastic convection in a massive star's core CZ.

A config file can be provided which overwrites any number of the options below, but command line flags can also be used if preferred.

Usage:
    bootstrap_rrbc.py [options]
    bootstrap_rrbc.py <config> [options]

Options:
    --Re=<Re>            The Reynolds number of the numerical diffusivities [default: 1e2]
    --Pr=<Prandtl>       The Prandtl number  of the numerical diffusivities [default: 1]
    --L=<Lmax>           The value of Lmax   [default: 14]
    --N=<Nmax>           The value of Nmax   [default: 15]

    --wall_hours=<t>     The number of hours to run for [default: 24]
    --safety=<s>         Timestep CFL safety factor [default: 0.4]

    --mesh=<n,m>         The processor mesh over which to distribute the cores
    --A0=<A>             Amplitude of initial noise [default: 1e-6]

    --label=<label>      A label to add to the end of the output directory

    --SBDF2              Use SBDF2 (default)
    --SBDF4              Use SBDF4

    --mesa_file=<f>      path to a .h5 file of ICCs, curated from a MESA model
    --restart=<chk_f>    path to a checkpoint file to restart from
"""
import time

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

import logging
logger = logging.getLogger(__name__)

from dedalus.tools.config import config
config['linear algebra']['MATRIX_FACTORIZER'] = 'SuperLUNaturalFactorizedTranspose'

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
radius    = 1
Lmax      = int(args['--L'])
Nmax      = int(args['--N'])
L_dealias = N_dealias = dealias = 3/2


max_dt = 1e-2
t_end = 100*max_dt
if args['--SBDF4']:
    ts = timesteppers.SBDF4
else:
    ts = timesteppers.SBDF2
dtype = np.float64
mesh = args['--mesh']
if mesh is not None:
    mesh = [int(m) for m in mesh.split(',')]

Re  = float(args['--Re'])
Pr  = 1
Pe  = Pr*Re

# Bases
c = coords.SphericalCoordinates('φ', 'θ', 'r')
d = distributor.Distributor((c,), mesh=mesh)
b = basis.BallBasis(c, (2*(Lmax+2), Lmax+1, Nmax+1), radius=radius, dtype=dtype)
bNCC = basis.BallBasis(c, (1, 1, Nmax+1), radius=radius, dtype=dtype)
b_S2 = b.S2_basis()
φ, θ, r = b.local_grids((dealias, dealias, dealias))

#Operators
div       = lambda A: operators.Divergence(A, index=0)
lap       = lambda A: operators.Laplacian(A, c)
grad      = lambda A: operators.Gradient(A, c)
dot       = lambda A, B: arithmetic.DotProduct(A, B)
cross     = lambda A, B: arithmetic.CrossProduct(A, B)
trace     = lambda A: operators.Trace(A)
ddt       = lambda A: operators.TimeDerivative(A)
transpose = lambda A: operators.TransposeComponents(A)
radComp   = lambda A: operators.RadialComponent(A)
angComp   = lambda A, index=1: operators.AngularComponent(A, index=index)

# Fields
u = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
p = field.Field(dist=d, bases=(b,), dtype=dtype)
s1 = field.Field(dist=d, bases=(b,), dtype=dtype)
tau_u = field.Field(dist=d, bases=(b_S2,), tensorsig=(c,), dtype=dtype)
tau_T = field.Field(dist=d, bases=(b_S2,), dtype=dtype)

#TODO: do viscous heating right
VH = field.Field(dist=d, bases=(b,), dtype=dtype)

#nccs
ln_ρ  = field.Field(dist=d, bases=(b.radial_basis,), dtype=dtype)
ln_ρT = field.Field(dist=d, bases=(b.radial_basis,), dtype=dtype)
inv_T = field.Field(dist=d, bases=(b,), dtype=dtype) #only on RHS, multiplies other terms
H_eff = field.Field(dist=d, bases=(b,), dtype=dtype)
g_eff = field.Field(dist=d, bases=(b.radial_basis,), tensorsig=(c,), dtype=dtype)

if args['--mesa_file'] is not None:
    φ1, θ1, r1 = b.local_grids((1, 1, 1))
    with h5py.File(args['--mesa_file'], 'r') as f:
        r_file = f['r'][()].flatten()
        r_slice = np.zeros_like(r_file.flatten(), dtype=bool)
        for this_r in r1.flatten():
            r_slice[this_r == r_file] = True
        ln_ρ['g']      = f['ln_ρ'][()][:,:,r_slice]
        ln_ρT['g']     = f['ln_ρT'][()][:,:,r_slice]
        H_eff['g']     = f['H_eff'][()][:,:,r_slice]
        inv_T['g']     = f['inv_T'][()][:,:,r_slice]
        g_eff['g'][2]  = f['g_eff'][()][:,:,r_slice]
else:
    logger.error("Must specify an initial condition file")
    import sys
    sys.exit()

for f in [u, s1, p, ln_ρ, ln_ρT, inv_T, H_eff, g_eff]:
    f.require_scales(dealias)

r_vec = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
r_vec.require_scales(dealias)
r_vec['g'][2] = r

# Boundary conditions


I_matrix = field.Field(dist=d, bases=(b.radial_basis,), tensorsig=(c,c,), dtype=dtype)
I_matrix['g'] = 0
for i in range(3):
    I_matrix['g'][i,i,:] = 1

stress1 = grad(u) + transpose(grad(u))
stress2 = -(2/3)*I_matrix*div(u)
stress  = stress1 + stress2

VH1 = trace(dot(stress1, stress1))
VH2 = -(2/3)*(div(u)*div(u))
VH  = VH1 + VH2

#TODO: Viscous heating

u_r_bc = radComp(u(r=1))
u_perp_bc = radComp(angComp(stress(r=1), index=1))

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

# Problem
def eq_eval(eq_str):
    return [eval(expr) for expr in split_equation(eq_str)]
problem = problems.IVP([p, u, s1, tau_u, tau_T])

problem.add_equation(eq_eval("div(u) + dot(u, grad(ln_ρ)) = 0"), condition="nθ != 0")
problem.add_equation(eq_eval("p = 0"), condition="nθ == 0")
problem.add_equation(eq_eval("ddt(u) + grad(p) + g_eff*s1 - (1/Re)*(div(stress) + dot(stress, grad(ln_ρ)))= - dot(u,grad(u))"), condition = "nθ != 0")
problem.add_equation(eq_eval("u = 0"), condition="nθ == 0")
problem.add_equation(eq_eval("ddt(s1) - (1/Pe)*(lap(s1) + dot(grad(s1), grad(ln_ρT))) = - dot(u, grad(s1)) + H_eff + inv_T*VH "), condition = "nθ != 0")
problem.add_equation(eq_eval("ddt(s1)                                                 = - dot(u, grad(s1)) + H_eff + inv_T*VH "), condition = "nθ == 0")
problem.add_equation(eq_eval("u_r_bc = 0"), condition="nθ != 0")
problem.add_equation(eq_eval("u_perp_bc = 0"), condition="nθ != 0")
problem.add_equation(eq_eval("tau_u = 0"), condition="nθ == 0")
problem.add_equation(eq_eval("s1(r=1) = 0"))

print("Problem built")

# Solver
solver = solvers.InitialValueSolver(problem, ts)
solver.stop_sim_time = t_end

# Add taus
alpha_BC = 0

def C(N, ell, deg):
    ab = (alpha_BC,ell+deg+0.5)
    cd = (2,       ell+deg+0.5)
    return dedalus_sphere.jacobi.coefficient_connection(N - ell//2 + 1,ab,cd)

def BC_rows(N, ell, num_comp):
    N_list = (np.arange(num_comp)+1)*(N - ell//2 + 1)
    return N_list

for subproblem in solver.subproblems:
    ell = subproblem.group[1]
    L = subproblem.left_perm.T @ subproblem.L_min
    shape = L.shape
    if dtype == np.complex128:
        N0, N1, N2, N3, N4 = BC_rows(Nmax, ell, 5)
        tau_columns = np.zeros((shape[0], 4))
        if ell != 0:
            tau_columns[N0:N1,0] = (C(Nmax, ell, -1))[:,-1]
            tau_columns[N1:N2,1] = (C(Nmax, ell, +1))[:,-1]
            tau_columns[N2:N3,2] = (C(Nmax, ell,  0))[:,-1]
            tau_columns[N3:N4,3] = (C(Nmax, ell,  0))[:,-1]
            L[:,-4:] = tau_columns
        else: # ell = 0
            tau_columns[N3:N4, 3] = (C(Nmax, ell, 0))[:,-1]
            L[:,-1:] = tau_columns[:,3:]
    elif dtype == np.float64:
        NL = Nmax - ell//2 + 1
        N0, N1, N2, N3, N4 = BC_rows(Nmax, ell, 5) * 2
        tau_columns = np.zeros((shape[0], 8))
        if ell != 0:
            tau_columns[N0:N0+NL,0] = (C(Nmax, ell, -1))[:,-1]
            tau_columns[N1:N1+NL,2] = (C(Nmax, ell, +1))[:,-1]
            tau_columns[N2:N2+NL,4] = (C(Nmax, ell,  0))[:,-1]
            tau_columns[N3:N3+NL,6] = (C(Nmax, ell,  0))[:,-1]
            tau_columns[N0+NL:N0+2*NL,1] = (C(Nmax, ell, -1))[:,-1]
            tau_columns[N1+NL:N1+2*NL,3] = (C(Nmax, ell, +1))[:,-1]
            tau_columns[N2+NL:N2+2*NL,5] = (C(Nmax, ell,  0))[:,-1]
            tau_columns[N3+NL:N3+2*NL,7] = (C(Nmax, ell,  0))[:,-1]
            L[:,-8:] = tau_columns
        else: # ell = 0
            tau_columns[N3:N3+NL,6] = (C(Nmax, ell,  0))[:,-1]
            tau_columns[N3+NL:N3+2*NL,7] = (C(Nmax, ell,  0))[:,-1]
            L[:,-2:] = tau_columns[:,6:]
    subproblem.L_min = subproblem.left_perm @ L
    if problem.STORE_EXPANDED_MATRICES:
        subproblem.expand_matrices(['M','L'])

# Analysis
t_list = []
E_list = []
weight_θ = b.local_colatitude_weights(dealias)
weight_r = b.local_radial_weights(dealias)
reducer = GlobalArrayReducer(d.comm_cart)
vol_test = np.sum(weight_r*weight_θ+0*p['g'])*np.pi/(Lmax+1)/L_dealias
vol_test = reducer.reduce_scalar(vol_test, MPI.SUM)
vol_correction = 4*np.pi/3/vol_test

#CFL setup
class BallCFL:
    """
    A CFL to calculate the appropriate magnitude of the timestep for a spherical simulation
    """

    def __init__(self, distributor, r, Lmax, max_dt, safety=0.1, threshold=0.1, cadence=1):
        """
        Initialize the CFL class. 

        # Arguments
            distributor (Dedalus Distributor) :
                The distributor which guides the d3 simulation
            r (NumPy array) :
                The local radial grid points
            Lmax (float) :
                The maximum L value achieved by the simulation
            max_dt (float) :
                The maximum timestep size allowed, in simulation units.
            safety (float) :
                A factor to apply to the CFL calculation to adjust the timestep size
            threshold (float) :
                A factor by which the magnitude of dt must change in order for the timestep size to change
            cadence (int) :
                the number of iterations to wait between CFL calculations
        """
        self.reducer   = GlobalArrayReducer(distributor.comm_cart)
        self.dr        = np.gradient(r[0,0])
        self.Lmax      = Lmax
        self.max_dt    = max_dt
        self.safety    = safety
        self.threshold = threshold
        self.cadence   = cadence
        logger.info("CFL initialized with: max dt={:.2g}, safety={:.2g}, threshold={:.2g}".format(max_dt, self.safety, self.threshold))

    def calculate_dt(self, u, dt_old, r_index=2, φ_index=0, θ_index=1):
        """
        Calculates what the timestep should be according to the CFL condition

        # Arguments
            u (Dedalus Field) :
                A Dedalus tensor field of the velocity
            dt_old (float) : 
                The current value of the timestep
            r_index, φ_index, θ_index (int) :
                The reference index (0, 1, 2) of the different bases, respectively
        """
        u.require_scales(dealias)
        local_freq  = np.abs(u['g'][r_index]/self.dr) + (np.abs(u['g'][φ_index]) + np.abs(u['g'][θ_index]))*(self.Lmax + 1)
        global_freq = self.reducer.global_max(local_freq)
        if global_freq == 0.:
            dt = np.inf
        else:
            dt = 1 / global_freq
            dt *= self.safety
            if dt > self.max_dt: dt = self.max_dt
            if dt < dt_old*(1+self.threshold) and dt > dt_old*(1-self.threshold): dt = dt_old
        return dt

CFL = BallCFL(d, r, Lmax, max_dt, safety=float(args['--safety']), threshold=0.1, cadence=1)
dt = max_dt

# Main loop
start_time = time.time()
while solver.ok:
    if solver.iteration % 10 == 0:
        E0 = np.sum(vol_correction*weight_r*weight_θ*u['g'].real**2)
        E0 = 0.5*E0*(np.pi)/(Lmax+1)/L_dealias
        E0 = reducer.reduce_scalar(E0, MPI.SUM)
        logger.info("t = %f, dt = %f, E = %e" %(solver.sim_time, dt, E0))
        t_list.append(solver.sim_time)
        E_list.append(E0)
    solver.step(dt)
    if solver.iteration % CFL.cadence == 0:
        dt = CFL.calculate_dt(u, dt)
end_time = time.time()
print('Run time:', end_time-start_time)
