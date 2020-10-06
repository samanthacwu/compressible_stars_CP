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

from output.averaging    import VolumeAverager, EquatorSlicer, PhiAverager, PhiThetaAverager
from output.writing      import ScalarWriter,  RadialProfileWriter, MeridionalSliceWriter, EquatorialSliceWriter, SphericalShellWriter

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

out_dir = './' + sys.argv[0].split('.py')[0]
out_dir += '_Re{}_{}x{}'.format(args['--Re'], args['--L'], args['--N'])
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

# Bases
c = coords.SphericalCoordinates('φ', 'θ', 'r')
d = distributor.Distributor((c,), mesh=mesh)
b = basis.BallBasis(c, (2*(Lmax+2), Lmax+1, Nmax+1), radius=radius, dtype=dtype)
bNCC = basis.BallBasis(c, (1, 1, Nmax+1), radius=radius, dtype=dtype)
b_S2 = b.S2_basis()
φ, θ, r = b.local_grids((dealias, dealias, dealias))
φg, θg, rg = b.global_grids((dealias, dealias, dealias))

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
ρ  = field.Field(dist=d, bases=(b,), dtype=dtype)

#nccs
ln_ρ  = field.Field(dist=d, bases=(b.radial_basis,), dtype=dtype)
ln_ρT = field.Field(dist=d, bases=(b.radial_basis,), dtype=dtype)
inv_T = field.Field(dist=d, bases=(b,), dtype=dtype) #only on RHS, multiplies other terms
H_eff = field.Field(dist=d, bases=(b,), dtype=dtype)
g_eff = field.Field(dist=d, bases=(b.radial_basis,), tensorsig=(c,), dtype=dtype)

grad_ln_ρ  = grad(ln_ρ).evaluate()
grad_ln_ρT = grad(ln_ρT).evaluate()

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
        g_eff['g']     = f['g_eff'][()][:,:,:,r_slice]
        ρ['g']         = np.exp(f['ln_ρ'][()][:,:,r_slice].reshape(r1.shape))

        t_buoy = np.sqrt(1/f['g_eff'][()].max())
else:
    logger.error("Must specify an initial condition file")
    import sys
    sys.exit()
logger.info('buoyancy time is {}'.format(t_buoy))
max_dt = 0.1*t_buoy
t_end = 1e3*t_buoy

for f in [u, s1, p, ln_ρ, ln_ρT, inv_T, H_eff, g_eff, ρ]:
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
stress1.store_last = True
momentum_viscous_terms = div(stress1) + dot(grad_ln_ρ, stress1) - (2/3)*(grad(div(u)) + grad_ln_ρ*div(u) )

trace_stress = trace(stress1)
trace_stress.store_last = True
VH1 = trace(dot(stress1, stress1))
VH2 = -(1/3)*(trace_stress*trace_stress)
VH  = VH1 + VH2

#TODO: Viscous heating

u_r_bc = radComp(u(r=1))
u_perp_bc = radComp(angComp(stress1(r=1), index=1))

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

problem.add_equation(eq_eval("div(u) + dot(u, grad_ln_ρ) = 0"), condition="nθ != 0")
problem.add_equation(eq_eval("p = 0"), condition="nθ == 0")
problem.add_equation(eq_eval("ddt(u) + grad(p) - g_eff*s1 - (1/Re)*momentum_viscous_terms = - dot(u, stress1)"), condition = "nθ != 0")
problem.add_equation(eq_eval("u = 0"), condition="nθ == 0")
problem.add_equation(eq_eval("ddt(s1) - (1/Pe)*(lap(s1) + dot(grad(s1), grad_ln_ρT)) = - dot(u, grad(s1)) + H_eff + (1/Re)*inv_T*VH "))
#problem.add_equation(eq_eval("ddt(s1)                                                 = - dot(u, grad(s1)) + H_eff + inv_T*VH "), condition = "nθ == 0")
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

# Analysis Setup
vol_averager       = VolumeAverager(b, d, p, dealias=dealias)
radial_averager    = PhiThetaAverager(b, d, dealias=dealias)
azimuthal_averager = PhiAverager(b, d, dealias=dealias)
equator_slicer     = EquatorSlicer(b, d, dealias=dealias)

def vol_avgmag_scalar(scalar_field, squared=False):
    if squared:
        f = scalar_field
    else:
        f = scalar_field**2
    return vol_averager(np.sqrt(f))

def vol_rms_scalar(scalar_field, squared=False):
    if squared:
        f = scalar_field
    else:
        f = scalar_field**2
    return np.sqrt(vol_averager(f))


class AnelasticSW(ScalarWriter):

    def __init__(self, *args, **kwargs):
        super(AnelasticSW, self).__init__(*args, **kwargs)
        self.ops = OrderedDict()
        self.ops['u·u'] = dot(u, u)
        self.fields = OrderedDict()

    def evaluate_tasks(self):
        for f in [s1, u]:
            s1.require_scales(dealias)
            u.require_scales(dealias)
        for k, op in self.ops.items():
            f = op.evaluate()
            f.require_scales(dealias)
            self.fields[k] = f['g']
        #KE & Reynolds
        self.tasks['KE']       = vol_averager(ρ['g']*self.fields['u·u']/2)
        self.tasks['Re_rms']   = Re*vol_rms_scalar(self.fields['u·u'], squared=True)
        self.tasks['Re_avg']   = Re*vol_avgmag_scalar(self.fields['u·u'], squared=True)

class AnelasticRPW(RadialProfileWriter):

    def __init__(self, *args, **kwargs):
        super(AnelasticRPW, self).__init__(*args, **kwargs)
        self.grad_s1_op = grad(s1)
        self.ds1_dr = field.Field(dist=d, bases=(b,), dtype=dtype)

    def evaluate_tasks(self):
        for f in [s1, u]:
            s1.require_scales(dealias)
            u.require_scales(dealias)
        self.tasks['s1'] = radial_averager(s1['g'])
        self.tasks['uφ'] = radial_averager(u['g'][0])
        self.tasks['uθ'] = radial_averager(u['g'][1])
        self.tasks['ur'] = radial_averager(u['g'][2])

        #Get fluxes for energy output
        self.ds1_dr['g'] = self.grad_s1_op.evaluate()['g'][2]
        self.ds1_dr.require_scales(dealias)
        s1.require_scales(dealias)
        self.tasks['J_cond'] = radial_averager(ρ['g']*self.ds1_dr['g']/Pe)
        self.tasks['J_conv'] = radial_averager(ρ['g']*u['g'][2]*s1['g'])

class AnelasticMSW(MeridionalSliceWriter):
    
    def evaluate_tasks(self):
        for f in [s1, u]:
            s1.require_scales(dealias)
            u.require_scales(dealias)
        self.tasks['s1']  = azimuthal_averager(s1['g'],  comm=True)
        self.tasks['uφ'] = azimuthal_averager(u['g'][0], comm=True)
        self.tasks['uθ'] = azimuthal_averager(u['g'][1], comm=True)
        self.tasks['ur'] = azimuthal_averager(u['g'][2], comm=True)

class AnelasticESW(EquatorialSliceWriter):

    def evaluate_tasks(self):
        for f in [s1, u]:
            s1.require_scales(dealias)
            u.require_scales(dealias)
        self.tasks['s1']  = equator_slicer(s1['g'])
        self.tasks['uφ'] = equator_slicer(u['g'][0])
        self.tasks['uθ'] = equator_slicer(u['g'][1])
        self.tasks['ur'] = equator_slicer(u['g'][2])

class AnelasticSSW(SphericalShellWriter):
    def __init__(self, *args, **kwargs):
        super(AnelasticSSW, self).__init__(*args, **kwargs)
        self.ops = OrderedDict()
        self.ops['s1_r0.95']  = s1(r=0.95)
        self.ops['s1_r0.5']   = s1(r=0.5)
        self.ops['ur_r0.95'] = radComp(u(r=0.95))
        self.ops['ur_r0.5']  = radComp(u(r=0.5))

        # Logic for local and global slicing
        φbool = np.zeros_like(φg, dtype=bool)
        θbool = np.zeros_like(θg, dtype=bool)
        for φl in φ.flatten():
            φbool[φl == φg] = 1
        for θl in θ.flatten():
            θbool[θl == θg] = 1
        self.local_slice_indices = φbool*θbool
        self.local_shape    = (φl*θl).shape
        self.global_shape   = (φg*θg).shape

        self.local_buff  = np.zeros(self.global_shape)
        self.global_buff = np.zeros(self.global_shape)

    def evaluate_tasks(self):
        for f in [s1, u]:
            s1.require_scales(dealias)
            u.require_scales(dealias)
        for k, op in self.ops.items():
            local_part = op.evaluate()['g'].real
            self.local_buff *= 0
            if local_part.shape[-1] == 1:
                self.local_buff[self.local_slice_indices] = local_part.flatten()
            d.comm_cart.Allreduce(self.local_buff, self.global_buff, op=MPI.SUM)
            self.tasks[k] = np.copy(self.global_buff)



scalarWriter  = AnelasticSW(b, d, out_dir,  write_dt=0.25*t_buoy, dealias=dealias)
profileWriter = AnelasticRPW(b, d, out_dir, write_dt=0.5*t_buoy, max_writes=200, dealias=dealias)
msliceWriter  = AnelasticMSW(b, d, out_dir, write_dt=0.5*t_buoy, max_writes=40, dealias=dealias)
esliceWriter  = AnelasticESW(b, d, out_dir, write_dt=0.5*t_buoy, max_writes=40, dealias=dealias)
sshellWriter  = AnelasticSSW(b, d, out_dir, write_dt=0.5*t_buoy, max_writes=40, dealias=dealias)
writers = [scalarWriter, esliceWriter]
#writers = [scalarWriter, profileWriter, msliceWriter, esliceWriter, sshellWriter]

checkpoint = solver.evaluator.add_file_handler('{:s}/checkpoint'.format(out_dir), max_writes=2, sim_dt=50*t_buoy)
checkpoint.add_task(s1, name='s1', scales=1, layout='c')
checkpoint.add_task(u, name='u', scales=1, layout='c')


imaginary_cadence = 100


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
        scalarWriter.evaluate_tasks()
        E0  = vol_averager.volume*scalarWriter.tasks['KE']
        Re0  = scalarWriter.tasks['Re_rms']
        logger.info("t = %f, dt = %f, Re = %e, E = %e" %(solver.sim_time, dt, Re0, E0))
    for writer in writers:
        writer.process(solver)
    solver.step(dt)
    if solver.iteration % CFL.cadence == 0:
        dt = CFL.calculate_dt(u, dt)

    if solver.iteration % imaginary_cadence in timestepper_history:
        for f in solver.state:
            f.require_grid_space()
end_time = time.time()
print('Run time:', end_time-start_time)
