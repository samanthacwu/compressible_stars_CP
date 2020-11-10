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
import dedalus_sphere
from mpi4py import MPI

from output.averaging    import VolumeAverager, EquatorSlicer, PhiAverager, PhiThetaAverager
from output.writing      import ScalarWriter,  RadialProfileWriter, MeridionalSliceWriter, EquatorialSliceWriter, SphericalShellWriter

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
mesh = args['--mesh']
if mesh is not None:
    mesh = [int(m) for m in mesh.split(',')]

Re  = float(args['--Re'])
Pr  = 1
Pe  = Pr*Re


if args['--mesa_file'] is not None:
    with h5py.File(args['--mesa_file'], 'r') as f:
        radius = f['maxR'][()]
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


if args['--mesa_file'] is not None:
    φ1, θ1, r1 = b.local_grids((1, 1, 1))
    with h5py.File(args['--mesa_file'], 'r') as f:
        r_file = f['r'][()].flatten()
        r_slice = np.zeros_like(r_file.flatten(), dtype=bool)
        for this_r in r1.flatten():
            r_slice[this_r == r_file] = True
        if np.prod(grad_s0['g'].shape) > 0:
            grad_s0['g']        = f['grad_s0'][()][:,:,:,r_slice].reshape(grad_s0['g'].shape)
            grad_ln_ρ['g']      = f['grad_ln_ρ'][()][:,:,:,r_slice].reshape(grad_s0['g'].shape)
            grad_ln_T['g']      = f['grad_ln_T'][()][:,:,:,r_slice].reshape(grad_s0['g'].shape)
        ln_ρ['g']      = f['ln_ρ'][()][:,:,r_slice]
        ln_T['g']      = f['ln_T'][()][:,:,r_slice]
        H_eff['g']     = f['H_eff'][()][:,:,r_slice]
        T_NCC['g']     = f['T'][()][:,:,r_slice]
        ρ['g']         = np.exp(f['ln_ρ'][()][:,:,r_slice].reshape(r1.shape))
        T['g']         = f['T'][()][:,:,r_slice].reshape(r1.shape)
        inv_T['g']     = 1/T['g']
        grad_s0_RHS['g'][2]        = f['grad_s0'][()][2,:,:,r_slice].reshape(r1.shape)

        t_buoy = 1
        max_grad_s0 = f['grad_s0'][()][2,0,0,-1]

else:
    logger.info("Using polytropic initial conditions")
    from scipy.interpolate import interp1d
    with h5py.File('polytropes/poly_nOuter1.6.h5', 'r') as f:
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

max_dt = 0.5/np.sqrt(max_grad_s0)

#import matplotlib.pyplot as plt
#plt.plot(r.flatten(), grad_s0['g'][2,0,0,:])
#plt.yscale('log')
#plt.show()

logger.info('buoyancy time is {}'.format(t_buoy))
t_end = float(args['--buoy_end_time'])*t_buoy

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

# Problem
def eq_eval(eq_str):
    return [eval(expr) for expr in split_equation(eq_str)]
problem = problems.IVP([p, u, s1, tau_u, tau_T])

problem.add_equation(eq_eval("div(u) + dot(u, grad_ln_ρ) = 0"), condition="nθ != 0")
problem.add_equation(eq_eval("p = 0"), condition="nθ == 0")
problem.add_equation(eq_eval("ddt(u) + grad(p) + grad(T_NCC)*s1 - (1/Re)*momentum_viscous_terms   = - dot(u, grad(u))"), condition = "nθ != 0")
problem.add_equation(eq_eval("u = 0"), condition="nθ == 0")
problem.add_equation(eq_eval("ddt(s1) + dot(u, grad_s0) - (1/Pe)*(lap(s1) + dot(grad(s1), (grad_ln_ρ + grad_ln_T))) = - dot(u, grad(s1)) + H_eff + (1/Re)*inv_T*VH "))
problem.add_equation(eq_eval("u_r_bc    = 0"), condition="nθ != 0")
problem.add_equation(eq_eval("u_perp_bc = 0"), condition="nθ != 0")
problem.add_equation(eq_eval("tau_u     = 0"), condition="nθ == 0")
problem.add_equation(eq_eval("therm_bc  = 0"))

logger.info("Problem built")
# Solver
solver = solvers.InitialValueSolver(problem, ts)
solver.stop_sim_time = t_end
logger.info("solver built")

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


## Check condition number and plot matrices
#import matplotlib.pyplot as plt
#plt.figure()
#for subproblem in solver.subproblems:
#    ell = subproblem.group[1]
#    M = subproblem.left_perm.T @ subproblem.M_min
#    L = subproblem.left_perm.T @ subproblem.L_min
#    plt.imshow(np.log10(np.abs(L.A)))
#    plt.colorbar()
#    plt.savefig("matrices/ell_%03i.png" %ell, dpi=300)
#    plt.clf()
#    print(subproblem.group, np.linalg.cond((M + L).A))



# Analysis Setup
vol_averager       = VolumeAverager(b, d, p, dealias=dealias, radius=radius)
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
        for f in [s1, u, ρ, T]:
            f.require_scales(dealias)
        for k, op in self.ops.items():
            f = op.evaluate()
            f.require_scales(dealias)
            self.fields[k] = f['g']
        #KE & Reynolds
        self.tasks['TE']       = vol_averager(ρ['g']*T['g']*s1['g'])
        self.tasks['s1']       = vol_averager(s1['g'])
        self.tasks['KE']       = vol_averager(ρ['g']*self.fields['u·u']/2)
        self.tasks['Re_rms']   = Re*vol_rms_scalar(self.fields['u·u'], squared=True)
        self.tasks['Re_avg']   = Re*vol_avgmag_scalar(self.fields['u·u'], squared=True)

class AnelasticRPW(RadialProfileWriter):

    def __init__(self, *args, **kwargs):
        super(AnelasticRPW, self).__init__(*args, **kwargs)
        self.ops = OrderedDict()
        self.fields = OrderedDict()
        self.ops['u·σ_r']   = dot(er, dot(u, σ_post))
        self.ops['u·u']     = dot(u, u)
        self.ops['div_u']   = div(u)
        self.ops['grad_s']  = dot(er, grad(s1))
        self.ops['ur']      = dot(er, u)
        self.fields = OrderedDict()
        self.ds1_dr = field.Field(dist=d, bases=(b,), dtype=dtype)
        for k in ['s1', 'uφ', 'uθ', 'ur', 'J_cond', 'J_conv', 'enth_flux', 'visc_flux', 'cond_flux', 'KE_flux', 'ρ_ur', 'N2_term']:
            self.tasks[k] = np.zeros_like(radial_averager.global_profile)

    def evaluate_tasks(self):
        for k, op in self.ops.items():
            f = op.evaluate()
            f.require_scales(dealias)
            self.fields[k] = f['g']

        for f in [s1, u, ρ, T]:
            f.require_scales(dealias)
        self.tasks['s1'][:] = radial_averager(s1['g'])[:]
        self.tasks['uφ'][:] = radial_averager(u['g'][0])[:]
        self.tasks['uθ'][:] = radial_averager(u['g'][1])[:]
        self.tasks['ρ_ur'][:] = radial_averager(ρ['g']*u['g'][2])[:]

        self.tasks['N2_term'][:] = radial_averager(ρ['g']*u['g'][2]*T['g']*grad_s0_RHS['g'][2])

        #Get fluxes for energy output
#        self.tasks['enth_flux'][:] = radial_averager(ρ['g']*u['g'][2,:]*(p['g'] + T['g']*s1['g'])) #need to subtract a 0.5 u dot u if I use dot(u, stress1) in mometum
        enthalpy = p['g'] - 0.5*self.fields['u·u'] + T['g']*s1['g']
        self.tasks['enth_flux'][:] = radial_averager(ρ['g']*self.fields['ur']*(enthalpy)) #need to subtract a 0.5 u dot u if I use dot(u, stress1) in mometum
        self.tasks['visc_flux'][:] = radial_averager(-ρ['g']*(self.fields['u·σ_r'])/Re)
        self.tasks['cond_flux'][:] = radial_averager(-ρ['g']*T['g']*self.fields['grad_s']/Pe)
        self.tasks['KE_flux'][:]   = radial_averager(0.5*ρ['g']*self.fields['ur']*self.fields['u·u'])

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
        self.ops['s1_r0.5']   = s1(r=0.5)
        self.ops['s1_r0.95']  = s1(r=0.95)
        self.ops['s1_r1.25']  = s1(r=1.25)
        self.ops['ur_r0.5']   = radComp(u(r=0.5))
        self.ops['ur_r0.95']  = radComp(u(r=0.95))
        self.ops['ur_r1.25']  = radComp(u(r=1.25))

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
            local_part = op.evaluate()
            local_part.require_scales(dealias)
            self.local_buff *= 0
            if local_part['g'].shape[-1] == 1:
                self.local_buff[self.local_slice_indices] = local_part['g'].flatten()
            d.comm_cart.Allreduce(self.local_buff, self.global_buff, op=MPI.SUM)
            self.tasks[k] = np.copy(self.global_buff)



scalarWriter  = AnelasticSW(b, d, out_dir,  write_dt=0.25*t_buoy, dealias=dealias)
profileWriter = AnelasticRPW(b, d, out_dir, write_dt=0.5*t_buoy, max_writes=200, dealias=dealias)
msliceWriter  = AnelasticMSW(b, d, out_dir, write_dt=0.5*t_buoy, max_writes=40, dealias=dealias)
esliceWriter  = AnelasticESW(b, d, out_dir, write_dt=0.5*t_buoy, max_writes=40, dealias=dealias)
sshellWriter  = AnelasticSSW(b, d, out_dir, write_dt=0.5*t_buoy, max_writes=40, dealias=dealias)
writers = [scalarWriter, esliceWriter, profileWriter, msliceWriter, sshellWriter]

checkpoint = solver.evaluator.add_file_handler('{:s}/checkpoint'.format(out_dir), max_writes=1, sim_dt=50*t_buoy)
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
    dt = CFL.calculate_dt(u, dt)
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
profileWriter.evaluate_tasks()
start_iter = solver.iteration
try:
    while solver.ok:
        if solver.iteration % 10 == 0:
            scalarWriter.evaluate_tasks()
            KE  = vol_averager.volume*scalarWriter.tasks['KE']
            TE  = vol_averager.volume*scalarWriter.tasks['TE']
            Re0  = scalarWriter.tasks['Re_rms']
            if d.comm_cart.rank == 0:
                surf_lum = (4*np.pi*rg**2*profileWriter.tasks['cond_flux'])[0,0,-1]
            else:
                surf_lum = 0
            logger.info("t = %f, dt = %f, Re = %e, KE / TE = %e / %e" %(solver.sim_time, dt, Re0, KE, TE))
        for writer in writers:
            writer.process(solver)
        solver.step(dt)
        if solver.iteration % CFL.cadence == 0:
            dt = CFL.calculate_dt(u, dt)

        if solver.iteration % imaginary_cadence in timestepper_history:
            for f in solver.state:
                f.require_grid_space()
except:
    logger.info('something went wrong in main loop, making final checkpoint')
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
    n_coeffs = 2*(Nmax+1)*(Lmax+1)**2
    n_cpu    = d.comm_cart.size
    dof_cycles_per_cpusec = n_coeffs*n_iter/(cpu_sec*n_cpu)
    logger.info('DOF-cycles/cpu-sec : {:e}'.format(dof_cycles_per_cpusec))
    logger.info('Run iterations: {:e}'.format(n_iter))
    logger.info('Sim end time: {:e}'.format(solver.sim_time))
    logger.info('Run time: {}'.format(cpu_sec))
