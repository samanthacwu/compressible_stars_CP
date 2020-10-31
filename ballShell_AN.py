"""
d3 script for anelastic convection in a massive star's core CZ.

A config file can be provided which overwrites any number of the options below, but command line flags can also be used if preferred.

Usage:
    ballShell_AN.py [options]
    ballShell_AN.py <config> [options]

Options:
    --Re=<Re>            The Reynolds number of the numerical diffusivities [default: 1e2]
    --Pr=<Prandtl>       The Prandtl number  of the numerical diffusivities [default: 1]
    --L=<Lmax>           The value of Lmax   [default: 14]
    --NB=<Nmax>          The ball value of Nmax   [default:  47]
    --NS=<Nmax>          The shell value of Nmax   [default: 23]

    --wall_hours=<t>     The number of hours to run for [default: 24]
    --buoy_end_time=<t>  Number of buoyancy times to run [default: 1e5]
    --safety=<s>         Timestep CFL safety factor [default: 0.3]

    --mesh=<n,m>         The processor mesh over which to distribute the cores
    --A0=<A>             Amplitude of initial noise [default: 1e-6]

    --label=<label>      A label to add to the end of the output directory

    --SBDF2              Use SBDF2 (default)
    --SBDF4              Use SBDF4

    --mesa_file=<f>      path to a .h5 file of ICCs, curated from a MESA model
    --restart=<chk_f>    path to a checkpoint file to restart from
    --restart_Re=<Re>    Re of the run being restarted from

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

from output.averaging    import BallShellVolumeAverager, EquatorSlicer, PhiAverager, PhiThetaAverager
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
NmaxB      = int(args['--NB'])
NmaxS      = int(args['--NS'])
L_dealias = N_dealias = dealias = 1

out_dir = './' + sys.argv[0].split('.py')[0]
if args['--mesa_file'] is None:
    out_dir += '_polytrope'
if args['--benchmark']:
    out_dir += '_benchmark'
out_dir += '_Re{}_{}x{}_{}x{}'.format(args['--Re'], args['--L'], args['--NB'], args['--L'], args['--NS'])
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
        r_inner = f['r_inner'][()]
        r_outer = f['r_outer'][()]
else:
    r_inner = 1.2
    r_outer = 2
logger.info('r_inner: {:.2f} / r_outer: {:.2f}'.format(r_inner, r_outer))

# Bases
c    = coords.SphericalCoordinates('φ', 'θ', 'r')
c_S2 = c.S2coordsys 
d    = distributor.Distributor((c,), mesh=mesh)
bB   = basis.BallBasis(c, (2*(Lmax+2), Lmax+1, NmaxB+1), radius=r_inner, dtype=dtype)
bS   = basis.SphericalShellBasis(c, (2*(Lmax+2), Lmax+1, NmaxS+1), radii=(r_inner, r_outer), dtype=dtype)
b_mid = bB.S2_basis(radius=r_inner)
b_midS = bS.S2_basis(radius=r_inner)
b_top = bS.S2_basis(radius=r_outer)
φB,  θB,  rB  = bB.local_grids((dealias, dealias, dealias))
φBg, θBg, rBg = bB.global_grids((dealias, dealias, dealias))
φS,  θS,  rS  = bS.local_grids((dealias, dealias, dealias))
φSg, θSg, rSg = bS.global_grids((dealias, dealias, dealias))

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
uB    = field.Field(dist=d, bases=(bB,), tensorsig=(c,), dtype=dtype)
pB    = field.Field(dist=d, bases=(bB,), dtype=dtype)
s1B   = field.Field(dist=d, bases=(bB,), dtype=dtype)
uS    = field.Field(dist=d, bases=(bS,), tensorsig=(c,), dtype=dtype)
pS    = field.Field(dist=d, bases=(bS,), dtype=dtype)
s1S   = field.Field(dist=d, bases=(bS,), dtype=dtype)

tB     = field.Field(dist=d, bases=(b_mid,), dtype=dtype)
tBt    = field.Field(dist=d, bases=(b_mid,), dtype=dtype,   tensorsig=(c,))
tS2_top = field.Field(dist=d, bases=(b_top,), dtype=dtype,  tensorsig=(c_S2,))
tS2_bot = field.Field(dist=d, bases=(b_midS,), dtype=dtype, tensorsig=(c_S2,))
tSu_bot = field.Field(dist=d, bases=(b_midS,), dtype=dtype)
tSu_top = field.Field(dist=d, bases=(b_top,), dtype=dtype)
tS_bot = field.Field(dist=d, bases=(b_midS,), dtype=dtype)
tS_top = field.Field(dist=d, bases=(b_top,), dtype=dtype)

ρB   = field.Field(dist=d, bases=(bB,), dtype=dtype)
TB   = field.Field(dist=d, bases=(bB,), dtype=dtype)
ρS   = field.Field(dist=d, bases=(bS,), dtype=dtype)
TS   = field.Field(dist=d, bases=(bS,), dtype=dtype)

#nccs
grad_ln_ρB    = field.Field(dist=d, bases=(bB.radial_basis,), tensorsig=(c,), dtype=dtype)
grad_s0B      = field.Field(dist=d, bases=(bB.radial_basis,), tensorsig=(c,), dtype=dtype)
grad_ln_TB    = field.Field(dist=d, bases=(bB.radial_basis,), tensorsig=(c,), dtype=dtype)
ln_ρB         = field.Field(dist=d, bases=(bB.radial_basis,), dtype=dtype)
ln_TB         = field.Field(dist=d, bases=(bB.radial_basis,), dtype=dtype)
T_NCCB        = field.Field(dist=d, bases=(bB.radial_basis,), dtype=dtype)
ρ_NCCB        = field.Field(dist=d, bases=(bB.radial_basis,), dtype=dtype)
inv_TB        = field.Field(dist=d, bases=(bB,), dtype=dtype) #only on RHS, multiplies other terms
H_effB        = field.Field(dist=d, bases=(bB,), dtype=dtype)
grad_s0_RHSB  = field.Field(dist=d, bases=(bB,), tensorsig=(c,), dtype=dtype)
grad_ln_ρS    = field.Field(dist=d, bases=(bS.radial_basis,), tensorsig=(c,), dtype=dtype)
grad_s0S      = field.Field(dist=d, bases=(bS.radial_basis,), tensorsig=(c,), dtype=dtype)
grad_ln_TS    = field.Field(dist=d, bases=(bS.radial_basis,), tensorsig=(c,), dtype=dtype)
ln_ρS         = field.Field(dist=d, bases=(bS.radial_basis,), dtype=dtype)
ln_TS         = field.Field(dist=d, bases=(bS.radial_basis,), dtype=dtype)
T_NCCS        = field.Field(dist=d, bases=(bS.radial_basis,), dtype=dtype)
ρ_NCCS        = field.Field(dist=d, bases=(bS.radial_basis,), dtype=dtype)
inv_TS        = field.Field(dist=d, bases=(bS,), dtype=dtype) #only on RHS, multiplies other terms
H_effS        = field.Field(dist=d, bases=(bS,), dtype=dtype)
grad_s0_RHSS  = field.Field(dist=d, bases=(bS,), tensorsig=(c,), dtype=dtype)


erB  = field.Field(dist=d, bases=(bB,), tensorsig=(c,), dtype=dtype)
erB['g'][2] = 1

if args['--mesa_file'] is not None:
    φB1, θB1, rB1 = bB.local_grids((1, 1, 1))
    φS1, θS1, rS1 = bS.local_grids((1, 1, 1))
    with h5py.File(args['--mesa_file'], 'r') as f:
        rB_file = f['rB'][()].flatten()
        rB_slice = np.zeros_like(rB_file.flatten(), dtype=bool)
        rS_file = f['rS'][()].flatten()
        rS_slice = np.zeros_like(rS_file.flatten(), dtype=bool)
        for this_r in rB1.flatten():
            rB_slice[this_r == rB_file] = True
        for this_r in rS1.flatten():
            rS_slice[this_r == rS_file] = True
        
        if np.prod(grad_s0B['g'].shape) > 0:
            grad_s0B['g']        = f['grad_s0B'][()][:,:,:,rB_slice].reshape(grad_s0B['g'].shape)
            grad_ln_ρB['g']      = f['grad_ln_ρB'][()][:,:,:,rB_slice].reshape(grad_s0B['g'].shape)
            grad_ln_TB['g']      = f['grad_ln_TB'][()][:,:,:,rB_slice].reshape(grad_s0B['g'].shape)
        if np.prod(grad_s0S['g'].shape) > 0:
            grad_s0S['g']        = f['grad_s0S'][()][:,:,:,rS_slice].reshape(grad_s0S['g'].shape)
            grad_ln_ρS['g']      = f['grad_ln_ρS'][()][:,:,:,rS_slice].reshape(grad_s0S['g'].shape)
            grad_ln_TS['g']      = f['grad_ln_TS'][()][:,:,:,rS_slice].reshape(grad_s0S['g'].shape)
        ln_ρB['g']      = f['ln_ρB'][()][:,:,rB_slice]
        ln_TB['g']      = f['ln_TB'][()][:,:,rB_slice]
        H_effB['g']     = f['H_effB'][()][:,:,rB_slice]
        T_NCCB['g']     = f['TB'][()][:,:,rB_slice]
        ρB['g']         = np.exp(f['ln_ρB'][()][:,:,rB_slice].reshape(rB1.shape))
        TB['g']         = f['TB'][()][:,:,rB_slice].reshape(rB1.shape)
        inv_TB['g']     = 1/TB['g']
        grad_s0_RHSB['g'][2]        = f['grad_s0B'][()][2,:,:,rB_slice].reshape(rB1.shape)

        ln_ρS['g']      = f['ln_ρS'][()][:,:,rS_slice]
        ln_TS['g']      = f['ln_TS'][()][:,:,rS_slice]
        H_effS['g']     = f['H_effS'][()][:,:,rS_slice]
        T_NCCS['g']     = f['TS'][()][:,:,rS_slice]
        ρS['g']         = np.exp(f['ln_ρS'][()][:,:,rS_slice].reshape(rS1.shape))
        TS['g']         = f['TS'][()][:,:,rS_slice].reshape(rS1.shape)
        inv_TS['g']     = 1/TS['g']
        grad_s0_RHSS['g'][2]        = f['grad_s0S'][()][2,:,:,rS_slice].reshape(rS1.shape)

        t_buoy = 1
else:
    logger.info("Using polytropic initial conditions")

    # "Polytrope" properties
    n_rho = 2
    gamma = 5/3
    gradT = (np.exp(n_rho * (1 - gamma)) - 1)/r_outer**2
    t_buoy = 1

    #Gaussian luminosity -- zero at r = 0 and r = 1
    mu = 0.5
    sig = 0.1

    T_func  = lambda r_val: 1 + gradT*r_val**2
    ρ_func  = lambda r_val: T_func(r_val)**(1/(gamma-1))
    L_func  = lambda r_val: np.exp(-(r_val - mu)**2/(2*sig**2))
    dL_func = lambda r_val: -(2*(r_val-mu)/(2*sig**2)) * L_func(r_val)
    H_func  = lambda r_val: dL_func(r_val) / (ρ_func(r_val) * T_func(r_val) * 4 * np.pi * r_val**2)

    for basis_r, basis_fields in zip((rB, rS), ((TB, T_NCCB, ρB, ρ_NCCB, inv_TB, ln_TB, ln_ρB, grad_ln_TB, grad_ln_ρB, H_effB, grad_s0B), (TS, T_NCCS, ρS, ρ_NCCS, inv_TS, ln_TS, ln_ρS, grad_ln_TS, grad_ln_ρS, H_effS, grad_s0S))):
        T, T_NCC, ρ, ρ_NCC, inv_T, ln_T, ln_ρ, grad_ln_T, grad_ln_ρ, H_eff, grad_s0 = basis_fields

        for f in [T, T_NCC, ρ, ρ_NCC, inv_T, ln_T, ln_ρ, grad_ln_T, grad_ln_ρ, H_eff, grad_s0]:
            f.require_scales(dealias)
        T['g'] = T_NCC['g'] = T_func(basis_r)
        ρ['g'] = ρ_NCC['g'] = ρ_func(basis_r)
        inv_T['g'] = 1/T['g']
        if np.prod(ln_T['g'].shape) > 0:
            ln_T['g'][:,:,:] = np.log(T['g'])[0,0,:]
            ln_ρ['g'][:,:,:] = np.log(ρ['g'])[0,0,:]

        grad_ln_T['g'][2]  = 2*gradT*basis_r/T['g'][0,0,:]
        grad_ln_ρ['g'][2]  = (1/(gamma-1))*grad_ln_T['g'][2]

        H_eff['g'] = H_func(basis_r) / H_func(0.4)

        grad_s0['g'][2,:,:,:]     = 1e2*zero_to_one(basis_r, 1, width=0.05)
#        grad_s0['c'][:,:,:,-1] = 0


#import matplotlib.pyplot as plt
#plt.plot(rB.flatten(), grad_ln_ρB['g'][2][0,0,:])
#plt.plot(rS.flatten(), grad_ln_ρS['g'][2][0,0,:])
#plt.show()

logger.info('buoyancy time is {}'.format(t_buoy))
if args['--benchmark']:
    max_dt = 0.035*t_buoy
else:
    max_dt = 0.25*t_buoy
t_end = float(args['--buoy_end_time'])*t_buoy

#for f in [u, s1, p, ln_ρ, ln_T, inv_T, H_eff, ρ]:
#    f.require_scales(dealias)

# Stress matrices & viscous terms
I_matrixB_post = field.Field(dist=d, bases=(bB,), tensorsig=(c,c,), dtype=dtype)
I_matrixB = field.Field(dist=d, bases=(bB.radial_basis,), tensorsig=(c,c,), dtype=dtype)
I_matrixB_post['g'] = 0
I_matrixB['g'] = 0
I_matrixS_post = field.Field(dist=d, bases=(bS,), tensorsig=(c,c,), dtype=dtype)
I_matrixS = field.Field(dist=d, bases=(bS.radial_basis,), tensorsig=(c,c,), dtype=dtype)
I_matrixS_post['g'] = 0
I_matrixS['g'] = 0
for i in range(3):
    I_matrixB_post['g'][i,i,:] = 1
    I_matrixB['g'][i,i,:] = 1
    I_matrixS_post['g'][i,i,:] = 1
    I_matrixS['g'][i,i,:] = 1

#Ball stress
EB = 0.5*(grad(uB) + transpose(grad(uB)))
EB.store_last = True
divUB = div(uB)
divUB.store_last = True
σB = 2*(EB - (1/3)*divUB*I_matrixB)
σ_postB = 2*(EB - (1/3)*divUB*I_matrixB_post)
momentum_viscous_termsB = div(σB) + dot(σB, grad_ln_ρB)

VHB  = 2*(trace(dot(EB, EB)) - (1/3)*divUB*divUB)

#Shell stress
ES = 0.5*(grad(uS) + transpose(grad(uS)))
ES.store_last = True
divUS = div(uS)
divUS.store_last = True
σS = 2*(ES - (1/3)*divUS*I_matrixS)
σ_postS = 2*(ES - (1/3)*divUS*I_matrixS_post)
momentum_viscous_termsS = div(σS) + dot(σS, grad_ln_ρS)

VHS  = 2*(trace(dot(ES, ES)) - (1/3)*divUS*divUS)


#Impenetrable, stress-free boundary conditions
u_r_bcB_mid    = pB(r=r_inner)
u_r_bcS_mid    = pS(r=r_inner)
u_perp_bcB_mid = angComp(radComp(σB(r=r_inner)), index=0)
u_perp_bcS_mid = angComp(radComp(σS(r=r_inner)), index=0)
uS_r_bc        = radComp(uS(r=r_outer))
u_perp_bcS_top = radComp(angComp(ES(r=r_outer), index=1))


# Problem
def eq_eval(eq_str):
    return [eval(expr) for expr in split_equation(eq_str)]
#Only velocity
problem = problems.IVP([pB, uB, pS, uS, s1B, s1S, tBt, tSu_bot, tS2_bot, tSu_top, tS2_top, tB, tS_bot, tS_top])

### Ball momentum
problem.add_equation(eq_eval("div(uB) + dot(uB, grad_ln_ρB) = 0"), condition="nθ != 0")
problem.add_equation(eq_eval("ddt(uB) + grad(pB) - T_NCCB*grad(s1B) - (1/Re)*momentum_viscous_termsB   = - dot(uB, grad(uB))"), condition = "nθ != 0")
### Shell momentum
problem.add_equation(eq_eval("div(uS) + dot(uS, grad_ln_ρS) = 0"), condition="nθ != 0")
problem.add_equation(eq_eval("ddt(uS) + grad(pS) - T_NCCS*grad(s1S) - (1/Re)*momentum_viscous_termsS   = - dot(uS, grad(uS))"), condition = "nθ != 0")
## ell == 0 momentum
problem.add_equation(eq_eval("pB = 0"), condition="nθ == 0")
problem.add_equation(eq_eval("uB = 0"), condition="nθ == 0")
problem.add_equation(eq_eval("pS = 0"), condition="nθ == 0")
problem.add_equation(eq_eval("uS = 0"), condition="nθ == 0")

### Ball energy
problem.add_equation(eq_eval("ddt(s1B) + dot(uB, grad_s0B) - (1/Pe)*(lap(s1B) + dot(grad(s1B), (grad_ln_ρB + grad_ln_TB))) = - dot(uB, grad(s1B)) + H_effB + (1/Re)*inv_TB*VHB "))
### Shell energy
problem.add_equation(eq_eval("ddt(s1S) + dot(uS, grad_s0S) - (1/Pe)*(lap(s1S) + dot(grad(s1S), (grad_ln_ρS + grad_ln_TS))) = - dot(uS, grad(s1S)) + H_effS + (1/Re)*inv_TS*VHS "))


#Velocity BCs ell != 0
problem.add_equation(eq_eval("uB(r=r_inner) - uS(r=r_inner)    = 0"),            condition="nθ != 0")
problem.add_equation(eq_eval("u_r_bcB_mid - u_r_bcS_mid    = 0"),            condition="nθ != 0")
#problem.add_equation(eq_eval("radComp(grad(uB)(r=r_inner) - grad(uS)(r=r_inner)) = 0"), condition="nθ != 0")
problem.add_equation(eq_eval("u_perp_bcB_mid - u_perp_bcS_mid = 0"), condition="nθ != 0")
problem.add_equation(eq_eval("uS_r_bc    = 0"),                      condition="nθ != 0")
problem.add_equation(eq_eval("u_perp_bcS_top    = 0"),               condition="nθ != 0")
# velocity BCs ell == 0
problem.add_equation(eq_eval("tBt     = 0"),                         condition="nθ == 0")
problem.add_equation(eq_eval("tSu_bot     = 0"), condition="nθ == 0")
problem.add_equation(eq_eval("tS2_bot     = 0"), condition="nθ == 0")
problem.add_equation(eq_eval("tSu_top     = 0"), condition="nθ == 0")
problem.add_equation(eq_eval("tS2_top     = 0"), condition="nθ == 0")

#Entropy BCs
problem.add_equation(eq_eval("s1B(r=r_inner) - s1S(r=r_inner) = 0"))
problem.add_equation(eq_eval("radComp(grad(s1B)(r=r_inner)) - radComp(grad(s1S)(r=r_inner))    = 0"))
problem.add_equation(eq_eval("s1S(r=r_outer)    = 0"))


logger.info("Problem built")
# Solver
solver = solvers.InitialValueSolver(problem, ts)
solver.stop_sim_time = t_end
logger.info("solver built")

# Add taus
alpha_BC_ball = 0

def C_ball(N, ell, deg):
    ab = (alpha_BC_ball,ell+deg+0.5)
    cd = (2,            ell+deg+0.5)
    return dedalus_sphere.jacobi.coefficient_connection(N - ell//2 + 1,ab,cd)

# ChebyshevV
alpha_BC_shell = (2-1/2, 2-1/2)

def C_shell(N):
    ab = alpha_BC_shell
    cd = (bS.radial_basis.alpha[0]+2,bS.radial_basis.alpha[1]+2)
    return dedalus_sphere.jacobi.coefficient_connection(N,ab,cd)

def BC_rows(N, num_comp):
    N_list = (np.arange(num_comp)+1)*(N + 1)
    return N_list

#Velocity only
for subproblem in solver.subproblems:
    ell = subproblem.group[1]
    L = subproblem.left_perm.T @ subproblem.L_min
    shape = L.shape
    NL = NmaxB - ell//2 + 1
    NS = bS.shape[-1]


    if dtype == np.complex128:
        tau_columns = np.zeros((shape[0], 12))
        N0, N1, N2, N3, N4 = BC_rows(NmaxB - ell//2, 5)
        N4, N5, N6, N7, N8 = N3 + BC_rows(NS-1, 5)
        if ell != 0:
            #ball
            tau_columns[:N0,   0] = (C_ball(NmaxB, ell,  0))[:,-1]
            tau_columns[N1:N2, 1] = (C_ball(NmaxB, ell, -1))[:,-1]
            tau_columns[N2:N3, 2] = (C_ball(NmaxB, ell, +1))[:,-1]
            tau_columns[N3:N4, 3] = (C_ball(NmaxB, ell,  0))[:,-1]
            #shell
            tau_columns[N4:N5, 4]  = (C_shell(NS))[:,-1]
            tau_columns[N4:N5, 8]  = (C_shell(NS))[:,-2]
            tau_columns[N6:N7, 5]  = (C_shell(NS))[:,-1]
            tau_columns[N6:N7, 9]  = (C_shell(NS))[:,-2]
            tau_columns[N7:N8, 6]  = (C_shell(NS))[:,-1]
            tau_columns[N7:N8, 10]  = (C_shell(NS))[:,-2]
            tau_columns[N8:N8+NS, 7]  = (C_shell(NS))[:,-1]
            tau_columns[N8:N8+NS, 11]  = (C_shell(NS))[:,-2]
            L[:,-12:] = tau_columns
        else:
            tau_columns[:N0,   0] = (C_ball(NmaxB, ell,  0))[:,-1]
            tau_columns[N4:N5, 4]  = (C_shell(NS))[:,-1]
            tau_columns[N4:N5, 8]  = (C_shell(NS))[:,-2]
            L[:,N8+NS+0] = tau_columns[:,0].reshape((shape[0],1))
            L[:,N8+NS+4] = tau_columns[:,4].reshape((shape[0],1))
            L[:,N8+NS+8] = tau_columns[:,8].reshape((shape[0],1))
    elif dtype == np.float64:
        tau_columns = np.zeros((shape[0], 24))
        N0, N1, N2, N3 = BC_rows(NmaxB - ell//2, 4) *2
        N4, N5, N6, N7 = N3 + BC_rows(NS-1, 4) * 2
        if ell != 0:
            #velocity
            #ball
            tau_columns[N0:N0+NL, 0] = (C_ball(NmaxB, ell, -1))[:,-1]
            tau_columns[N1:N1+NL, 1] = (C_ball(NmaxB, ell, +1))[:,-1]
            tau_columns[N2:N2+NL, 2] = (C_ball(NmaxB, ell,  0))[:,-1]
            tau_columns[N0+NL:N0+2*NL, 3] = (C_ball(NmaxB, ell, -1))[:,-1]
            tau_columns[N1+NL:N1+2*NL, 4] = (C_ball(NmaxB, ell, +1))[:,-1]
            tau_columns[N2+NL:N2+2*NL, 5] = (C_ball(NmaxB, ell,  0))[:,-1]
            #shell
            tau_columns[N4:N4+NS, 6]   = (C_shell(NS))[:,-1]
            tau_columns[N4:N4+NS, 7]   = (C_shell(NS))[:,-2]
            tau_columns[N5:N5+NS, 10]  = (C_shell(NS))[:,-1]
            tau_columns[N5:N5+NS, 11]  = (C_shell(NS))[:,-2]
            tau_columns[N6:N6+NS, 14]  = (C_shell(NS))[:,-1]
            tau_columns[N6:N6+NS, 15]  = (C_shell(NS))[:,-2]
            tau_columns[N4+NS:N4+2*NS, 8]  = (C_shell(NS))[:,-1]
            tau_columns[N4+NS:N4+2*NS, 9] = (C_shell(NS))[:,-2]
            tau_columns[N5+NS:N5+2*NS, 12] = (C_shell(NS))[:,-1]
            tau_columns[N5+NS:N5+2*NS, 13] = (C_shell(NS))[:,-2]
            tau_columns[N6+NS:N6+2*NS, 16] = (C_shell(NS))[:,-1]
            tau_columns[N6+NS:N6+2*NS, 17] = (C_shell(NS))[:,-2]

            #entropy
            N8 = N7+2*NL
            tau_columns[N7:N7+NL, 18]      = (C_ball(NmaxB, ell,  0))[:,-1]
            tau_columns[N7+NL:N7+2*NL, 19] = (C_ball(NmaxB, ell,  0))[:,-1]
            tau_columns[N8:N8+NS, 20]  = (C_shell(NS))[:,-1]
            tau_columns[N8:N8+NS, 21]  = (C_shell(NS))[:,-2]
            tau_columns[N8+NS:N8+2*NS, 22] = (C_shell(NS))[:,-1]
            tau_columns[N8+NS:N8+2*NS, 23] = (C_shell(NS))[:,-2]

            L[:,-24:] = tau_columns
        else:
            N8 = N7+2*NL
            tau_columns[N7:N7+NL, 18]      = (C_ball(NmaxB, ell,  0))[:,-1]
            tau_columns[N7+NL:N7+2*NL, 19] = (C_ball(NmaxB, ell,  0))[:,-1]
            tau_columns[N8:N8+NS, 20]  = (C_shell(NS))[:,-1]
            tau_columns[N8:N8+NS, 21]  = (C_shell(NS))[:,-2]
            tau_columns[N8+NS:N8+2*NS, 22] = (C_shell(NS))[:,-1]
            tau_columns[N8+NS:N8+2*NS, 23] = (C_shell(NS))[:,-2]

            L[:,-6:] = tau_columns[:,-6:].reshape((shape[0], 6))
          
    L.eliminate_zeros()
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
vol_averager       = BallShellVolumeAverager(bB, bS, d, pB, pS, dealias=dealias, ball_radius=r_inner, shell_radius=r_outer)
ball_radial_averager    = PhiThetaAverager(bB, d, dealias=dealias)
ball_azimuthal_averager = PhiAverager(bB, d, dealias=dealias)
ball_equator_slicer     = EquatorSlicer(bB, d, dealias=dealias)
shell_radial_averager    = PhiThetaAverager(bS, d, dealias=dealias)
shell_azimuthal_averager = PhiAverager(bS, d, dealias=dealias)
shell_equator_slicer     = EquatorSlicer(bS, d, dealias=dealias)


def vol_avgmag_scalar(scalar_ball_field, scalar_shell_field, squared=False):
    if squared:
        fB = scalar_ball_field
        fS = scalar_shell_field
    else:
        fB = scalar_ball_field**2
        fS = scalar_shell_field**2
    return vol_averager(np.sqrt(fB), np.sqrt(fS))

def vol_rms_scalar(scalar_ball_field, scalar_shell_field, squared=False):
    if squared:
        fB = scalar_ball_field
        fS = scalar_shell_field
    else:
        fB = scalar_ball_field**2
        fS = scalar_shell_field**2
    return np.sqrt(vol_averager(fB, fS))


class AnelasticSW(ScalarWriter):

    def __init__(self, *args, **kwargs):
        super(AnelasticSW, self).__init__(*args, **kwargs)
        self.ops = OrderedDict()
        self.ops['uB·uB'] = dot(uB, uB)
        self.ops['uS·uS'] = dot(uS, uS)
        self.fields = OrderedDict()

    def evaluate_tasks(self):
        for f in [uB, s1B, s1S, uS]:
            f.require_scales(dealias)
        for k, op in self.ops.items():
            f = op.evaluate()
            f.require_scales(dealias)
            self.fields[k] = f['g']
        #KE & Reynolds
        self.tasks['TE']       = vol_averager(ρB['g']*TB['g']*s1B['g'], ρS['g']*TS['g']*s1S['g'])
        self.tasks['s1']       = vol_averager(s1B['g'], s1S['g'])
        self.tasks['KE']       = vol_averager(ρB['g']*self.fields['uB·uB']/2, ρS['g']*self.fields['uS·uS']/2)
        self.tasks['Re_rms']   = Re*vol_rms_scalar(self.fields['uB·uB'], self.fields['uS·uS'], squared=True)
        self.tasks['Re_avg']   = Re*vol_avgmag_scalar(self.fields['uB·uB'], self.fields['uS·uS'], squared=True)

class AnelasticRPW(RadialProfileWriter):

    def __init__(self, *args, **kwargs):
        super(AnelasticRPW, self).__init__(*args, **kwargs)
        self.ops = OrderedDict()
        self.fields = OrderedDict()
        self.ops['u·σ_r']   = dot(erB, dot(uB, σ_postB))
        self.ops['u·u']     = dot(uB, uB)
        self.ops['div_u']   = div(uB)
        self.ops['grad_s']  = dot(erB, grad(s1B))
        self.ops['ur']      = dot(erB, uB)
        self.fields = OrderedDict()
        for k in ['s1', 'uφ', 'uθ', 'ur', 'J_cond', 'J_conv', 'enth_flux', 'visc_flux', 'cond_flux', 'KE_flux', 'ρ_ur', 'N2_term']:
            self.tasks[k] = np.zeros_like(ball_radial_averager.global_profile)

    def evaluate_tasks(self):
        for k, op in self.ops.items():
            f = op.evaluate()
            f.require_scales(dealias)
            self.fields[k] = f['g']

        for f in [s1B, uB, ρB, TB]:
            f.require_scales(dealias)
        self.tasks['s1'][:] = ball_radial_averager(s1B['g'])[:]
        self.tasks['uφ'][:] = ball_radial_averager(uB['g'][0])[:]
        self.tasks['uθ'][:] = ball_radial_averager(uB['g'][1])[:]
        self.tasks['ρ_ur'][:] = ball_radial_averager(ρB['g']*uB['g'][2])[:]

        self.tasks['N2_term'][:] = ball_radial_averager(ρB['g']*uB['g'][2]*TB['g']*grad_s0_RHSB['g'][2])

        #Get fluxes for energy output
        self.tasks['enth_flux'][:] = ball_radial_averager(ρB['g']*self.fields['ur']*(pB['g']))
        self.tasks['visc_flux'][:] = ball_radial_averager(-ρB['g']*(self.fields['u·σ_r'])/Re)
        self.tasks['cond_flux'][:] = ball_radial_averager(-ρB['g']*TB['g']*self.fields['grad_s']/Pe)
        self.tasks['KE_flux'][:]   = ball_radial_averager(0.5*ρB['g']*self.fields['ur']*self.fields['u·u'])

class AnelasticMSW(MeridionalSliceWriter):
    
    def evaluate_tasks(self):
        for f in [s1B, uB]:
            f.require_scales(dealias)
        self.tasks['s1']  = ball_azimuthal_averager(s1B['g'],  comm=True)
        self.tasks['uφ'] = ball_azimuthal_averager(uB['g'][0], comm=True)
        self.tasks['uθ'] = ball_azimuthal_averager(uB['g'][1], comm=True)
        self.tasks['ur'] = ball_azimuthal_averager(uB['g'][2], comm=True)

class AnelasticBallESW(EquatorialSliceWriter):

    def evaluate_tasks(self):
        for f in [s1B, uB]:
            f.require_scales(dealias)
        self.tasks['s1_B'] = ball_equator_slicer(s1B['g'])
        self.tasks['uφ_B'] = ball_equator_slicer(uB['g'][0])
        self.tasks['uθ_B'] = ball_equator_slicer(uB['g'][1])
        self.tasks['ur_B'] = ball_equator_slicer(uB['g'][2])

class AnelasticShellESW(EquatorialSliceWriter):

    def evaluate_tasks(self):
        for f in [s1S, uS]:
            f.require_scales(dealias)
        self.tasks['s1_S'] = shell_equator_slicer(s1S['g'])
        self.tasks['uφ_S'] = shell_equator_slicer(uS['g'][0])
        self.tasks['uθ_S'] = shell_equator_slicer(uS['g'][1])
        self.tasks['ur_S'] = shell_equator_slicer(uS['g'][2])



class AnelasticSSW(SphericalShellWriter):
    def __init__(self, *args, **kwargs):
        super(AnelasticSSW, self).__init__(*args, **kwargs)
        self.ops = OrderedDict()
        self.ops['s1_r0.95'] = s1B(r=0.95)
        self.ops['s1_r0.5']  = s1B(r=0.5)
        self.ops['s1_r0.25'] = s1B(r=0.25)
        self.ops['ur_r0.95'] = radComp(uB(r=0.95))
        self.ops['ur_r0.5']  = radComp(uB(r=0.5))
        self.ops['ur_r0.25'] = radComp(uB(r=0.25))

        # Logic for local and global slicing
        φbool = np.zeros_like(φBg, dtype=bool)
        θbool = np.zeros_like(θBg, dtype=bool)
        for φBl in φB.flatten():
            φbool[φBl == φBg] = 1
        for θBl in θB.flatten():
            θbool[θBl == θBg] = 1
        self.local_slice_indices = φbool*θbool
        self.local_shape    = (φBl*θBl).shape
        self.global_shape   = (φBg*θBg).shape

        self.local_buff  = np.zeros(self.global_shape)
        self.global_buff = np.zeros(self.global_shape)

    def evaluate_tasks(self):
        for f in [s1B, uB]:
            f.require_scales(dealias)
        for k, op in self.ops.items():
            local_part = op.evaluate()
            local_part.require_scales(dealias)
            self.local_buff *= 0
            if local_part['g'].shape[-1] == 1:
                self.local_buff[self.local_slice_indices] = local_part['g'].flatten()
            d.comm_cart.Allreduce(self.local_buff, self.global_buff, op=MPI.SUM)
            self.tasks[k] = np.copy(self.global_buff)



scalarWriter  = AnelasticSW(bB, d, out_dir,  write_dt=0.25*t_buoy, dealias=dealias)
profileWriter = AnelasticRPW(bB, d, out_dir, write_dt=0.5*t_buoy, max_writes=200, dealias=dealias)
msliceWriter  = AnelasticMSW(bB, d, out_dir, write_dt=0.5*t_buoy, max_writes=40, dealias=dealias)
esliceWriterB = AnelasticBallESW(bB, d, out_dir, filename='eq_sliceB', write_dt=0.1*t_buoy, max_writes=40, dealias=dealias)
esliceWriterS = AnelasticShellESW(bS, d, out_dir, filename='eq_sliceS', write_dt=0.1*t_buoy, max_writes=40, dealias=dealias)
sshellWriter  = AnelasticSSW(bB, d, out_dir, write_dt=0.5*t_buoy, max_writes=40, dealias=dealias)
writers = [scalarWriter, esliceWriterB, esliceWriterS, profileWriter, msliceWriter, sshellWriter]

ball_checkpoint = solver.evaluator.add_file_handler('{:s}/ball_checkpoint'.format(out_dir), max_writes=1, sim_dt=10*t_buoy)
ball_checkpoint.add_task(s1B, name='s1B', scales=1, layout='c')
ball_checkpoint.add_task(uB, name='uB', scales=1, layout='c')
ball_checkpoint.add_task(s1S, name='s1S', scales=1, layout='c')
ball_checkpoint.add_task(uS, name='uS', scales=1, layout='c')


imaginary_cadence = 100


#CFL setup
class BallCFL:
    """
    A CFL to calculate the appropriate magnitude of the timestep for a spherical simulation
    """

    def __init__(self, distributor, r, Lmax, max_dt, safety=0.1, threshold=0.1, cadence=1, radius_cutoff=None):
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
            radius_cutoff (float) :
                Radial value beyond which to ignore CFL constraints
        """
        self.reducer   = GlobalArrayReducer(distributor.comm_cart)
        self.dr        = np.gradient(r[0,0])
        self.r         = r[0,0]
        self.Lmax      = Lmax
        self.max_dt    = max_dt
        self.safety    = safety
        self.threshold = threshold
        self.cadence   = cadence
        self.radius_cutoff = radius_cutoff
        if self.radius_cutoff is not None:
            self.bad_radius = self.r >= self.radius_cutoff
            self.any_good = bool(1-np.max(self.bad_radius))
            print(self.r, self.bad_radius, self.any_good, self.radius_cutoff)
        logger.info("CFL initialized with: max dt={:.2g}, safety={:.2g}, threshold={:.2g}, cutoff={}".format(max_dt, self.safety, self.threshold, self.radius_cutoff))

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
        if self.radius_cutoff is not None:
            if not self.any_good:
                local_freq *= 0
            else:
                local_freq[:,:,self.bad_radius] *= 0
        global_freq = self.reducer.global_max(local_freq)
        if global_freq == 0.:
            dt = np.inf
        else:
            dt = 1 / global_freq
            dt *= self.safety
            if dt > self.max_dt: dt = self.max_dt
            if dt < dt_old*(1+self.threshold) and dt > dt_old*(1-self.threshold): dt = dt_old
        return dt

CFLB = BallCFL(d, rB, Lmax, max_dt, safety=float(args['--safety']), threshold=0.1, cadence=1, radius_cutoff=1.05)
CFLS = BallCFL(d, rS, Lmax, max_dt, safety=float(args['--safety']), threshold=0.1, cadence=1)
dt = max_dt

if args['--restart'] is not None:
    fname = args['--restart']
    fdir = fname.split('.h5')[0]
    check_name = fdir.split('/')[-1]
    #Try to just load the loal piece file

    restart_Re = args['--restart_Re']
    if restart_Re is not None:
        restart_Re = float(restart_Re)
        vel_factor = restart_Re/Re
    else:
        vel_factor = 1

    import h5py
    with h5py.File('{}/{}_p{}.h5'.format(fdir, check_name, d.comm_cart.rank), 'r') as f:
        s1B.set_scales(1)
        uB.set_scales(1)
        s1S.set_scales(1)
        uS.set_scales(1)
        s1B['c'] = f['tasks/s1B'][()][-1,:]
        uB['c'] = f['tasks/uB'][()][-1,:]
        s1S['c'] = f['tasks/s1S'][()][-1,:]
        uS['c'] = f['tasks/uS'][()][-1,:]

        uB['g'] *= vel_factor
        uS['g'] *= vel_factor
        s1B.require_scales(dealias)
        uB.require_scales(dealias)
        s1S.require_scales(dealias)
        uS.require_scales(dealias)
    dt = CFLB.calculate_dt(uB, dt)
#    dt = np.min((CFLB.calculate_dt(uB, dt), CFLS.calculate_dt(uS, dt)))
else:
    if args['--benchmark']:
        #Marti benchmark-like ICs
        A0 = 1e-3
        s1B['g'] = A0*np.sqrt(35/np.pi)*(rB/r_outer)**3*(1-(rB/r_outer)**2)*(np.cos(3*φB)+np.sin(3*φB))*np.sin(θB)**3
        s1S['g'] = A0*np.sqrt(35/np.pi)*(rS/r_outer)**3*(1-(rS/r_outer)**2)*(np.cos(3*φS)+np.sin(3*φS))*np.sin(θS)**3
    else:
        # Initial conditions
        A0   = float(1e-6)
        seed = 42 + d.comm_cart.rank
        rand = np.random.RandomState(seed=seed)
        filter_scale = 0.25

        # Generate noise & filter it
        s1B['g'] = A0*rand.standard_normal(s1B['g'].shape)*np.sin(2*np.pi*rB)
        s1B.require_scales(filter_scale)
        s1B['c']
        s1B['g']
        s1B.require_scales(dealias)



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
            logger.info("t = %f, dt = %f, Re = %e, KE / TE = %e / %e" %(solver.sim_time, dt, Re0, KE, TE))
        for writer in writers:
            writer.process(solver)
        solver.step(dt)
        if solver.iteration % CFLB.cadence == 0:
            dt = CFLB.calculate_dt(uB, dt)
#            dt = np.min((CFLB.calculate_dt(uB, dt), CFLS.calculate_dt(uS, dt)))

        if solver.iteration % imaginary_cadence in timestepper_history:
            for f in solver.state:
                f.require_grid_space()
except:
    logger.info('something went wrong in main loop, making final checkpoint')
    raise
finally:
    fcheckpoint = solver.evaluator.add_file_handler('{:s}/final_checkpoint'.format(out_dir), max_writes=1, iter=1)
    fcheckpoint.add_task(s1B, name='s1B', scales=1, layout='c')
    fcheckpoint.add_task(uB, name='uB', scales=1, layout='c')
    fcheckpoint.add_task(s1S, name='s1S', scales=1, layout='c')
    fcheckpoint.add_task(uS, name='uS', scales=1, layout='c')
    solver.step(1e-5*dt)

    end_time = time.time()
    end_iter = solver.iteration
    cpu_sec  = end_time - start_time
    n_iter   = end_iter - start_iter

    #TODO: Make the end-of-sim report better
    n_coeffs = 2*(NmaxB + NmaxS + 2)*(Lmax+1)**2
    n_cpu    = d.comm_cart.size
    dof_cycles_per_cpusec = n_coeffs*n_iter/(cpu_sec*n_cpu)
    logger.info('DOF-cycles/cpu-sec : {:e}'.format(dof_cycles_per_cpusec))
    logger.info('Run iterations: {:e}'.format(n_iter))
    logger.info('Sim end time: {:e}'.format(solver.sim_time))
    logger.info('Run time: {}'.format(cpu_sec))
