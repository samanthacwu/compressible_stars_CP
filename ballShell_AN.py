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
    --N=<Nmax>           The value of Nmax   [default: 63]

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
Lmax      = int(args['--L'])
Nmax      = int(args['--N'])
L_dealias = N_dealias = dealias = 1

out_dir = './' + sys.argv[0].split('.py')[0]
if args['--mesa_file'] is None:
    out_dir += '_polytrope'
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
        maxR = f['maxR'][()]
else:
    maxR = 1.5
radius    = maxR

# Bases
c    = coords.SphericalCoordinates('φ', 'θ', 'r')
c_S2 = c.S2coordsys 
d    = distributor.Distributor((c,), mesh=mesh)
bB   = basis.BallBasis(c, (2*(Lmax+2), Lmax+1, Nmax+1), radius=1, dtype=dtype)
bS   = basis.SphericalShellBasis(c, (2*(Lmax+2), Lmax+1, Nmax+1), radii=(1, radius), dtype=dtype)
b_mid = bB.S2_basis(radius=1)
b_midS = bS.S2_basis(radius=1)
b_top = bS.S2_basis(radius=radius)
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
tSt_top = field.Field(dist=d, bases=(b_top,), dtype=dtype,  tensorsig=(c,))
tSt_bot = field.Field(dist=d, bases=(b_midS,), dtype=dtype, tensorsig=(c,))
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
inv_TB        = field.Field(dist=d, bases=(bB,), dtype=dtype) #only on RHS, multiplies other terms
H_effB        = field.Field(dist=d, bases=(bB,), dtype=dtype)
grad_s0_RHSB  = field.Field(dist=d, bases=(bB,), tensorsig=(c,), dtype=dtype)
grad_ln_ρS    = field.Field(dist=d, bases=(bS.radial_basis,), tensorsig=(c,), dtype=dtype)
grad_s0S      = field.Field(dist=d, bases=(bS.radial_basis,), tensorsig=(c,), dtype=dtype)
grad_ln_TS    = field.Field(dist=d, bases=(bS.radial_basis,), tensorsig=(c,), dtype=dtype)
ln_ρS         = field.Field(dist=d, bases=(bS.radial_basis,), dtype=dtype)
ln_TS         = field.Field(dist=d, bases=(bS.radial_basis,), dtype=dtype)
T_NCCS        = field.Field(dist=d, bases=(bS.radial_basis,), dtype=dtype)
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
    H_effB['g'] = 1
    T_NCCB['g'] = 1
    TB['g']     = 1
    ρB['g']     = 1
    H_effS['g'] = 1
    T_NCCS['g'] = 1
    TS['g']     = 1
    ρS['g']     = 1

    t_buoy = 1

logger.info('buoyancy time is {}'.format(t_buoy))
max_dt = 0.5*t_buoy
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
u_perp_bcB_mid = radComp(angComp(EB(r=1), index=1))
u_perp_bcS_mid = radComp(angComp(ES(r=1), index=1))
uS_r_bc        = radComp(uS(r=radius))
u_perp_bcS_top = radComp(angComp(ES(r=radius), index=1))

# Problem
def eq_eval(eq_str):
    return [eval(expr) for expr in split_equation(eq_str)]
#Only velocity
problem = problems.IVP([s1B, pB, uB, s1S, pS, uS, tB,  tBt, tS_bot, tS_top,  tSt_bot, tSt_top])

### Ball energy
problem.add_equation(eq_eval("ddt(s1B) + dot(uB, grad_s0B) - (1/Pe)*(lap(s1B) + dot(grad(s1B), (grad_ln_ρB + grad_ln_TB))) = - dot(uB, grad(s1B)) + H_effB + (1/Re)*inv_TB*VHB "))
### Ball momentum
problem.add_equation(eq_eval("div(uB) + dot(uB, grad_ln_ρB) = 0"), condition="nθ != 0")
problem.add_equation(eq_eval("ddt(uB) + grad(pB) - T_NCCB*grad(s1B) - (1/Re)*momentum_viscous_termsB   = - dot(uB, grad(uB))"), condition = "nθ != 0")
problem.add_equation(eq_eval("pB = 0"), condition="nθ == 0")
problem.add_equation(eq_eval("uB = 0"), condition="nθ == 0")
### Shell energy
problem.add_equation(eq_eval("ddt(s1S) + dot(uS, grad_s0S) - (1/Pe)*(lap(s1S) + dot(grad(s1S), (grad_ln_ρS + grad_ln_TS))) = - dot(uS, grad(s1S)) + H_effS + (1/Re)*inv_TS*VHS "))
### Shell momentum
problem.add_equation(eq_eval("div(uS) + dot(uS, grad_ln_ρS) = 0"), condition="nθ != 0")
problem.add_equation(eq_eval("ddt(uS) + grad(pS) - T_NCCS*grad(s1S) - (1/Re)*momentum_viscous_termsS   = - dot(uS, grad(uS))"), condition = "nθ != 0")
## ell == 0
problem.add_equation(eq_eval("pS = 0"), condition="nθ == 0")
problem.add_equation(eq_eval("uS = 0"), condition="nθ == 0")


problem.add_equation(eq_eval("s1B(r=1) = 0"))
problem.add_equation(eq_eval("uB(r=1) - uS(r=1)    = 0"),            condition="nθ != 0")
problem.add_equation(eq_eval("radComp(grad(s1B)(r=1) - grad(s1S)(r=1))    = 0"))
problem.add_equation(eq_eval("s1S(r=radius)    = 0"))
problem.add_equation(eq_eval("u_perp_bcB_mid - u_perp_bcS_mid = 0"), condition="nθ != 0")
problem.add_equation(eq_eval("pB(r=1) - pS(r=1)    = 0"),            condition="nθ != 0")
problem.add_equation(eq_eval("uS_r_bc    = 0"),                      condition="nθ != 0")
problem.add_equation(eq_eval("u_perp_bcS_top    = 0"),               condition="nθ != 0")
problem.add_equation(eq_eval("tBt     = 0"), condition="nθ == 0")
problem.add_equation(eq_eval("tSt_bot     = 0"), condition="nθ == 0")
problem.add_equation(eq_eval("tSt_top     = 0"), condition="nθ == 0")



#
#problem = problems.IVP([pB, uB, s1B, pS, uS, s1S, tBt, tB, tSt_bot, tSt_top, tS_bot, tS_top])
#
### Ball energy
#problem.add_equation(eq_eval("ddt(s1B) + dot(uB, grad_s0B) - (1/Pe)*(lap(s1B) + dot(grad(s1B), (grad_ln_ρB + grad_ln_TB))) = - dot(uB, grad(s1B)) + H_effB + (1/Re)*inv_TB*VHB "))
### Shell energy
#problem.add_equation(eq_eval("ddt(s1S) + dot(uS, grad_s0S) - (1/Pe)*(lap(s1S) + dot(grad(s1S), (grad_ln_ρS + grad_ln_TS))) = - dot(uS, grad(s1S)) + H_effS + (1/Re)*inv_TS*VHS "))
### ell == 0
#problem.add_equation(eq_eval("pB = 0"), condition="nθ == 0")
#problem.add_equation(eq_eval("uB = 0"), condition="nθ == 0")
#problem.add_equation(eq_eval("pS = 0"), condition="nθ == 0")
#problem.add_equation(eq_eval("uS = 0"), condition="nθ == 0")
#
#problem.add_equation(eq_eval("uB(r=1) - uS(r=1)    = 0"),            condition="nθ != 0")
#problem.add_equation(eq_eval("s1B(r=1) - s1S(r=1)    = 0"))
#problem.add_equation(eq_eval("u_perp_bcB_mid - u_perp_bcS_mid = 0"), condition="nθ != 0")
#problem.add_equation(eq_eval("pB(r=1) - pS(r=1)    = 0"),            condition="nθ != 0")
#problem.add_equation(eq_eval("uS_r_bc    = 0"),                      condition="nθ != 0")
#problem.add_equation(eq_eval("u_perp_bcS_top    = 0"),               condition="nθ != 0")
#problem.add_equation(eq_eval("radComp(grad(s1B)(r=1) - grad(s1S)(r=1))    = 0"))
#problem.add_equation(eq_eval("s1S(r=radius)    = 0"))
#problem.add_equation(eq_eval("tBt     = 0"), condition="nθ == 0")
#problem.add_equation(eq_eval("tSt_bot     = 0"), condition="nθ == 0")
#problem.add_equation(eq_eval("tSt_top     = 0"), condition="nθ == 0")

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
    NL = Nmax - ell//2 + 1
    NS = bS.shape[-1]


    if dtype == np.complex128:
        tau_columns = np.zeros((shape[0], 12))
        N0, N1, N2, N3, N4 = BC_rows(Nmax - ell//2, 5)
        N4, N5, N6, N7, N8 = N3 + BC_rows(NS-1, 5)
        if ell != 0:
            #ball
            tau_columns[:N0,   0] = (C_ball(Nmax, ell,  0))[:,-1]
            tau_columns[N1:N2, 1] = (C_ball(Nmax, ell, -1))[:,-1]
            tau_columns[N2:N3, 2] = (C_ball(Nmax, ell, +1))[:,-1]
            tau_columns[N3:N4, 3] = (C_ball(Nmax, ell,  0))[:,-1]
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
            tau_columns[:N0,   0] = (C_ball(Nmax, ell,  0))[:,-1]
            tau_columns[N4:N5, 4]  = (C_shell(NS))[:,-1]
            tau_columns[N4:N5, 8]  = (C_shell(NS))[:,-2]
            L[:,N8+NS+0] = tau_columns[:,0].reshape((shape[0],1))
            L[:,N8+NS+4] = tau_columns[:,4].reshape((shape[0],1))
            L[:,N8+NS+8] = tau_columns[:,8].reshape((shape[0],1))
    elif dtype == np.float64:
        tau_columns = np.zeros((shape[0], 24))
        N0, N1, N2, N3, N4 = BC_rows(Nmax - ell//2, 5) *2
        N4, N5, N6, N7, N8 = N3 + BC_rows(NS-1, 5) * 2
        print(N1, N2, N3, N4, N6, N7, N8)
        if ell != 0:
            #ball
            tau_columns[:NL, 0] = (C_ball(Nmax, ell,  0))[:,-1]
            tau_columns[N1:N1+NL, 2] = (C_ball(Nmax, ell, -1))[:,-1]
            tau_columns[N2:N2+NL, 4] = (C_ball(Nmax, ell, +1))[:,-1]
            tau_columns[N3:N3+NL, 6] = (C_ball(Nmax, ell,  0))[:,-1]
            tau_columns[NL:2*NL, 1] = (C_ball(Nmax, ell,  0))[:,-1]
            tau_columns[N1+NL:N1+2*NL, 3] = (C_ball(Nmax, ell, -1))[:,-1]
            tau_columns[N2+NL:N2+2*NL, 5] = (C_ball(Nmax, ell, +1))[:,-1]
            tau_columns[N3+NL:N3+2*NL, 7] = (C_ball(Nmax, ell,  0))[:,-1]
            #shell
            tau_columns[N4:N4+NS, 8]   = (C_shell(NS))[:,-1]
            tau_columns[N4:N4+NS, 10]  = (C_shell(NS))[:,-2]
            tau_columns[N6:N6+NS, 12]  = (C_shell(NS))[:,-1]
            tau_columns[N6:N6+NS, 14]  = (C_shell(NS))[:,-2]
            tau_columns[N7:N7+NS, 16]  = (C_shell(NS))[:,-1]
            tau_columns[N7:N7+NS, 18]  = (C_shell(NS))[:,-2]
            tau_columns[N8:N8+NS, 20]  = (C_shell(NS))[:,-1]
            tau_columns[N8:N8+NS, 22]  = (C_shell(NS))[:,-2]
            tau_columns[N4+NS:N4+2*NS, 9] = (C_shell(NS))[:,-1]
            tau_columns[N4+NS:N4+2*NS, 11] = (C_shell(NS))[:,-2]
            tau_columns[N6+NS:N6+2*NS, 13] = (C_shell(NS))[:,-1]
            tau_columns[N6+NS:N6+2*NS, 15] = (C_shell(NS))[:,-2]
            tau_columns[N7+NS:N7+2*NS, 17] = (C_shell(NS))[:,-1]
            tau_columns[N7+NS:N7+2*NS, 19] = (C_shell(NS))[:,-2]
            tau_columns[N8+NS:N8+2*NS, 21] = (C_shell(NS))[:,-1]
            tau_columns[N8+NS:N8+2*NS, 23] = (C_shell(NS))[:,-2]

            L[:,-24:] = tau_columns
        else:
            tau_columns[:NL, 0] = (C_ball(Nmax, ell,  0))[:,-1]
            tau_columns[NL:2*NL, 1] = (C_ball(Nmax, ell,  0))[:,-1]
            tau_columns[N4:N4+NS, 8]  = (C_shell(NS))[:,-1]
            tau_columns[N4:N4+NS, 10]  = (C_shell(NS))[:,-2]
            tau_columns[N4+NS:N4+2*NS, 9] = (C_shell(NS))[:,-1]
            tau_columns[N4+NS:N4+2*NS, 11] = (C_shell(NS))[:,-2]
            L[:,N8+2*NS+0]  = tau_columns[:,0].reshape((shape[0],1))
            L[:,N8+2*NS+1]  = tau_columns[:,1].reshape((shape[0],1))
            L[:,N8+2*NS+8]  = tau_columns[:,8].reshape((shape[0],1))
            L[:,N8+2*NS+9]  = tau_columns[:,9].reshape((shape[0],1))
            L[:,N8+2*NS+10] = tau_columns[:,10].reshape((shape[0],1))
            L[:,N8+2*NS+11] = tau_columns[:,11].reshape((shape[0],1))
           
    L.eliminate_zeros()
    subproblem.L_min = subproblem.left_perm @ L
    if problem.STORE_EXPANDED_MATRICES:
        subproblem.expand_matrices(['M','L'])



# Analysis Setup
vol_averager       = VolumeAverager(bB, d, pB, dealias=dealias, radius=1)
radial_averager    = PhiThetaAverager(bB, d, dealias=dealias)
azimuthal_averager = PhiAverager(bB, d, dealias=dealias)
equator_slicer     = EquatorSlicer(bB, d, dealias=dealias)

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
        self.ops['u·u'] = dot(uB, uB)
        self.fields = OrderedDict()

    def evaluate_tasks(self):
        for f in [s1B, uB, ρB, TB]:
            f.require_scales(dealias)
        for k, op in self.ops.items():
            f = op.evaluate()
            f.require_scales(dealias)
            self.fields[k] = f['g']
        #KE & Reynolds
        self.tasks['TE']       = vol_averager(ρB['g']*TB['g']*s1B['g'])
        self.tasks['s1']       = vol_averager(s1B['g'])
        self.tasks['KE']       = vol_averager(ρB['g']*self.fields['u·u']/2)
        self.tasks['Re_rms']   = Re*vol_rms_scalar(self.fields['u·u'], squared=True)
        self.tasks['Re_avg']   = Re*vol_avgmag_scalar(self.fields['u·u'], squared=True)

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
            self.tasks[k] = np.zeros_like(radial_averager.global_profile)

    def evaluate_tasks(self):
        for k, op in self.ops.items():
            f = op.evaluate()
            f.require_scales(dealias)
            self.fields[k] = f['g']

        for f in [s1B, uB, ρB, TB]:
            f.require_scales(dealias)
        self.tasks['s1'][:] = radial_averager(s1B['g'])[:]
        self.tasks['uφ'][:] = radial_averager(uB['g'][0])[:]
        self.tasks['uθ'][:] = radial_averager(uB['g'][1])[:]
        self.tasks['ρ_ur'][:] = radial_averager(ρB['g']*uB['g'][2])[:]

        self.tasks['N2_term'][:] = radial_averager(ρB['g']*uB['g'][2]*TB['g']*grad_s0_RHSB['g'][2])

        #Get fluxes for energy output
        self.tasks['enth_flux'][:] = radial_averager(ρB['g']*self.fields['ur']*(pB['g']))
        self.tasks['visc_flux'][:] = radial_averager(-ρB['g']*(self.fields['u·σ_r'])/Re)
        self.tasks['cond_flux'][:] = radial_averager(-ρB['g']*TB['g']*self.fields['grad_s']/Pe)
        self.tasks['KE_flux'][:]   = radial_averager(0.5*ρB['g']*self.fields['ur']*self.fields['u·u'])

class AnelasticMSW(MeridionalSliceWriter):
    
    def evaluate_tasks(self):
        for f in [s1B, uB]:
            f.require_scales(dealias)
        self.tasks['s1']  = azimuthal_averager(s1B['g'],  comm=True)
        self.tasks['uφ'] = azimuthal_averager(uB['g'][0], comm=True)
        self.tasks['uθ'] = azimuthal_averager(uB['g'][1], comm=True)
        self.tasks['ur'] = azimuthal_averager(uB['g'][2], comm=True)

class AnelasticESW(EquatorialSliceWriter):

    def evaluate_tasks(self):
        for f in [s1B, uB]:
            f.require_scales(dealias)
        self.tasks['s1']  = equator_slicer(s1B['g'])
        self.tasks['uφ'] = equator_slicer(uB['g'][0])
        self.tasks['uθ'] = equator_slicer(uB['g'][1])
        self.tasks['ur'] = equator_slicer(uB['g'][2])

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
esliceWriter  = AnelasticESW(bB, d, out_dir, write_dt=0.5*t_buoy, max_writes=40, dealias=dealias)
sshellWriter  = AnelasticSSW(bB, d, out_dir, write_dt=0.5*t_buoy, max_writes=40, dealias=dealias)
writers = [scalarWriter, esliceWriter, profileWriter, msliceWriter, sshellWriter]

ball_checkpoint = solver.evaluator.add_file_handler('{:s}/ball_checkpoint'.format(out_dir), max_writes=1, sim_dt=50*t_buoy)
ball_checkpoint.add_task(s1B, name='s1', scales=1, layout='c')
ball_checkpoint.add_task(uB, name='u', scales=1, layout='c')


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

CFLB = BallCFL(d, rB, Lmax, max_dt, safety=float(args['--safety']), threshold=0.1, cadence=1)
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
    dt = CFLB.calculate_dt(u, dt)
else:
    # Initial conditions
    A0   = float(1e-6)
    seed = 42 + d.comm_cart.rank
    rand = np.random.RandomState(seed=seed)
    filter_scale = 0.25

    # Generate noise & filter it
    s1B['g'] = A0*rand.standard_normal(s1B['g'].shape)
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

        if solver.iteration % imaginary_cadence in timestepper_history:
            for f in solver.state:
                f.require_grid_space()
except:
    logger.info('something went wrong in main loop, making final checkpoint')
    raise
finally:
    fcheckpoint = solver.evaluator.add_file_handler('{:s}/final_checkpoint'.format(out_dir), max_writes=1, iter=1)
    fcheckpoint.add_task(s1B, name='s1', scales=1, layout='c')
    fcheckpoint.add_task(uB, name='u', scales=1, layout='c')
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
