"""
d3 script for anelastic convection in a massive star's core CZ.

A config file can be provided which overwrites any number of the options below, but command line flags can also be used if preferred.

Usage:
    ballShell_AN.py [options]
    ballShell_AN.py <config> [options]

Options:
    --Re=<Re>            The Reynolds number of the numerical diffusivities [default: 5e1]
    --Pr=<Prandtl>       The Prandtl number  of the numerical diffusivities [default: 1]
    --L=<Lmax>           The value of Lmax   [default: 6]
    --NB=<Nmax>          The ball value of Nmax   [default: 23]
    --NS=<Nmax>          The shell value of Nmax   [default: 7]

    --wall_hours=<t>     The number of hours to run for [default: 24]
    --buoy_end_time=<t>  Number of buoyancy times to run [default: 1e5]
    --safety=<s>         Timestep CFL safety factor [default: 0.4]

    --mesh=<n,m>         The processor mesh over which to distribute the cores
    --A0=<A>             Amplitude of initial noise [default: 1e-6]

    --label=<label>      A label to add to the end of the output directory

    --SBDF2              Use SBDF2 (default)
    --RK222              Use RK222
    --SBDF4              Use SBDF4

    --mesa_file=<f>      path to a .h5 file of ICCs, curated from a MESA model
    --restart=<chk_f>    path to a checkpoint file to restart from

    --benchmark          If flagged, do a simple benchmark problem for comparison with the ball-shell

    --boost=<b>          Inverse Mach number boost squared [default: 1]
    --grad_s_rhs         Move grad_s0 term of energy eqn to RHS
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

from d3_outputs.averaging    import BallVolumeAverager, BallShellVolumeAverager, EquatorSlicer, PhiAverager, PhiThetaAverager, SphericalShellCommunicator
from d3_outputs.writing      import HandlerWriter, BallShellHandlerWriter

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
elif args['--RK222']:
    ts = timesteppers.RK222
    timestepper_history = [0, ]
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
    r_inner = 1.1
    r_outer = 1.5
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
curl      = lambda A: operators.Curl(A)
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
tSt_bot = field.Field(dist=d, bases=(b_mid,), dtype=dtype, tensorsig=(c,))
tS_bot = field.Field(dist=d, bases=(b_midS,), dtype=dtype)
tS_top = field.Field(dist=d, bases=(b_top,), dtype=dtype)

ρB   = field.Field(dist=d, bases=(bB,), dtype=dtype)
TB   = field.Field(dist=d, bases=(bB,), dtype=dtype)
ρS   = field.Field(dist=d, bases=(bS,), dtype=dtype)
TS   = field.Field(dist=d, bases=(bS,), dtype=dtype)

#nccs
grad_ln_ρB    = field.Field(dist=d, bases=(bB.radial_basis,), tensorsig=(c,), dtype=dtype)
grad_ln_TB    = field.Field(dist=d, bases=(bB.radial_basis,), tensorsig=(c,), dtype=dtype)
ln_ρB         = field.Field(dist=d, bases=(bB.radial_basis,), dtype=dtype)
ln_TB         = field.Field(dist=d, bases=(bB.radial_basis,), dtype=dtype)
T_NCCB        = field.Field(dist=d, bases=(bB.radial_basis,), dtype=dtype)
ρ_NCCB        = field.Field(dist=d, bases=(bB.radial_basis,), dtype=dtype)
inv_PeB   = field.Field(dist=d, bases=(bB.radial_basis,), dtype=dtype)
inv_TB        = field.Field(dist=d, bases=(bB,), dtype=dtype) #only on RHS, multiplies other terms
H_effB        = field.Field(dist=d, bases=(bB,), dtype=dtype)
grad_ln_ρS    = field.Field(dist=d, bases=(bS.radial_basis,), tensorsig=(c,), dtype=dtype)
grad_ln_TS    = field.Field(dist=d, bases=(bS.radial_basis,), tensorsig=(c,), dtype=dtype)
ln_ρS         = field.Field(dist=d, bases=(bS.radial_basis,), dtype=dtype)
ln_TS         = field.Field(dist=d, bases=(bS.radial_basis,), dtype=dtype)
T_NCCS        = field.Field(dist=d, bases=(bS.radial_basis,), dtype=dtype)
ρ_NCCS        = field.Field(dist=d, bases=(bS.radial_basis,), dtype=dtype)
inv_PeS   = field.Field(dist=d, bases=(bS.radial_basis,), dtype=dtype)
inv_TS        = field.Field(dist=d, bases=(bS,), dtype=dtype) #only on RHS, multiplies other terms
H_effS        = field.Field(dist=d, bases=(bS,), dtype=dtype)

if args['--grad_s_rhs']:
    grad_s0B      = field.Field(dist=d, bases=(bB,), tensorsig=(c,), dtype=dtype)
    grad_s0S      = field.Field(dist=d, bases=(bS,), tensorsig=(c,), dtype=dtype)
else:
    grad_s0B      = field.Field(dist=d, bases=(bB.radial_basis,), tensorsig=(c,), dtype=dtype)
    grad_s0S      = field.Field(dist=d, bases=(bS.radial_basis,), tensorsig=(c,), dtype=dtype)
    

# Get local slices
slicesB     = d.layouts[-1].slices(s1B.domain, 1)
slicesS     = d.layouts[-1].slices(s1S.domain, 1)
slicesB = [slice(slicesB[0][:,0,0][0], slicesB[0][:,0,0][-1]+1, 1), slice(slicesB[1][0,:,0][0], slicesB[1][0,:,0][-1]+1, 1), slice(slicesB[2][0,0,:][0], slicesB[2][0,0,:][-1]+1, 1)]
slicesS = [slice(slicesS[0][:,0,0][0], slicesS[0][:,0,0][-1]+1, 1), slice(slicesS[1][0,:,0][0], slicesS[1][0,:,0][-1]+1, 1), slice(slicesS[2][0,0,:][0], slicesS[2][0,0,:][-1]+1, 1)]

erB  = field.Field(dist=d, bases=(bB,), tensorsig=(c,), dtype=dtype)
erB['g'][2] = 1
erS  = field.Field(dist=d, bases=(bS,), tensorsig=(c,), dtype=dtype)
erS['g'][2] = 1

grads0_boost = float(args['--boost'])#1/100
logger.info("Boost: {}".format(grads0_boost))

if args['--mesa_file'] is not None:
    with h5py.File(args['--mesa_file'], 'r') as f:
        if np.prod(grad_s0B['g'].shape) > 0:
            grad_s0B['g']        = np.expand_dims(np.expand_dims(np.expand_dims(f['grad_s0B'][:,:,:,slicesB[-1]].reshape(grad_s0B['g'].shape), axis=0), axis=0), axis=0)
            grad_ln_ρB['g']      = f['grad_ln_ρB'][:,:,:,slicesB[-1]].reshape(grad_s0B['g'].shape)
            grad_ln_TB['g']      = f['grad_ln_TB'][:,:,:,slicesB[-1]].reshape(grad_s0B['g'].shape)
        if np.prod(grad_s0S['g'].shape) > 0:
            grad_s0S['g']        = np.expand_dims(np.expand_dims(np.expand_dims(f['grad_s0S'][:,:,:,slicesS[-1]].reshape(grad_s0S['g'].shape), axis=0), axis=0), axis=0)
            grad_ln_ρS['g']      = f['grad_ln_ρS'][:,:,:,slicesS[-1]].reshape(grad_s0S['g'].shape)
            grad_ln_TS['g']      = f['grad_ln_TS'][:,:,:,slicesS[-1]].reshape(grad_s0S['g'].shape)
        inv_PeB['g']= f['inv_Pe_radB'][:,:,slicesB[-1]]
        ln_ρB['g']      = f['ln_ρB'][:,:,slicesB[-1]]
        ln_TB['g']      = f['ln_TB'][:,:,slicesB[-1]]
        H_effB['g']     = f['H_effB'][:,:,slicesB[-1]]
        T_NCCB['g']     = f['TB'][:,:,slicesB[-1]]
        ρB['g']         = np.expand_dims(np.expand_dims(np.exp(f['ln_ρB'][:,:,slicesB[-1]]), axis=0), axis=0)
        TB['g']         = np.expand_dims(np.expand_dims(f['TB'][:,:,slicesB[-1]], axis=0), axis=0)
        inv_TB['g']     = 1/TB['g']

        inv_PeS['g']= f['inv_Pe_radS'][:,:,slicesS[-1]]
        ln_ρS['g']      = f['ln_ρS'][:,:,slicesS[-1]]
        ln_TS['g']      = f['ln_TS'][:,:,slicesS[-1]]
        H_effS['g']     = f['H_effS'][:,:,slicesS[-1]]
        T_NCCS['g']     = f['TS'][:,:,slicesS[-1]]
        ρS['g']         = np.expand_dims(np.expand_dims(np.exp(f['ln_ρS'][:,:,slicesS[-1]]), axis=0), axis=0)
        TS['g']         = np.expand_dims(np.expand_dims(f['TS'][:,:,slicesS[-1]], axis=0), axis=0)
        inv_TS['g']     = 1/TS['g']


        grad_s0B['g'] *= grads0_boost
        grad_s0S['g'] *= grads0_boost

        max_dt = f['max_dt'][()] / np.sqrt(grads0_boost)
        t_buoy = 1
else:
    logger.info("Using polytropic initial conditions")
    from scipy.interpolate import interp1d
    with h5py.File('polytropes/poly_nOuter1.6.h5', 'r') as f:
        T_func = interp1d(f['r'][()], f['T'][()])
        ρ_func = interp1d(f['r'][()], f['ρ'][()])
        grad_s0_func = interp1d(f['r'][()], f['grad_s0'][()])
        H_eff_func   = interp1d(f['r'][()], f['H_eff'][()])
    max_grad_s0 = grad_s0_func(r_outer)
    max_dt = 2/np.sqrt(max_grad_s0)
    t_buoy      = 1


    for basis_r, basis_fields in zip((rB, rS), ((TB, T_NCCB, ρB, ρ_NCCB, inv_TB, ln_TB, ln_ρB, grad_ln_TB, grad_ln_ρB, H_effB, grad_s0B), (TS, T_NCCS, ρS, ρ_NCCS, inv_TS, ln_TS, ln_ρS, grad_ln_TS, grad_ln_ρS, H_effS, grad_s0S))):
        T, T_NCC, ρ, ρ_NCC, inv_T, ln_T, ln_ρ, grad_ln_T, grad_ln_ρ, H_eff, grad_s0 = basis_fields

        grad_s0['g'][2]     = grad_s0_func(basis_r)#*zero_to_one(r, 0.5, width=0.1)
        T['g']           = T_func(basis_r)
        T_NCC['g']       = T_func(basis_r)
        ρ['g']           = ρ_func(basis_r)
        ρ_NCC['g']       = ρ_func(basis_r)
        inv_T['g']       = T_func(basis_r)
        H_eff['g']       = H_eff_func(basis_r)
        ln_T['g']        = np.log(T_func(basis_r))
        ln_ρ['g']        = np.log(ρ_func(basis_r))
        grad_ln_ρ['g']        = grad(ln_ρ).evaluate()['g']
        grad_ln_T['g']        = grad(ln_T).evaluate()['g']

inv_PeB['g'] += 1/Pe
inv_PeS['g'] += 1/Pe


logger.info('buoyancy time is {}'.format(t_buoy))
t_end = float(args['--buoy_end_time'])*t_buoy

# Stress matrices & viscous terms
I_matrixB = field.Field(dist=d, bases=(bB.radial_basis,), tensorsig=(c,c,), dtype=dtype)
I_matrixB['g'] = 0
I_matrixS = field.Field(dist=d, bases=(bS.radial_basis,), tensorsig=(c,c,), dtype=dtype)
I_matrixS['g'] = 0
for i in range(3):
    I_matrixB['g'][i,i,:] = 1
    I_matrixS['g'][i,i,:] = 1

#Ball stress
EB = 0.5*(grad(uB) + transpose(grad(uB)))
EB.store_last = True
divUB = div(uB)
divUB.store_last = True
σB = 2*(EB - (1/3)*divUB*I_matrixB)
momentum_viscous_termsB = div(σB) + dot(σB, grad_ln_ρB)

VHB  = 2*(trace(dot(EB, EB)) - (1/3)*divUB*divUB)

#Shell stress
ES = 0.5*(grad(uS) + transpose(grad(uS)))
ES.store_last = True
divUS = div(uS)
divUS.store_last = True
σS = 2*(ES - (1/3)*divUS*I_matrixS)
momentum_viscous_termsS = div(σS) + dot(σS, grad_ln_ρS)

VHS  = 2*(trace(dot(ES, ES)) - (1/3)*divUS*divUS)


#Impenetrable, stress-free boundary conditions
u_r_bcB_mid    = pB(r=r_inner)
u_r_bcS_mid    = pS(r=r_inner)
u_perp_bcB_mid = angComp(radComp(σB(r=r_inner)), index=0)
u_perp_bcS_mid = angComp(radComp(σS(r=r_inner)), index=0)
uS_r_bc        = radComp(uS(r=r_outer))
u_perp_bcS_top = radComp(angComp(ES(r=r_outer), index=1))

#Initial conditions
if args['--restart'] is not None:
    fname = args['--restart']
    fdir = fname.split('.h5')[0]
    check_name = fdir.split('/')[-1]
    #Try to just load the loal piece file

    import h5py
    with h5py.File('{}/{}_p{}.h5'.format(fdir, check_name, d.comm_cart.rank), 'r') as f:
        s1B.set_scales(1)
        pB.set_scales(1)
        uB.set_scales(1)
        s1S.set_scales(1)
        pS.set_scales(1)
        uS.set_scales(1)
        s1B['c'] = f['tasks/s1B'][()][-1,:]
        pB['c'] = f['tasks/pB'][()][-1,:]
        uB['c'] = f['tasks/uB'][()][-1,:]
        s1S['c'] = f['tasks/s1S'][()][-1,:]
        pS['c'] = f['tasks/pS'][()][-1,:]
        uS['c'] = f['tasks/uS'][()][-1,:]

        s1B.require_scales(dealias)
        pB.require_scales(dealias)
        uB.require_scales(dealias)
        s1S.require_scales(dealias)
        pS.require_scales(dealias)
        uS.require_scales(dealias)
else:
    if args['--benchmark']:
        #Marti benchmark-like ICs
        A0 = 1e-3
        s1B['g'] = A0*np.sqrt(35/np.pi)*(rB/r_outer)**3*(1-(rB/r_outer)**2)*(np.cos(φB)+np.sin(φB))*np.sin(θB)**3
        s1S['g'] = A0*np.sqrt(35/np.pi)*(rS/r_outer)**3*(1-(rS/r_outer)**2)*(np.cos(φS)+np.sin(φS))*np.sin(θS)**3
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



H_effB = operators.Grid(H_effB).evaluate()
H_effS = operators.Grid(H_effS).evaluate()


# Problem
def eq_eval(eq_str):
    return [eval(expr) for expr in split_equation(eq_str)]
#Only velocity
problem = problems.IVP([pB, uB, pS, uS, s1B, s1S, tBt, tSt_bot, tSt_top, tB, tS_bot, tS_top])

### Ball momentum
problem.add_equation(eq_eval("div(uB) + dot(uB, grad_ln_ρB) = 0"), condition="nθ != 0")
#problem.add_equation(eq_eval("ddt(uB) + grad(pB) - T_NCCB*grad(s1B) - (1/Re)*momentum_viscous_termsB   = - dot(uB, grad(uB))"), condition = "nθ != 0")
problem.add_equation(eq_eval("ddt(uB) + grad(pB) + grad(T_NCCB)*s1B - (1/Re)*momentum_viscous_termsB  = cross(uB, curl(uB))"), condition = "nθ != 0")
### Shell momentum
problem.add_equation(eq_eval("div(uS) + dot(uS, grad_ln_ρS) = 0"), condition="nθ != 0")
#problem.add_equation(eq_eval("ddt(uS) + grad(pS) - T_NCCS*grad(s1S) - (1/Re)*momentum_viscous_termsS   = - dot(uS, grad(uS))"), condition = "nθ != 0")
problem.add_equation(eq_eval("ddt(uS) + grad(pS) + grad(T_NCCS)*s1S - (1/Re)*momentum_viscous_termsS = cross(uS, curl(uS))"), condition = "nθ != 0")
## ell == 0 momentum
problem.add_equation(eq_eval("pB = 0"), condition="nθ == 0")
problem.add_equation(eq_eval("uB = 0"), condition="nθ == 0")
problem.add_equation(eq_eval("pS = 0"), condition="nθ == 0")
problem.add_equation(eq_eval("uS = 0"), condition="nθ == 0")

### Ball energy
problem.add_equation(eq_eval("ddt(s1B) + dot(uB, grad_s0B) - (inv_PeB)*(lap(s1B) + dot(grad(s1B), (grad_ln_ρB + grad_ln_TB))) - dot(grad(s1B), grad(inv_PeB)) = - dot(uB, grad(s1B)) + H_effB + (1/Re)*inv_TB*VHB "))
### Shell energy
problem.add_equation(eq_eval("ddt(s1S) + dot(uS, grad_s0S) - (inv_PeS)*(lap(s1S) + dot(grad(s1S), (grad_ln_ρS + grad_ln_TS))) - dot(grad(s1S), grad(inv_PeS)) = - dot(uS, grad(s1S)) + H_effS + (1/Re)*inv_TS*VHS "))


#Velocity BCs ell != 0
problem.add_equation(eq_eval("uB(r=r_inner) - uS(r=r_inner)    = 0"),            condition="nθ != 0")
problem.add_equation(eq_eval("u_r_bcB_mid - u_r_bcS_mid    = 0"),            condition="nθ != 0")
#problem.add_equation(eq_eval("radComp(grad(uB)(r=r_inner) - grad(uS)(r=r_inner)) = 0"), condition="nθ != 0")
problem.add_equation(eq_eval("u_perp_bcB_mid - u_perp_bcS_mid = 0"), condition="nθ != 0")
problem.add_equation(eq_eval("uS_r_bc    = 0"),                      condition="nθ != 0")
problem.add_equation(eq_eval("u_perp_bcS_top    = 0"),               condition="nθ != 0")
# velocity BCs ell == 0
problem.add_equation(eq_eval("tBt     = 0"),                         condition="nθ == 0")
problem.add_equation(eq_eval("tSt_bot     = 0"), condition="nθ == 0")
problem.add_equation(eq_eval("tSt_top     = 0"), condition="nθ == 0")

#Entropy BCs
problem.add_equation(eq_eval("s1B(r=r_inner) - s1S(r=r_inner) = 0"))
problem.add_equation(eq_eval("radComp(grad(s1B)(r=r_inner)) - radComp(grad(s1S)(r=r_inner))    = 0"))
problem.add_equation(eq_eval("radComp(grad(s1S)(r=r_outer))    = 0"))


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


## Analysis Setup
scalar_dt = 0.25*t_buoy
lum_dt   = 0.5*t_buoy
visual_dt = 0.05*t_buoy
outer_shell_dt = max_dt
onesB = field.Field(dist=d, bases=(bB,), dtype=dtype)
onesS = field.Field(dist=d, bases=(bS,), dtype=dtype)
r_valsB = field.Field(dist=d, bases=(bB,), dtype=dtype)
r_valsS = field.Field(dist=d, bases=(bS,), dtype=dtype)
eφB = field.Field(dist=d, bases=(bB,), tensorsig=(c,), dtype=dtype)
eθB = field.Field(dist=d, bases=(bB,), tensorsig=(c,), dtype=dtype)
eφB['g'][0] = 1
eθB['g'][1] = 1
eφS = field.Field(dist=d, bases=(bS,), tensorsig=(c,), dtype=dtype)
eθS = field.Field(dist=d, bases=(bS,), tensorsig=(c,), dtype=dtype)
eφS['g'][0] = 1
eθS['g'][1] = 1
onesB['g']  = 1
onesS['g']  = 1
r_valsB['g'] = rB
r_valsS['g'] = rS
eφB = operators.Grid(eφB).evaluate()
eθB = operators.Grid(eθB).evaluate()
erB = operators.Grid(erB).evaluate()
eφS = operators.Grid(eφS).evaluate()
eθS = operators.Grid(eθS).evaluate()
erS = operators.Grid(erS).evaluate()
onesB = operators.Grid(onesB).evaluate()
onesS = operators.Grid(onesS).evaluate()
r_valsB = operators.Grid(r_valsB).evaluate()
r_valsS = operators.Grid(r_valsS).evaluate()
uφB = dot(uB, eφB)
uθB = dot(uB, eθB)
urB = dot(erB, uB)
uφS = dot(uS, eφS)
uθS = dot(uS, eθS)
urS = dot(erS, uS)
hB = pB - 0.5*dot(uB,uB) + TB*s1B
hS = pS - 0.5*dot(uS,uS) + TS*s1S
pomega_hat_B = pB - 0.5*dot(uB,uB)
pomega_hat_S = pS - 0.5*dot(uS,uS)

I_matrixB_post = field.Field(dist=d, bases=(bB,), tensorsig=(c,c,), dtype=dtype)
I_matrixB_post['g'] = 0
I_matrixS_post = field.Field(dist=d, bases=(bS,), tensorsig=(c,c,), dtype=dtype)
I_matrixS_post['g'] = 0
for i in range(3):
    I_matrixB_post['g'][i,i,:] = 1
    I_matrixS_post['g'][i,i,:] = 1
σ_postB = 2*(EB - (1/3)*divUB*I_matrixB_post)
σ_postS = 2*(ES - (1/3)*divUS*I_matrixS_post)

visc_fluxB_r = 2*(dot(erB, dot(uB, EB)) - (1/3) * urB * divUB)
visc_fluxS_r = 2*(dot(erS, dot(uS, ES)) - (1/3) * urS * divUS)



visualsB = solver.evaluator.add_dictionary_handler(sim_dt=visual_dt)
visualsB.add_task(uφB, name='uφB', layout='g')
visualsB.add_task(uθB, name='uθB', layout='g')
visualsB.add_task(urB, name='urB', layout='g')
visualsB.add_task(onesB*s1B,     name='s1B',  layout='g')
visualsS = solver.evaluator.add_dictionary_handler(sim_dt=visual_dt)
visualsS.add_task(uφS, name='uφS', layout='g')
visualsS.add_task(uθS, name='uθS', layout='g')
visualsS.add_task(urS, name='urS', layout='g')
visualsS.add_task(onesS*s1S,     name='s1S',  layout='g')

re_ball = solver.evaluator.add_dictionary_handler(iter=10)
re_ball.add_task(Re*(dot(uB,uB))**(1/2), name='Re_avg_ball', layout='g')
scalars = solver.evaluator.add_dictionary_handler(sim_dt=scalar_dt)#, iter=10)
scalars.add_task(Re*(dot(uB,uB))**(1/2), name='Re_avg_ball', layout='g')
scalars.add_task(Re*(dot(uS,uS))**(1/2), name='Re_avg_shell', layout='g')
scalars.add_task(ρB*dot(uB, uB)/2,    name='KE_ball', layout='g')
scalars.add_task(ρS*dot(uS, uS)/2,    name='KE_shell', layout='g')
scalars.add_task(ρB*TB*s1B,           name='TE_ball', layout='g')
scalars.add_task(ρS*TS*s1S,           name='TE_shell', layout='g')

#Radial averages (lums)

ball_lums = solver.evaluator.add_dictionary_handler(sim_dt=lum_dt)
shell_lums = solver.evaluator.add_dictionary_handler(sim_dt=lum_dt)
ball_lums.add_task((4*np.pi*r_valsB**2)*(ρB*urB*pomega_hat_B), name='wave_lumB', layout='g')
ball_lums.add_task((4*np.pi*r_valsB**2)*(ρB*urB*hB), name='enth_lumB', layout='g')
ball_lums.add_task((4*np.pi*r_valsB**2)*(-ρB*visc_fluxB_r/Re), name='visc_lumB', layout='g')
ball_lums.add_task((4*np.pi*r_valsB**2)*(-ρB*TB*dot(erB, grad(s1B))/Pe), name='cond_lumB', layout='g')
ball_lums.add_task((4*np.pi*r_valsB**2)*(0.5*ρB*urB*dot(uB, uB)), name='KE_lumB', layout='g')
shell_lums.add_task((4*np.pi*r_valsS**2)*(ρS*urS*pomega_hat_S), name='wave_lumS', layout='g')
shell_lums.add_task((4*np.pi*r_valsS**2)*(ρS*urS*hS), name='enth_lumS', layout='g')
shell_lums.add_task((4*np.pi*r_valsS**2)*(-ρS*visc_fluxS_r/Re), name='visc_lumS', layout='g')
shell_lums.add_task((4*np.pi*r_valsS**2)*(-ρS*TS*dot(erS, grad(s1S))/Pe), name='cond_lumS', layout='g')
shell_lums.add_task((4*np.pi*r_valsS**2)*(0.5*ρS*urS*dot(uS, uS)), name='KE_lumS', layout='g')

shell_visualsB = solver.evaluator.add_dictionary_handler(sim_dt=visual_dt)
shell_visualsB.add_task(uφB(r=0.5),         name='uφB_r0.5', layout='g')
shell_visualsB.add_task(uθB(r=0.5),         name='uθB_r0.5', layout='g')              
shell_visualsB.add_task(urB(r=0.5),         name='urB_r0.5', layout='g')                      
shell_visualsB.add_task(s1B(r=0.5), name='s1B_r0.5',  layout='g')
shell_visualsB.add_task(uφB(r=1),         name='uφB_r1', layout='g')
shell_visualsB.add_task(uθB(r=1),         name='uθB_r1', layout='g')              
shell_visualsB.add_task(urB(r=1),         name='urB_r1', layout='g')                      
shell_visualsB.add_task(s1B(r=1), name='s1B_r1',  layout='g')
shell_visualsS = solver.evaluator.add_dictionary_handler(sim_dt=visual_dt)
shell_visualsS.add_task(uφS(r=0.95*r_outer), name='uφS_near_surf', layout='g')              
shell_visualsS.add_task(uθS(r=0.95*r_outer), name='uθS_near_surf', layout='g')              
shell_visualsS.add_task(urS(r=0.95*r_outer), name='urS_near_surf', layout='g')                      
shell_visualsS.add_task(s1S(r=0.95*r_outer),     name='s1S_near_surf',  layout='g')           

surface_shells = solver.evaluator.add_dictionary_handler(sim_dt=max_dt)
surface_shells.add_task(uφS(r=r_outer), name='u_phi_surf', layout='g')
surface_shells.add_task(uθS(r=r_outer), name='u_theta_surf', layout='g')
surface_shells.add_task(s1S(r=r_outer), name='s1_surf', layout='g')

vol_averagerB      = BallVolumeAverager(pB)
vol_averager       = BallShellVolumeAverager(pB, pS)
radial_averagerB    = PhiThetaAverager(pB)
radial_averagerS    = PhiThetaAverager(pS)
equator_slicerB     = EquatorSlicer(pB)
equator_slicerS     = EquatorSlicer(pS)
shell_commB         = SphericalShellCommunicator(pB)
shell_commS         = SphericalShellCommunicator(pS)

scalarWriter  = BallShellHandlerWriter(scalars, vol_averager,  out_dir, 'scalar')    
profileWriterB = HandlerWriter(ball_lums, radial_averagerB,    out_dir, 'profilesB', max_writes=100)
profileWriterS = HandlerWriter(shell_lums, radial_averagerS,    out_dir, 'profilesS', max_writes=100)
esliceWriterB  = HandlerWriter(visualsB, equator_slicerB,     out_dir, 'eq_sliceB', max_writes=40)  
esliceWriterS  = HandlerWriter(visualsS, equator_slicerS,     out_dir, 'eq_sliceS', max_writes=40)  
sshellWriterB  = HandlerWriter(shell_visualsB, shell_commB,          out_dir, 'shell_sliceB', max_writes=100)
sshellWriterS  = HandlerWriter(shell_visualsS, shell_commS,          out_dir, 'shell_sliceS', max_writes=100)
surface_shellWriter = HandlerWriter(surface_shells, shell_commS, out_dir, 'surface_shells', max_writes=100)
writers = [scalarWriter, profileWriterB, profileWriterS, esliceWriterB, esliceWriterS, sshellWriterB, sshellWriterS, surface_shellWriter]

ball_checkpoint = solver.evaluator.add_file_handler('{:s}/ball_checkpoint'.format(out_dir), max_writes=1, sim_dt=10*t_buoy)
ball_checkpoint.add_task(s1B, name='s1B', scales=1, layout='c')
ball_checkpoint.add_task(pB, name='pB', scales=1, layout='c')
ball_checkpoint.add_task(uB, name='uB', scales=1, layout='c')
ball_checkpoint.add_task(s1S, name='s1S', scales=1, layout='c')
ball_checkpoint.add_task(pS, name='pS', scales=1, layout='c')
ball_checkpoint.add_task(uS, name='uS', scales=1, layout='c')

imaginary_cadence = 100

#CFL setup
from dedalus.extras.flow_tools import CFL

my_cfl = CFL(solver, max_dt, safety=float(args['--safety']), cadence=1, max_dt=max_dt, min_change=0.1, max_change=1.5, threshold=0.1)
my_cfl.add_velocity(uB)

dt = max_dt

# Main loop
start_time = time.time()
start_iter = solver.iteration
try:
    while solver.ok:
        solver.step(dt)
        for writer in writers:
            writer.process(solver, dt)
        dt = my_cfl.compute_dt()
        int_frac = int(np.ceil(max_dt/dt))
        dt = max_dt/int_frac

        if solver.iteration % imaginary_cadence in timestepper_history:
            for f in solver.state:
                f.require_grid_space()

        if solver.iteration % 10 == 0:
            KE  = vol_averager.volume*scalarWriter.tasks['KE']
            TE  = vol_averager.volume*scalarWriter.tasks['TE']
            Re0 = vol_averagerB(re_ball.fields['Re_avg_ball']['g'], comm=True)
            logger.info("t = %f, dt = %f, Re = %e, KE / TE = %e / %e" %(solver.sim_time, dt, Re0, KE, TE))

except:
    logger.info('something went wrong in main loop, making final checkpoint')
    raise
finally:
    fcheckpoint = solver.evaluator.add_file_handler('{:s}/final_checkpoint'.format(out_dir), max_writes=1, iter=1)
    fcheckpoint.add_task(s1B, name='s1B', scales=1, layout='c')
    fcheckpoint.add_task(uB, name='uB', scales=1, layout='c')
    fcheckpoint.add_task(pB, name='pB', scales=1, layout='c')
    fcheckpoint.add_task(s1S, name='s1S', scales=1, layout='c')
    fcheckpoint.add_task(uS, name='uS', scales=1, layout='c')
    fcheckpoint.add_task(pS, name='pS', scales=1, layout='c')
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
