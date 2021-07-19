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

    --mesh=<n,m>         The processor mesh over which to distribute the cores

    --SBDF2              Use SBDF2 (default)
    --RK222              Use RK222
    --SBDF4              Use SBDF4
    --safety=<s>         Timestep CFL safety factor [default: 0.35]
    --CFL_max_r=<r>      zero out velocities above this radius value for CFL

    --mesa_file=<f>      path to a .h5 file of ICCs, curated from a MESA model
    --restart=<chk_f>    path to a checkpoint file to restart from
    --benchmark          If flagged, use simple benchmark initial conditions
    --sponge             If flagged, add a "sponge" layer in the shell that damps out waves.

    --boost=<b>          A factor by which to boost grad_s0 [default: 1]
    --label=<label>      A label to add to the end of the output directory
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

from d3_outputs.extra_ops    import BallVolumeAverager, ShellVolumeAverager, EquatorSlicer, PhiAverager, PhiThetaAverager, OutputRadialInterpolate, GridSlicer, MeridionSlicer
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
NmaxB      = int(args['--NB'])
NmaxS      = int(args['--NS'])
N_dealias = 1.5
L_dealias = 1.5

out_dir = './' + sys.argv[0].split('.py')[0]
if args['--sponge']:
    out_dir += '_sponge'
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
ncpu = MPI.COMM_WORLD.size
if mesh is not None:
    mesh = mesh.split(',')
    mesh = [int(mesh[0]), int(mesh[1])]
else:
    log2 = np.log2(ncpu)
    if log2 == int(log2):
        mesh = [int(2**np.ceil(log2/2)),int(2**np.floor(log2/2))]
    logger.info("running on processor mesh={}".format(mesh))

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

dealias_tuple = (L_dealias, L_dealias, N_dealias)

# Bases
c    = coords.SphericalCoordinates('φ', 'θ', 'r')
c_S2 = c.S2coordsys 
d    = distributor.Distributor((c,), mesh=mesh)
bB   = basis.BallBasis(c, (2*(Lmax+2), Lmax+1, NmaxB+1), radius=r_inner, dtype=dtype, dealias=dealias_tuple)
bS   = basis.SphericalShellBasis(c, (2*(Lmax+2), Lmax+1, NmaxS+1), radii=(r_inner, r_outer), dtype=dtype, dealias=dealias_tuple)
b_mid = bB.S2_basis(radius=r_inner)
b_midS = bS.S2_basis(radius=r_inner)
b_top = bS.S2_basis(radius=r_outer)
φB,  θB,  rB  = bB.local_grids(dealias_tuple)
φBg, θBg, rBg = bB.global_grids(dealias_tuple)
φS,  θS,  rS  = bS.local_grids(dealias_tuple)
φSg, θSg, rSg = bS.global_grids(dealias_tuple)

kBtau = 0
kBrhs = 0
kStau = 0
kSrhs = 0
bB_tau = bB._new_k(kBtau)
bB_rhs = bB._new_k(kBrhs)
bS_tau = bS._new_k(kStau)
bS_rhs = bS._new_k(kSrhs)

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
LiftTauB   = lambda A: operators.LiftTau(A, bB_tau, -1)
LiftTauS   = lambda A, n: operators.LiftTau(A, bS_tau, n)
Grid = operators.Coeff
Coeff = operators.Coeff
Conv = operators.Convert
RHSB = lambda A: Coeff(Conv(A, bB_rhs))
RHSS = lambda A: Coeff(Conv(A, bS_rhs))



# Fields
tB     = field.Field(dist=d, bases=(b_mid,), dtype=dtype)
tBt    = field.Field(dist=d, bases=(b_mid,), dtype=dtype,   tensorsig=(c,))
tSt_top = field.Field(dist=d, bases=(b_top,), dtype=dtype,  tensorsig=(c,))
tSt_bot = field.Field(dist=d, bases=(b_mid,), dtype=dtype, tensorsig=(c,))
tS_bot = field.Field(dist=d, bases=(b_midS,), dtype=dtype)
tS_top = field.Field(dist=d, bases=(b_top,), dtype=dtype)

#nccs & fields
scalar_variables = ['p', 's1']
tensor_variables = ['u']
tensor_nccs = ['grad_ln_ρ', 'grad_ln_T', 'grad_T', 'grad_inv_Pe', 'grad_s0']
scalar_nccs = ['ln_ρ', 'ln_T', 'inv_Pe']
rhs_fields  = ['ρ', 'T', 'inv_T', 'H_eff']
tensor_rhs_fields  = ['er']
for label, b in zip(('B', 'S'), (bB, bS)):
    for field_name in scalar_variables + rhs_fields:
        locals()[field_name+label] = field.Field(dist=d, bases=(b,), dtype=dtype)
    for field_name in tensor_variables + tensor_rhs_fields:
        locals()[field_name+label] = field.Field(dist=d, bases=(b,), tensorsig=(c,), dtype=dtype)
    for field_name in scalar_nccs:
        locals()[field_name+label] = field.Field(dist=d, bases=(b.radial_basis,), dtype=dtype)
    for field_name in tensor_nccs:
        locals()[field_name+label] = field.Field(dist=d, bases=(b.radial_basis,), tensorsig=(c,), dtype=dtype)

if args['--sponge']:
    L_shell = r_outer - r_inner
    spongeS = field.Field(dist=d, bases=(bS.radial_basis,), dtype=dtype)
    spongeS.require_scales(dealias_tuple)
    spongeS['g'] = zero_to_one(rS, r_inner + 2*L_shell/3, 0.1*L_shell)
# Get local slices
slicesB     = GridSlicer(pB, dealias=(1,1,1))
slicesS     = GridSlicer(pS, dealias=(1,1,1))

erB['g'][2] = 1
erS['g'][2] = 1

grads0_boost = float(args['--boost'])#1/100
logger.info("Boost: {}".format(grads0_boost))

if args['--mesa_file'] is not None:
    with h5py.File(args['--mesa_file'], 'r') as f:
        if np.prod(grad_s0B['g'].shape) > 0:
            grad_s0B['g']        = np.expand_dims(np.expand_dims(np.expand_dims(f['grad_s0B'][:,:,:,slicesB[-1]].reshape(grad_s0B['g'].shape), axis=0), axis=0), axis=0)
            grad_ln_ρB['g']      = f['grad_ln_ρB'][:,:,:,slicesB[-1]].reshape(grad_s0B['g'].shape)
            grad_ln_TB['g']      = f['grad_ln_TB'][:,:,:,slicesB[-1]].reshape(grad_s0B['g'].shape)
            grad_TB['g']         = f['grad_TB'][:,:,:,slicesB[-1]].reshape(grad_s0B['g'].shape)
            grad_inv_PeB['g']    = f['grad_inv_Pe_radB'][:, :,:,slicesB[-1]].reshape(grad_s0B['g'].shape)
        if np.prod(grad_s0S['g'].shape) > 0:
            grad_s0S['g']        = np.expand_dims(np.expand_dims(np.expand_dims(f['grad_s0S'][:,:,:,slicesS[-1]].reshape(grad_s0S['g'].shape), axis=0), axis=0), axis=0)
            grad_ln_ρS['g']      = f['grad_ln_ρS'][:,:,:,slicesS[-1]].reshape(grad_s0S['g'].shape)
            grad_ln_TS['g']      = f['grad_ln_TS'][:,:,:,slicesS[-1]].reshape(grad_s0S['g'].shape)
            grad_TS['g']         = f['grad_TS'][:,:,:,slicesS[-1]].reshape(grad_s0S['g'].shape)
            grad_inv_PeS['g']    = f['grad_inv_Pe_radS'][:,:,:,slicesS[-1]].reshape(grad_s0S['g'].shape)
        inv_PeB['g']= f['inv_Pe_radB'][:,:,slicesB[-1]]
        ln_ρB['g']      = f['ln_ρB'][:,:,slicesB[-1]]
        ln_TB['g']      = f['ln_TB'][:,:,slicesB[-1]]
        H_effB['g']     = f['H_effB'][:,:,slicesB[-1]]
        ρB['g']         = np.expand_dims(np.expand_dims(np.exp(f['ln_ρB'][:,:,slicesB[-1]]), axis=0), axis=0)
        TB['g']         = np.expand_dims(np.expand_dims(f['TB'][:,:,slicesB[-1]], axis=0), axis=0)
        inv_TB['g']     = 1/TB['g']

        inv_PeS['g']= f['inv_Pe_radS'][:,:,slicesS[-1]]
        ln_ρS['g']      = f['ln_ρS'][:,:,slicesS[-1]]
        ln_TS['g']      = f['ln_TS'][:,:,slicesS[-1]]
        H_effS['g']     = f['H_effS'][:,:,slicesS[-1]]
        ρS['g']         = np.expand_dims(np.expand_dims(np.exp(f['ln_ρS'][:,:,slicesS[-1]]), axis=0), axis=0)
        TS['g']         = np.expand_dims(np.expand_dims(f['TS'][:,:,slicesS[-1]], axis=0), axis=0)
        inv_TS['g']     = 1/TS['g']


        grad_s0B['g'] *= grads0_boost
        grad_s0S['g'] *= grads0_boost

        max_dt = f['max_dt'][()] / np.sqrt(grads0_boost)
        t_buoy = 1

        if args['--sponge']:
            f_brunt = f['tau'][()]*np.sqrt(f['N2max_shell'][()])/(2*np.pi)
            spongeS['g'] *= f_brunt
#            logger.info('sponge layer max_dt {:.3e} vs max_dt {:.3e}'.format(0.25/f_brunt, max_dt))
#            max_dt = np.min((max_dt, 0.25/f_brunt))

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


    for basis_r, basis_fields in zip((rB, rS), ((TB, ρB, inv_TB, ln_TB, ln_ρB, grad_ln_TB, grad_ln_ρB, grad_TB, H_effB, grad_s0B), (TS, ρS, inv_TS, ln_TS, ln_ρS, grad_ln_TS, grad_ln_ρS, grad_TS, H_effS, grad_s0S))):
        for f in basis_fields:
            f.require_scales(dealias_tuple)
        T, ρ, inv_T, ln_T, ln_ρ, grad_ln_T, grad_ln_ρ, grad_T, H_eff, grad_s0 = basis_fields

        grad_s0['g'][2]     = grad_s0_func(basis_r)#*zero_to_one(r, 0.5, width=0.1)
        T['g']           = T_func(basis_r)
        grad_T['g']      = grad(T).evaluate()['g']
        ρ['g']           = ρ_func(basis_r)
        inv_T['g']       = T_func(basis_r)
        H_eff['g']       = H_eff_func(basis_r)
        ln_T['g']        = np.log(T_func(basis_r))
        ln_ρ['g']        = np.log(ρ_func(basis_r))
        grad_ln_ρ['g']        = grad(ln_ρ).evaluate()['g']
        grad_ln_T['g']        = grad(ln_T).evaluate()['g']

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

for f in [s1B, s1S, uB, uS]:
    f.require_scales(dealias_tuple)

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

        s1B.require_scales(dealias_tuple)
        pB.require_scales(dealias_tuple)
        uB.require_scales(dealias_tuple)
        s1S.require_scales(dealias_tuple)
        pS.require_scales(dealias_tuple)
        uS.require_scales(dealias_tuple)
else:
    if args['--benchmark']:
        #Marti benchmark-like ICs
        A0 = 1e-3
        s1B['g'] = A0*np.sqrt(35/np.pi)*(rB/r_outer)**3*(1-(rB/r_outer)**2)*(np.cos(φB)+np.sin(φB))*np.sin(θB)**3
        s1S['g'] = A0*np.sqrt(35/np.pi)*(rS/r_outer)**3*(1-(rS/r_outer)**2)*(np.cos(φS)+np.sin(φS))*np.sin(θS)**3
    else:
        # Initial conditions
        A0   = float(1e-3)
        seed = 42 + d.comm_cart.rank
        rand = np.random.RandomState(seed=seed)
        filter_scale = 0.25

        # Generate noise & filter it
        s1B['g'] = A0*rand.standard_normal(s1B['g'].shape)*np.sin(2*np.pi*rB)
        s1B.require_scales(filter_scale)
        s1B['c']
        s1B['g']
        s1B.require_scales(dealias_tuple)

H_effB = operators.Grid(H_effB).evaluate()
H_effS = operators.Grid(H_effS).evaluate()
inv_TB = operators.Grid(inv_TB).evaluate()
inv_TS = operators.Grid(inv_TS).evaluate()

# Problem
def eq_eval(eq_str):
    return [eval(expr) for expr in split_equation(eq_str)]
#Only velocity
problem = problems.IVP([pB, uB, pS, uS, s1B, s1S, tBt, tSt_bot, tSt_top, tB, tS_bot, tS_top])

### Ball momentum
problem.add_equation(eq_eval("div(uB) + dot(uB, grad_ln_ρB) = 0"), condition="nθ != 0")
problem.add_equation(eq_eval("ddt(uB) + grad(pB) + grad_TB*s1B - (1/Re)*momentum_viscous_termsB + LiftTauB(tBt) = RHSB(cross(uB, curl(uB)))"), condition = "nθ != 0")
### Shell momentum
problem.add_equation(eq_eval("div(uS) + dot(uS, grad_ln_ρS) = 0"), condition="nθ != 0")
if args['--sponge']:
    problem.add_equation(eq_eval("ddt(uS) + grad(pS) + grad_TS*s1S - (1/Re)*momentum_viscous_termsS + spongeS*uS + LiftTauS(tSt_bot, -1) + LiftTauS(tSt_top, -2) = RHSS( cross(uS, curl(uS)) )"), condition = "nθ != 0")
else:
    problem.add_equation(eq_eval("ddt(uS) + grad(pS) + grad_TS*s1S - (1/Re)*momentum_viscous_termsS + LiftTauS(tSt_bot, -1) + LiftTauS(tSt_top, -2) = RHSS( cross(uS, curl(uS)) )"), condition = "nθ != 0")
## ell == 0 momentum
problem.add_equation(eq_eval("pB = 0"), condition="nθ == 0")
problem.add_equation(eq_eval("uB = 0"), condition="nθ == 0")
problem.add_equation(eq_eval("pS = 0"), condition="nθ == 0")
problem.add_equation(eq_eval("uS = 0"), condition="nθ == 0")

### Ball energy
grads1B = grad(s1B)
grads1S = grad(s1S)
grads1B.store_last = True
grads1S.store_last = True
problem.add_equation(eq_eval("ddt(s1B) + dot(uB, grad_s0B) - (inv_PeB)*(lap(s1B) + dot(grads1B, (grad_ln_ρB + grad_ln_TB))) - dot(grads1B, grad_inv_PeB) + LiftTauB(tB) = RHSB( - dot(uB, grads1B) + H_effB + (1/Re)*inv_TB*VHB ) "))
### Shell energy
problem.add_equation(eq_eval("ddt(s1S) + dot(uS, grad_s0S) - (inv_PeS)*(lap(s1S) + dot(grads1S, (grad_ln_ρS + grad_ln_TS))) - dot(grads1S, grad_inv_PeS) + LiftTauS(tS_bot, -1) + LiftTauS(tS_top, -2)  = RHSS( - dot(uS, grads1S) + H_effS + (1/Re)*inv_TS*VHS ) "))


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
problem.add_equation(eq_eval("radComp(grads1B(r=r_inner)) - radComp(grads1S(r=r_inner))    = 0"))
problem.add_equation(eq_eval("radComp(grads1S(r=r_outer))    = 0"))


logger.info("Problem built")
# Solver
solver = solvers.InitialValueSolver(problem, ts)
solver.stop_sim_time = t_end
logger.info("solver built")

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
r_valsB = field.Field(dist=d, bases=(bB,), dtype=dtype)
r_valsS = field.Field(dist=d, bases=(bS,), dtype=dtype)
for f in [r_valsB, r_valsS]:
    f.require_scales(dealias_tuple)
r_valsB['g'] = rB
r_valsS['g'] = rS
erB = operators.Grid(erB).evaluate()
erS = operators.Grid(erS).evaluate()
r_valsB = operators.Grid(r_valsB).evaluate()
r_valsS = operators.Grid(r_valsS).evaluate()
urB = dot(erB, uB)
urS = dot(erS, uS)
hB = pB - 0.5*dot(uB,uB) + TB*s1B
hS = pS - 0.5*dot(uS,uS) + TS*s1S
pomega_hat_B = pB - 0.5*dot(uB,uB)
pomega_hat_S = pS - 0.5*dot(uS,uS)

visc_fluxB_r = 2*(dot(erB, dot(uB, EB)) - (1/3) * urB * divUB)
visc_fluxS_r = 2*(dot(erS, dot(uS, ES)) - (1/3) * urS * divUS)

# Angular momentum, etc. 
ezB = field.Field(dist=d, bases=(bB,), tensorsig=(c,), dtype=dtype)
ezB.set_scales(dealias_tuple)
ezB['g'][1] = -np.sin(θB)
ezB['g'][2] =  np.cos(θB)
ezB = operators.Grid(ezB).evaluate()

exB = field.Field(dist=d, bases=(bB,), tensorsig=(c,), dtype=dtype)
exB.set_scales(dealias_tuple)
exB['g'][0] = -np.sin(φB)
exB['g'][1] = np.cos(θB)*np.cos(φB)
exB['g'][2] = np.sin(θB)*np.cos(φB)
exB = operators.Grid(exB).evaluate()

eyB = field.Field(dist=d, bases=(bB,), tensorsig=(c,), dtype=dtype)
eyB.set_scales(dealias_tuple)
eyB['g'][0] = np.cos(φB)
eyB['g'][1] = np.cos(θB)*np.sin(φB)
eyB['g'][2] = np.sin(θB)*np.sin(φB)
eyB = operators.Grid(eyB).evaluate()

rB_vec_post = field.Field(dist=d, bases=(bB,), tensorsig=(c,), dtype=dtype)
rB_vec_post.set_scales(dealias_tuple)
rB_vec_post['g'][2] = rB
L_AMB = cross(rB_vec_post, ρB*uB)
Lx_AMB = dot(exB, L_AMB)
Ly_AMB = dot(eyB, L_AMB)
Lz_AMB = dot(ezB, L_AMB)

ezS = field.Field(dist=d, bases=(bS,), tensorsig=(c,), dtype=dtype)
ezS.set_scales(dealias_tuple)
ezS['g'][1] = -np.sin(θS)
ezS['g'][2] =  np.cos(θS)
ezS = operators.Grid(ezS).evaluate()

exS = field.Field(dist=d, bases=(bS,), tensorsig=(c,), dtype=dtype)
exS.set_scales(dealias_tuple)
exS['g'][0] = -np.sin(φS)
exS['g'][1] = np.cos(θS)*np.cos(φS)
exS['g'][2] = np.sin(θS)*np.cos(φS)
exS = operators.Grid(exS).evaluate()

eyS = field.Field(dist=d, bases=(bS,), tensorsig=(c,), dtype=dtype)
eyS.set_scales(dealias_tuple)
eyS['g'][0] = np.cos(φS)
eyS['g'][1] = np.cos(θS)*np.sin(φS)
eyS['g'][2] = np.sin(θS)*np.sin(φS)
eyS = operators.Grid(eyS).evaluate()

rS_vec_post = field.Field(dist=d, bases=(bS,), tensorsig=(c,), dtype=dtype)
rS_vec_post.set_scales(dealias_tuple)
rS_vec_post['g'][2] = rS
L_AMS = cross(rS_vec_post, ρS*uS)
Lx_AMS = dot(exS, L_AMS)
Ly_AMS = dot(eyS, L_AMS)
Lz_AMS = dot(ezS, L_AMS)



uB_squared = dot(uB, uB)
uS_squared = dot(uS, uS)
uB_squared.store_last = True
uS_squared.store_last = True

vol_averagerB      = BallVolumeAverager(pB)
vol_averagerS      = ShellVolumeAverager(pS)
profile_averagerB    = PhiThetaAverager(pB)
profile_averagerS    = PhiThetaAverager(pS)
equator_slicerB     = EquatorSlicer(pB)
equator_slicerS     = EquatorSlicer(pS)
mer_slicer1B          = MeridionSlicer(pB, phi_target=0)
mer_slicer2B          = MeridionSlicer(pB, phi_target=np.pi/2)
mer_slicer3B          = MeridionSlicer(pB, phi_target=np.pi)
mer_slicer4B          = MeridionSlicer(pB, phi_target=3*np.pi/2)
mer_slicer1S          = MeridionSlicer(pS, phi_target=0)
mer_slicer2S          = MeridionSlicer(pS, phi_target=np.pi/2)
mer_slicer3S          = MeridionSlicer(pS, phi_target=np.pi)
mer_slicer4S          = MeridionSlicer(pS, phi_target=3*np.pi/2)
ORI = OutputRadialInterpolate
analysis_tasks = []

slices = d3FileHandler(solver, '{:s}/slices'.format(out_dir), sim_dt=visual_dt, max_writes=40)
slices.add_task(uB, name='uB_eq', layout='g', extra_op=equator_slicerB)
slices.add_task(uS, name='uS_eq', layout='g', extra_op=equator_slicerS)
slices.add_task(s1B, name='s1B_eq', layout='g', extra_op=equator_slicerB)
slices.add_task(s1S, name='s1S_eq', layout='g', extra_op=equator_slicerS)
for fd, name in zip((uB, s1B), ('uB', 's1B')):
    for radius, r_str in zip((0.5, 1), ('0.5', '1')):
        operation = fd(r=radius)
        slices.add_task(operation, name=name+'(r={})'.format(r_str), layout='g', extra_op=ORI(pB, operation))
for fd, name in zip((uS, s1S), ('uS', 's1S')):
    for radius, r_str in zip((0.95*r_outer,), ('0.95R',)):
        operation = fd(r=radius)
        slices.add_task(operation, name=name+'(r={})'.format(r_str), layout='g', extra_op=ORI(pS, operation))
for extra_op, name in zip([mer_slicer1B, mer_slicer2B, mer_slicer3B, mer_slicer4B], ['(phi=0)', '(phi=0.5*pi)', '(phi=pi)', '(phi=1.5*pi)',]):
    slices.add_task(uB,  extra_op=extra_op, name='uB' +name, layout='g', extra_op_comm=False)
    slices.add_task(s1B,  extra_op=extra_op, name='s1B' +name, layout='g', extra_op_comm=False)
for extra_op, name in zip([mer_slicer1S, mer_slicer2S, mer_slicer3S, mer_slicer4S], ['(phi=0)', '(phi=0.5*pi)', '(phi=pi)', '(phi=1.5*pi)',]):
    slices.add_task(uS,  extra_op=extra_op, name='uS' +name, layout='g', extra_op_comm=False)
    slices.add_task(s1S,  extra_op=extra_op, name='s1S' +name, layout='g', extra_op_comm=False)


analysis_tasks.append(slices)

re_ball = solver.evaluator.add_dictionary_handler(iter=10)
re_ball.add_task(Re*(uB_squared)**(1/2), name='Re_avg_ball', layout='g', scales=dealias_tuple)

scalars = d3FileHandler(solver, '{:s}/scalars'.format(out_dir), sim_dt=scalar_dt, max_writes=np.inf)
scalars.add_task(Re*(uB_squared)**(1/2), name='Re_avg_ball',  layout='g', extra_op = vol_averagerB, extra_op_comm=True)
scalars.add_task(Re*(uS_squared)**(1/2), name='Re_avg_shell', layout='g', extra_op = vol_averagerS, extra_op_comm=True)
scalars.add_task(ρB*uB_squared/2,    name='KE_ball',   layout='g', extra_op = vol_averagerB, extra_op_comm=True)
scalars.add_task(ρS*uS_squared/2,    name='KE_shell',  layout='g', extra_op = vol_averagerS, extra_op_comm=True)
scalars.add_task(ρB*TB*s1B,           name='TE_ball',  layout='g', extra_op = vol_averagerB, extra_op_comm=True)
scalars.add_task(ρS*TS*s1S,           name='TE_shell', layout='g', extra_op = vol_averagerS, extra_op_comm=True)
scalars.add_task(Lx_AMB, name='Lx_AM_ball', layout='g', extra_op=vol_averagerB, extra_op_comm=True)
scalars.add_task(Ly_AMB, name='Ly_AM_ball', layout='g', extra_op=vol_averagerB, extra_op_comm=True)
scalars.add_task(Lz_AMB, name='Lz_AM_ball', layout='g', extra_op=vol_averagerB, extra_op_comm=True)
scalars.add_task(Lx_AMS, name='Lx_AM_shell', layout='g', extra_op=vol_averagerS, extra_op_comm=True)
scalars.add_task(Ly_AMS, name='Ly_AM_shell', layout='g', extra_op=vol_averagerS, extra_op_comm=True)
scalars.add_task(Lz_AMS, name='Lz_AM_shell', layout='g', extra_op=vol_averagerS, extra_op_comm=True)
analysis_tasks.append(scalars)

profiles = d3FileHandler(solver, '{:s}/profiles'.format(out_dir), sim_dt=visual_dt, max_writes=100)
profiles.add_task((4*np.pi*r_valsB**2)*(ρB*urB*pomega_hat_B),         name='wave_lumB', layout='g', extra_op = profile_averagerB, extra_op_comm=True)
profiles.add_task((4*np.pi*r_valsS**2)*(ρS*urS*pomega_hat_S),         name='wave_lumS', layout='g', extra_op = profile_averagerS, extra_op_comm=True)
profiles.add_task((4*np.pi*r_valsB**2)*(ρB*urB*hB),                   name='enth_lumB', layout='g', extra_op = profile_averagerB, extra_op_comm=True)
profiles.add_task((4*np.pi*r_valsS**2)*(ρS*urS*hS),                   name='enth_lumS', layout='g', extra_op = profile_averagerS, extra_op_comm=True)
profiles.add_task((4*np.pi*r_valsB**2)*(-ρB*visc_fluxB_r/Re),         name='visc_lumB', layout='g', extra_op = profile_averagerB, extra_op_comm=True)
profiles.add_task((4*np.pi*r_valsS**2)*(-ρS*visc_fluxS_r/Re),         name='visc_lumS', layout='g', extra_op = profile_averagerS, extra_op_comm=True)
profiles.add_task((4*np.pi*r_valsB**2)*(-ρB*TB*dot(erB, grads1B)/Pe), name='cond_lumB', layout='g', extra_op = profile_averagerB, extra_op_comm=True)
profiles.add_task((4*np.pi*r_valsS**2)*(-ρS*TS*dot(erS, grads1S)/Pe), name='cond_lumS', layout='g', extra_op = profile_averagerS, extra_op_comm=True)
profiles.add_task((4*np.pi*r_valsB**2)*(0.5*ρB*urB*uB_squared),       name='KE_lumB',   layout='g', extra_op = profile_averagerB, extra_op_comm=True)
profiles.add_task((4*np.pi*r_valsS**2)*(0.5*ρS*urS*uS_squared),       name='KE_lumS',   layout='g', extra_op = profile_averagerS, extra_op_comm=True)
analysis_tasks.append(profiles)

if args['--sponge']:
    surface_shell_slices = d3FileHandler(solver, '{:s}/wave_shell_slices'.format(out_dir), sim_dt=max_dt, max_writes=20)
    for rval in [0.90, 1.05]:
        surface_shell_slices.add_task(radComp(uB(r=rval)),  extra_op=ORI(pB, radComp(uB(r=rval))), name='u(r={})'.format(rval), layout='g')
        surface_shell_slices.add_task(pomega_hat_B(r=rval), extra_op=ORI(pB, pomega_hat_B(r=rval)), name='pomega(r={})'.format(rval),    layout='g')
    for rval in [1.15, 1.60]:
        surface_shell_slices.add_task(radComp(uS(r=rval)),  extra_op=ORI(pS, radComp(uB(r=rval))), name='u(r={})'.format(rval), layout='g')
        surface_shell_slices.add_task(pomega_hat_S(r=rval), extra_op=ORI(pS, pomega_hat_B(r=rval)), name='pomega(r={})'.format(rval),    layout='g')
    analysis_tasks.append(surface_shell_slices)
else:
    surface_shell_slices = d3FileHandler(solver, '{:s}/surface_shell_slices'.format(out_dir), sim_dt=max_dt, max_writes=20)
    surface_shell_slices.add_task(s1S(r=r_outer),         name='s1_surf',    layout='g', extra_op=ORI(pS, s1S(r=r_outer)))
    analysis_tasks.append(surface_shell_slices)

checkpoint = d3FileHandler(solver, '{:s}/checkpoint'.format(out_dir), max_writes=1, sim_dt=10*t_buoy)
checkpoint.add_task(s1B, name='s1B', scales=1, layout='c')
checkpoint.add_task(pB, name='pB', scales=1, layout='c')
checkpoint.add_task(uB, name='uB', scales=1, layout='c')
checkpoint.add_task(s1S, name='s1S', scales=1, layout='c')
checkpoint.add_task(pS, name='pS', scales=1, layout='c')
checkpoint.add_task(uS, name='uS', scales=1, layout='c')

imaginary_cadence = 100

#CFL setup
from dedalus.extras.flow_tools import CFL
heaviside_cfl = field.Field(dist=d, bases=(bB,), dtype=dtype)
heaviside_cfl.require_scales(dealias_tuple)
heaviside_cfl['g'] = 1
if args['--CFL_max_r'] is not None:
    if np.sum(rB > float(args['--CFL_max_r'])) > 0:
        heaviside_cfl['g'][:,:, rB.flatten() > float(args['--CFL_max_r'])] = 0
initial_max_dt = 0.5*t_buoy
my_cfl = CFL(solver, initial_max_dt, safety=float(args['--safety']), cadence=1, max_dt=initial_max_dt, min_change=0.1, max_change=1.5, threshold=0.1)
my_cfl.add_velocity(heaviside_cfl*uB)

dt = initial_max_dt

#startup iterations
for i in range(10):
    logger.info('startup iteration {}'.format(i))
    solver.step(dt)

# Main loop
start_time = time.time()
start_iter = solver.iteration
max_dt_check = True
try:
    while solver.ok:
        solver.step(dt)
        dt = my_cfl.compute_dt()
        base2_frac = 2**(np.ceil(np.log2(max_dt/dt)))
        dt = max_dt/base2_frac

        if solver.iteration % imaginary_cadence in timestepper_history:
            for f in solver.state:
                f.require_grid_space()

        if solver.iteration % 10 == 0:
            Re0 = vol_averagerB(re_ball.fields['Re_avg_ball'], comm=True)
            logger.info("t = %f, dt = %f, Re = %e" %(solver.sim_time, dt, Re0))
            if max_dt_check and Re0 > 1:
                my_cfl.max_dt = max_dt
                max_dt_check = False

except:
    logger.info('something went wrong in main loop, making final checkpoint')
    raise
finally:
    end_time = time.time()
    end_iter = solver.iteration
    cpu_sec  = end_time - start_time
    n_iter   = end_iter - start_iter


    fcheckpoint = d3FileHandler(solver, '{:s}/final_checkpoint'.format(out_dir), max_writes=1, iter=1)
    fcheckpoint.add_task(s1B, name='s1B', scales=1, layout='c')
    fcheckpoint.add_task(uB, name='uB', scales=1, layout='c')
    fcheckpoint.add_task(pB, name='pB', scales=1, layout='c')
    fcheckpoint.add_task(s1S, name='s1S', scales=1, layout='c')
    fcheckpoint.add_task(uS, name='uS', scales=1, layout='c')
    fcheckpoint.add_task(pS, name='pS', scales=1, layout='c')
    solver.step(1e-5*dt)

    #TODO: Make the end-of-sim report better
    n_coeffs = 2*(NmaxB + NmaxS + 2)*(Lmax+1)*(Lmax+2)
    n_cpu    = d.comm_cart.size
    dof_cycles_per_cpusec = n_coeffs*n_iter/(cpu_sec*n_cpu)
    logger.info('DOF-cycles/cpu-sec : {:e}'.format(dof_cycles_per_cpusec))
    logger.info('Run iterations: {:e}'.format(n_iter))
    logger.info('Sim end time: {:e}'.format(solver.sim_time))
    logger.info('Run time: {}'.format(cpu_sec))

from d3_outputs import post
for t in analysis_tasks:
    post.merge_analysis(t.base_path)
