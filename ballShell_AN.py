"""
d3 script for anelastic convection in a massive star.
Includes a ball and shell domain to simulate both the core and envelope.

A config file can be provided which overwrites any number of the options below, but command line flags can also be used if preferred.
Config files take precedence over command line args.

Usage:
    ballShell_AN.py [options]
    ballShell_AN.py <config> [options]

Options:
    --Re=<Re>            The Reynolds number of the numerical diffusivities [default: 2e2]
    --Pr=<Prandtl>       The Prandtl number  of the numerical diffusivities [default: 1]
    --L=<Lmax>           Spherical Harmonic degrees of freedom (Lmax+1)   [default: 16]
    --NB=<Nmax>          Radial degrees of freedom in ball (Nmax+1)   [default: 24]
    --NS=<Nmax>          Radial degrees of freedom in shell (Nmax+1)   [default: 8]
    --sponge             If flagged, add a damping layer in the shell that damps out waves.

    --wall_hours=<t>     Max number of wall hours to run simulation for [default: 24]
    --buoy_end_time=<t>  Max number of buoyancy time units to simulate [default: 1e5]

    --mesh=<n,m>         The processor mesh over which to distribute the cores

    --RK222              Use RK222 (default is SBDF2)
    --SBDF4              Use SBDF4 (default is SBDF2)
    --safety=<s>         Timestep CFL safety factor [default: 0.2]
    --CFL_max_r=<r>      zero out velocities above this radius value for CFL

    --mesa_file=<f>      path to a .h5 file of ICCs, curated from a MESA model
    --restart=<chk_f>    path to a checkpoint file to restart from
    --A0=<A>             Amplitude of random noise initial conditions [default: 1e-6]

    --label=<label>      A label to add to the end of the output directory

    --rotation_time=<t>  Rotation timescale, in days (for MESA file) or sim units (for polytrope)
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
import dedalus.public as d3
from mpi4py import MPI

import logging
logger = logging.getLogger(__name__)

# Define smooth Heaviside functions
from scipy.special import erf
def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
    return -(one_to_zero(*args, **kwargs) - 1)

# Read options
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
nθ = int(args['--L'])
nφ = int(2*nθ)
nrB = int(args['--NB'])
nrS = int(args['--NS'])
resolutionB = (nφ, nθ, nrB)
resolutionS = (nφ, nθ, nrS)
L_dealias = N_dealias = dealias = 1.5
dtype = np.float64
Re  = float(args['--Re'])
Pr  = 1
Pe  = Pr*Re
mesa_file = args['--mesa_file']
wall_hours = float(args['--wall_hours'])
buoy_end_time = float(args['--buoy_end_time'])
sponge = args['--sponge']

# rotation
rotation_time = args['--rotation_time']
if rotation_time is not None:
    rotation_time = float(rotation_time)
    dimensional_Ω = 2*np.pi / rotation_time  #radians / day [in MESA units]

# Initial conditions
restart = args['--restart']
A0 = float(args['--A0'])

# Timestepper
if args['--SBDF4']:
    ts = d3.SBDF4
    timestepper_history = [0, 1, 2, 3]
elif args['--RK222']:
    ts = d3.RK222
    timestepper_history = [0, ]
else:
    ts = d3.SBDF2
    timestepper_history = [0, 1,]
hermitian_cadence = 100
safety = float(args['--safety'])
CFL_max_r = args['--CFL_max_r']
if CFL_max_r is not None:
    CFL_max_r = float(CFL_max_r)
else:
    CFL_max_r = np.inf

# Processor mesh
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

# Output directory
out_dir = './' + sys.argv[0].split('.py')[0]
if sponge:
    out_dir += '_sponge'
if mesa_file is None:
    out_dir += '_polytrope'
if rotation_time is not None:
    out_dir += '_rotation{}'.format(rotation_time)

out_dir += '_Re{}_{}x{}x{}+{}'.format(args['--Re'], nφ, nθ, nrB, nrS)
if args['--label'] is not None:
    out_dir += '_{:s}'.format(args['--label'])
logger.info('saving data to {:s}'.format(out_dir))
if MPI.COMM_WORLD.rank == 0:
    if not os.path.exists('{:s}/'.format(out_dir)):
        os.makedirs('{:s}/'.format(out_dir))

# Read in domain bound values
if mesa_file is not None:
    with h5py.File(args['--mesa_file'], 'r') as f:
        r_inner = f['r_inner'][()]
        r_outer = f['r_outer'][()]
else:
    r_inner = 1.1
    r_outer = 1.5
logger.info('r_inner: {:.2f} / r_outer: {:.2f}'.format(r_inner, r_outer))

# Bases
coords  = d3.SphericalCoordinates('φ', 'θ', 'r')
dist    = d3.Distributor((coords,), mesh=mesh, dtype=dtype)
basisB  = d3.BallBasis(coords, resolutionB, radius=r_inner, dtype=dtype, dealias=(L_dealias, L_dealias, N_dealias))
basisS  = d3.ShellBasis(coords, resolutionS, radii=(r_inner, r_outer), dtype=dtype, dealias=(L_dealias, L_dealias, N_dealias))
radial_basisB = basisB.radial_basis
radial_basisS = basisS.radial_basis
b_midB = basisB.S2_basis(radius=r_inner)
b_midS = basisS.S2_basis(radius=r_inner)
b_topS = basisS.S2_basis(radius=r_outer)
φB,  θB,  rB  = basisB.local_grids(basisB.dealias)
φ1B,  θ1B,  r1B  = basisB.local_grids((1,1,1))
φBg, θBg, rBg = basisB.global_grids(basisB.dealias)
φS,  θS,  rS  = basisS.local_grids(basisS.dealias)
φ1S,  θ1S,  r1S  = basisS.local_grids((1,1,1))
φSg, θSg, rSg = basisS.global_grids(basisS.dealias)

# Fields - taus
tB      = dist.Field(name='tau_s1B', bases=b_midB)
tBt     = dist.VectorField(coords, name='tau_uB',  bases=b_midB)
tSt_top = dist.VectorField(coords, name='tau_uBt', bases=b_topS)
tSt_bot = dist.VectorField(coords, name='tau_uBb', bases=b_midB)
tS_top  = dist.Field(name='tau_s1St',bases=b_topS)
tS_bot  = dist.Field(name='tau_s1Sb',bases=b_midS)

# Fields - Problem Variables
uB = dist.VectorField(coords, name='uB', bases=basisB)
uS = dist.VectorField(coords, name='uS', bases=basisS)
pB, s1B = [dist.Field(name=n+'B', bases=basisB) for n in ['p', 's1']]
pS, s1S = [dist.Field(name=n+'S', bases=basisS) for n in ['p', 's1']]

# Fields - nccs / constants
grad_ln_ρB, grad_ln_TB, grad_s0B, grad_TB, grad_inv_PeB \
          = [dist.VectorField(coords, name=n+'B', bases=radial_basisB) for n in ['grad_ln_ρ', 'grad_ln_T', 'grad_s0', 'grad_T', 'grad_inv_Pe']]
grad_ln_ρS, grad_ln_TS, grad_s0S, grad_TS, grad_inv_PeS\
            = [dist.VectorField(coords, name=n+'S', bases=radial_basisS) for n in ['grad_ln_ρ', 'grad_ln_T', 'grad_s0', 'grad_T', 'grad_inv_Pe']]
ln_ρB, ln_TB, inv_PeB = [dist.Field(name=n+'B', bases=radial_basisB) for n in ['ln_ρ', 'ln_T', 'inv_Pe']] 
ln_ρS, ln_TS, inv_PeS = [dist.Field(name=n+'S', bases=radial_basisS) for n in ['ln_ρ', 'ln_T', 'inv_Pe']] 
inv_TB, HB, ρB, TB = [dist.Field(name=n+'B', bases=basisB) for n in ['inv_T', 'H', 'ρ', 'T']]
inv_TS, HS, ρS, TS = [dist.Field(name=n+'S', bases=basisS) for n in ['inv_T', 'H', 'ρ', 'T']]

if sponge:
    L_shell = r_outer - r_inner
    spongeS = dist.Field(name='spongeS', bases=radial_basisS)
    spongeS['g'] = zero_to_one(r1S, r_inner + 2*L_shell/3, 0.1*L_shell)

# Fields - unit vectors & (NCC) identity matrix
eφB, eθB, erB = [dist.VectorField(coords, name=n+'B') for n in ['eφ', 'eθ', 'er']]
eφS, eθS, erS = [dist.VectorField(coords, name=n+'S') for n in ['eφ', 'eθ', 'er']]
I_matrixB = dist.TensorField(coords, name='I_matrixB')
I_matrixS = dist.TensorField(coords, name='I_matrixS')
for f in [eφB, eθB, erB, I_matrixB, eφS, eθS, erS, I_matrixS]: f['g'] = 0
eφB['g'][0] = 1
eθB['g'][1] = 1
erB['g'][2] = 1
eφS['g'][0] = 1
eθS['g'][1] = 1
erS['g'][2] = 1
for i in range(3):
    I_matrixB['g'][i,i] = 1
    I_matrixS['g'][i,i] = 1

# Cartesian unit vectors for post
exB, eyB, ezB = [dist.VectorField(coords, name=n+'B', bases=basisB) for n in ['ex', 'ey', 'ez']]
exS, eyS, ezS = [dist.VectorField(coords, name=n+'S', bases=basisS) for n in ['ex', 'ey', 'ez']]

exB['g'][0] = -np.sin(φ1B)
exB['g'][1] = np.cos(θ1B)*np.cos(φ1B)
exB['g'][2] = np.sin(θ1B)*np.cos(φ1B)
exS['g'][0] = -np.sin(φ1S)
exS['g'][1] = np.cos(θ1S)*np.cos(φ1S)
exS['g'][2] = np.sin(θ1S)*np.cos(φ1S)

eyB['g'][0] = np.cos(φ1B)
eyB['g'][1] = np.cos(θ1B)*np.sin(φ1B)
eyB['g'][2] = np.sin(θ1B)*np.sin(φ1B)
eyS['g'][0] = np.cos(φ1S)
eyS['g'][1] = np.cos(θ1S)*np.sin(φ1S)
eyS['g'][2] = np.sin(θ1S)*np.sin(φ1S)

ezB['g'][0] = 0
ezB['g'][1] = -np.sin(θ1B)
ezB['g'][2] =  np.cos(θ1B)
ezS['g'][0] = 0
ezS['g'][1] = -np.sin(θ1S)
ezS['g'][2] =  np.cos(θ1S)

exB, eyB, ezB, exS, eyS, ezS = [d3.Grid(f).evaluate() for f in [exB, eyB, ezB, exS, eyS, ezS]]

# Load MESA NCC file or setup NCCs using polytrope
grid_slicesB  = dist.layouts[-1].slices(uB.domain, N_dealias)
grid_slicesS  = dist.layouts[-1].slices(uS.domain, N_dealias)
grad_s0B.require_scales(basisB.dealias)
grad_s0S.require_scales(basisS.dealias)
local_vncc_shapeB = grad_s0B['g'].shape
local_vncc_shapeS = grad_s0S['g'].shape
if mesa_file is not None:
    for field in [grad_s0B, grad_ln_ρB, grad_ln_TB, grad_TB, grad_inv_PeB, HB, ln_ρB, ln_TB, inv_PeB, ρB, TB, inv_TB]:
        field.require_scales(basisB.dealias)
    for field in [grad_s0S, grad_ln_ρS, grad_ln_TS, grad_TS, grad_inv_PeS, HS, ln_ρS, ln_TS, inv_PeS, ρS, TS, inv_TS]:
        field.require_scales(basisS.dealias)
    with h5py.File(mesa_file, 'r') as f:
        if np.prod(local_vncc_shapeB) > 0:
            grad_s0B['g']      = f['grad_s0B'][:,0,0,grid_slicesB[-1]][:,None,None,:]
            grad_ln_ρB['g']    = f['grad_ln_ρB'][:,0,0,grid_slicesB[-1]][:,None,None,:]
            grad_ln_TB['g']    = f['grad_ln_TB'][:,0,0,grid_slicesB[-1]][:,None,None,:]
            grad_TB['g']       = f['grad_TB'][:,0,0,grid_slicesB[-1]][:,None,None,:]
            grad_inv_PeB['g']  = f['grad_inv_Pe_radB'][:,0,0,grid_slicesB[-1]][:,None,None,:]
        if np.prod(grad_s0S['g'].shape) > 0:
            grad_s0S['g']     = f['grad_s0S'][:,0,0,grid_slicesS[-1]][:,None,None,:]
            grad_ln_ρS['g']   = f['grad_ln_ρS'][:,0,0,grid_slicesS[-1]][:,None,None,:]
            grad_ln_TS['g']   = f['grad_ln_TS'][:,0,0,grid_slicesS[-1]][:,None,None,:]
            grad_TS['g']      = f['grad_TS'][:,0,0,grid_slicesS[-1]][:,None,None,:]
            grad_inv_PeS['g'] = f['grad_inv_Pe_radS'][:,0,0,grid_slicesS[-1]][:,None,None,:]
        inv_PeB['g']= f['inv_Pe_radB'][:,:,grid_slicesB[-1]]
        ln_ρB['g']      = f['ln_ρB'][:,:,grid_slicesB[-1]]
        ln_TB['g']      = f['ln_TB'][:,:,grid_slicesB[-1]]
        HB['g']         = f['H_effB'][:,:,grid_slicesB[-1]]
        ρB['g']         = np.exp(f['ln_ρB'][:,:,grid_slicesB[-1]])[None,None,:]
        TB['g']         = f['TB'][:,:,grid_slicesB[-1]][None,None,:]
        inv_TB['g']     = 1/TB['g']

        inv_PeS['g']= f['inv_Pe_radS'][:,:,grid_slicesS[-1]]
        ln_ρS['g']      = f['ln_ρS'][:,:,grid_slicesS[-1]]
        ln_TS['g']      = f['ln_TS'][:,:,grid_slicesS[-1]]
        HS['g']          = f['H_effS'][:,:,grid_slicesS[-1]]
        ρS['g']         = np.exp(f['ln_ρS'][:,:,grid_slicesS[-1]])[None,None,:]
        TS['g']         = f['TS'][:,:,grid_slicesS[-1]][None,None,:]
        inv_TS['g']     = 1/TS['g']

        max_dt = f['max_dt'][()]
        t_buoy = 1 #assume nondimensionalization on heating ~ buoyancy time

        if rotation_time is not None:
            sim_tau_sec = f['tau'][()]
            sim_tau_day = sim_tau_sec / (60*60*24)
            Ω = sim_tau_day * dimensional_Ω 
            t_rot = 1/(2*Ω)
        else:
            t_rot = np.inf

        if sponge:
            f_brunt = f['tau'][()]*np.sqrt(f['N2max_shell'][()])/(2*np.pi)
            spongeS['g'] *= f_brunt

else:
    logger.info("Using polytropic initial conditions")
    from scipy.interpolate import interp1d
    with h5py.File('polytropes/poly_nOuter1.6.h5', 'r') as f:
        T_func = interp1d(f['r'][()], f['T'][()])
        ρ_func = interp1d(f['r'][()], f['ρ'][()])
        grad_s0_func = interp1d(f['r'][()], f['grad_s0'][()])
        H_func   = interp1d(f['r'][()], f['H_eff'][()])
    max_grad_s0 = grad_s0_func(r_outer)
    max_dt = 2/np.sqrt(max_grad_s0)
    t_buoy      = 1
    if rotation_time is not None:
        Ω = dimensional_Ω 
        t_rot = 1/(2*Ω)
    else:
        t_rot = np.inf
        

    for r1, basis_fields, local_vncc_shape, basis  in zip((r1B, r1S), ((TB, ρB, HB, inv_TB, ln_TB, ln_ρB, inv_PeB, grad_ln_TB, grad_ln_ρB, grad_TB, grad_s0B, grad_inv_PeB), \
                                             (TS, ρS, HS, inv_TS, ln_TS, ln_ρS, inv_PeS, grad_ln_TS, grad_ln_ρS, grad_TS, grad_s0S, grad_inv_PeS)), \
                                             (local_vncc_shapeB, local_vncc_shapeS), (basisB, basisS)):
        T, ρ, H, inv_T, ln_T, ln_ρ, inv_Pe, grad_ln_T, grad_ln_ρ, grad_T, grad_s0, grad_inv_Pe = basis_fields

        T['g']       = T_func(r1)
        ρ['g']       = ρ_func(r1)
        H['g']       = H_func(r1)
        inv_T['g']   = 1/T_func(r1)

        grad_ln_ρ_full = (d3.grad(ρ)/ρ).evaluate()
        grad_T_full = d3.grad(T).evaluate()
        grad_ln_T_full = (grad_T_full/T).evaluate()
        if np.prod(local_vncc_shape) > 0:
            grad_s0.require_scales(1)
            grad_s0['g'][2]  = grad_s0_func(r1)
            for f in [grad_ln_ρ, grad_ln_T, grad_T]: f.require_scales(basis.dealias)
            grad_ln_ρ['g']   = grad_ln_ρ_full['g'][:,0,0,None,None,:]
            grad_ln_T['g']   = grad_ln_T_full['g'][:,0,0,None,None,:]
            grad_T['g']      = grad_T_full['g'][:,0,0,None,None,:]
            grad_inv_Pe['g'] = 0
        ln_T['g']        = np.log(T_func(r1))
        ln_ρ['g']        = np.log(ρ_func(r1))
        inv_Pe['g']      = 1/Pe

if rotation_time is not None:
    logger.info("Running with Coriolis Omega = {:.3e}".format(Ω))

#Stress matrices & viscous terms (assumes uniform kinematic viscosity; so dynamic viscosity mu = const * rho)
divUB = d3.div(uB)
EB = 0.5*(d3.grad(uB) + d3.transpose(d3.grad(uB)))
σB = 2*(EB - (1/3)*divUB*I_matrixB)
visc_div_stressB = d3.div(σB) + d3.dot(σB, grad_ln_ρB)
VHB  = 2*(d3.trace(d3.dot(EB, EB)) - (1/3)*divUB*divUB)

divUS = d3.div(uS)
ES = 0.5*(d3.grad(uS) + d3.transpose(d3.grad(uS)))
σS = 2*(ES - (1/3)*divUS*I_matrixS)
visc_div_stressS = d3.div(σS) + d3.dot(σS, grad_ln_ρS)
VHS  = 2*(d3.trace(d3.dot(ES, ES)) - (1/3)*divUS*divUS)

# Grid-lock some operators / define grad's
HB = d3.Grid(HB).evaluate()
HS = d3.Grid(HS).evaluate()
inv_TB = d3.Grid(inv_TB).evaluate()
inv_TS = d3.Grid(inv_TS).evaluate()
grad_s1B = d3.grad(s1B)
grad_s1S = d3.grad(s1S)

## Boundary conditions
# Matching boundary conditions at ball-shell
u_match_bc      = uB(r=r_inner) - uS(r=r_inner)
p_match_bc      = pB(r=r_inner) - pS(r=r_inner)
stress_match_bc = d3.angular(d3.radial(σB(r=r_inner) - σS(r=r_inner)), index=0)
stress_match_bc.name = 'stress_match_bc'
#stress_match_bc = d3.angular(d3.radial(σB(r=r_inner)), index=0) - d3.angular(d3.radial(σS(r=r_inner)), index=0)
s_match_bc      = s1B(r=r_inner) - s1S(r=r_inner)
grad_s_match_bc = d3.radial(grad_s1B(r=r_inner) - grad_s1S(r=r_inner))
#grad_s_match_bc = d3.radial(grad_s1B(r=r_inner)) - d3.radial(grad_s1S(r=r_inner))
# Surface: Impenetrable, stress-free, no entropy gradient
impenetrable = d3.radial(uS(r=r_outer))
stress_free  = d3.angular(d3.radial(ES(r=r_outer)), index=0)
stress_free.name = 'stress_free'
grad_s_surface = d3.radial(grad_s1S(r=r_outer))

# Rotation and damping terms
if rotation_time is not None:
    rotation_termB = -2*Ω*d3.cross(ezB, uB)
    rotation_termS = -2*Ω*d3.cross(ezS, uS)
else:
    rotation_termB = 0
    rotation_termS = 0

if args['--sponge']:
    sponge_termS = spongeS*uS
else:
    sponge_termS = 0

# Lift operators for boundary conditions
lift_basisB = basisB.clone_with(k=0)
lift_basisS = basisS.clone_with(k=2)
liftB   = lambda A: d3.LiftTau(A, lift_basisB, -1)
liftS   = lambda A, n: d3.LiftTau(A, lift_basisS, n)
BC_uB = liftB(tBt)
BC_uS = liftS(tSt_bot, -1) + liftS(tSt_top, -2)
BC_s1B = liftB(tB)
BC_s1S = liftS(tS_bot, -1) + liftS(tS_top, -2)


# Problem
problem = d3.IVP([pB, uB, pS, uS, s1B, s1S, tBt, tSt_bot, tSt_top, tB, tS_bot, tS_top], namespace=locals())

# Equations
### Ball momentum
problem.add_equation("div(uB) + dot(uB, grad_ln_ρB) = 0", condition="nθ != 0")
problem.add_equation("dt(uB) + grad(pB) + grad_TB*s1B - (1/Re)*visc_div_stressB                + BC_uB = cross(uB, curl(uB)) + rotation_termB", condition = "nθ != 0")
### Shell momentum
problem.add_equation("div(uS) + dot(uS, grad_ln_ρS) = 0", condition="nθ != 0")
problem.add_equation("dt(uS) + grad(pS) + grad_TS*s1S - (1/Re)*visc_div_stressS + sponge_termS + BC_uS = cross(uS, curl(uS)) + rotation_termS", condition = "nθ != 0")
## ell == 0 momentum
problem.add_equation("pB = 0", condition="nθ == 0")
problem.add_equation("uB = 0", condition="nθ == 0")
problem.add_equation("pS = 0", condition="nθ == 0")
problem.add_equation("uS = 0", condition="nθ == 0")
### Ball energy
problem.add_equation("dt(s1B) + dot(uB, grad_s0B) - (inv_PeB)*(lap(s1B) + dot(grad_s1B, (grad_ln_ρB + grad_ln_TB))) - dot(grad_s1B, grad_inv_PeB) + BC_s1B = - dot(uB, grad_s1B) + HB + (1/Re)*inv_TB*VHB ")
### Shell energy
problem.add_equation("dt(s1S) + dot(uS, grad_s0S) - (inv_PeS)*(lap(s1S) + dot(grad_s1S, (grad_ln_ρS + grad_ln_TS))) - dot(grad_s1S, grad_inv_PeS) + BC_s1S = - dot(uS, grad_s1S) + HS + (1/Re)*inv_TS*VHS ")

# Boundary Conditions
# Velocity BCs ell != 0
problem.add_equation("u_match_bc = 0", condition="nθ != 0")
problem.add_equation("p_match_bc = 0", condition="nθ != 0")
problem.add_equation("stress_match_bc = 0", condition="nθ != 0")
problem.add_equation("impenetrable = 0", condition="nθ != 0")
problem.add_equation("stress_free = 0", condition="nθ != 0")
# velocity BCs ell == 0
problem.add_equation("tBt = 0", condition="nθ == 0")
problem.add_equation("tSt_bot = 0", condition="nθ == 0")
problem.add_equation("tSt_top = 0", condition="nθ == 0")

# Entropy BCs
problem.add_equation("s_match_bc = 0")
problem.add_equation("grad_s_match_bc = 0")
problem.add_equation("grad_s_surface = 0")

logger.info("Problem built")
# Solver
solver = problem.build_solver(ts)
solver.stop_sim_time = buoy_end_time*t_buoy
solver.stop_wall_time = wall_hours * 60 * 60
logger.info("solver built")

# Initial conditions / Checkpoint
write_mode = 'overwrite'
timestep = None
if restart is not None:
    write, timestep = solver.load_state(restart)
    write_mode = 'append'
else:
    # Initial conditions
    seed = 42 + dist.comm_cart.rank
    rand = np.random.RandomState(seed=seed)
    filter_scale = 0.25

    # Generate noise & filter it
#    s1B['g'] = A0*rand.standard_normal(s1B['g'].shape)*one_to_zero(r1B, 0.9*r_inner, width=0.05*r_inner)
    s1B['g'] = A0*rand.standard_normal(s1B['g'].shape)
    s1B.require_scales(filter_scale)
    s1B['c']
    s1B['g']
    s1B.require_scales(basisB.dealias)
    s1B['g'] *= one_to_zero(rB, 0.9*r_inner, width=0.04*r_inner)

## Analysis Setup
# Cadence
scalar_dt = 0.25*t_buoy
lum_dt   = 0.5*t_buoy
visual_dt = 0.05*t_buoy
outer_shell_dt = max_dt

# Operators, extra fields
r_valsB = dist.Field(name='r_valsB', bases=basisB)
r_valsS = dist.Field(name='r_valsS', bases=basisS)
r_valsB['g'] = r1B
r_valsS['g'] = r1S
erB = d3.Grid(erB).evaluate()
erS = d3.Grid(erS).evaluate()
r_valsB = d3.Grid(r_valsB).evaluate()
r_valsS = d3.Grid(r_valsS).evaluate()
uB_squared = d3.dot(uB, uB)
uS_squared = d3.dot(uS, uS)
urB = d3.dot(erB, uB)
urS = d3.dot(erS, uS)
hB = pB - 0.5*uB_squared + TB*s1B
hS = pS - 0.5*uS_squared + TS*s1S
pomega_hat_B = pB - 0.5*uB_squared
pomega_hat_S = pS - 0.5*uS_squared

visc_fluxB_r = 2*d3.dot(erB, d3.dot(uB, EB) - (1/3) * uB * divUB)
visc_fluxS_r = 2*d3.dot(erS, d3.dot(uS, ES) - (1/3) * uS * divUS)

# Angular momentum
rB_vec_post = dist.VectorField(coords, name='rB_vec_post', bases=basisB)
rB_vec_post['g'][2] = r1B
L_AMB = d3.cross(rB_vec_post, ρB*uB)
Lx_AMB = d3.dot(exB, L_AMB)
Ly_AMB = d3.dot(eyB, L_AMB)
Lz_AMB = d3.dot(ezB, L_AMB)

rS_vec_post = dist.VectorField(coords, name='rS_vec_post', bases=basisS)
rS_vec_post['g'][2] = r1S
L_AMS = d3.cross(rS_vec_post, ρS*uS)
Lx_AMS = d3.dot(exS, L_AMS)
Ly_AMS = d3.dot(eyS, L_AMS)
Lz_AMS = d3.dot(ezS, L_AMS)

# Averaging Operations
volume  = (4/3)*np.pi*r_outer**3
volumeB = (4/3)*np.pi*r_inner**3
volumeS = volume - volumeB

az_avg = lambda A: d3.Average(A, coords.coords[0])
s2_avg = lambda A: d3.Average(A, coords.S2coordsys)
vol_avgB = lambda A: d3.Integrate(A/volumeB, coords)
vol_avgS = lambda A: d3.Integrate(A/volumeS, coords)
luminosityB = lambda A: (4*np.pi*r_valsB**2) * s2_avg(A)
luminosityS = lambda A: (4*np.pi*r_valsS**2) * s2_avg(A)

analysis_tasks = []

slices = solver.evaluator.add_file_handler('{:s}/slices'.format(out_dir), sim_dt=visual_dt, max_writes=40)
slices.add_task(uB(θ=np.pi/2), name='uB_eq', layout='g')
slices.add_task(uS(θ=np.pi/2), name='uS_eq', layout='g')
slices.add_task(s1B(θ=np.pi/2), name='s1B_eq', layout='g')
slices.add_task(s1S(θ=np.pi/2), name='s1S_eq', layout='g')
for fd, name in zip((uB, s1B), ('uB', 's1B')):
    for radius, r_str in zip((0.5, 1), ('0.5', '1')):
        operation = fd(r=radius)
        slices.add_task(operation, name=name+'(r={})'.format(r_str), layout='g')
for fd, name in zip((uS, s1S), ('uS', 's1S')):
    for radius, r_str in zip((0.95*r_outer,), ('0.95R',)):
        operation = fd(r=radius)
        slices.add_task(operation, name=name+'(r={})'.format(r_str), layout='g')
for az_val, name in zip([0, np.pi/2, np.pi, 3*np.pi/2], ['(phi=0)', '(phi=0.5*pi)', '(phi=pi)', '(phi=1.5*pi)',]):
    slices.add_task(uB(φ=az_val),  name='uB' +name, layout='g')
    slices.add_task(s1B(φ=az_val), name='s1B'+name, layout='g')
for az_val, name in zip([0, np.pi/2, np.pi, 3*np.pi/2], ['(phi=0)', '(phi=0.5*pi)', '(phi=pi)', '(phi=1.5*pi)',]):
    slices.add_task(uS(φ=az_val),  name='uS' +name, layout='g')
    slices.add_task(s1S(φ=az_val), name='s1S'+name, layout='g')
analysis_tasks.append(slices)

scalars = solver.evaluator.add_file_handler('{:s}/scalars'.format(out_dir), sim_dt=scalar_dt, max_writes=np.inf)
scalars.add_task(vol_avgB(Re*(uB_squared)**(1/2)), name='Re_avg_ball',  layout='g')
scalars.add_task(vol_avgS(Re*(uS_squared)**(1/2)), name='Re_avg_shell', layout='g')
scalars.add_task(vol_avgB(ρB*uB_squared/2),    name='KE_ball',   layout='g')
scalars.add_task(vol_avgS(ρS*uS_squared/2),    name='KE_shell',  layout='g')
scalars.add_task(vol_avgB(ρB*TB*s1B),           name='TE_ball',  layout='g')
scalars.add_task(vol_avgS(ρS*TS*s1S),           name='TE_shell', layout='g')
scalars.add_task(vol_avgB(Lx_AMB), name='Lx_AM_ball', layout='g')
scalars.add_task(vol_avgB(Ly_AMB), name='Ly_AM_ball', layout='g')
scalars.add_task(vol_avgB(Lz_AMB), name='Lz_AM_ball', layout='g')
scalars.add_task(vol_avgS(Lx_AMS), name='Lx_AM_shell', layout='g')
scalars.add_task(vol_avgS(Ly_AMS), name='Ly_AM_shell', layout='g')
scalars.add_task(vol_avgS(Lz_AMS), name='Lz_AM_shell', layout='g')
analysis_tasks.append(scalars)

profiles = solver.evaluator.add_file_handler('{:s}/profiles'.format(out_dir), sim_dt=visual_dt, max_writes=100)
profiles.add_task(luminosityB(ρB*urB*pomega_hat_B),         name='wave_lumB', layout='g')
profiles.add_task(luminosityS(ρS*urS*pomega_hat_S),         name='wave_lumS', layout='g')
profiles.add_task(luminosityB(ρB*urB*hB),                   name='enth_lumB', layout='g')
profiles.add_task(luminosityS(ρS*urS*hS),                   name='enth_lumS', layout='g')
profiles.add_task(luminosityB(-ρB*visc_fluxB_r/Re),         name='visc_lumB', layout='g')
profiles.add_task(luminosityS(-ρS*visc_fluxS_r/Re),         name='visc_lumS', layout='g')
profiles.add_task(luminosityB(-ρB*TB*d3.dot(erB, grad_s1B)/Pe), name='cond_lumB', layout='g')
profiles.add_task(luminosityS(-ρS*TS*d3.dot(erS, grad_s1S)/Pe), name='cond_lumS', layout='g')
profiles.add_task(luminosityB(0.5*ρB*urB*uB_squared),       name='KE_lumB',   layout='g')
profiles.add_task(luminosityS(0.5*ρS*urS*uS_squared),       name='KE_lumS',   layout='g')
analysis_tasks.append(profiles)

if args['--sponge']:
    surface_shell_slices = solver.evaluator.add_file_handler('{:s}/wave_shell_slices'.format(out_dir), sim_dt=100000*max_dt, max_writes=20)
    for rval in [0.90, 1.05]:
        surface_shell_slices.add_task(d3.radial(uB(r=rval)), name='u(r={})'.format(rval), layout='g')
        surface_shell_slices.add_task(pomega_hat_B(r=rval), name='pomega(r={})'.format(rval),    layout='g')
    for rval in [1.15, 1.60]:
        surface_shell_slices.add_task(d3.radial(uS(r=rval)), name='u(r={})'.format(rval), layout='g')
        surface_shell_slices.add_task(pomega_hat_S(r=rval), name='pomega(r={})'.format(rval),    layout='g')
    analysis_tasks.append(surface_shell_slices)
else:
    surface_shell_slices = solver.evaluator.add_file_handler('{:s}/wave_shell_slices'.format(out_dir), sim_dt=100000*max_dt, max_writes=20)
    surface_shell_slices.add_task(s1S(r=r_outer),         name='s1_surf',    layout='g')
    analysis_tasks.append(surface_shell_slices)

checkpoint = solver.evaluator.add_file_handler('{:s}/checkpoint'.format(out_dir), max_writes=1, sim_dt=10*t_buoy)
checkpoint.add_tasks(solver.state, layout='g')

re_ball = solver.evaluator.add_dictionary_handler(iter=10)
re_ball.add_task(vol_avgB(Re*(uB_squared)**(1/2)), name='Re_avg_ball', layout='g')

#CFL setup
heaviside_cfl = dist.Field(name='heaviside_cfl', bases=basisB)
heaviside_cfl['g'] = 1
if np.sum(rB > CFL_max_r) > 0:
    heaviside_cfl['g'][:,:, r1B.flatten() > CFL_max_r] = 0
heaviside_cfl = d3.Grid(heaviside_cfl).evaluate()

initial_max_dt = np.min((visual_dt, t_rot*0.5))
while initial_max_dt < max_dt:
    max_dt /= 2
if timestep is None:
    timestep = initial_max_dt
my_cfl = d3.CFL(solver, timestep, safety=safety, cadence=1, max_dt=initial_max_dt, min_change=0.1, max_change=1.5, threshold=0.1)
my_cfl.add_velocity(heaviside_cfl*uB)

#startup iterations
for i in range(10):
    solver.step(timestep)
    logger.info("startup iteration %d, t = %f, timestep = %f" %(i, solver.sim_time, timestep))
    timestep = my_cfl.compute_timestep()

# Main loop
start_time = time.time()
start_iter = solver.iteration
max_dt_check = True
current_max_dt = my_cfl.max_dt
slice_process = False
just_wrote    = False
slice_time = np.inf
Re0 = 0
try:
    while solver.ok:
        timestep = my_cfl.compute_timestep()

        if just_wrote:
            just_wrote = False
            num_steps = np.ceil(outer_shell_dt / timestep)
            timestep = current_max_dt = my_cfl.stored_dt = outer_shell_dt/num_steps
        elif max_dt_check:
            timestep = np.min((timestep, current_max_dt))
        else:
            my_cfl.stored_dt = timestep = current_max_dt

        t_future = solver.sim_time + timestep
        if t_future >= slice_time*(1-1e-8):
           slice_process = True

        if solver.iteration % hermitian_cadence in timestepper_history:
            for f in solver.state:
                f.require_grid_space()

        solver.step(timestep)

        if solver.iteration % 10 == 0:
            Re_avg = re_ball.fields['Re_avg_ball']
            if dist.comm_cart.rank == 0:
                Re0 = Re_avg['g'].min()
            else:
                Re0 = None
            Re0 = dist.comm_cart.bcast(Re0, root=0)
            logger.info("t = %f, timestep = %f, Re = %e" %(solver.sim_time, timestep, Re0))
        if max_dt_check and timestep < outer_shell_dt:
            my_cfl.max_dt = max_dt
            max_dt_check = False
            just_wrote = True
            slice_time = solver.sim_time + outer_shell_dt

        if slice_process:
            slice_process = False
            wall_time = time.time() - solver.start_time
            solver.evaluator.evaluate_handlers([surface_shell_slices],wall_time=wall_time, sim_time=solver.sim_time, iteration=solver.iteration,world_time = time.time(),timestep=timestep)
            slice_time = solver.sim_time + outer_shell_dt
            just_wrote = True

        if np.isnan(Re0):
            logger.info('exiting with NaN')
            break

except:
    logger.info('something went wrong in main loop, making final checkpoint')
    raise
finally:
    end_time = time.time()
    end_iter = solver.iteration
    cpu_sec  = end_time - start_time
    n_iter   = end_iter - start_iter


    fcheckpoint = solver.evaluator.add_file_handler('{:s}/final_checkpoint'.format(out_dir), max_writes=1, sim_dt=10*t_buoy)
    fcheckpoint.add_tasks(solver.state, layout='g')
    solver.step(timestep)

    #TODO: Make the end-of-sim report better
    n_coeffs = np.prod(resolutionB) + np.prod(resolutionS)
    n_cpu    = dist.comm_cart.size
    dof_cycles_per_cpusec = n_coeffs*n_iter/(cpu_sec*n_cpu)
    logger.info('DOF-cycles/cpu-sec : {:e}'.format(dof_cycles_per_cpusec))
    logger.info('Run iterations: {:e}'.format(n_iter))
    logger.info('Sim end time: {:e}'.format(solver.sim_time))
    logger.info('Run time: {}'.format(cpu_sec))
