"""
d3 script for eigenvalue problem of anelastic convection / waves in a massive star.

Usage:
    evp_ballShell_AN.py [options]
    evp_ballShell_AN.py <config> [options]

Options:
    --Re=<Re>            The Reynolds number of the numerical diffusivities [default: 5e1]
    --Pr=<Prandtl>       The Prandtl number  of the numerical diffusivities [default: 1]
    --L=<Lmax>           Angular resolution (Lmax   [default: 1]
    --NB=<Nmax>          The ball radial degrees of freedom (Nmax+1)   [default: 64]
    --NS=<Nmax>          The shell radial degrees of freedom (Nmax+1)   [default: 64]
    --NB_hi=<Nmax>       The hires-ball radial degrees of freedom (Nmax+1)
    --NS_hi=<Nmax>       The hires-shell radial degrees of freedom (Nmax+1)

    --label=<label>      A label to add to the end of the output directory

    --mesa_file=<f>      path to a .h5 file of ICCs, curated from a MESA model
    --mesa_file_hi=<f>   path to a .h5 file of ICCs, curated from a MESA model (for hires solve)
"""
import gc
import os
import sys
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
from docopt import docopt
from configparser import ConfigParser
import dedalus.public as d3
from mpi4py import MPI
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import logging
logger = logging.getLogger(__name__)

# Define smooth Heaviside functions
from scipy.special import erf
def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
    return -(one_to_zero(*args, **kwargs) - 1)

# Read in parameters and create output directory
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
Lmax = int(args['--L'])
nθ = Lmax + 1
nφ = 2*nθ
nrB = int(args['--NB'])
nrS = int(args['--NS'])
resolutionB = (nφ, nθ, nrB)
resolutionS = (nφ, nθ, nrS)
Re  = float(args['--Re'])
Pr  = 1
Pe  = Pr*Re
mesa_file  = args['--mesa_file']

if args['--NB_hi'] is not None and args['--NS_hi'] is not None:
    nrB_hi = int(args['--NB_hi'])
    nrS_hi = int(args['--NS_hi'])
    mesa_file_hi = args['--mesa_file_hi']
    resolutionB_hi = (nφ, nθ, nrB_hi)
    resolutionS_hi = (nφ, nθ, nrS_hi)
else:
    nrB_hi = nrS_hi = mesa_file_hi = None

out_dir = './' + sys.argv[0].split('.py')[0]
if args['--mesa_file'] is None:
    out_dir += '_polytrope'
out_dir += '_Re{}_Lmax{}_nr{}+{}'.format(args['--Re'], Lmax, nrB, nrS)
if nrB_hi is not None and nrS_hi is not None:
    out_dir += '_nrhi{}+{}'.format(nrB_hi, nrS_hi)
if args['--label'] is not None:
    out_dir += '_{:s}'.format(args['--label'])
logger.info('saving data to {:s}'.format(out_dir))
if MPI.COMM_WORLD.rank == 0:
    if not os.path.exists('{:s}/'.format(out_dir)):
        os.makedirs('{:s}/'.format(out_dir))

if mesa_file is not None:
    with h5py.File(mesa_file, 'r') as f:
        r_inner = f['r_inner'][()]
        r_outer = f['r_outer'][()]
        tau_s = f['tau'][()]
        tau = tau_s/(60*60*24)
        N2_mesa = f['N2_mesa'][()]
        r_mesa = f['r_mesa'][()]
        L_mesa = f['L'][()]
else:
    r_inner = 1.1
    r_outer = 1.5
    tau = 1

logger.info('r_inner: {:.2f} / r_outer: {:.2f}'.format(r_inner, r_outer))

def build_solver(resolutionB, resolutionS, r_inner, r_outer,  mesa_file, dtype=np.complex128, Re=Re):
    """
    Builds a BallShell solver by creating fields and adding equations.

    Arguments:
    ----------
        bB, bS : Bases objects
            The Ball (B) and Shell (S) bases
        b_midB, b_midS, b_top: S2 Bases objects
            S2 bases at the middle interface (for B and S) and at the top (for S)
        mesa_file : string
            string containing path to the MESA-based NCC file.
    """
    # Bases
    coords  = d3.SphericalCoordinates('φ', 'θ', 'r')
    dist    = d3.Distributor((coords,), mesh=None, dtype=dtype, comm=MPI.COMM_SELF)
    basisB  = d3.BallBasis(coords, resolutionB, radius=r_inner, dtype=dtype, dealias=(1,1,1))
    basisS  = d3.ShellBasis(coords, resolutionS, radii=(r_inner, r_outer), dtype=dtype, dealias=(1,1,1))
    radial_basisB = basisB.radial_basis
    radial_basisS = basisS.radial_basis
    b_midB = basisB.S2_basis(radius=r_inner)
    b_midS = basisS.S2_basis(radius=r_inner)
    b_topS = basisS.S2_basis(radius=r_outer)
    φB, θB, rB  = basisB.local_grids((1,1,1))
    φS, θS, rS  = basisS.local_grids((1,1,1))
    shell_ell = basisS.local_ell
    shell_m   = basisS.local_m

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

    # Load MESA NCC file or setup NCCs using polytrope
    if mesa_file is not None:
        #Set scales equal to file
        mesa_file_dealias = float(mesa_file.split('_de')[-1].split('.h5')[0])
        load_dealias_tuple = (1, 1, mesa_file_dealias)
        for field in [grad_s0B, grad_ln_ρB, grad_ln_TB, grad_TB, grad_inv_PeB, HB, ln_ρB, ln_TB, inv_PeB, ρB, TB, inv_TB]:
            field.require_scales(load_dealias_tuple)
        for field in [grad_s0S, grad_ln_ρS, grad_ln_TS, grad_TS, grad_inv_PeS, HS, ln_ρS, ln_TS, inv_PeS, ρS, TS, inv_TS]:
            field.require_scales(load_dealias_tuple)
        #Load fields
        with h5py.File(mesa_file, 'r') as f:
            grad_s0B['g']      = f['grad_s0B'][:,0,0,:][:,None,None,:]
            grad_ln_ρB['g']    = f['grad_ln_ρB'][:,0,0,:][:,None,None,:]
            grad_ln_TB['g']    = f['grad_ln_TB'][:,0,0,:][:,None,None,:]
            grad_TB['g']       = f['grad_TB'][:,0,0,:][:,None,None,:]   
            grad_inv_PeB['g']  = f['grad_inv_Pe_radB'][:,0,0,:][:,None,None,:]
            grad_s0S['g']     = f['grad_s0S'][:,0,0,:][:,None,None,:]
            grad_ln_ρS['g']   = f['grad_ln_ρS'][:,0,0,:][:,None,None,:]
            grad_ln_TS['g']   = f['grad_ln_TS'][:,0,0,:][:,None,None,:]
            grad_TS['g']      = f['grad_TS'][:,0,0,:][:,None,None,:]   
            grad_inv_PeS['g'] = f['grad_inv_Pe_radS'][:,0,0,:][:,None,None,:]
            inv_PeB['g']= f['inv_Pe_radB'][:,0,0,:][:,None,None,:]
            ln_ρB['g']      = f['ln_ρB'][:,0,0,:][:,None,None,:]  
            ln_TB['g']      = f['ln_TB'][:,0,0,:][:,None,None,:]  
            HB['g']         = f['H_effB'][:,0,0,:][:,None,None,:] 
            ρB['g']         = np.exp(f['ln_ρB'][:,0,0,:][:,None,None,:])
            TB['g']         = f['TB'][:,0,0,:][:,None,None,:]
            inv_TB['g']     = 1/TB['g']

            inv_PeS['g']  = f['inv_Pe_radS'][:,0,0,:][:,None,None,:] 
            ln_ρS['g']      = f['ln_ρS'][:,0,0,:][:,None,None,:]
            ln_TS['g']      = f['ln_TS'][:,0,0,:][:,None,None,:]
            HS['g']          = f['H_effS'][:,0,0,:][:,None,None,:]
            ρS['g']         = np.exp(f['ln_ρS'][:,0,0,:][:,None,None,:])
            TS['g']         = f['TS'][:,0,0,:][:,None,None,:]
            inv_TS['g']     = 1/TS['g']

        #Revert to scales=1
        for field in [grad_s0B, grad_ln_ρB, grad_ln_TB, grad_TB, grad_inv_PeB, HB, ln_ρB, ln_TB, inv_PeB, ρB, TB, inv_TB]:
            field.require_scales((1,1,1))
        for field in [grad_s0S, grad_ln_ρS, grad_ln_TS, grad_TS, grad_inv_PeS, HS, ln_ρS, ln_TS, inv_PeS, ρS, TS, inv_TS]:
            field.require_scales((1,1,1))
    else:
        logger.info("Using polytropic initial conditions")
        #Load and interpolate file
        from scipy.interpolate import interp1d
        with h5py.File('polytropes/poly_nOuter1.6.h5', 'r') as f:
            T_func = interp1d(f['r'][()], f['T'][()])
            ρ_func = interp1d(f['r'][()], f['ρ'][()])
            grad_s0_func = interp1d(f['r'][()], f['grad_s0'][()])
            H_func   = interp1d(f['r'][()], f['H_eff'][()])

        #Set data in ball and shell
        for r, basis_fields, basis  in zip((rB, rS), ((TB, ρB, HB, inv_TB, ln_TB, ln_ρB, inv_PeB, grad_ln_TB, grad_ln_ρB, grad_TB, grad_s0B, grad_inv_PeB), \
                                                 (TS, ρS, HS, inv_TS, ln_TS, ln_ρS, inv_PeS, grad_ln_TS, grad_ln_ρS, grad_TS, grad_s0S, grad_inv_PeS)), \
                                                 (basisB, basisS)):
            T, ρ, H, inv_T, ln_T, ln_ρ, inv_Pe, grad_ln_T, grad_ln_ρ, grad_T, grad_s0, grad_inv_Pe = basis_fields

            T['g']       = T_func(r)
            ρ['g']       = ρ_func(r)
            H['g']       = H_func(r)
            inv_T['g']   = 1/T_func(r)

            grad_ln_ρ_full = (d3.grad(ρ)/ρ).evaluate()
            grad_T_full = d3.grad(T).evaluate()
            grad_ln_T_full = (grad_T_full/T).evaluate()
            grad_s0.require_scales(1)
            grad_s0['g'][2]  = grad_s0_func(r)
            grad_ln_ρ['g']   = grad_ln_ρ_full['g'][:,0,0,:][:,None,None,:]
            grad_ln_T['g']   = grad_ln_T_full['g'][:,0,0,:][:,None,None,:]
            grad_T['g']      = grad_T_full['g'][:,0,0,:][:,None,None,:]
            grad_inv_Pe['g'] = 0
            ln_T['g']        = np.log(T_func(r))
            ln_ρ['g']        = np.log(ρ_func(r))
            inv_Pe['g']      = 1/Pe

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
    s_match_bc      = s1B(r=r_inner) - s1S(r=r_inner)
    grad_s_match_bc = d3.radial(grad_s1B(r=r_inner) - grad_s1S(r=r_inner))
    # Surface: Impenetrable, stress-free, no entropy gradient
    impenetrable = d3.radial(uS(r=r_outer))
    stress_free  = d3.angular(d3.radial(ES(r=r_outer)), index=0)
    grad_s_surface = d3.radial(grad_s1S(r=r_outer))

    # Lift operators for boundary conditions
    lift_basisB = basisB.clone_with(k=0)
    lift_basisS = basisS.clone_with(k=2)
    liftB   = lambda A: d3.LiftTau(A, lift_basisB, -1)
    liftS   = lambda A, n: d3.LiftTau(A, lift_basisS, n)
    BC_uB = liftB(tBt)
    BC_uS = liftS(tSt_bot, -1) + liftS(tSt_top, -2)
    BC_s1B = liftB(tB)
    BC_s1S = liftS(tS_bot, -1) + liftS(tS_top, -2)

    omega = dist.Field(name='ω')
    ddt = lambda A: -1j * omega * A
    problem = d3.EVP([pB, uB, pS, uS, s1B, s1S, tBt, tSt_bot, tSt_top, tB, tS_bot, tS_top], omega, namespace=locals())

    # Equations
    ### Ball momentum
    problem.add_equation("div(uB) + dot(uB, grad_ln_ρB) = 0", condition="nθ != 0")
    problem.add_equation("ddt(uB) + grad(pB) + grad_TB*s1B - (1/Re)*visc_div_stressB + BC_uB = cross(uB, curl(uB))", condition = "nθ != 0")
    ### Shell momentum
    problem.add_equation("div(uS) + dot(uS, grad_ln_ρS) = 0", condition="nθ != 0")
    problem.add_equation("ddt(uS) + grad(pS) + grad_TS*s1S - (1/Re)*visc_div_stressS + BC_uS = cross(uS, curl(uS))", condition = "nθ != 0")
    ## ell == 0 momentum
    problem.add_equation("pB = 0", condition="nθ == 0")
    problem.add_equation("uB = 0", condition="nθ == 0")
    problem.add_equation("pS = 0", condition="nθ == 0")
    problem.add_equation("uS = 0", condition="nθ == 0")
    ### Ball energy
    problem.add_equation("ddt(s1B) + dot(uB, grad_s0B) - (inv_PeB)*(lap(s1B) + dot(grad_s1B, (grad_ln_ρB + grad_ln_TB))) - dot(grad_s1B, grad_inv_PeB) + BC_s1B = - dot(uB, grad_s1B) + HB + (1/Re)*inv_TB*VHB ")
    ### Shell energy
    problem.add_equation("ddt(s1S) + dot(uS, grad_s0S) - (inv_PeS)*(lap(s1S) + dot(grad_s1S, (grad_ln_ρS + grad_ln_TS))) - dot(grad_s1S, grad_inv_PeS) + BC_s1S = - dot(uS, grad_s1S) + HS + (1/Re)*inv_TS*VHS ")

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
    solver = problem.build_solver()
    logger.info("solver built")

    for subproblem in solver.subproblems:
        M = subproblem.left_perm.T @ subproblem.M_min
        L = subproblem.left_perm.T @ subproblem.L_min
    return solver, locals()

def solve_dense(solver, ell):
    """
    Do a dense eigenvalue solve at a specified ell.
    Sort the eigenvalues and eigenvectors according to damping rate.
    """
    for subproblem in solver.subproblems:
        this_ell = subproblem.group[1]
        if this_ell != ell:
            continue
        #TODO: Output to file.
        logger.info("solving ell = {}".format(ell))
        solver.solve_dense(subproblem)

        values = solver.eigenvalues 
#        values *= -1j
        vectors = solver.eigenvectors

        #filter out nans
        cond1 = np.isfinite(values)
        values = values[cond1]
        vectors = vectors[:, cond1]

        #Only take positive frequencies
        cond2 = values.real > 0
        values = values[cond2]
        vectors = vectors[:, cond2]

#        #Sort by decay timescales
#        order = np.argsort(-values.imag)
        #Sort by real frequency magnitude
        order = np.argsort(1/values.real)
        values = values[order]
        vectors = vectors[:, order]

        #Update solver
        solver.eigenvalues = values
        solver.eigenvectors = vectors
        return solver


solver1, namespace1 = build_solver(resolutionB, resolutionS, r_inner, r_outer, mesa_file)
if nrB_hi is not None and nrS_hi is not None:
    solver2, namespace2 = build_solver(resolutionB_hi, resolutionS_hi, r_inner, r_outer, mesa_file_hi)

#def check_eigen(solver1, solver2, subsystems1, subsystems2, namespace1, namespace2, cutoff=1e-2):
#    """
#    Compare eigenvalues and eigenvectors between a hi-res and lo-res solve.
#    Only keep the solutions that match to within the specified cutoff between the two cases.
#    """
#    good_values1 = []
#    good_values2 = []
#    cutoff2 = np.sqrt(cutoff)
#
#    ρB2 = namespace2['ρB']
#    ρS2 = namespace2['ρS']
#    bB1 = namespace1['bB']
#    bS1 = namespace1['bS']
#    bB2 = namespace2['bB']
#    bS2 = namespace2['bS']
#    uB1 = namespace1['uB']
#    uS1 = namespace1['uS']
#    uB2 = namespace2['uB']
#    uS2 = namespace2['uS']
##    φB1,  θB1,  rB1  = bB1.local_grids((1, 1, 1))
#    φB1,  θB1,  rB1  = bB1.local_grids((1, 1, (NmaxB_hires)/(NmaxB)))
#    φB2,  θB2,  rB2  = bB2.local_grids((1, 1, 1))
#    φS1_0,  θS1_0,  rS1_0  = bS1.local_grids((1, 1, 1))
#    φS1,  θS1,  rS1  = bS1.local_grids((1, 1, (NmaxS_hires)/(NmaxS)))
#    φS2,  θS2,  rS2  = bS2.local_grids((1, 1, 1))
#    weight_rB2 = bB2.radial_basis.local_weights(dealias)
#    weight_rS2 = bS2.radial_basis.local_weights(dealias)*rS2**2
#
#    radial_weights_2 = np.concatenate((weight_rB2.flatten(), weight_rS2.flatten()), axis=-1)
#    ρ2 = np.concatenate((ρB2['g'][0,0,:].flatten(), ρS2['g'][0,0,:].flatten()))
#    r1 = np.concatenate((rB1.flatten(), rS1.flatten()))
#    r2 = np.concatenate((rB2.flatten(), rS2.flatten()))
#    good1 = (shell_ell1 == subsystems1[0].group[1])*(shell_m1 == subsystems1[0].group[0])
#    good2 = (shell_ell2 == subsystems2[0].group[1])*(shell_m2 == subsystems2[0].group[0])
#
#    for i, v1 in enumerate(solver1.eigenvalues):
#        for j, v2 in enumerate(solver2.eigenvalues):
#            real_goodness = np.abs(v1.real - v2.real)/np.abs(v1.real).min()
#            goodness = np.abs(v1 - v2)/np.abs(v1).min()
#            if goodness < cutoff:# or (j == 0 and (i == 2 or i == 3)):# and (np.abs(v1.imag - v2.imag)/np.abs(v1.imag)).min() < 1e-1:
#                print(v1/(2*np.pi), v2/(2*np.pi))
#                
##                print((np.abs(v1 - v2)/np.abs(v1)).min())
#                solver1.set_state(i, subsystems1[0])
#                solver2.set_state(j, subsystems2[0])
#                uB1.require_scales((1, 1, (NmaxB_hires)/(NmaxB)))
#                uS1.require_scales((1, 1, (NmaxS_hires)/(NmaxS)))
#
#                #Get eigenvectors
#                for f in [uB1, uS1, uB2, uS2]:
#                    f['c']
#                    f.towards_grid_space()
#                ef_uB1_pm = uB1.data[:,good1,:].squeeze()
#                ef_uS1_pm = uS1.data[:,good1,:].squeeze()
#                ef_uB2_pm = uB2.data[:,good2,:].squeeze()
#                ef_uS2_pm = uS2.data[:,good2,:].squeeze()
#
#                ef_u1_pm = np.concatenate((ef_uB1_pm, ef_uS1_pm), axis=-1)
#                ef_u2_pm = np.concatenate((ef_uB2_pm, ef_uS2_pm), axis=-1)
#
#                ix1 = np.argmax(np.abs(ef_u1_pm[2,:]))
#                ef_u1_pm /= ef_u1_pm[2,ix1]
#                ix1 = np.argmax(np.abs(ef_u2_pm[2,:]))
#                ef_u2_pm /= ef_u2_pm[2,ix1]
#
#                ef_u1 = np.zeros_like(ef_u1_pm)
#                ef_u2 = np.zeros_like(ef_u2_pm)
#                for u, u_pm in zip((ef_u1, ef_u2), (ef_u1_pm, ef_u2_pm)):
#                    u[0,:] = (1j/np.sqrt(2))*(u_pm[1,:] - u_pm[0,:])
#                    u[1,:] = ( 1/np.sqrt(2))*(u_pm[1,:] + u_pm[0,:])
#                    u[2,:] = u_pm[2,:]
#
#                #If mode KE is inside of the convection zone then it's a bad mode.
#                mode_KE = ρ2*np.sum(ef_u2*np.conj(ef_u2), axis=0).real/2
#                cz_KE = np.sum((mode_KE*radial_weights_2)[r2 <= 1])
#                tot_KE = np.sum((mode_KE*radial_weights_2))
##                plt.plot(r1, ef_u1[0,:].real, c='k')
##                plt.plot(r2, ef_u2[0,:].real, c='k', ls='--')
##                plt.plot(r1, ef_u1[0,:].imag, c='r')
##                plt.plot(r2, ef_u2[0,:].imag, c='r', ls='--')
##                plt.show()
#
#                cz_KE_frac = cz_KE/tot_KE
#                vector_diff = np.max(np.abs(ef_u1 - ef_u2))
#                print('vdiff', vector_diff, 'czfrac', cz_KE_frac.real)
#                if vector_diff < cutoff2 and cz_KE_frac < 0.5:
#                    good_values1.append(i)
#                    good_values2.append(j)
#
#
#    solver1.eigenvalues = solver1.eigenvalues[good_values1]
#    solver2.eigenvalues = solver2.eigenvalues[good_values2]
#    solver1.eigenvectors = solver1.eigenvectors[:, good_values1]
#    solver2.eigenvectors = solver2.eigenvectors[:, good_values2]
#    return solver1, solver2
#
#r1 = np.concatenate((rB1.flatten(), rS1.flatten()))
#radial_weights_1 = np.concatenate((weight_rB1.flatten(), weight_rS1.flatten()), axis=-1)
#def IP(velocity1, velocity2, density):
#    """ Integrate the bra-ket of two eigenfunctions of velocity. """
#    int_field = np.sum(velocity1*np.conj(velocity2), axis=0)
#    return np.sum(density*int_field*full_weight_1)
#
#def calculate_duals(velocity_list, density):
#    """
#    Calculate the dual basis of the velocity eigenvectors.
#    """
#    velocity_list = np.array(velocity_list)
#    n_modes = velocity_list.shape[0]
#    IP_matrix = np.zeros((n_modes, n_modes), dtype=np.complex128)
#    for i in range(n_modes):
#        if i % 10 == 0: logger.info("duals {}/{}".format(i, n_modes))
#        for j in range(n_modes):
#            IP_matrix[i,j] = IP(velocity_list[i], velocity_list[j], density)
#    
#    print('dual IP matrix cond: {:.3e}'.format(np.linalg.cond(IP_matrix)))
#    IP_inv = np.linalg.inv(IP_matrix)
#
#    vel_dual = np.zeros_like(velocity_list)
#    for i in range(3):
#        vel_dual[:,i,:] = np.einsum('ij,ik->kj', velocity_list[:,i,:], np.conj(IP_inv))
#
#    return vel_dual

#only solve ell = 1 one right now.
from scipy.interpolate import interp1d

for i in range(Lmax):
    ell = i + 1
    logger.info('solving lores eigenvalue with nr = ({}, {})'.format(nrB, nrS))
    solver1 = solve_dense(solver1, ell)
    subsystem1 = None
    for sbsys in solver1.subsystems:
        ss_m, ss_ell, r_couple = sbsys.group
        if ss_ell == ell and ss_m == 1:
            subsystem1 = sbsys
            break
    print(solver1.eigenvalues)


    if nrB_hi is not None and nrS_hi is not None:
        logger.info('solving hires eigenvalue with nr ({}, {})'.format(nrB_hi, nrS_hi))
        solver2 = solve_dense(solver2, ell)
        subsystems2 = []
        for subsystem in solver2.eigenvalue_subproblem.subsystems:
            ss_m, ss_ell, r_couple = subsystem.group
            if ss_ell == ell and ss_m == 1:
                subsystems2.append(subsystem)
                break
        logger.info('cleaning bad eigenvalues')
        solver1, solver2 = check_eigen(solver1, solver2, subsystems1, subsystems2, namespace1, namespace2)

#TODO: fix old post-evp processing so it works in modern-d3
#    depths = []
#    for om in solver1.eigenvalues.real:
#        Lambda = np.sqrt(ell*(ell+1))
#        kr_cm = np.sqrt(N2_mesa)*Lambda/(r_mesa* (om/tau_s))
#        v_group = (om/tau_s) / kr_cm
#        inv_Pe = np.ones_like(r_mesa) / Pe
#        inv_Pe[r_mesa/L_mesa > 1.1] = interp1d(namespace1['rS'].flatten(), namespace1['inv_PeS']['g'][0,0,:], bounds_error=False, fill_value='extrapolate')(r_mesa[r_mesa/L_mesa > 1.1]/L_mesa)
#        k_rad = (L_mesa**2 / tau_s) * inv_Pe
#        gamma_rad = k_rad * kr_cm**2
#        depth_integrand = np.gradient(r_mesa) * gamma_rad/v_group
#
#        opt_depth = 0
#        for i, rv in enumerate(r_mesa):
#            if rv/L_mesa > 1.0 and rv/L_mesa < r_outer:
#                opt_depth += depth_integrand[i]
#        depths.append(opt_depth)
#
#        
#
#    bS = namespace1['bS']
#    bB = namespace1['bB']
#    ρS = namespace1['ρS']
#    ρB = namespace1['ρB']
#    ρ  = np.concatenate((ρB['g'][0,0,:], ρS['g'][0,0,:]))
#    pS = namespace1['pS']
#    pB = namespace1['pB']
#    uS = namespace1['uS']
#    uB = namespace1['uB']
#    s1S = namespace1['s1S']
#    s1B = namespace1['s1B']
#    for f in [ρS, ρB, uB, uS, s1S, s1B]:
#        f.require_scales((1,1,1))
#    pomega_hat_B = pB - 0.5*dot(uB,uB)
#    pomega_hat_S = pS - 0.5*dot(uS,uS)
#
#    ball_avg = BallVolumeAverager(s1B)
#    shell_avg = ShellVolumeAverager(s1S)
#    s1_surf = s1S(r=r_outer)
#
#    KEB  = field.Field(dist=d, bases=(bB,), dtype=dtype)
#    KES  = field.Field(dist=d, bases=(bS,), dtype=dtype)
#
#    integ_energies = np.zeros_like(   solver1.eigenvalues, dtype=np.float64) 
#    s1_amplitudes = np.zeros_like(solver1.eigenvalues, dtype=np.float64)  
#    velocity_eigenfunctions = []
#    entropy_eigenfunctions = []
#    wave_flux_eigenfunctions = []
#
#    subsystem = subsystem1
#    print('using subsystem ', subsystem.group, ' for eigenvectors')
#    for i, e in enumerate(solver1.eigenvalues):
#        good = (shell_ell1 == ell)*(shell_m1 == subsystem.group[0])
#        solver1.set_state(i, subsystem)
#
#        #Get eigenvectors
#        pomB = pomega_hat_B.evaluate()
#        pomS = pomega_hat_S.evaluate()
#        for f in [uB, uS, s1B, s1S, pomB, pomS]:
#            f['c']
#            f.towards_grid_space()
#        ef_uB_pm = uB.data[:,good,:].squeeze()
#        ef_uS_pm = uS.data[:,good,:].squeeze()
#        ef_s1B = s1B.data[good,:].squeeze()
#        ef_s1S = s1S.data[good,:].squeeze()
#        ef_pomB = s1B.data[good,:].squeeze()
#        ef_pomS = s1S.data[good,:].squeeze()
#
#        #normalize & store eigenvectors
#        shift = np.max((np.abs(ef_uB_pm[2,:]).max(), np.abs(ef_uS_pm[2,:]).max()))
#        for data in [ef_uB_pm, ef_uS_pm, ef_s1B, ef_s1S, ef_pomB, ef_pomS]:
#            data /= shift
#
#        ef_uB = np.zeros_like(ef_uB_pm)
#        ef_uS = np.zeros_like(ef_uS_pm)
#        for u, u_pm in zip((ef_uB, ef_uS), (ef_uB_pm, ef_uS_pm)):
#            u[0,:] = (1j/np.sqrt(2))*(u_pm[1,:] - u_pm[0,:])
#            u[1,:] = ( 1/np.sqrt(2))*(u_pm[1,:] + u_pm[0,:])
#            u[2,:] = u_pm[2,:]
#
#        full_ef_u = np.concatenate((ef_uB, ef_uS), axis=-1)
#        full_ef_s1 = np.concatenate((ef_s1B, ef_s1S), axis=-1)
#        velocity_eigenfunctions.append(full_ef_u)
#        entropy_eigenfunctions.append(full_ef_s1)
#
#        #Wave flux
#        wave_fluxB = (ρB['g'][0,0,:]*ef_uB[2,:]*np.conj(ef_pomB)).squeeze()
#        wave_fluxS = (ρS['g'][0,0,:]*ef_uS[2,:]*np.conj(ef_pomS)).squeeze()
#        wave_flux_eig = np.concatenate((wave_fluxB, wave_fluxS), axis=-1)
#        wave_flux_eigenfunctions.append(wave_flux_eig)
#
#        #Kinetic energy
#        KES['g'] = (ρS['g'][0,0,:]*np.sum(ef_uS*np.conj(ef_uS), axis=0)).real/2
#        KEB['g'] = (ρB['g'][0,0,:]*np.sum(ef_uB*np.conj(ef_uB), axis=0)).real/2
#        integ_energy = ball_avg(KEB)[0]*ball_avg.volume + shell_avg(KES)[0]*shell_avg.volume
#        integ_energies[i] = integ_energy.real / 2 #factor of 2 accounts for spherical harmonic integration
##        KES['g'] = (ρS['g'][0,0,:]*np.sum(uS['g']*np.conj(uS['g']), axis=0)).real/2/shift**2
##        KEB['g'] = (ρB['g'][0,0,:]*np.sum(uB['g']*np.conj(uB['g']), axis=0)).real/2/shift**2
##        old_integ_energy = ball_avg(KEB)[0]*ball_avg.volume + shell_avg(KES)[0]*shell_avg.volume
#
#        #Surface entropy perturbations
##        s1_surf_value = np.sqrt(np.sum(np.abs(s1_surf.evaluate()['g']/shift)**2*weight1)/np.sum(weight1))
##        old_s1_amplitudes = s1_surf_value.real
#        s1S['g'] = 0
#        s1S['c']
#        s1S['g'] = ef_s1S
#        s1_surf_vals = s1_surf.evaluate()['g'] / np.sqrt(2) #sqrt(2) accounts for spherical harmonic integration
#        s1_amplitudes[i] = np.abs(s1_surf_vals.max())
##        print(subsystem.group, s1_amplitudes[i], old_s1_amplitudes, integ_energies[i], old_integ_energy, s1_amplitudes[i]/old_s1_amplitudes, integ_energies[i]/old_integ_energy)
#
#    with h5py.File('{:s}/ell{:03d}_eigenvalues.h5'.format(out_dir, ell), 'w') as f:
#        f['good_evalues'] = solver1.eigenvalues
#        f['good_omegas']  = solver1.eigenvalues.real
#        f['good_evalues_inv_day'] = solver1.eigenvalues/tau
#        f['good_omegas_inv_day']  = solver1.eigenvalues.real/tau
#        f['s1_amplitudes']  = s1_amplitudes
#        f['integ_energies'] = integ_energies
#        f['wave_flux_eigenfunctions'] = np.array(wave_flux_eigenfunctions)
#        f['velocity_eigenfunctions'] = np.array(velocity_eigenfunctions)
#        f['entropy_eigenfunctions'] = np.array(entropy_eigenfunctions)
#        f['rB'] = namespace1['rB']
#        f['rS'] = namespace1['rS']
#        f['ρB'] = namespace1['ρB']['g']
#        f['ρS'] = namespace1['ρS']['g']
#        f['depths'] = np.array(depths)
#
#    velocity_duals = calculate_duals(velocity_eigenfunctions, ρ)
#    with h5py.File('{:s}/duals_ell{:03d}_eigenvalues.h5'.format(out_dir, ell), 'w') as f:
#        f['good_evalues'] = solver1.eigenvalues
#        f['good_omegas']  = solver1.eigenvalues.real
#        f['good_evalues_inv_day'] = solver1.eigenvalues/tau
#        f['good_omegas_inv_day']  = solver1.eigenvalues.real/tau
#        f['s1_amplitudes']  = s1_amplitudes
#        f['integ_energies'] = integ_energies
#        f['wave_flux_eigenfunctions'] = np.array(wave_flux_eigenfunctions)
#        f['velocity_eigenfunctions'] = np.array(velocity_eigenfunctions)
#        f['entropy_eigenfunctions'] = np.array(entropy_eigenfunctions)
#        f['velocity_duals'] = velocity_duals
#        f['rB'] = namespace1['rB']
#        f['rS'] = namespace1['rS']
#        f['ρB'] = namespace1['ρB']['g']
#        f['ρS'] = namespace1['ρS']['g']
#        f['depths'] = np.array(depths)
#
#
#    gc.collect()
