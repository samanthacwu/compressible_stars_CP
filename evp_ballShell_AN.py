"""
d3 script for anelastic convection in a massive star's core CZ.

A config file can be provided which overwrites any number of the options below, but command line flags can also be used if preferred.

Usage:
    ballShell_AN.py [options]
    ballShell_AN.py <config> [options]

Options:
    --Re=<Re>            The Reynolds number of the numerical diffusivities [default: 5e1]
    --Pr=<Prandtl>       The Prandtl number  of the numerical diffusivities [default: 1]
    --L=<Lmax>           The value of Lmax   [default: 1]
    --NB=<Nmax>          The ball value of Nmax   [default: 63]
    --NS=<Nmax>          The shell value of Nmax   [default: 63]
    --NB_hires=<Nmax>    The ball value of Nmax
    --NS_hires=<Nmax>    The shell value of Nmax

    --wall_hours=<t>     The number of hours to run for [default: 24]
    --buoy_end_time=<t>  Number of buoyancy times to run [default: 1e5]
    --safety=<s>         Timestep CFL safety factor [default: 0.4]

    --label=<label>      A label to add to the end of the output directory

    --mesa_file=<f>      path to a .h5 file of ICCs, curated from a MESA model
    --mesa_file_hires=<f>      path to a .h5 file of ICCs, curated from a MESA model
    --restart=<chk_f>    path to a checkpoint file to restart from

    --boost=<b>          Inverse Mach number boost squared [default: 1]

    --freq_power=<p>     Power exponent of wave frequency for Shiode comparison [default: -6.5]
"""
import gc
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
import matplotlib.pyplot as plt
from scipy.linalg import eig
from scipy.interpolate import interp1d

from d3_outputs.extra_ops    import BallVolumeAverager, ShellVolumeAverager, EquatorSlicer, PhiAverager, PhiThetaAverager, OutputRadialInterpolate, GridSlicer
from d3_outputs.writing      import d3FileHandler

import logging
logger = logging.getLogger(__name__)

from dedalus.tools.config import config
config['linear algebra']['MATRIX_FACTORIZER'] = 'SuperLUNaturalFactorizedTranspose'

#TODO: Use locals() to make this script (esp build_solver) cleaner


from scipy.special import erf
def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
    return -(one_to_zero(*args, **kwargs) - 1)


def build_solver(bB, bS, b_midB, b_midS, b_top, mesa_file):
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
    
    LiftTauB   = lambda A: operators.LiftTau(A, bB, -1)
    LiftTauS   = lambda A, n: operators.LiftTau(A, bS, n)

    φB,  θB,  rB  = bB.local_grids((dealias, dealias, dealias))
    φS,  θS,  rS  = bS.local_grids((dealias, dealias, dealias))
    # Fields
    uB    = field.Field(dist=d, bases=(bB,), tensorsig=(c,), dtype=dtype)
    pB    = field.Field(dist=d, bases=(bB,), dtype=dtype)
    s1B   = field.Field(dist=d, bases=(bB,), dtype=dtype)
    uS    = field.Field(dist=d, bases=(bS,), tensorsig=(c,), dtype=dtype)
    pS    = field.Field(dist=d, bases=(bS,), dtype=dtype)
    s1S   = field.Field(dist=d, bases=(bS,), dtype=dtype)

    tB     = field.Field(dist=d, bases=(b_midB,), dtype=dtype)
    tBt    = field.Field(dist=d, bases=(b_midB,), dtype=dtype,   tensorsig=(c,))
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
    grad_ln_TB    = field.Field(dist=d, bases=(bB.radial_basis,), tensorsig=(c,), dtype=dtype)
    ln_ρB         = field.Field(dist=d, bases=(bB.radial_basis,), dtype=dtype)
    ln_TB         = field.Field(dist=d, bases=(bB.radial_basis,), dtype=dtype)
    grad_TB        = field.Field(dist=d, bases=(bB.radial_basis,), tensorsig=(c,), dtype=dtype)
    inv_PeB   = field.Field(dist=d, bases=(bB.radial_basis,), dtype=dtype)
    grad_inv_PeB  = field.Field(dist=d, bases=(bB.radial_basis,), tensorsig=(c,), dtype=dtype)
    grad_ln_ρS    = field.Field(dist=d, bases=(bS.radial_basis,), tensorsig=(c,), dtype=dtype)
    grad_ln_TS    = field.Field(dist=d, bases=(bS.radial_basis,), tensorsig=(c,), dtype=dtype)
    ln_ρS         = field.Field(dist=d, bases=(bS.radial_basis,), dtype=dtype)
    ln_TS         = field.Field(dist=d, bases=(bS.radial_basis,), dtype=dtype)
    grad_TS       = field.Field(dist=d, bases=(bS.radial_basis,), tensorsig=(c,), dtype=dtype)
    inv_PeS   = field.Field(dist=d, bases=(bS.radial_basis,), dtype=dtype)
    grad_inv_PeS  = field.Field(dist=d, bases=(bS.radial_basis,), tensorsig=(c,), dtype=dtype)

    grad_s0B      = field.Field(dist=d, bases=(bB.radial_basis,), tensorsig=(c,), dtype=dtype)
    grad_s0S      = field.Field(dist=d, bases=(bS.radial_basis,), tensorsig=(c,), dtype=dtype)



    # Get local slices
    slicesB     = GridSlicer(pB)
    slicesS     = GridSlicer(pS)

    grads0_boost = float(args['--boost'])#1/100
    logger.info("Boost: {}".format(grads0_boost))

    if mesa_file is not None:
        with h5py.File(mesa_file, 'r') as f:
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
            ρB['g']         = np.expand_dims(np.expand_dims(np.exp(f['ln_ρB'][:,:,slicesB[-1]]), axis=0), axis=0)
            TB['g']         = np.expand_dims(np.expand_dims(f['TB'][:,:,slicesB[-1]], axis=0), axis=0)

            inv_PeS['g']= f['inv_Pe_radS'][:,:,slicesS[-1]]
            ln_ρS['g']      = f['ln_ρS'][:,:,slicesS[-1]]
            ln_TS['g']      = f['ln_TS'][:,:,slicesS[-1]]
            ρS['g']         = np.expand_dims(np.expand_dims(np.exp(f['ln_ρS'][:,:,slicesS[-1]]), axis=0), axis=0)
            TS['g']         = np.expand_dims(np.expand_dims(f['TS'][:,:,slicesS[-1]], axis=0), axis=0)


#            grad_inv_PeB['g'] = 0
#            grad_inv_PeS['g'] = 0
#            inv_PeB['g'] = 1/Pe
#            inv_PeS['g'] = 1/Pe


            grad_s0B['g'] *= grads0_boost
            grad_s0S['g'] *= grads0_boost

            max_dt = f['max_dt'][()] / np.sqrt(grads0_boost)
            t_buoy = 1
    else:
        raise NotImplementedError()

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

    omega = field.Field(name='omega', dist=d, dtype=dtype)
    ddt       = lambda A: -1j * omega * A

    grads1B = grad(s1B)
    grads1S = grad(s1S)
    grads1B.store_last = True
    grads1S.store_last = True

    # Problem
    these_locals = locals()
    def eq_eval(eq_str, namespace=these_locals):
        for k, i in namespace.items():
            try:
                locals()[k] = i 
            except:
                print('failed {}'.format(k))
        exprs = []
        for expr in split_equation(eq_str):
            exprs.append(eval(expr))
        return exprs

    problem = problems.EVP([pB, uB, pS, uS, s1B, s1S, tBt, tSt_bot, tSt_top, tB, tS_bot, tS_top], omega)

    ### Ball momentum
    problem.add_equation(eq_eval("div(uB) + dot(uB, grad_ln_ρB) = 0"), condition="nθ != 0")
    problem.add_equation(eq_eval("ddt(uB) + grad(pB) + grad_TB*s1B - (1/Re)*momentum_viscous_termsB + LiftTauB(tBt)  = 0"), condition = "nθ != 0")
    ### Shell momentum
    problem.add_equation(eq_eval("div(uS) + dot(uS, grad_ln_ρS) = 0"), condition="nθ != 0")
    problem.add_equation(eq_eval("ddt(uS) + grad(pS) + grad_TS*s1S - (1/Re)*momentum_viscous_termsS + LiftTauS(tSt_bot, -1) + LiftTauS(tSt_top, -2) = 0"), condition = "nθ != 0")
    ## ell == 0 momentum
    problem.add_equation(eq_eval("pB = 0"), condition="nθ == 0")
    problem.add_equation(eq_eval("uB = 0"), condition="nθ == 0")
    problem.add_equation(eq_eval("pS = 0"), condition="nθ == 0")
    problem.add_equation(eq_eval("uS = 0"), condition="nθ == 0")

    ### Ball energy
    problem.add_equation(eq_eval("ddt(s1B) + dot(uB, grad_s0B) - (inv_PeB)*(lap(s1B) + dot(grads1B, (grad_ln_ρB + grad_ln_TB))) - dot(grads1B, grad_inv_PeB) + LiftTauB(tB) = 0 "))
    ### Shell energy
    problem.add_equation(eq_eval("ddt(s1S) + dot(uS, grad_s0S) - (inv_PeS)*(lap(s1S) + dot(grads1S, (grad_ln_ρS + grad_ln_TS))) - dot(grads1S, grad_inv_PeS) + LiftTauS(tS_bot, -1) + LiftTauS(tS_top, -2) = 0 "))


    #Velocity BCs ell != 0
    problem.add_equation(eq_eval("uB(r=r_inner) - uS(r=r_inner)    = 0"),            condition="nθ != 0")
    problem.add_equation(eq_eval("u_r_bcB_mid - u_r_bcS_mid    = 0"),            condition="nθ != 0")
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
    print(problem.dtype)
    solver = solvers.EigenvalueSolver(problem)
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

def check_eigen(solver1, solver2, subsystems1, subsystems2, namespace1, namespace2, cutoff=1e-2):
    """
    Compare eigenvalues and eigenvectors between a hi-res and lo-res solve.
    Only keep the solutions that match to within the specified cutoff between the two cases.
    """
    good_values1 = []
    good_values2 = []
    cutoff2 = np.sqrt(cutoff)

    ρB2 = namespace2['ρB']
    ρS2 = namespace2['ρS']
    bB1 = namespace1['bB']
    bS1 = namespace1['bS']
    bB2 = namespace2['bB']
    bS2 = namespace2['bS']
    uB1 = namespace1['uB']
    uS1 = namespace1['uS']
    uB2 = namespace2['uB']
    uS2 = namespace2['uS']
#    φB1,  θB1,  rB1  = bB1.local_grids((1, 1, 1))
    φB1,  θB1,  rB1  = bB1.local_grids((1, 1, (NmaxB_hires+1)/(NmaxB+1)))
    φB2,  θB2,  rB2  = bB2.local_grids((1, 1, 1))
    φS1_0,  θS1_0,  rS1_0  = bS1.local_grids((1, 1, 1))
    φS1,  θS1,  rS1  = bS1.local_grids((1, 1, (NmaxS_hires+1)/(NmaxS+1)))
    φS2,  θS2,  rS2  = bS2.local_grids((1, 1, 1))
    weight_rB2 = bB2.radial_basis.local_weights(dealias)
    weight_rS2 = bS2.radial_basis.local_weights(dealias)*rS2**2

    radial_weights_2 = np.concatenate((weight_rB2.flatten(), weight_rS2.flatten()), axis=-1)
    ρ2 = np.concatenate((ρB2['g'][0,0,:].flatten(), ρS2['g'][0,0,:].flatten()))
    r1 = np.concatenate((rB1.flatten(), rS1.flatten()))
    r2 = np.concatenate((rB2.flatten(), rS2.flatten()))
    good1 = (shell_ell1 == subsystems1[0].group[1])*(shell_m1 == subsystems1[0].group[0])
    good2 = (shell_ell2 == subsystems2[0].group[1])*(shell_m2 == subsystems2[0].group[0])

    for i, v1 in enumerate(solver1.eigenvalues):
        for j, v2 in enumerate(solver2.eigenvalues):
            real_goodness = np.abs(v1.real - v2.real)/np.abs(v1.real).min()
            goodness = np.abs(v1 - v2)/np.abs(v1).min()
            if goodness < cutoff:# or (j == 0 and (i == 2 or i == 3)):# and (np.abs(v1.imag - v2.imag)/np.abs(v1.imag)).min() < 1e-1:
                print(v1/(2*np.pi), v2/(2*np.pi))
                
#                print((np.abs(v1 - v2)/np.abs(v1)).min())
                solver1.set_state(i, subsystems1[0])
                solver2.set_state(j, subsystems2[0])
                uB1.require_scales((1, 1, (NmaxB_hires+1)/(NmaxB+1)))
                uS1.require_scales((1, 1, (NmaxS_hires+1)/(NmaxS+1)))

                #Get eigenvectors
                for f in [uB1, uS1, uB2, uS2]:
                    f['c']
                    f.towards_grid_space()
                ef_uB1_pm = uB1.data[:,good1,:].squeeze()
                ef_uS1_pm = uS1.data[:,good1,:].squeeze()
                ef_uB2_pm = uB2.data[:,good2,:].squeeze()
                ef_uS2_pm = uS2.data[:,good2,:].squeeze()

                ef_u1_pm = np.concatenate((ef_uB1_pm, ef_uS1_pm), axis=-1)
                ef_u2_pm = np.concatenate((ef_uB2_pm, ef_uS2_pm), axis=-1)

                ix1 = np.argmax(np.abs(ef_u1_pm[2,:]))
                ef_u1_pm /= ef_u1_pm[2,ix1]
                ix1 = np.argmax(np.abs(ef_u2_pm[2,:]))
                ef_u2_pm /= ef_u2_pm[2,ix1]

                ef_u1 = np.zeros_like(ef_u1_pm)
                ef_u2 = np.zeros_like(ef_u2_pm)
                for u, u_pm in zip((ef_u1, ef_u2), (ef_u1_pm, ef_u2_pm)):
                    u[0,:] = (1j/np.sqrt(2))*(u_pm[1,:] - u_pm[0,:])
                    u[1,:] = ( 1/np.sqrt(2))*(u_pm[1,:] + u_pm[0,:])
                    u[2,:] = u_pm[2,:]

                #If mode KE is inside of the convection zone then it's a bad mode.
                mode_KE = ρ2*np.sum(ef_u2*np.conj(ef_u2), axis=0).real/2
                cz_KE = np.sum((mode_KE*radial_weights_2)[r2 <= 1])
                tot_KE = np.sum((mode_KE*radial_weights_2))
#                plt.plot(r1, ef_u1[0,:].real, c='k')
#                plt.plot(r2, ef_u2[0,:].real, c='k', ls='--')
#                plt.plot(r1, ef_u1[0,:].imag, c='r')
#                plt.plot(r2, ef_u2[0,:].imag, c='r', ls='--')
#                plt.show()

                cz_KE_frac = cz_KE/tot_KE
                vector_diff = np.max(np.abs(ef_u1 - ef_u2))
                print('vdiff', vector_diff, 'czfrac', cz_KE_frac.real)
                if vector_diff < cutoff2 and cz_KE_frac < 0.5:
                    good_values1.append(i)
                    good_values2.append(j)


    solver1.eigenvalues = solver1.eigenvalues[good_values1]
    solver2.eigenvalues = solver2.eigenvalues[good_values2]
    solver1.eigenvectors = solver1.eigenvectors[:, good_values1]
    solver2.eigenvectors = solver2.eigenvectors[:, good_values2]
    return solver1, solver2

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
mesa_file1  = args['--mesa_file']
L_dealias = N_dealias = dealias = 1

if args['--NB_hires'] is not None and args['--NS_hires'] is not None:
    NmaxB_hires = int(args['--NB_hires'])
    NmaxS_hires = int(args['--NS_hires'])
    mesa_file_hires = args['--mesa_file_hires']
else:
    NmaxB_hires = None
    NmaxS_hires = None
    mesa_file_hires = None


out_dir = './' + sys.argv[0].split('.py')[0]
if args['--mesa_file'] is None:
    out_dir += '_polytrope'
out_dir += '_Re{}_{}x{}_{}x{}'.format(args['--Re'], args['--L'], args['--NB'], args['--L'], args['--NS'])
if args['--label'] is not None:
    out_dir += '_{:s}'.format(args['--label'])
logger.info('saving data to {:s}'.format(out_dir))
if MPI.COMM_WORLD.rank == 0:
    if not os.path.exists('{:s}/'.format(out_dir)):
        os.makedirs('{:s}/'.format(out_dir))


dtype = np.complex128

Re  = float(args['--Re'])
Pr  = 1
Pe  = Pr*Re


if mesa_file1 is not None:
    with h5py.File(mesa_file1, 'r') as f:
        r_inner = f['r_inner'][()]
        r_outer = f['r_outer'][()]
else:
    r_inner = 1.1
    r_outer = 1.5
logger.info('r_inner: {:.2f} / r_outer: {:.2f}'.format(r_inner, r_outer))

# Bases
c    = coords.SphericalCoordinates('φ', 'θ', 'r')
d    = distributor.Distributor((c,), mesh=None)
bB1   = basis.BallBasis(c, (2*(Lmax+1), Lmax+1, NmaxB+1), radius=r_inner, dtype=dtype)
bS1   = basis.SphericalShellBasis(c, (2*(Lmax+1), Lmax+1, NmaxS+1), radii=(r_inner, r_outer), dtype=dtype)
b_midB1 = bB1.S2_basis(radius=r_inner)
b_midS1 = bS1.S2_basis(radius=r_inner)
b_top1 = bS1.S2_basis(radius=r_outer)
φB1,  θB1,  rB1  = bB1.local_grids((dealias, dealias, dealias))
φS1,  θS1,  rS1  = bS1.local_grids((dealias, dealias, dealias))
shell_ell1 = bS1.local_ell
shell_m1 = bS1.local_m

weight_φ1 = np.gradient(φS1.flatten()).reshape(φS1.shape)
weight_θ1 = bS1.local_colatitude_weights(dealias)
weight_rB1 = bB1.radial_basis.local_weights(dealias)
weight_rS1 = bS1.radial_basis.local_weights(dealias)*rS1**2
weight1 = weight_θ1 * weight_φ1
volume1 = np.sum(weight1)

if NmaxB_hires is not None:
    bB2   = basis.BallBasis(c, (2*(Lmax+1), Lmax+1, NmaxB_hires+1), radius=r_inner, dtype=dtype)
    bS2   = basis.SphericalShellBasis(c, (2*(Lmax+1), Lmax+1, NmaxS_hires+1), radii=(r_inner, r_outer), dtype=dtype)
    b_midB2 = bB2.S2_basis(radius=r_inner)
    b_midS2 = bS2.S2_basis(radius=r_inner)
    b_top2 = bS2.S2_basis(radius=r_outer)
    φB2,  θB2,  rB2  = bB2.local_grids((dealias, dealias, dealias))
    φS2,  θS2,  rS2  = bS2.local_grids((dealias, dealias, dealias))
    shell_ell2 = bS2.local_ell
    shell_m2 = bS2.local_m

    weight_φ2 = np.gradient(φS2.flatten()).reshape(φS2.shape)
    weight_θ2 = bS2.local_colatitude_weights(dealias)
    weight_rB2 = bB2.radial_basis.local_weights(dealias)
    weight_rS2 = bS2.radial_basis.local_weights(dealias)*rS2**2
    weight2 = weight_θ2 * weight_φ2
    volume2 = np.sum(weight2)

#Operators
div       = lambda A: operators.Divergence(A, index=0)
lap       = lambda A: operators.Laplacian(A, c)
grad      = lambda A: operators.Gradient(A, c)
dot       = lambda A, B: arithmetic.DotProduct(A, B)
curl      = lambda A: operators.Curl(A)
cross     = lambda A, B: arithmetic.CrossProduct(A, B)
trace     = lambda A: operators.Trace(A)
transpose = lambda A: operators.TransposeComponents(A)
radComp   = lambda A: operators.RadialComponent(A)
angComp   = lambda A, index=1: operators.AngularComponent(A, index=index)

r1 = np.concatenate((rB1.flatten(), rS1.flatten()))
radial_weights_1 = np.concatenate((weight_rB1.flatten(), weight_rS1.flatten()), axis=-1)
def IP(velocity1, velocity2):
    """ Integrate the bra-ket of two eigenfunctions of velocity. """
    int_field = np.sum(velocity1*np.conj(velocity2), axis=0)
    return np.sum(int_field*radial_weights_1)/np.sum(radial_weights_1)

def calculate_duals(velocity_list):
    """
    Calculate the dual basis of the velocity eigenvectors.
    """
    velocity_list = np.array(velocity_list)
    n_modes = velocity_list.shape[0]
    IP_matrix = np.zeros((n_modes, n_modes), dtype=np.complex128)
    for i in range(n_modes):
        if i % 10 == 0: logger.info("duals {}/{}".format(i, n_modes))
        for j in range(n_modes):
            IP_matrix[i,j] = IP(velocity_list[i], velocity_list[j])
    
    print('dual IP matrix cond: {:.3e}'.format(np.linalg.cond(IP_matrix)))
    IP_inv = np.linalg.inv(IP_matrix)

    vel_dual = np.zeros_like(velocity_list)
    for i in range(3):
        vel_dual[:,i,:] = np.einsum('ij,ik->kj', velocity_list[:,i,:], np.conj(IP_inv))

    return vel_dual

if mesa_file1 is not None:
    with h5py.File(mesa_file1, 'r') as f:
        tau_s = f['tau'][()]
        tau = tau_s/(60*60*24)
        N2_mesa = f['N2_mesa'][()]
        r_mesa = f['r_mesa'][()]
        L_mesa = f['L'][()]
else:
    tau = 1
logger.info('using tau = {} days'.format(tau))
#only solve ell = 1 one right now.
from scipy.interpolate import interp1d


solver1, namespace1 = build_solver(bB1, bS1, b_midB1, b_midS1, b_top1, mesa_file1)
if NmaxB_hires is not None:
    solver2, namespace2 = build_solver(bB2, bS2, b_midB2, b_midS2, b_top2, mesa_file_hires)
for i in range(Lmax):
    ell = i + 1
    logger.info('solving lores eigenvalue with NmaxB {}'.format(NmaxB))
    solver1 = solve_dense(solver1, ell)
#    print(solver1.eigenvalues)
    subsystems1 = []
    for subsystem in solver1.subsystems:
        ss_m, ss_ell, r_couple = subsystem.group
        if ss_ell == ell and ss_m == 1:
            subsystems1.append(subsystem)
        if ss_ell == 0 and ss_m == 0:
            subsystem1_0 = subsystem

    if NmaxB_hires is not None:
        logger.info('solving hires eigenvalue with NmaxB {}'.format(NmaxB_hires))
        solver2 = solve_dense(solver2, ell)
#        print(solver2.eigenvalues)
        subsystems2 = []
        for subsystem in solver2.eigenvalue_subproblem.subsystems:
            ss_m, ss_ell, r_couple = subsystem.group
            if ss_ell == ell and ss_m == 1:
                subsystems2.append(subsystem)
                break
        logger.info('cleaning bad eigenvalues')
        solver1, solver2 = check_eigen(solver1, solver2, subsystems1, subsystems2, namespace1, namespace2)
    print(solver1.eigenvalues.real/tau/(2*np.pi))
    print(solver2.eigenvalues.real/tau/(2*np.pi))

    depths = []
    for om in solver1.eigenvalues.real:
        Lambda = np.sqrt(ell*(ell+1))
        kr_cm = np.sqrt(N2_mesa)*Lambda/(r_mesa* (om/tau_s))
        v_group = (om/tau_s) / kr_cm
        inv_Pe = np.ones_like(r_mesa) / Pe
        inv_Pe[r_mesa/L_mesa > 1.1] = interp1d(namespace1['rS'].flatten(), namespace1['inv_PeS']['g'][0,0,:], bounds_error=False, fill_value='extrapolate')(r_mesa[r_mesa/L_mesa > 1.1]/L_mesa)
        k_rad = (L_mesa**2 / tau_s) * inv_Pe
        gamma_rad = k_rad * kr_cm**2
        depth_integrand = np.gradient(r_mesa) * gamma_rad/v_group

        opt_depth = 0
        for i, rv in enumerate(r_mesa):
            if rv/L_mesa > 1.0 and rv/L_mesa < r_outer:
                opt_depth += depth_integrand[i]
        depths.append(opt_depth)
    print(depths)

        

    bS = namespace1['bS']
    bB = namespace1['bB']
    ρS = namespace1['ρS']
    ρB = namespace1['ρB']
    pS = namespace1['pS']
    pB = namespace1['pB']
    uS = namespace1['uS']
    uB = namespace1['uB']
    s1S = namespace1['s1S']
    s1B = namespace1['s1B']
    for f in [ρS, ρB, uB, uS, s1S, s1B]:
        f.require_scales((1,1,1))
    pomega_hat_B = pB - 0.5*dot(uB,uB)
    pomega_hat_S = pS - 0.5*dot(uS,uS)

    ball_avg = BallVolumeAverager(s1B)
    shell_avg = ShellVolumeAverager(s1S)
    s1_surf = s1S(r=r_outer)

    KEB  = field.Field(dist=d, bases=(bB,), dtype=dtype)
    KES  = field.Field(dist=d, bases=(bS,), dtype=dtype)

    integ_energies = np.zeros_like(   solver1.eigenvalues, dtype=np.float64) 
    s1_amplitudes = np.zeros_like(solver1.eigenvalues, dtype=np.float64)  
    velocity_eigenfunctions = []
    entropy_eigenfunctions = []
    wave_flux_eigenfunctions = []

    subsystem = subsystems1[0]
    print('using subsystem ', subsystem.group, ' for eigenvectors')
    for i, e in enumerate(solver1.eigenvalues):
        good = (shell_ell1 == ell)*(shell_m1 == subsystem.group[0])
        solver1.set_state(i, subsystem)

        #Get eigenvectors
        pomB = pomega_hat_B.evaluate()
        pomS = pomega_hat_S.evaluate()
        for f in [uB, uS, s1B, s1S, pomB, pomS]:
            f['c']
            f.towards_grid_space()
        ef_uB_pm = uB.data[:,good,:].squeeze()
        ef_uS_pm = uS.data[:,good,:].squeeze()
        ef_s1B = s1B.data[good,:].squeeze()
        ef_s1S = s1S.data[good,:].squeeze()
        ef_pomB = s1B.data[good,:].squeeze()
        ef_pomS = s1S.data[good,:].squeeze()

        #normalize & store eigenvectors
        shift = np.max((np.abs(ef_uB_pm[2,:]).max(), np.abs(ef_uS_pm[2,:]).max()))
        for data in [ef_uB_pm, ef_uS_pm, ef_s1B, ef_s1S, ef_pomB, ef_pomS]:
            data /= shift

        ef_uB = np.zeros_like(ef_uB_pm)
        ef_uS = np.zeros_like(ef_uS_pm)
        for u, u_pm in zip((ef_uB, ef_uS), (ef_uB_pm, ef_uS_pm)):
            u[0,:] = (1j/np.sqrt(2))*(u_pm[1,:] - u_pm[0,:])
            u[1,:] = ( 1/np.sqrt(2))*(u_pm[1,:] + u_pm[0,:])
            u[2,:] = u_pm[2,:]

        full_ef_u = np.concatenate((ef_uB, ef_uS), axis=-1)
        full_ef_s1 = np.concatenate((ef_s1B, ef_s1S), axis=-1)
        velocity_eigenfunctions.append(full_ef_u)
        entropy_eigenfunctions.append(full_ef_s1)

        #Wave flux
        wave_fluxB = (ρB['g'][0,0,:]*ef_uB[2,:]*np.conj(ef_pomB)).squeeze()
        wave_fluxS = (ρS['g'][0,0,:]*ef_uS[2,:]*np.conj(ef_pomS)).squeeze()
        wave_flux_eig = np.concatenate((wave_fluxB, wave_fluxS), axis=-1)
        wave_flux_eigenfunctions.append(wave_flux_eig)

        #Kinetic energy
        KES['g'] = (ρS['g'][0,0,:]*np.sum(ef_uS*np.conj(ef_uS), axis=0)).real/2
        KEB['g'] = (ρB['g'][0,0,:]*np.sum(ef_uB*np.conj(ef_uB), axis=0)).real/2
        integ_energy = ball_avg(KEB)[0]*ball_avg.volume + shell_avg(KES)[0]*shell_avg.volume
        integ_energies[i] = integ_energy.real / 2 #factor of 2 accounts for spherical harmonic integration
#        KES['g'] = (ρS['g'][0,0,:]*np.sum(uS['g']*np.conj(uS['g']), axis=0)).real/2/shift**2
#        KEB['g'] = (ρB['g'][0,0,:]*np.sum(uB['g']*np.conj(uB['g']), axis=0)).real/2/shift**2
#        old_integ_energy = ball_avg(KEB)[0]*ball_avg.volume + shell_avg(KES)[0]*shell_avg.volume

        #Surface entropy perturbations
#        s1_surf_value = np.sqrt(np.sum(np.abs(s1_surf.evaluate()['g']/shift)**2*weight1)/np.sum(weight1))
#        old_s1_amplitudes = s1_surf_value.real
        s1S['g'] = 0
        s1S['c']
        s1S['g'] = ef_s1S
        s1_surf_vals = s1_surf.evaluate()['g'] / np.sqrt(2) #sqrt(2) accounts for spherical harmonic integration
        s1_amplitudes[i] = np.abs(s1_surf_vals.max())
#        print(subsystem.group, s1_amplitudes[i], old_s1_amplitudes, integ_energies[i], old_integ_energy, s1_amplitudes[i]/old_s1_amplitudes, integ_energies[i]/old_integ_energy)

    with h5py.File('{:s}/ell{:03d}_eigenvalues.h5'.format(out_dir, ell), 'w') as f:
        f['good_evalues'] = solver1.eigenvalues/tau
        f['good_omegas']  = solver1.eigenvalues.real/tau
        f['s1_amplitudes']  = s1_amplitudes
        f['integ_energies'] = integ_energies
        f['wave_flux_eigenfunctions'] = np.array(wave_flux_eigenfunctions)
        f['velocity_eigenfunctions'] = np.array(velocity_eigenfunctions)
        f['entropy_eigenfunctions'] = np.array(entropy_eigenfunctions)
        f['rB'] = namespace1['rB']
        f['rS'] = namespace1['rS']
        f['ρB'] = namespace1['ρB']['g']
        f['ρS'] = namespace1['ρS']['g']
        f['depths'] = np.array(depths)

    velocity_duals = calculate_duals(velocity_eigenfunctions)
    with h5py.File('{:s}/duals_ell{:03d}_eigenvalues.h5'.format(out_dir, ell), 'w') as f:
        f['good_evalues'] = solver1.eigenvalues/tau
        f['good_omegas']  = solver1.eigenvalues.real/tau
        f['s1_amplitudes']  = s1_amplitudes
        f['integ_energies'] = integ_energies
        f['wave_flux_eigenfunctions'] = np.array(wave_flux_eigenfunctions)
        f['velocity_eigenfunctions'] = np.array(velocity_eigenfunctions)
        f['entropy_eigenfunctions'] = np.array(entropy_eigenfunctions)
        f['velocity_duals'] = velocity_duals
        f['rB'] = namespace1['rB']
        f['rS'] = namespace1['rS']
        f['ρB'] = namespace1['ρB']['g']
        f['ρS'] = namespace1['ρS']['g']
        f['depths'] = np.array(depths)


    gc.collect()
