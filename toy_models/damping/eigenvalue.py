"""
Dedalus script simulating internally-heated Boussinesq convection in the ball.
This script demonstrates soving an initial value problem in the ball. It can be
ran serially or in parallel, and uses the built-in analysis framework to save
data snapshots to HDF5 files. The `plot_ball.py` script can be used to produce
plots from the saved data. The simulation should take roughly 15 cpu-minutes to run.

The strength of gravity is proportional to radius, as for a constant density ball.
The problem is non-dimensionalized using the ball radius and freefall time, so
the resulting thermal diffusivity and viscosity are related to the Prandtl
and Rayleigh numbers as:

    kappa = (Rayleigh * Prandtl)**(-1/2)
    nu = (Rayleigh / Prandtl)**(-1/2)

We use stress-free boundary conditions, and maintain a constant flux on the outer
boundary. The convection is driven by the internal heating term with a conductive
equilibrium of T(r) = 1 - r**2.

For incompressible hydro in the ball, we need one tau term each for the velocity
and temperature. Here we choose to lift them to the original (k=0) basis.

The simulation will run to t=10, about the time for the first convective plumes
to hit the top boundary. After running this initial simulation, you can restart
the simulation with the command line option '--restart'.

To run, restart, and plot using e.g. 4 processes:
    $ mpiexec -n 4 python3 internally_heated_convection.py
    $ mpiexec -n 4 python3 internally_heated_convection.py --restart
    $ mpiexec -n 4 python3 plot_ball.py slices/*.h5
"""

import time
import sys
import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

from d3_stars.simulations.evp_functions import solve_dense, combine_eigvecs, calculate_duals, scipy_sparse_eigs, matrix_info

def combine_eigvecs(solver, field, ell_m_bool, scales=None, shift=True):
    if len(field.tensorsig) == 1:
        vector = True
    else:
        vector = False

    if scales is not None:
        field.change_scales(scales)
    else:
        field.change_scales((1,1,1))

    field['c']
    field.towards_grid_space()
    ef = field.data[:,ell_m_bool,:].squeeze()

    if shift:
        if vector:
            #Getting argmax then indexing ensures we're not off by a negative sign flip.
            ix = np.argmax(np.abs(ef[2,:]))
            divisor = ef[2,:][ix]
        else:
            ix = np.argmax(np.abs(ef))
            divisor = ef[ix]

        ef /= divisor

    if vector:
        #go from plus/minus to theta/phi
        ef_u = np.zeros_like(ef)
        ef_u[0,:] = (1j/np.sqrt(2))*(ef[1,:] - ef[0,:])
        ef_u[1,:] = ( 1/np.sqrt(2))*(ef[1,:] + ef[0,:])
        ef_u[2,:] = ef[2,:]
        ef[:] = ef_u[:]

    return ef

def solve_sparse(eigenvalues, solver, A, B):
        """
        Perform targeted sparse eigenvalue search for selected pencil.
        Parameters
        ----------
        pencil : pencil object
            Pencil for which to solve the EVP
        N : int
            Number of eigenmodes to solver for.  Note: the dense method may
            be more efficient for finding large numbers of eigenmodes.
        target : complex
            Target eigenvalue for search.
        rebuild_coeffs : bool, optional
            Flag to rebuild cached coefficient matrices (default: False)
        Other keyword options passed to scipy.sparse.linalg.eigs.
        """
        N = 1
        full_evals = []
        full_evecs = []
        start_time = time.time()
        for i, target in enumerate(eigenvalues):
            evalue, evec = scipy_sparse_eigs(A=A, B=B, N=N, target=target, matsolver=solver.matsolver)
            full_evals.append(evalue.ravel())
            full_evecs.append(evec.ravel())
        end_time = time.time()
        logger.info('sparse solve done in {:.2f} sec'.format(end_time - start_time))
        solver.eigenvalues = np.array(full_evals)
        solver.eigenvectors = np.swapaxes(np.array(full_evecs), 0, 1)
#        logger.info('shapes: {} / {}'.format(solver.eigenvalues.shape, solver.eigenvectors.shape))
        return solver


def check_eigen(solver, solver_lo, cutoff=1e-2, r_cz=1):
    """
    Compare eigenvalues and eigenvectors between a hi-res and lo-res solve.
    Only keep the solutions that match to within the specified cutoff between the two cases.
    """
    for sbsys in solver.subsystems:
        ss_m, ss_ell, r_couple = sbsys.group
        if ss_ell == ell and ss_m == 1:
            subsystem = sbsys
            break
    for sbsys in solver_lo.subsystems:
        ss_m, ss_ell, r_couple = sbsys.group
        if ss_ell == ell and ss_m == 1:
            subsystem_lo = sbsys
            break

    for subproblem in solver.subproblems:
        this_ell = subproblem.group[1]
        if this_ell != ell:
            continue
        sp = subproblem
        matrix_info(sp)
        A = (sp.L_min @ sp.pre_right).A
        B = - (sp.M_min @ sp.pre_right).A
        solver.eigenvalue_subproblem = sp
        sparse_args = (solver, A, B)
        break

    logger.info('finding good eigenvalues with cutoff {}'.format(cutoff))
    a_field = solver.problem.namespace['T']
    shape = list(a_field['c'].shape[:2])
    good = np.zeros(shape, bool)
    for i in range(shape[0]):
        for j in range(shape[1]):
            grid_space = (False,False)
            elements = (np.array((i,)),np.array((j,)))
            m, this_ell = a_field.domain.bases[0].sphere_basis.elements_to_groups(grid_space, elements)
            if this_ell == ell and m == 1:
                good[i,j] = True

    hires_r = solver.problem.namespace['r'].ravel()
    dr = np.gradient(hires_r)

    good_values = []
    for i, v1 in enumerate(solver_lo.eigenvalues):
        if np.abs(v1.real) < 0.1*np.abs(v1.imag):
            logger.info('skipping eigenvalue {}; damps very quickly'.format(v1))
            continue
        solver = solve_sparse([v1,], *sparse_args)
        v2 = solver.eigenvalues[0]
        real_goodness = np.abs(v1.real - v2.real)/np.abs(v1.real).min()
        goodness = np.abs(v1 - v2)/np.abs(v1).min()

        print('goodness: {:.3e}'.format(goodness.max()))

        if goodness < cutoff:
            
            solver.set_state(0, subsystem)
            solver_lo.set_state(0, subsystem_lo)

            ef_u1 = combine_eigvecs(solver_lo, solver_lo.problem.namespace['u'], good, scales=(1,1,3/2))
            ef_u2 = combine_eigvecs(solver, solver.problem.namespace['u'], good)

            #If mode KE is inside of the convection zone then it's a bad mode.
            mode_KE = np.sum(ef_u2*np.conj(ef_u2), axis=0).real/2
            cz_KE = np.sum((mode_KE*4*np.pi*hires_r**2*dr)[hires_r <= r_cz])
            tot_KE = np.sum((mode_KE*4*np.pi*hires_r**2*dr))
            cz_KE_frac = cz_KE/tot_KE
            vector_diff = np.max(np.abs(ef_u1 - ef_u2))

            print(vector_diff)
            if vector_diff < np.sqrt(cutoff):
                logger.info('good evalue w/ vdiff {} and czfrac {}'.format(vector_diff, cz_KE_frac.real))
                if cz_KE_frac.real > 0.5:
                    logger.info('evalue is in the CZ, skipping')
                elif cz_KE_frac.real < 1e-3:
                    logger.info('evalue is spurious, skipping')
                else:
                    good_values.append(i)

    solver.eigenvalues  = solver.eigenvalues[good_values]
    solver.eigenvectors = solver.eigenvectors[:, good_values]



from scipy.special import erf
def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
    return -(one_to_zero(*args, **kwargs) - 1)

# Allow restarting via command line
restart = (len(sys.argv) > 1 and sys.argv[1] == '--restart')

# Parameters
Nphi, Ntheta, Nr = 128, 64, 96
Rayleigh = 1e6
Prandtl = 1
dealias = 1
S=100
dtype = np.complex128
mesh = None
kappa = (Rayleigh * Prandtl)**(-1/2)
nu = (Rayleigh / Prandtl)**(-1/2)

r_transition=1
radius=2

resolutions = [(4, 4, Nr), (4, 4, int(Nr*3/2))]

# Bases
coords = d3.SphericalCoordinates('phi', 'theta', 'r')
dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)

solvers = []

for rn, res in enumerate(resolutions):
    ns = dict()
    ns.update(locals())
    ns['basis'] = basis = d3.BallBasis(coords, shape=res, radius=radius, dealias=dealias, dtype=dtype)
    ns['S2_basis'] = S2_basis = basis.S2_basis()

    # Fields
    ns['u'] = u = dist.VectorField(coords, name='u',bases=basis)
    ns['p'] = p = dist.Field(name='p', bases=basis)
    ns['T'] = T = dist.Field(name='T', bases=basis)
    ns['tau_p'] = tau_p = dist.Field(name='tau_p')
    ns['tau_u'] = tau_u = dist.VectorField(coords, name='tau u', bases=S2_basis)
    ns['tau_T'] = tau_T = dist.Field(name='tau T', bases=S2_basis)

    # Substitutions
    ns['phi'], ns['theta'], ns['r'] = phi, theta, r = dist.local_grids(basis)
    ns['r_vec'] = r_vec = dist.VectorField(coords, bases=basis.radial_basis)
    r_vec['g'][2] = r

    ns['grad_T0_source'] = grad_T0_source = dist.VectorField(coords, bases=basis.radial_basis)
    grad_T0_source['g'][2] = S * r# * zero_to_one(r, r_transition, width=0.1)

    ns['lift'] = lift = lambda A: d3.Lift(A, basis, -1)
    ns['strain_rate'] = strain_rate = d3.grad(u) + d3.trans(d3.grad(u))
    ns['shear_stress'] = shear_stress = d3.angular(d3.radial(strain_rate(r=radius), index=1))


    # Problem
    ns['omega'] = omega = dist.Field(name='omega')
    ns['dt'] = dt = lambda A: -1j * omega * A
    problem = d3.EVP([p, u, T, tau_p, tau_u, tau_T], namespace=ns, eigenvalue=omega)
    problem.add_equation("div(u) + tau_p = 0")
    problem.add_equation("dt(u) - nu*lap(u) - r_vec*T + grad(p) + lift(tau_u) = 0")
    problem.add_equation("dt(T) + u@grad_T0_source - kappa*lap(T) + lift(tau_T) = 0")
    problem.add_equation("shear_stress = 0")  # Stress free
    problem.add_equation("radial(u(r=radius)) = 0")  # No penetration
    problem.add_equation("radial(grad(T)(r=radius)) = 0")
    problem.add_equation("integ(p) = 0")  # Pressure gauge

    # Solver
    solver = problem.build_solver()
    solvers.append(solver)

for ell in range(1,3):
    for sn, solver in enumerate(solvers):
        if sn == 0:
            solve_dense(solver, ell)
        else:
            solver.eigenvalues = solvers[0].eigenvalues
            solver.eigenvectors = solvers[0].eigenvectors
            check_eigen(solver, solvers[0])

