import os, gc, time, sys
import numpy as np
import dedalus.public as d3
import logging
import h5py
logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt

from d3_stars.simulations.evp_functions import solve_dense, combine_eigvecs, scipy_sparse_eigs, matrix_info

def calculate_duals(vel_ef_lists, basis, dist):
    """
    Calculate the dual basis of the velocity eigenvectors.
    """
    vel_ef_lists = list(vel_ef_lists)
    work_field = dist.Field(bases=basis)
    int_field = d3.integ(work_field)

    def IP(velocity1, velocity2):
        """ Integrate the bra-ket of two eigenfunctions of velocity. """
        velocity1 = velocity1
        velocity2 = velocity2
        work_field['g'] = np.sum(np.conj(velocity1)*velocity2, axis=0)
        return int_field.evaluate()['g'].min()


    n_modes = len(vel_ef_lists)
    IP_matrix = np.zeros((n_modes, n_modes), dtype=np.complex128)
    for i in range(n_modes):
        vel_ef_lists[i] = np.array(vel_ef_lists[i])
        if i % 1 == 0: logger.info("duals {}/{}".format(i, n_modes))
        for j in range(n_modes):
            IP_matrix[i,j] = IP(vel_ef_lists[i], vel_ef_lists[j])

    logger.info('dual IP matrix cond: {:.3e}'.format(np.linalg.cond(IP_matrix)))
    IP_inv = np.linalg.inv(IP_matrix)

    #TODO:fix
    total_nr = vel_ef_lists[0].shape[-1]
    vel_duals = np.zeros((n_modes, 3, total_nr), dtype=np.complex128)
    vel_efs = np.zeros((n_modes, 3, total_nr), dtype=np.complex128)
    for n in range(n_modes):
        vel_efs[n, :, :] = vel_ef_lists[n]
    for j in range(3): #velocity dimensions
        vel_duals[:,j,:] = np.einsum('ij,jk->ik', np.conj(IP_inv), vel_efs[:,j,:])

    #Check that velocity duals were evaluated correctly
    IP_check = np.zeros_like(IP_matrix)
    for i in range(n_modes):
        for j in range(n_modes):
            IP_check[i,j] = IP(vel_duals[i], vel_ef_lists[j])
    I_matrix = np.zeros_like(IP_matrix)
    for i in range(I_matrix.shape[0]):
        I_matrix[i,i] = 1

    if np.allclose(I_matrix.real, IP_check.real, rtol=1e-5, atol=1e-5):
        logger.info('velocity duals properly calculated')
    else:
        logger.info('something went wrong in velocity dual calc; IP_check info:')
        for i in range(n_modes):
            logger.info('{}, {}'.format(np.sum(IP_check[i,:].real), IP_check[i,i].real))
        raise ValueError("Something went wrong in calculating the dual basis.")

    return vel_duals

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
    if vector:
        ef = field.data[:,ell_m_bool,:].squeeze()
    else:
        ef = field.data[ell_m_bool,:].squeeze()

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
            solver_lo.set_state(i, subsystem_lo)

            ef_u1 = combine_eigvecs(solver_lo, solver_lo.problem.namespace['u'], good, scales=(1,1,3/2))
            ef_u2 = combine_eigvecs(solver, solver.problem.namespace['u'], good)

#            plt.plot(hires_r, ef_u1[2,:])
#            plt.plot(hires_r, ef_u2[2,:])
#            plt.show()

            #If mode KE is inside of the convection zone then it's a bad mode.
            mode_KE = np.sum(ef_u2*np.conj(ef_u2), axis=0).real/2
            cz_KE = np.sum((mode_KE*4*np.pi*hires_r**2*dr)[hires_r <= r_cz])
            tot_KE = np.sum((mode_KE*4*np.pi*hires_r**2*dr))
            cz_KE_frac = cz_KE/tot_KE
            vector_diff = np.max(np.abs(ef_u1 - ef_u2))

#            print(vector_diff)
            if vector_diff < np.sqrt(cutoff):
                logger.info('good evalue w/ vdiff {} and czfrac {}'.format(vector_diff, cz_KE_frac.real))
                if cz_KE_frac.real > 0.5:
                    logger.info('evalue is in the CZ, skipping')
                elif cz_KE_frac.real < 1e-3:
                    logger.info('evalue is spurious, skipping')
                else:
                    good_values.append(i)

    solver_lo.eigenvalues  = solver_lo.eigenvalues[good_values]
    solver_lo.eigenvectors = solver_lo.eigenvectors[:, good_values]



from scipy.special import erf
def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
    return -(one_to_zero(*args, **kwargs) - 1)

# Allow restarting via command line
restart = (len(sys.argv) > 1 and sys.argv[1] == '--restart')

# Parameters
Nphi, Ntheta, Nr = 4, 16, 128
Rayleigh = 1e8
Prandtl = 1
dealias = 1
S=100
dtype = np.complex128
mesh = None
kappa = (Rayleigh * Prandtl)**(-1/2)
nu = (Rayleigh / Prandtl)**(-1/2)

r_transition=1
radius=2

resolutions = [(4, Ntheta, Nr), (4, Ntheta, int(Nr*3/2))]

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
    grad_T0_source['g'][2] = S * r * zero_to_one(r, r_transition, width=0.1)

    ns['lift'] = lift = lambda A: d3.Lift(A, basis, -1)
    ns['strain_rate'] = strain_rate = d3.grad(u) + d3.trans(d3.grad(u))
    ns['shear_stress'] = shear_stress = d3.angular(d3.radial(strain_rate(r=radius), index=1))


    ns['damper'] = damper = dist.Field(bases=basis.radial_basis)
    damper.change_scales(basis.dealias)
    damper['g'] = zero_to_one(r, radius*0.925, width=radius*0.025)

    N2 = (r_vec@grad_T0_source).evaluate()
    ns['f_bv_max'] = f_bv_max = np.sqrt(N2['g'].max())/(2*np.pi)

    # Problem
    ns['omega'] = omega = dist.Field(name='omega')
    ns['dt'] = dt = lambda A: -1j * omega * A
    problem = d3.EVP([p, u, T, tau_p, tau_u, tau_T], namespace=ns, eigenvalue=omega)
    problem.add_equation("div(u) + tau_p = 0")
    problem.add_equation("dt(u) + u*damper*f_bv_max - nu*lap(u) - r_vec*T + grad(p) + lift(tau_u) = 0")
    problem.add_equation("dt(T) + u@grad_T0_source - kappa*lap(T) + lift(tau_T) = 0")
    problem.add_equation("shear_stress = 0")  # Stress free
    problem.add_equation("radial(u(r=radius)) = 0")  # No penetration
    problem.add_equation("radial(grad(T)(r=radius)) = 0")
    problem.add_equation("integ(p) = 0")  # Pressure gauge

    # Solver
    solver = problem.build_solver()
    solvers.append(solver)

for ell in range(1,Ntheta):
    for sn, solver in enumerate(solvers):
        if sn == 0:
            solve_dense(solver, ell)
        else:
            solver.eigenvalues = solvers[0].eigenvalues
            solver.eigenvectors = solvers[0].eigenvectors
            check_eigen(solver, solvers[0])
    
    solver = solvers[0]
    
    locals().update(solver.problem.namespace)

    #Calculate 'optical depths' of each mode.
    shape = list(T['c'].shape[:2])
    good = np.zeros(shape, bool)
    for i in range(shape[0]):
        for j in range(shape[1]):
            grid_space = (False,False)
            elements = (np.array((i,)),np.array((j,)))
            m, this_ell = basis.sphere_basis.elements_to_groups(grid_space, elements)
            if this_ell == ell and m == 1:
                good[i,j] = True

    integ_energy_op = 0.5*d3.dot(u, u)
    T_surf = T(r=radius)

    integ_energies = np.zeros_like(solver.eigenvalues, dtype=np.float64) 
    T_amplitudes = np.zeros_like(solver.eigenvalues, dtype=np.float64)  
    velocity_eigenfunctions = []
    temperature_eigenfunctions = []
    wave_flux_eigenfunctions = []

    for sbsys in solver.subsystems:
        ss_m, ss_ell, r_couple = sbsys.group
        if ss_ell == ell and ss_m == 1:
            subsystem = sbsys
            break


    for i, e in enumerate(solver.eigenvalues):
        solver.set_state(i, subsystem)

        ef_u = combine_eigvecs(solver, solver.problem.namespace['u'], good, shift=False)
        ef_T = combine_eigvecs(solver, solver.problem.namespace['T'], good, shift=False)
        ef_p = combine_eigvecs(solver, solver.problem.namespace['p'], good, shift=False)

        #normalize & store eigenvectors
        shift = np.max(np.abs(ef_u[2,:]))
        for data in [ef_u, ef_T, ef_p]:
            data[:] /= shift

        velocity_eigenfunctions.append(ef_u)
        temperature_eigenfunctions.append(ef_T)

        #Wave flux
        wave_flux = ef_u[2,:]*np.conj(ef_p).squeeze()
        wave_flux_eigenfunctions.append(wave_flux)

    #            #Kinetic energy
        integ_energy = integ_energy_op.evaluate()
        integ_energies[i] = integ_energy['g'].min().real / 2 #factor of 2 accounts for spherical harmonic integration (we're treating the field like an ell = 0 one)

        #Surface temperature perturbations
        T['g'] = ef_T
        T_surf_vals = T_surf.evaluate()['g'] / np.sqrt(2) #sqrt(2) accounts for spherical harmonic integration
        T_amplitudes[i] = np.abs(T_surf_vals.max())

    out_dir = 'eigenvalues'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with h5py.File('{:s}/ell{:03d}_eigenvalues.h5'.format(out_dir, ell), 'w') as f:
        f['good_evalues'] = solver.eigenvalues
        f['good_omegas']  = solver.eigenvalues.real
        f['T_amplitudes']  = T_amplitudes
        f['integ_energies'] = integ_energies
        f['wave_flux_eigenfunctions'] = np.array(wave_flux_eigenfunctions)
        f['velocity_eigenfunctions'] = np.array(velocity_eigenfunctions)
        f['temperature_eigenfunctions'] = np.array(temperature_eigenfunctions)
        f['r'] = r

    velocity_duals = calculate_duals(velocity_eigenfunctions, basis, dist)
    with h5py.File('{:s}/ell{:03d}_eigenvalues.h5'.format(out_dir, ell), 'r') as f:
        with h5py.File('{:s}/duals_ell{:03d}_eigenvalues.h5'.format(out_dir, ell), 'w') as df:
            for k in f.keys():
                df.create_dataset(k, data=f[k])
            df['velocity_duals'] = velocity_duals

    gc.collect()
