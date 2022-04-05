import time
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

from .anelastic_functions import make_bases, make_fields, fill_structure, get_anelastic_variables, set_anelastic_problem

from scipy.sparse import linalg as spla
from scipy import sparse
def scipy_sparse_eigs(A, B, N, target, matsolver, **kw):
    """
    Perform targeted eigenmode search using the scipy/ARPACK sparse solver
    for the reformulated generalized eigenvalue problem
        A.x = λ B.x  ==>  (A - σB)^I B.x = (1/(λ-σ)) x
    for eigenvalues λ near the target σ.
    Parameters
    ----------
    A, B : scipy sparse matrices
        Sparse matrices for generalized eigenvalue problem
    N : int
        Number of eigenmodes to return
    target : complex
        Target σ for eigenvalue search
    matsolver : matrix solver class
        Class implementing solve method for solving sparse systems.
    Other keyword options passed to scipy.sparse.linalg.eigs.

    Taken from dedalus' d3-s2-ncc branch on apr 1, 2022
    """
    # Build sparse linear operator representing (A - σB)^I B = C^I B = D
    C = sparse.csr_matrix(A - target * B)
    solver = matsolver(C)
    def matvec(x):
        return solver.solve(B.dot(x))
    D = spla.LinearOperator(dtype=A.dtype, shape=A.shape, matvec=matvec)
    # Solve using scipy sparse algorithm
    evals, evecs = spla.eigs(D, k=N, which='LM', sigma=None, **kw)
    # Rectify eigenvalues
    evals = 1 / evals + target
    return evals, evecs


def matrix_info(subproblem):
    if not os.path.exists('evp_matrices'):
        os.mkdir('evp_matrices')

    sp = subproblem
    ell = subproblem.group[1]
    AL = (sp.pre_left.T @ sp.L_min).A
    BL = - (sp.pre_left.T @ sp.M_min).A
    plt.imshow(np.log10(np.abs(AL)))
    plt.colorbar()
    plt.savefig("evp_matrices/ell_A_%03i.png" %ell, dpi=600)
    plt.clf()
    plt.imshow(np.log10(np.abs(BL)))
    plt.colorbar()
    plt.savefig("evp_matrices/ell_B_%03i.png" %ell, dpi=600)
    plt.clf()

    A = (sp.L_min @ sp.pre_right).A
    B = - (sp.M_min @ sp.pre_right).A
    plt.imshow(np.log10(np.abs(A)))
    plt.colorbar()
    plt.savefig("evp_matrices/cond_ell_A_%03i.png" %ell, dpi=600)
    plt.clf()
    plt.imshow(np.log10(np.abs(B)))
    plt.colorbar()
    plt.savefig("evp_matrices/cond_ell_B_%03i.png" %ell, dpi=600)
    plt.clf()

    condition_number_A = np.linalg.cond(A)
    condition_number_B = np.linalg.cond(B)
    condition_number_ML = np.linalg.cond(((sp.L_min + 0.5*sp.M_min) @ sp.pre_right).A)
    rank_A = np.linalg.matrix_rank(A)
    rank_B = np.linalg.matrix_rank(B)
    rank_ML = np.linalg.matrix_rank(((sp.L_min + 0.5*sp.M_min) @ sp.pre_right).A)
    logger.info("condition numbers A / B = {:.3e} / {:.3e}".format(condition_number_A, condition_number_B))
    logger.info("rank A = {:.3e} / {:.3e}, rank B = {:.3e} / {:.3e}".format(rank_A, np.max(A.shape), rank_B, np.max(B.shape)))
    logger.info("condition number M + L= {:.3e}".format(condition_number_ML))
    logger.info("rank M + L = {:.3e} / {:.3e}".format(rank_ML, np.max(((sp.L_min + 0.5*sp.M_min) @ sp.pre_right).A.shape)))


def solve_dense(solver, ell):
    """
    Do a dense eigenvalue solve at a specified ell.
    Sort the eigenvalues and eigenvectors according to damping rate.
    """
    for subproblem in solver.subproblems:
        this_ell = subproblem.group[1]
        if this_ell != ell:
            continue

        sp = subproblem
        matrix_info(sp)
        ell = subproblem.group[1]
        logger.info("dense solving ell = {}".format(ell))
        start_time = time.time()
        solver.solve_dense(subproblem)
        end_time = time.time()
        logger.info('dense solve done in {:.2f} sec'.format(end_time - start_time))

        values = solver.eigenvalues 
        vectors = solver.eigenvectors

        #filter out nans
        cond1 = np.isfinite(values)
        values = values[cond1]
        vectors = vectors[:, cond1]

        #Only take positive frequencies
        cond2 = values.real > 0
        values = values[cond2]
        vectors = vectors[:, cond2]

        #Sort by real frequency magnitude
        order = np.argsort(1/values.real)
        values = values[order]
        vectors = vectors[:, order]

        #Get rid of purely diffusive modes
        goodvals = np.abs(values.real/values.imag) > 1e-5
        values = values[goodvals]
        vectors = vectors[:, goodvals]

        #Update solver
        solver.eigenvalues = values
        solver.eigenvectors = vectors
        return solver

def solve_sparse(solver, ell, eigenvalues):
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

    """
    Do a dense eigenvalue solve at a specified ell.
    Sort the eigenvalues and eigenvectors according to damping rate.
    """
    for subproblem in solver.subproblems:
        this_ell = subproblem.group[1]
        if this_ell != ell:
            continue

        ell = subproblem.group[1]
        sp = subproblem
        matrix_info(sp)

        A = (sp.L_min @ sp.pre_right).A
        B = - (sp.M_min @ sp.pre_right).A
        logger.info("sparse solving ell = {}".format(ell))

        N = 1
        A = (sp.L_min @ sp.pre_right).A
        B = - (sp.M_min @ sp.pre_right).A

        full_evals = []
        full_evecs = []
        start_time = time.time()
        for i, target in enumerate(eigenvalues):
            evalue, evec = scipy_sparse_eigs(A=A, B=B, N=N, target=target, matsolver=solver.matsolver)
            full_evals.append(evalue.ravel())
            full_evecs.append(evec.ravel())
            print('sparse hi: {} / target: {}'.format(evalue, target))
        end_time = time.time()
        logger.info('sparse solve done in {:.2f} sec'.format(end_time - start_time))
        solver.eigenvalues = np.array(full_evals)
        solver.eigenvectors = np.swapaxes(np.array(full_evecs), 0, 1)
        solver.eigenvalue_subproblem = sp
        logger.info('shapes: {} / {}'.format(solver.eigenvalues.shape, solver.eigenvectors.shape))

        return solver

def combine_eigvecs(field, ell_m_bool, bases, namespace, scales=None, shift=True):
    fields = []
    ef_base = []
    vector = False
    for i, bn in enumerate(bases.keys()):
        fields.append(namespace['{}_{}'.format(field, bn)])
        this_field = fields[-1]


        if len(this_field.tensorsig) == 1:
            vector = True

        if scales is not None:
            this_field.change_scales(scales[i])
        else:
            this_field.change_scales((1,1,1))
        this_field['c']
        this_field.towards_grid_space()

        if vector:
            ef_base.append(this_field.data[:,ell_m_bool,:].squeeze())
        else:
            ef_base.append(this_field.data[ell_m_bool,:].squeeze())

    ef = np.concatenate(ef_base, axis=-1)

    if shift:
        if vector:
            #Getting argmax then indexing ensures we're not off by a negative sign flip.
            ix = np.argmax(np.abs(ef[2,:]))
            divisor = ef[2,:][ix]
        else:
            ix = np.argmax(np.abs(ef))
            divisor = ef[ix]

        ef /= divisor
        for i in range(len(bases.keys())):
            ef_base[i] /= divisor

    if vector:
        #go from plus/minus to theta/phi
        ef_u = np.zeros_like(ef)
        ef_u[0,:] = (1j/np.sqrt(2))*(ef[1,:] - ef[0,:])
        ef_u[1,:] = ( 1/np.sqrt(2))*(ef[1,:] + ef[0,:])
        ef_u[2,:] = ef[2,:]
        ef[:] = ef_u[:]

    return ef, ef_base



def check_eigen(solver1, solver2, bases1, bases2, namespace1, namespace2, ell, r_cz=1, cutoff=1e-2):
    """
    Compare eigenvalues and eigenvectors between a hi-res and lo-res solve.
    Only keep the solutions that match to within the specified cutoff between the two cases.
    """
    good_values1 = []
    good_values2 = []

    scales = []
    rho2 = []
    r1 = []
    r2 = []
    for i, bn in enumerate(bases1.keys()):
        scales.append((1, 1, namespace2['resolutions_hi'][i][-1]/namespace1['resolutions'][i][-1]))
        rho2.append(namespace2['rho_{}'.format(bn)]['g'][0,0,:])
        r1.append(namespace1['r_{}'.format(bn)].flatten())
        r2.append(namespace2['r_{}'.format(bn)].flatten())

    rho2 = np.concatenate(rho2)
    r1 = np.concatenate(r1)
    r2 = np.concatenate(r2)

    for sbsys in solver1.subsystems:
        ss_m, ss_ell, r_couple = sbsys.group
        if ss_ell == ell and ss_m == 1:
            subsystem1 = sbsys
            break
    for sbsys in solver2.subsystems:
        ss_m, ss_ell, r_couple = sbsys.group
        if ss_ell == ell and ss_m == 1:
            subsystem2 = sbsys
            break

    print('finding good eigenvalues with cutoff {}'.format(cutoff))
    shape = list(namespace1['s1_B']['c'].shape[:2])
    good = np.zeros(shape, bool)
    for i in range(shape[0]):
        for j in range(shape[1]):
            grid_space = (False,False)
            elements = (np.array((i,)),np.array((j,)))
            m, this_ell = bases1['B'].sphere_basis.elements_to_groups(grid_space, elements)
            if this_ell == ell and m == 1:
                good[i,j] = True

    rough_dr2 = np.gradient(r2, edge_order=2)

    for i, v1 in enumerate(solver1.eigenvalues):
        if np.abs(v1.real) < 3*np.abs(v1.imag):
            print('skipping eigenvalue {}; damps very quickly'.format(v1))
            continue
        for j, v2 in enumerate(solver2.eigenvalues):
            real_goodness = np.abs(v1.real - v2.real)/np.abs(v1.real).min()
            goodness = np.abs(v1 - v2)/np.abs(v1).min()
            if goodness < cutoff:# or (j == 0 and (i == 2 or i == 3)):# and (np.abs(v1.imag - v2.imag)/np.abs(v1.imag)).min() < 1e-1:
                
                solver1.set_state(i, subsystem1)
                print(solver1.eigenvectors.shape, solver2.eigenvectors.shape)
                solver2.set_state(j, subsystem2)

                ef_u1, ef_u1_pieces = combine_eigvecs('u', good, bases1, namespace1, scales=scales)
                ef_u2, ef_u2_pieces = combine_eigvecs('u', good, bases2, namespace2)

                #If mode KE is inside of the convection zone then it's a bad mode.
                mode_KE = rho2*np.sum(ef_u2*np.conj(ef_u2), axis=0).real/2
                cz_KE = np.sum((mode_KE*4*np.pi*r2**2*rough_dr2)[r2 <= r_cz])
                tot_KE = np.sum((mode_KE*4*np.pi*r2**2*rough_dr2))
                cz_KE_frac = cz_KE/tot_KE
                vector_diff = np.max(np.abs(ef_u1 - ef_u2))
                print(v1/(2*np.pi), v2/(2*np.pi), vector_diff)
#                if vector_diff < np.sqrt(cutoff) and cz_KE_frac < 0.5:
                plt.figure()
                plt.plot(r2, ef_u1[2,:])
                plt.plot(r2, ef_u2[2,:])
                plt.title(v1)
                plt.show()

                if vector_diff < np.sqrt(cutoff):
                    print('good evalue w/ vdiff', vector_diff, 'czfrac', cz_KE_frac.real)
                    if cz_KE_frac.real > 0.5:
                        print('evalue is in the CZ, skipping')
                    elif cz_KE_frac.real < 1e-3:
                        print('evalue is spurious, skipping')
                    else:
                        good_values1.append(i)
                        good_values2.append(j)


    solver1.eigenvalues = solver1.eigenvalues[good_values1]
    solver2.eigenvalues = solver2.eigenvalues[good_values2]
    solver1.eigenvectors = solver1.eigenvectors[:, good_values1]
    solver2.eigenvectors = solver2.eigenvectors[:, good_values2]
    return solver1, solver2

def calculate_duals(vel_ef_lists, bases, namespace):
    """
    Calculate the dual basis of the velocity eigenvectors.
    """
    work_fields = []
    vel_ef_lists = list(vel_ef_lists)
    int_field = None
    for i, bn in enumerate(bases.keys()):
        work_fields.append(namespace['dist'].Field(bases=bases[bn]))
        if int_field is None:
            int_field = d3.integ(namespace['rho_{}'.format(bn)]*work_fields[-1])
        else:
            int_field += d3.integ(namespace['rho_{}'.format(bn)]*work_fields[-1])

    def IP(velocity_list1, velocity_list2):
        """ Integrate the bra-ket of two eigenfunctions of velocity. """
        for i, bn in enumerate(bases.keys()):
            velocity1 = velocity_list1[i]
            velocity2 = velocity_list2[i]
            work_fields[i]['g'] = np.sum(np.conj(velocity1)*velocity2, axis=0)
        return int_field.evaluate()['g'].min()


    n_modes = len(vel_ef_lists)
    IP_matrix = np.zeros((n_modes, n_modes), dtype=np.complex128)
    for i in range(n_modes):
        for j, bn in enumerate(bases.keys()):
            vel_ef_lists[i][j] = np.array(vel_ef_lists[i][j])
        if i % 1 == 0: logger.info("duals {}/{}".format(i, n_modes))
        for j in range(n_modes):
            velocity_list1 = []
            velocity_list2 = []
            for k, bn in enumerate(bases.keys()):
                velocity_list1.append(vel_ef_lists[i][k])
                velocity_list2.append(vel_ef_lists[j][k])
            IP_matrix[i,j] = IP(velocity_list1, velocity_list2)
    
    print('dual IP matrix cond: {:.3e}'.format(np.linalg.cond(IP_matrix)))
    IP_inv = np.linalg.inv(IP_matrix)

    total_nr = 0
    nr_slices = []
    for i, bn in enumerate(bases.keys()):
        this_nr = vel_ef_lists[0][i].shape[-1]
        total_nr += this_nr
        if i == 0:
            nr_slices.append(slice(0, this_nr, 1))
        else:
            start = nr_slices[-1].stop
            nr_slices.append(slice(start, start+this_nr, 1))

    vel_duals = np.zeros((n_modes, 3, total_nr), dtype=np.complex128)
    vel_efs = np.zeros((n_modes, 3, total_nr), dtype=np.complex128)
    for i, bn in enumerate(bases.keys()):
        r_slice = nr_slices[i]
        for n in range(n_modes):
            vel_efs[n, :, r_slice] = vel_ef_lists[n][i]
    for j in range(3): #velocity dimensions
        vel_duals[:,j,:] = np.einsum('ij,jk->ik', np.conj(IP_inv), vel_efs[:,j,:])

    #Check that velocity duals were evaluated correctly
    IP_check = np.zeros_like(IP_matrix)
    for i in range(n_modes):
        for j in range(n_modes):
            velocity_duals = []
            velocity_efs = []
            for k, bn in enumerate(bases.keys()):
                r_slice = nr_slices[k]
                velocity_duals.append(vel_duals[i,:,r_slice])
                velocity_efs.append(vel_efs[j,:,r_slice])
            IP_check[i,j] = IP(velocity_duals, velocity_efs)
    I_matrix = np.zeros_like(IP_matrix)
    for i in range(I_matrix.shape[0]):
        I_matrix[i,i] = 1

    if np.allclose(I_matrix.real, IP_check.real, rtol=1e-5, atol=1e-5):
        print('velocity duals properly calculated')
    else:
        for i in range(n_modes):
            print(np.sum(IP_check[i,:].real), IP_check[i,i].real)
        raise ValueError("Something went wrong in calculating the dual basis.")

    return vel_duals



# Define smooth Heaviside functions
from scipy.special import erf
def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
    return -(one_to_zero(*args, **kwargs) - 1)

if __name__ == '__main__':
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
    ntheta = Lmax + 1
    nphi = 2*ntheta
    nrB = int(args['--nrB'])
    nrS1 = int(args['--nrS1'])
    nrS2 = int(args['--nrS2'])
    resolutionB = (nphi, ntheta, nrB)
    resolutionS1 = (nphi, ntheta, nrS1)
    resolutionS2 = (nphi, ntheta, nrS2)
    Re  = float(args['--Re'])
    Pr  = 1
    Pe  = Pr*Re
    ncc_file  = args['--ncc_file']

    do_hires = (args['--nrB_hi'] is not None and args['--nrS1_hi'] is not None and args['--nrS2_hi'] is not None)

    if do_hires:
        nrB_hi = int(args['--nrB_hi'])
        nrS1_hi = int(args['--nrS1_hi'])
        nrS2_hi = int(args['--nrS2_hi'])
        ncc_file_hi = args['--ncc_file_hi']
        resolutionB_hi = (nphi, ntheta, nrB_hi)
        resolutionS1_hi = (nphi, ntheta, nrS1_hi)
        resolutionS2_hi = (nphi, ntheta, nrS2_hi)
    else:
        nrB_hi = nrS1_hi = nrS2_hi = ncc_file_hi = None

    out_dir = './' + sys.argv[0].split('.py')[0]
    if args['--ncc_file'] is None:
        out_dir += '_polytrope'
    out_dir += '_Re{}_Lmax{}_nr{}+{}+{}'.format(args['--Re'], Lmax, nrB, nrS1, nrS2)
    if do_hires:
        out_dir += '_nrhi{}+{}+{}'.format(nrB_hi, nrS1_hi, nrS2_hi)
    if args['--label'] is not None:
        out_dir += '_{:s}'.format(args['--label'])
    logger.info('saving data to {:s}'.format(out_dir))
    if MPI.COMM_WORLD.rank == 0:
        if not os.path.exists('{:s}/'.format(out_dir)):
            os.makedirs('{:s}/'.format(out_dir))

    if ncc_file is not None:
        with h5py.File(ncc_file, 'r') as f:
            r_stitch = f['r_stitch'][()]
            r_outer = f['r_outer'][()]
            tau_nd = f['tau_nd'][()]
            m_nd = f['m_nd'][()]
            L_nd = f['L_nd'][()]
            T_nd = f['T_nd'][()]
            rho_nd = f['rho_nd'][()]
            s_nd = f['s_nd'][()]

            tau_day = tau_nd/(60*60*24)
            N2_mesa = f['N2_mesa'][()]
            S1_mesa = f['S1_mesa'][()]
            r_mesa = f['r_mesa'][()]
            Re_shift = f['Re_shift'][()]
    else:
        r_stitch = (1.1, 1.4)
        r_outer = 1.5
        tau_day = 1
        Re_shift = 1

    Re *= Re_shift
    Pe *= Re_shift

    sponge = False
    do_rotation = False
    dealias = (1,1,1) 

    logger.info('r_stitch: {} / r_outer: {:.2f}'.format(r_stitch, r_outer))
    logger.info('ncc file {}'.format(ncc_file))
    logger.info('hires ncc file {}'.format(ncc_file_hi))

    #Create bases
    resolutions = (resolutionB, resolutionS1, resolutionS2)
    stitch_radii = r_stitch
    radius = r_outer
    coords, dist, bases, bases_keys = make_bases(resolutions, stitch_radii, radius, dealias=dealias, dtype=np.complex128, mesh=None)

    vec_fields = ['u',]
    scalar_fields = ['p', 's1', 'inv_T', 'H', 'rho', 'T']
    vec_taus = ['tau_u']
    scalar_taus = ['tau_s']
    vec_nccs = ['grad_ln_rho', 'grad_ln_T', 'grad_s0', 'grad_T', 'grad_chi_rad']
    scalar_nccs = ['ln_rho', 'ln_T', 'chi_rad', 'sponge', 'nu_diff']

    variables = make_fields(bases, coords, dist, 
                            vec_fields=vec_fields, scalar_fields=scalar_fields, 
                            vec_taus=vec_taus, scalar_taus=scalar_taus, 
                            vec_nccs=vec_nccs, scalar_nccs=scalar_nccs,
                            sponge=sponge, do_rotation=do_rotation)


    variables, timescales = fill_structure(bases, dist, variables, ncc_file, r_outer, Pe, 
                                            vec_fields=vec_fields, vec_nccs=vec_nccs, scalar_nccs=scalar_nccs,
                                            sponge=sponge, do_rotation=do_rotation)

#    grad_s0_func = lambda r: 1e6*r
#    grad_T0_func = lambda r: -r
#    for i, bn in enumerate(bases_keys):
#        variables['grad_s0_{}'.format(bn)]['g'][2,:] = grad_s0_func(variables['r1_{}'.format(bn)])
#        variables['grad_T_{}'.format(bn)]['g'][2,:] = grad_T0_func(variables['r1_{}'.format(bn)])
#        variables['grad_chi_rad_{}'.format(bn)]['g'] = 0
#        variables['chi_rad_{}'.format(bn)]['g'] = 1/Re
#        variables['grad_ln_rho_{}'.format(bn)]['g'] = 0
#        variables['grad_ln_T_{}'.format(bn)]['g'] = 0
#
#        grad_s0 = variables['grad_s0_{}'.format(bn)]
#        print(grad_s0['g'], grad_s0['c'])
 
    variables.update(locals())
    omega = dist.Field(name='omega')
    variables['dt'] = lambda A: -1j * omega * A

    prob_variables = get_anelastic_variables(bases, bases_keys, variables)

    problem = d3.EVP(prob_variables, eigenvalue=omega, namespace=variables)

    problem = set_anelastic_problem(problem, bases, bases_keys, stitch_radii=stitch_radii)
    logger.info('problem built')

    if ncc_file is not None:
        ncc_cutoff = float(ncc_file.split('.h5')[0].split('_cutoff')[-1])
    else:
        ncc_cutoff=1e-8
    logger.info('using ncc cutoff {:.2e}'.format(ncc_cutoff))
    solver = problem.build_solver(ncc_cutoff=ncc_cutoff)
    logger.info('solver built')

    if do_hires:
        resolutions_hi = (resolutionB_hi, resolutionS1_hi, resolutionS2_hi)
        coords_hi, dist_hi, bases_hi, bases_keys_hi = make_bases(resolutions_hi, stitch_radii, radius, dealias=dealias, dtype=np.complex128, mesh=None)
        variables_hi = make_fields(bases_hi, coords_hi, dist_hi, 
                                vec_fields=vec_fields, scalar_fields=scalar_fields, 
                                vec_taus=vec_taus, scalar_taus=scalar_taus, 
                                vec_nccs=vec_nccs, scalar_nccs=scalar_nccs,
                                sponge=sponge, do_rotation=do_rotation)


        variables_hi, timescales_hi = fill_structure(bases_hi, dist_hi, variables_hi, ncc_file_hi, r_outer, Pe, 
                                                vec_fields=vec_fields, vec_nccs=vec_nccs, scalar_nccs=scalar_nccs,
                                                sponge=sponge, do_rotation=do_rotation)

#        for i, bn in enumerate(bases_keys):
#            variables_hi['grad_s0_{}'.format(bn)]['g'][2,:] = grad_s0_func(variables_hi['r_{}'.format(bn)])
#            variables_hi['grad_T_{}'.format(bn)]['g'][2,:] = grad_T0_func(variables_hi['r_{}'.format(bn)])
#            variables_hi['chi_rad_{}'.format(bn)]['g'] = 1/Re
#            variables_hi['grad_chi_rad_{}'.format(bn)]['g'] = 0
#            variables_hi['grad_ln_rho_{}'.format(bn)]['g'] = 0
#            variables_hi['grad_ln_T_{}'.format(bn)]['g'] = 0
        variables_hi.update(locals())
        omega_hi = dist_hi.Field(name='omega_hi')
        variables_hi['dt'] = lambda A: -1j * omega_hi * A
        prob_variables_hi = get_anelastic_variables(bases_hi, bases_keys_hi, variables_hi)
        problem_hi = d3.EVP(prob_variables_hi, eigenvalue=omega_hi, namespace=variables_hi)
        problem_hi = set_anelastic_problem(problem_hi, bases_hi, bases_keys_hi, stitch_radii=stitch_radii)
        logger.info('hires problem built')


        logger.info('using ncc cutoff {:.2e}'.format(ncc_cutoff))
        solver_hi = problem_hi.build_solver(ncc_cutoff=ncc_cutoff)
        logger.info('hires solver built')

    #ell = 1 solve
    for i in range(Lmax):
        ell = i + 1
        if not args['--duals_only']:
            logger.info('solving lores eigenvalue with nr = ({}, {}, {})'.format(nrB, nrS1, nrS2))
            solve_dense(solver, ell)

            if do_hires:
                logger.info('solving hires eigenvalue with nr = ({}, {}, {})'.format(nrB_hi, nrS1_hi, nrS2_hi))
                print(solver.eigenvectors.shape)
                solver_hi = solve_sparse(solver_hi, ell, solver.eigenvalues)
                solver, solver_hi = check_eigen(solver, solver_hi, bases, bases_hi, variables, variables_hi, ell, cutoff=1e-2)
        
            #Calculate 'optical depths' of each mode.
            #TODO: Fix this
            depths = []
            if ncc_file is not None:
                chi_rad = np.zeros_like(r_mesa)
                for i, bn in enumerate(bases_keys):
                    local_r_inner, local_r_outer = 0, 0
                    if i == 0:
                        local_r_inner = 0
                    else:
                        local_r_inner = stitch_radii[i-1]
                    if len(bases_keys) > 1 and i < len(bases_keys) - 1:
                        local_r_outer = stitch_radii[i]
                    else:
                        local_r_outer = radius
                    r_mesa_nd = r_mesa/L_nd
                    good_r = (r_mesa_nd > local_r_inner)*(r_mesa_nd <= local_r_outer)
                    chi_rad[good_r] = interp1d(variables['r_{}'.format(bn)].flatten(), variables['chi_rad_{}'.format(bn)]['g'][0,0,:], 
                                               bounds_error=False, fill_value='extrapolate')(r_mesa_nd[good_r])
                chi_rad *= (L_nd**2 / tau_nd)

                # from Shiode et al 2013 eqns 4-8 
                for om in solver.eigenvalues.real:
                    dim_om = (om.real/tau_nd)
                    Lambda = np.sqrt(ell*(ell+1))
                    kr_cm = np.sqrt(N2_mesa)*Lambda/(r_mesa* (om/tau_nd))
                    v_group = (om/tau_nd) / kr_cm
                    gamma_rad = chi_rad * kr_cm**2

                    lamb_freq = np.sqrt(ell*(ell+1) / 2) * S1_mesa
                    wave_cavity = (dim_om < np.sqrt(N2_mesa))*(dim_om < lamb_freq)

                    depth_integrand = np.zeros_like(gamma_rad)
                    depth_integrand[wave_cavity] = (gamma_rad/v_group)[wave_cavity]

                    #No optical depth in CZs, or outside of simulation domain...
                    depth_integrand[r_mesa/L_nd > r_outer] = 0

                    #Numpy integrate
                    opt_depth = np.trapz(depth_integrand, x=r_mesa)
                    depths.append(opt_depth)

                good_omegas = solver.eigenvalues.real
                smooth_oms = np.logspace(np.log10(good_omegas.min())-1, np.log10(good_omegas.max())+1, 100)
                smooth_depths = np.zeros_like(smooth_oms)
                # from Shiode et al 2013 eqns 4-8 
                for i, om in enumerate(smooth_oms):
                    dim_om = (om.real/tau_nd)
                    Lambda = np.sqrt(ell*(ell+1))
                    kr_cm = np.sqrt(N2_mesa)*Lambda/(r_mesa* (om/tau_nd))
                    v_group = (om/tau_nd) / kr_cm
                    gamma_rad = chi_rad * kr_cm**2

                    lamb_freq = np.sqrt(ell*(ell+1) / 2) * S1_mesa
                    wave_cavity = (dim_om < np.sqrt(N2_mesa))*(dim_om < lamb_freq)

                    depth_integrand = np.zeros_like(gamma_rad)
                    depth_integrand[wave_cavity] = (gamma_rad/v_group)[wave_cavity]

                    #No optical depth in CZs, or outside of simulation domain...
                    depth_integrand[r_mesa/L_nd > r_outer] = 0

                    #Numpy integrate
                    opt_depth = np.trapz(depth_integrand, x=r_mesa)
                    smooth_depths[i] = opt_depth



            shape = list(variables['s1_B']['c'].shape[:2])
            good = np.zeros(shape, bool)
            for i in range(shape[0]):
                for j in range(shape[1]):
                    grid_space = (False,False)
                    elements = (np.array((i,)),np.array((j,)))
                    m, this_ell = bases['B'].sphere_basis.elements_to_groups(grid_space, elements)
                    if this_ell == ell and m == 1:
                        good[i,j] = True

            integ_energy_op = None
            rho_fields = []
            for i, bn in enumerate(bases_keys):
                p, u = [variables['{}_{}'.format(f, bn)] for f in ['p', 'u']]
                variables['pomega_hat_{}'.format(bn)] = p - 0.5*d3.dot(u,u)
                variables['KE_{}'.format(bn)] = dist.Field(bases=bases[bn], name='KE_{}'.format(bn))
                rho_fields.append(variables['rho_{}'.format(bn)]['g'][0,0,:])

                if integ_energy_op is None:
                    integ_energy_op = d3.integ(variables['KE_{}'.format(bn)])
                else:
                    integ_energy_op += d3.integ(variables['KE_{}'.format(bn)])
            s1_surf = variables['s1_S2'](r=radius)
            rho_full = np.concatenate(rho_fields, axis=-1)

            integ_energies = np.zeros_like(solver.eigenvalues, dtype=np.float64) 
            s1_amplitudes = np.zeros_like(solver.eigenvalues, dtype=np.float64)  
            velocity_eigenfunctions = []
            velocity_eigenfunctions_pieces = []
            entropy_eigenfunctions = []
            wave_flux_eigenfunctions = []

            for sbsys in solver.subsystems:
                ss_m, ss_ell, r_couple = sbsys.group
                if ss_ell == ell and ss_m == 1:
                    subsystem = sbsys
                    break

            for i, e in enumerate(solver.eigenvalues):
                solver.set_state(i, subsystem)

                #Get eigenvectors
                for j, bn in enumerate(bases_keys):
                    variables['pomega_hat_field_{}'.format(bn)] = variables['pomega_hat_{}'.format(bn)].evaluate()

                ef_u, ef_u_pieces = combine_eigvecs('u', good, bases, variables, shift=False)
                ef_s1, ef_s1_pieces = combine_eigvecs('s1', good, bases, variables, shift=False)
                ef_pom, ef_pom_pieces = combine_eigvecs('pomega_hat_field', good, bases, variables, shift=False)

                #normalize & store eigenvectors
                shift = np.max(np.abs(ef_u[2,:]))
                for data in [ef_u, ef_s1, ef_pom]:
                    data[:] /= shift
                for piece_tuple in [ef_u_pieces, ef_s1_pieces, ef_pom_pieces]:
                    for data in piece_tuple:
                        data[:] /= shift

                velocity_eigenfunctions.append(ef_u)
                velocity_eigenfunctions_pieces.append(ef_u_pieces)
                entropy_eigenfunctions.append(ef_s1)

                #Wave flux
                wave_flux = rho_full*ef_u[2,:]*np.conj(ef_pom).squeeze()
                wave_flux_eigenfunctions.append(wave_flux)

    #            #Kinetic energy
                for j, bn in enumerate(bases_keys):
                    rho = variables['rho_{}'.format(bn)]['g'][0,0,:]
                    u_squared = np.sum(ef_u_pieces[j]*np.conj(ef_u_pieces[j]), axis=0)
                    variables['KE_{}'.format(bn)]['g'] = (rho*u_squared.real/2)[None,None,:]
                integ_energy = integ_energy_op.evaluate()
                integ_energies[i] = integ_energy['g'].min().real / 2 #factor of 2 accounts for spherical harmonic integration (we're treating the field like an ell = 0 one)

                #Surface entropy perturbations
                for j, bn in enumerate(bases_keys):
                    variables['s1_{}'.format(bn)]['g'] = ef_s1_pieces[j]
                s1_surf_vals = s1_surf.evaluate()['g'] / np.sqrt(2) #sqrt(2) accounts for spherical harmonic integration
                s1_amplitudes[i] = np.abs(s1_surf_vals.max())

            print(s1_amplitudes)

            with h5py.File('{:s}/ell{:03d}_eigenvalues.h5'.format(out_dir, ell), 'w') as f:
                f['good_evalues'] = solver.eigenvalues
                f['good_omegas']  = solver.eigenvalues.real
                f['good_evalues_inv_day'] = solver.eigenvalues/tau_day
                f['good_omegas_inv_day']  = solver.eigenvalues.real/tau_day
                f['s1_amplitudes']  = s1_amplitudes
                f['integ_energies'] = integ_energies
                f['wave_flux_eigenfunctions'] = np.array(wave_flux_eigenfunctions)
                f['velocity_eigenfunctions'] = np.array(velocity_eigenfunctions)
                f['entropy_eigenfunctions'] = np.array(entropy_eigenfunctions)
                for i, bn in enumerate(bases_keys):
                    f['r_{}'.format(bn)] = variables['r1_{}'.format(bn)]
                    f['rho_{}'.format(bn)] = variables['rho_{}'.format(bn)]['g']
                    for j in range(len(solver.eigenvalues)):
                        f['velocity_eigenfunctions_piece_{}_{}'.format(j, bn)] = velocity_eigenfunctions_pieces[j][i]
                f['rho_full'] = rho_full
                f['depths'] = np.array(depths)
                f['smooth_oms'] = smooth_oms
                f['smooth_depths'] = smooth_depths

                #Pass through nondimensionalization

                if ncc_file is not None:
                    f['tau_nd'] = tau_nd 
                    f['m_nd']   = m_nd   
                    f['L_nd']   = L_nd   
                    f['T_nd']   = T_nd   
                    f['rho_nd'] = rho_nd 
                    f['s_nd']   = s_nd   
        else: #duals only
            velocity_eigenfunctions_pieces = []
            with h5py.File('{:s}/ell{:03d}_eigenvalues.h5'.format(out_dir, ell), 'r') as f:
                for i in range(len(f.keys())):
                    key = 'velocity_eigenfunctions_piece_{}'.format(i)
                    these_pieces = []
                    for j, bn in enumerate(bases_keys):
                        this_key = '{}_{}'.format(key, bn)
                        if this_key in f.keys():
                            these_pieces.append(f[this_key][()])
                    if len(these_pieces) > 0:
                        velocity_eigenfunctions_pieces.append(these_pieces)
        #Calculate duals
        velocity_duals = calculate_duals(velocity_eigenfunctions_pieces, bases, variables)
        with h5py.File('{:s}/ell{:03d}_eigenvalues.h5'.format(out_dir, ell), 'r') as f:
            with h5py.File('{:s}/duals_ell{:03d}_eigenvalues.h5'.format(out_dir, ell), 'w') as df:
                for k in f.keys():
                    df.create_dataset(k, data=f[k])
                df['velocity_duals'] = velocity_duals

        gc.collect()
