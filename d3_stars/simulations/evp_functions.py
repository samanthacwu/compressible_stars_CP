from operator import itemgetter
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
from .parser import parse_std_config

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



def calculate_duals(vel_ef_lists, bases, namespace, dist):
    """
    Calculate the dual basis of the velocity eigenvectors.
    """
    work_fields = []
    vel_ef_lists = list(vel_ef_lists)
    int_field = None
    for i, bn in enumerate(bases.keys()):
        work_fields.append(dist.Field(bases=bases[bn]))
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
    
    logger.info('dual IP matrix cond: {:.3e}'.format(np.linalg.cond(IP_matrix)))
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
        logger.info('velocity duals properly calculated')
    else:
        logger.info('something went wrong in velocity dual calc; IP_check info:')
        for i in range(n_modes):
            logger.info('{}, {}'.format(np.sum(IP_check[i,:].real), IP_check[i,i].real))
        raise ValueError("Something went wrong in calculating the dual basis.")

    return vel_duals

def calculate_optical_depths(solver, bases_keys, stitch_radii, radius, ncc_file, variables, ell=1):
    #Calculate 'optical depths' of each mode.
    good_omegas = solver.eigenvalues.real
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

    return depths, smooth_oms, smooth_depths




# Define smooth Heaviside functions
from scipy.special import erf
def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
    return -(one_to_zero(*args, **kwargs) - 1)


class StellarEVP():

    def __init__(self, is_hires=False):
        # Read in parameters and create output directory
        config, raw_config, star_dir, star_file = parse_std_config('controls.cfg')
        Lmax = config['lmax']
        self.ells = np.arange(Lmax) + 1 #skip ell = 0
        ntheta = Lmax + 1
        nphi = 4#2*ntheta
        hires_factor = config['hires_factor']
        self.do_hires = hires_factor != 1

        self.out_dir = 'eigenvalues'
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)

        # Parameters
        ncc_cutoff = config['ncc_cutoff']
        resolutions = []
        for nr in config['nr']:
            if is_hires:
                resolutions.append((nphi, ntheta, int(hires_factor*nr)))
            else:
                resolutions.append((nphi, ntheta, nr))
        self.nr = config['nr']
        Re  = config['reynolds_target'] 
        Pr  = config['prandtl']
        Pe  = Pr*Re
        self.ncc_file  = star_file

        if self.ncc_file is not None:
            with h5py.File(self.ncc_file, 'r') as f:
                r_stitch = f['r_stitch'][()]
                self.r_outer = f['r_outer'][()]
                tau_nd = f['tau_nd'][()]
                m_nd = f['m_nd'][()]
                L_nd = f['L_nd'][()]
                T_nd = f['T_nd'][()]
                rho_nd = f['rho_nd'][()]
                s_nd = f['s_nd'][()]

                self.tau_day = tau_nd/(60*60*24)
                N2_mesa = f['N2_mesa'][()]
                S1_mesa = f['S1_mesa'][()]
                r_mesa = f['r_mesa'][()]
                Re_shift = f['Re_shift'][()]
        else:
            r_stitch = (1.1, 1.4)
            self.r_outer = 1.5
            self.tau_day = 1
            Re_shift = 1

        Re *= Re_shift
        Pe *= Re_shift

        sponge = False
        do_rotation = False
        dealias = (1,1,1) 

        logger.info('r_stitch: {} / r_outer: {:.2f}'.format(r_stitch, self.r_outer))
        logger.info('ncc file {}'.format(self.ncc_file))

        #Create bases
        self.stitch_radii = r_stitch
        radius = self.r_outer
        coords, self.dist, bases, self.bases_keys = make_bases(resolutions, self.stitch_radii, radius, dealias=dealias, dtype=np.complex128, mesh=None)


        vec_fields = ['u',]
        scalar_fields = ['p', 's1', 'inv_T', 'H', 'rho', 'T']
        vec_taus = ['tau_u']
        scalar_taus = ['tau_s']
        vec_nccs = ['grad_ln_rho', 'grad_ln_T', 'grad_s0', 'grad_T', 'grad_chi_rad']
        scalar_nccs = ['ln_rho', 'ln_T', 'chi_rad', 'sponge', 'nu_diff']

        self.namespace = make_fields(bases, coords, self.dist, 
                                vec_fields=vec_fields, scalar_fields=scalar_fields, 
                                vec_taus=vec_taus, scalar_taus=scalar_taus, 
                                vec_nccs=vec_nccs, scalar_nccs=scalar_nccs,
                                sponge=sponge, do_rotation=do_rotation)

        r_scale = config['n_dealias']/dealias[0]
        if is_hires:
            r_scale /= hires_factor
        self.namespace, timescales = fill_structure(bases, self.dist, self.namespace, self.ncc_file, self.r_outer, Pe, 
                                                vec_fields=vec_fields, vec_nccs=vec_nccs, scalar_nccs=scalar_nccs,
                                                sponge=sponge, do_rotation=do_rotation, scales=(1,1,r_scale))

        self.namespace.update(locals())
        omega = self.dist.Field(name='omega')
        self.namespace['dt'] = lambda A: -1j * omega * A

        prob_variables = get_anelastic_variables(bases, self.bases_keys, self.namespace)

        self.problem = d3.EVP(prob_variables, eigenvalue=omega, namespace=self.namespace)
        self.problem = set_anelastic_problem(self.problem, bases, self.bases_keys, stitch_radii=self.stitch_radii)
        logger.info('problem built')

        logger.info('using ncc cutoff {:.2e}'.format(ncc_cutoff))
        self.solver = self.problem.build_solver(ncc_cutoff=ncc_cutoff)
        logger.info('solver built')

        if self.do_hires and not is_hires:
            self.hires_EVP = StellarEVP(is_hires=True)

        self.bases = bases
        self.hires_factor = hires_factor

    def solve(self, ell):
        logger.info('solving eigenvalue with nr = {} at ell = {}'.format(self.nr, ell))
        self.solver = solve_dense(self.solver, ell)
        self.ell = ell

        for sbsys in self.solver.subsystems:
            ss_m, ss_ell, r_couple = sbsys.group
            if ss_ell == ell and ss_m == 1:
                self.subsystem = sbsys
                break

    def setup_sparse_solve(self, ell):
        for sbsys in self.solver.subsystems:
            ss_m, ss_ell, r_couple = sbsys.group
            if ss_ell == ell and ss_m == 1:
                self.subsystem = sbsys
                break

        for subproblem in self.solver.subproblems:
            this_ell = subproblem.group[1]
            if this_ell != ell:
                continue

            sp = subproblem

            ell = subproblem.group[1]
            matrix_info(sp)

            self.A = (sp.L_min @ sp.pre_right).A
            self.B = - (sp.M_min @ sp.pre_right).A
            self.solver.eigenvalue_subproblem = sp
            break

    def solve_sparse(self, eigenvalues):
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
            evalue, evec = scipy_sparse_eigs(A=self.A, B=self.B, N=N, target=target, matsolver=self.solver.matsolver)
            full_evals.append(evalue.ravel())
            full_evecs.append(evec.ravel())
        end_time = time.time()
        logger.info('sparse solve done in {:.2f} sec'.format(end_time - start_time))
        self.solver.eigenvalues = np.array(full_evals)
        self.solver.eigenvectors = np.swapaxes(np.array(full_evecs), 0, 1)
        logger.info('shapes: {} / {}'.format(self.solver.eigenvalues.shape, self.solver.eigenvectors.shape))

        return self.solver



    def check_eigen(self, cutoff=1e-2, r_cz=1):
        """
        Compare eigenvalues and eigenvectors between a hi-res and lo-res solve.
        Only keep the solutions that match to within the specified cutoff between the two cases.
        """
        if not self.do_hires:
            logger.info("hires_scales = 1; skipping check_eigen()")
            return
        
        self.hires_EVP.setup_sparse_solve(self.ell)

        scales = []
        hires_rho = []
        hires_r = []
        for i, bn in enumerate(self.bases.keys()):
            hires_rho.append(self.hires_EVP.namespace['rho_{}'.format(bn)]['g'][0,0,:])
            hires_r.append(self.hires_EVP.namespace['r_{}'.format(bn)].ravel())

            scales.append((1,1,self.hires_factor))

        hires_rho = np.concatenate(hires_rho)
        hires_r = np.concatenate(hires_r)
        dr = np.gradient(hires_r, edge_order=2)

        logger.info('finding good eigenvalues with cutoff {}'.format(cutoff))
        shape = list(self.namespace['s1_B']['c'].shape[:2])
        good = np.zeros(shape, bool)
        for i in range(shape[0]):
            for j in range(shape[1]):
                grid_space = (False,False)
                elements = (np.array((i,)),np.array((j,)))
                m, this_ell = self.bases['B'].sphere_basis.elements_to_groups(grid_space, elements)
                if this_ell == self.ell and m == 1:
                    good[i,j] = True

        good_values = []
        for i, v1 in enumerate(self.solver.eigenvalues):
            if np.abs(v1.real) < 3*np.abs(v1.imag):
                logger.info('skipping eigenvalue {}; damps very quickly'.format(v1))
                continue
            self.hires_EVP.solver = self.hires_EVP.solve_sparse([v1,])
            v2 = self.hires_EVP.solver.eigenvalues[0]
            real_goodness = np.abs(v1.real - v2.real)/np.abs(v1.real).min()
            goodness = np.abs(v1 - v2)/np.abs(v1).min()

            if goodness < cutoff:
                
                self.solver.set_state(i, self.subsystem)
                self.hires_EVP.solver.set_state(0, self.hires_EVP.subsystem)

                ef_u1, ef_u1_pieces = combine_eigvecs('u', good, self.bases, self.namespace, scales=scales)
                ef_u2, ef_u2_pieces = combine_eigvecs('u', good, self.hires_EVP.bases, self.hires_EVP.namespace)

                #If mode KE is inside of the convection zone then it's a bad mode.
                mode_KE = hires_rho*np.sum(ef_u2*np.conj(ef_u2), axis=0).real/2
                cz_KE = np.sum((mode_KE*4*np.pi*hires_r**2*dr)[hires_r <= r_cz])
                tot_KE = np.sum((mode_KE*4*np.pi*hires_r**2*dr))
                cz_KE_frac = cz_KE/tot_KE
                vector_diff = np.max(np.abs(ef_u1 - ef_u2))

#                plt.figure()
#                plt.plot(hires_r, ef_u1[2,:])
#                plt.plot(hires_r, ef_u2[2,:])
#                plt.title(v1)
#                plt.show()

                if vector_diff < np.sqrt(cutoff):
                    logger.info('good evalue w/ vdiff', vector_diff, 'czfrac', cz_KE_frac.real)
                    if cz_KE_frac.real > 0.5:
                        logger.info('evalue is in the CZ, skipping')
                    elif cz_KE_frac.real < 1e-3:
                        logger.info('evalue is spurious, skipping')
                    else:
                        good_values.append(i)

        self.solver.eigenvalues  = self.solver.eigenvalues[good_values]
        self.solver.eigenvectors = self.solver.eigenvectors[:, good_values]



    def output(self):
        #Calculate 'optical depths' of each mode.
        depths, smooth_oms, smooth_depths = calculate_optical_depths(self.solver, self.bases_keys, self.stitch_radii, self.r_outer, self.ncc_file, self.namespace, ell=self.ell)

        shape = list(self.namespace['s1_B']['c'].shape[:2])
        good = np.zeros(shape, bool)
        for i in range(shape[0]):
            for j in range(shape[1]):
                grid_space = (False,False)
                elements = (np.array((i,)),np.array((j,)))
                m, this_ell = self.bases['B'].sphere_basis.elements_to_groups(grid_space, elements)
                if this_ell == self.ell and m == 1:
                    good[i,j] = True

        integ_energy_op = None
        rho_fields = []
        s1_surf = None
        for i, bn in enumerate(self.bases_keys):
            p, u = [self.namespace['{}_{}'.format(f, bn)] for f in ['p', 'u']]
            self.namespace['pomega_hat_{}'.format(bn)] = p - 0.5*d3.dot(u,u)
            self.namespace['KE_{}'.format(bn)] = self.dist.Field(bases=self.bases[bn], name='KE_{}'.format(bn))
            rho_fields.append(self.namespace['rho_{}'.format(bn)]['g'][0,0,:])

            if integ_energy_op is None:
                integ_energy_op = d3.integ(self.namespace['KE_{}'.format(bn)])
            else:
                integ_energy_op += d3.integ(self.namespace['KE_{}'.format(bn)])

            if i == len(self.bases_keys) - 1:
                s1_surf = self.namespace['s1_{}'.format(bn)](r=self.r_outer)
        rho_full = np.concatenate(rho_fields, axis=-1)

        integ_energies = np.zeros_like(self.solver.eigenvalues, dtype=np.float64) 
        s1_amplitudes = np.zeros_like(self.solver.eigenvalues, dtype=np.float64)  
        velocity_eigenfunctions = []
        velocity_eigenfunctions_pieces = []
        entropy_eigenfunctions = []
        wave_flux_eigenfunctions = []

        for i, e in enumerate(self.solver.eigenvalues):
            self.solver.set_state(i, self.subsystem)

            #Get eigenvectors
            for j, bn in enumerate(self.bases_keys):
                self.namespace['pomega_hat_field_{}'.format(bn)] = self.namespace['pomega_hat_{}'.format(bn)].evaluate()

            ef_u, ef_u_pieces = combine_eigvecs('u', good, self.bases, self.namespace, shift=False)
            ef_s1, ef_s1_pieces = combine_eigvecs('s1', good, self.bases, self.namespace, shift=False)
            ef_pom, ef_pom_pieces = combine_eigvecs('pomega_hat_field', good, self.bases, self.namespace, shift=False)

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
            for j, bn in enumerate(self.bases_keys):
                rho = self.namespace['rho_{}'.format(bn)]['g'][0,0,:]
                u_squared = np.sum(ef_u_pieces[j]*np.conj(ef_u_pieces[j]), axis=0)
                self.namespace['KE_{}'.format(bn)]['g'] = (rho*u_squared.real/2)[None,None,:]
            integ_energy = integ_energy_op.evaluate()
            integ_energies[i] = integ_energy['g'].min().real / 2 #factor of 2 accounts for spherical harmonic integration (we're treating the field like an ell = 0 one)

            #Surface entropy perturbations
            for j, bn in enumerate(self.bases_keys):
                self.namespace['s1_{}'.format(bn)]['g'] = ef_s1_pieces[j]
            s1_surf_vals = s1_surf.evaluate()['g'] / np.sqrt(2) #sqrt(2) accounts for spherical harmonic integration
            s1_amplitudes[i] = np.abs(s1_surf_vals.max())

        with h5py.File('{:s}/ell{:03d}_eigenvalues.h5'.format(self.out_dir, self.ell), 'w') as f:
            f['good_evalues'] = self.solver.eigenvalues
            f['good_omegas']  = self.solver.eigenvalues.real
            f['good_evalues_inv_day'] = self.solver.eigenvalues/self.tau_day
            f['good_omegas_inv_day']  = self.solver.eigenvalues.real/self.tau_day
            f['s1_amplitudes']  = s1_amplitudes
            f['integ_energies'] = integ_energies
            f['wave_flux_eigenfunctions'] = np.array(wave_flux_eigenfunctions)
            f['velocity_eigenfunctions'] = np.array(velocity_eigenfunctions)
            f['entropy_eigenfunctions'] = np.array(entropy_eigenfunctions)
            for i, bn in enumerate(self.bases_keys):
                f['r_{}'.format(bn)] = self.namespace['r1_{}'.format(bn)]
                f['rho_{}'.format(bn)] = self.namespace['rho_{}'.format(bn)]['g']
                for j in range(len(self.solver.eigenvalues)):
                    f['velocity_eigenfunctions_piece_{}_{}'.format(j, bn)] = velocity_eigenfunctions_pieces[j][i]
            f['rho_full'] = rho_full
            f['depths'] = np.array(depths)
            f['smooth_oms'] = smooth_oms
            f['smooth_depths'] = smooth_depths

            #Pass through nondimensionalization

            if self.ncc_file is not None:
                with h5py.File(self.ncc_file, 'r') as nccf:
                    f['tau_nd'] = nccf['tau_nd'][()]
                    f['m_nd']   = nccf['m_nd'][()]
                    f['L_nd']   = nccf['L_nd'][()]   
                    f['T_nd']   = nccf['T_nd'][()]   
                    f['rho_nd'] = nccf['rho_nd'][()] 
                    f['s_nd']   = nccf['s_nd'][()]   

    def get_duals(self):
        velocity_eigenfunctions_pieces = []
        with h5py.File('{:s}/ell{:03d}_eigenvalues.h5'.format(self.out_dir, self.ell), 'r') as f:
            for i in range(len(f.keys())):
                key = 'velocity_eigenfunctions_piece_{}'.format(i)
                these_pieces = []
                for j, bn in enumerate(self.bases_keys):
                    this_key = '{}_{}'.format(key, bn)
                    if this_key in f.keys():
                        these_pieces.append(f[this_key][()])
                if len(these_pieces) > 0:
                    velocity_eigenfunctions_pieces.append(these_pieces)
        #Calculate duals
        velocity_duals = calculate_duals(velocity_eigenfunctions_pieces, self.bases, self.namespace, self.dist)
        with h5py.File('{:s}/ell{:03d}_eigenvalues.h5'.format(self.out_dir, self.ell), 'r') as f:
            with h5py.File('{:s}/duals_ell{:03d}_eigenvalues.h5'.format(self.out_dir, self.ell), 'w') as df:
                for k in f.keys():
                    df.create_dataset(k, data=f[k])
                df['velocity_duals'] = velocity_duals

        gc.collect()
