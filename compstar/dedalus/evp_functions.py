import time
import gc
import os

import h5py
import numpy as np
import dedalus.public as d3

import logging
logger = logging.getLogger(__name__)

from dedalus.core import subsystems
from dedalus.tools.array import csr_matvec, scipy_sparse_eigs
from .compressible_functions import SphericalCompressibleProblem
from .parser import name_star
from compstar.tools.general import one_to_zero, zero_to_one
import compstar.defaults.config as config



def SBDF2_gamma_eff(gam, om, dt):
    """ Calculates the effective value of a damped wave that is explicitly timestepped via SBDF2. """
    # \tilde{omega}
    omega_tilde = om - 1j*gam
    # A_+
    A_plus = (2 + np.sqrt(1-2*1j*omega_tilde*dt)) / (3 + 2*1j*omega_tilde*dt)
    #\tilde{omega}_{eff}
    omega_tilde_eff = (1/(-1j*dt))*np.log(A_plus)

    return -omega_tilde_eff.imag, omega_tilde_eff.real

def solve_dense(solver, ell, group_index=1, lamb_freq=None, bruntN2=None):
    """
    Do a dense eigenvalue solve at a specified ell.
    Sort the eigenvalues and eigenvectors according to damping rate.
    """
    for subproblem in solver.subproblems:
        this_ell = subproblem.group[group_index]
        if this_ell != ell:
            continue

        sp = subproblem
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

        #Only keep values with positive real omega
        posvals = values.real > 0
        values = values[posvals]
        vectors = vectors[:, posvals]

        #Sort by real frequency magnitude
        order = np.argsort(1/values.real)
        values = values[order]
        vectors = vectors[:, order]

        #Only keep g modes
        g_modes = np.zeros_like(values, dtype=bool)
        for i in range(len(g_modes)):
            g_modes[i] = np.sum((np.abs(values[i].real) < lamb_freq)*(np.abs(values[i].real) < np.sqrt(bruntN2))) > 0

        values = values[g_modes]
        vectors = vectors[:, g_modes]

        #Get rid of purely diffusive modes
        goodvals = np.abs(values.real/values.imag) > 1e-4
        values = values[goodvals]
        vectors = vectors[:, goodvals]

        #Update solver
        solver.eigenvalues = values
        solver.eigenvectors = vectors
        return solver

def clean_eigvecs(fieldname, bases_keys, namespace, scales=None, shift=True):

    pieces = []
    for bn in bases_keys:
        field = namespace['{}_{}'.format(fieldname, bn)].evaluate()
        if len(field.tensorsig) == 1:
            vector = True
        else:
            vector = False

        if scales is not None:
            field.change_scales(scales)
        else:
            field.change_scales((1,1,1))

        pieces.append(field)
    arrays = [p['g'] for p in pieces]
    full_ef = np.concatenate(arrays, axis=-1)
    if shift:
        if vector:
            #Getting argmax then indexing ensures we're not off by a negative sign flip.
            ix = np.unravel_index(np.argmax(np.abs(full_ef[2,:])), full_ef[2,:].shape)
            divisor = full_ef[2,:][ix]
        else:
            ix = np.unravel_index(np.argmax(np.abs(full_ef)), full_ef.shape)
            divisor = full_ef[ix]

        full_ef /= divisor
        for i in range(len(pieces)):
            pieces[i]['g'] /= divisor

    return full_ef, pieces

def calculate_duals(vel_ef_list, bases, dist, IP=None, max_cond=None, discard_cond_cutoff=None):
    """
    Calculate the dual basis of the velocity eigenvectors.
    """
    vel_efs = np.array(vel_ef_list)
    if IP is None:
        raise ValueError("Must specify IP")

    def split_eigfunc_list(ef_list):
        pieces = []
        prev_n = 0
        for bn, basis in bases.items():
            nr_b = basis.radial_basis.radial_size
            new_n = prev_n + nr_b
            pieces.append(ef_list[:,prev_n:new_n][:,None,None,:])
            prev_n = new_n
        return pieces

    n_modes = vel_efs.shape[0]
    IP_matrix = np.zeros((n_modes, n_modes), dtype=np.complex128)
    cond = 0
    discard = 0
    for i in range(n_modes):
        vel_efs_pieces_0 = split_eigfunc_list(vel_ef_list[i])
        if i % 1 == 0: logger.info("duals {}/{}; cond: {:.1e}".format(i, n_modes, cond))
        for j in range(n_modes):
            vel_efs_pieces_1 = split_eigfunc_list(vel_ef_list[j])
            IP_matrix[i,j] = IP(vel_efs_pieces_0, vel_efs_pieces_1)
        cond = np.linalg.cond(IP_matrix[:i+1,:i+1])
        if discard_cond_cutoff is not None and cond > discard_cond_cutoff:
            discard += 1
        if max_cond is not None and i > 0:
            if cond > max_cond:
                n_modes = i
                IP_matrix = IP_matrix[:n_modes,:n_modes]
                break
    cond = np.linalg.cond(IP_matrix)
#    while max_cond is not None and cond > max_cond:
#        n_modes -= 1
#        IP_matrix = IP_matrix[:n_modes, :n_modes]
#        cond = np.linalg.cond(IP_matrix)

    vel_efs = vel_efs[:n_modes,:]

    logger.info('dual IP matrix cond, nmodes, discard: {:.3e},  {}, {}'.format(cond, n_modes, discard))
    IP_inv = np.linalg.inv(IP_matrix)

    vel_shape = vel_efs.shape[2:]
    spatial_shape = tuple(vel_shape)
    vel_duals = np.zeros((n_modes, 3, *spatial_shape), dtype=np.complex128)
    for j in range(3): #velocity dimensions
        vel_duals[:,j,:] = np.einsum('ij,jk->ik', np.conj(IP_inv), vel_efs[:,j,:])

    #Check that velocity duals were evaluated correctly
    IP_check = np.zeros_like(IP_matrix)
    for i in range(n_modes):
        vel_duals_pieces = split_eigfunc_list(vel_duals[i])
        for j in range(n_modes):
            vel_efs_pieces = split_eigfunc_list(vel_efs[j])
            IP_check[i,j] = IP(vel_duals_pieces, vel_efs_pieces)
    I_matrix = np.eye(IP_matrix.shape[0])

    if np.allclose(I_matrix.real, IP_check.real, rtol=1e-3, atol=1e-3):
        logger.info('velocity duals properly calculated')
    else:
        logger.info('something went wrong in velocity dual calc; IP_check info:')
        for i in range(n_modes):
            logger.info('{}, {}'.format(np.sum(IP_check[i,:].real), IP_check[i,i].real))
        raise ValueError("Something went wrong in calculating the dual basis.")

    return vel_duals, discard

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

        for om in solver.eigenvalues.real:
            lamb_freq = np.sqrt(ell*(ell+1) / 2) * S1_mesa
            dim_om = np.abs(om.real/tau_nd)
            wave_cavity = (dim_om < np.sqrt(N2_mesa))*(dim_om < lamb_freq)
            depth_integrand = np.zeros_like(lamb_freq)

            # from Shiode et al 2013 eqns 4-8 
            # This is the low-diffusion limit
            Lambda = np.sqrt(ell*(ell+1))
            kr_cm = np.sqrt(N2_mesa)*Lambda/(r_mesa* (om/tau_nd))
            v_group = (om/tau_nd) / kr_cm
            gamma_rad = chi_rad * kr_cm**2

            depth_integrand[wave_cavity] = (gamma_rad/v_group)[wave_cavity]

            # from Lecoanet et al 2015 eqn 12. This is the more universal function
            k_perp = Lambda/r_mesa
            kz = np.sqrt(-k_perp**2 + 1j*(dim_om/(2*chi_rad))*(1 - np.sqrt(1 + 1j*4*(N2_mesa*chi_rad*k_perp**2 / dim_om**3))))
            depth_integrand[wave_cavity] = -kz[wave_cavity].imag


            #No optical depth in CZs, or outside of simulation domain...
            depth_integrand[r_mesa/L_nd > r_outer] = 0

            #Numpy integrate
            opt_depth = np.trapz(depth_integrand, x=r_mesa)
            depths.append(opt_depth)
        
        smooth_oms = np.logspace(np.log10(np.abs(good_omegas).min())-1, np.log10(good_omegas.max())+1, 100)
        smooth_depths = np.zeros_like(smooth_oms)
        # from Shiode et al 2013 eqns 4-8 
        for i, om in enumerate(smooth_oms):
            dim_om = np.abs(om.real/tau_nd)
            Lambda = np.sqrt(ell*(ell+1))
            kr_cm = np.sqrt(N2_mesa)*Lambda/(r_mesa* (om/tau_nd))
            v_group = (om/tau_nd) / kr_cm
            gamma_rad = chi_rad * kr_cm**2

            lamb_freq = np.sqrt(ell*(ell+1) / 2) * S1_mesa
            wave_cavity = (dim_om < np.sqrt(N2_mesa))*(dim_om < lamb_freq)

            depth_integrand = np.zeros_like(gamma_rad)
#            depth_integrand[wave_cavity] = (gamma_rad/v_group)[wave_cavity]

            # from Lecoanet et al 2015 eqn 12. This is the more universal function
            k_perp = Lambda/r_mesa
            kz = np.sqrt(-k_perp**2 + 1j*(dim_om/(2*chi_rad))*(1 - np.sqrt(1 + 1j*4*(N2_mesa*chi_rad*k_perp**2 / dim_om**3))))
            depth_integrand[wave_cavity] = -kz[wave_cavity].imag



            #No optical depth in CZs, or outside of simulation domain...
            depth_integrand[r_mesa/L_nd > r_outer] = 0

            #Numpy integrate
            opt_depth = np.trapz(depth_integrand, x=r_mesa)
            smooth_depths[i] = opt_depth

    return depths, smooth_oms, smooth_depths



class StellarEVP():

    def __init__(self, is_hires=False):
        # Read in parameters and create output directory
        star_dir, star_file = name_star()
        Lmax = config.eigenvalue['Lmax']
        self.ells = np.arange(Lmax) + 1 #skip ell = 0
        ntheta = Lmax + 1
        nphi = 4
        hires_factor = config.eigenvalue['hires_factor']
        base_radial_scale = config.eigenvalue['radial_scale']
        self.do_hires = hires_factor != 1

        self.out_dir = 'eigenvalues'
        if not os.path.exists(self.out_dir):
            os.mkdir(self.out_dir)

        # Parameters
        ncc_cutoff = config.numerics['ncc_cutoff']
        resolutions = []
        for nr in config.star['nr']:
            if is_hires:
                resolutions.append((nphi, ntheta, int(base_radial_scale*hires_factor*nr)))
            else:
                resolutions.append((nphi, ntheta, int(base_radial_scale*nr)))
        self.nr = base_radial_scale*np.array(config.star['nr'])
        Re  = config.numerics['reynolds_target'] 
        Pr  = config.numerics['prandtl']
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

        self.nondim_S1 = S1_mesa[r_mesa/L_nd < self.r_outer] * tau_nd
        self.nondim_N2 = N2_mesa[r_mesa/L_nd < self.r_outer] * tau_nd**2

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

        self.compressible = SphericalCompressibleProblem(resolutions, self.stitch_radii, radius, self.ncc_file, dealias=dealias, dtype=np.complex128, mesh=None, sponge=sponge, do_rotation=do_rotation)
        self.compressible.make_fields()

        r_scale = config.numerics['N_dealias']/dealias[0]/base_radial_scale
        if is_hires:
            r_scale /= base_radial_scale*hires_factor
        variables, timescales = self.compressible.fill_structure(scales=(1,1,r_scale))
        self.compressible.set_substitutions(EVP=True)

        variables = self.compressible.namespace
        self.coords, self.dist, self.bases, self.bases_keys = self.compressible.coords, self.compressible.dist, self.compressible.bases, self.compressible.bases_keys


        if sponge:
            for i, bn in enumerate(self.compressible.bases.keys()):
                variables['sponge_{}'.format(bn)]['g'] *= tau_factor
        t_kep, t_heat, t_rot = timescales
        logger.info('timescales -- t_kep {}, t_heat {}, t_rot {}'.format(t_kep, t_heat, t_rot))


        # Problem
        self.namespace = variables
        omega = self.dist.Field(name='omega')
        self.namespace['dt'] = lambda A: -1j * omega * A
        self.namespace.update(locals())
        prob_variables = self.compressible.get_compressible_variables()
        self.problem = d3.EVP(prob_variables, eigenvalue=omega, namespace=self.namespace)
        self.problem = self.compressible.set_compressible_problem(self.problem)
        logger.info('problem built')

        logger.info('using ncc cutoff {:.2e}'.format(ncc_cutoff))
        self.solver = self.problem.build_solver(ncc_cutoff=ncc_cutoff)
        logger.info('solver built')

        if self.do_hires and not is_hires:
            self.hires_EVP = StellarEVP(is_hires=True)

        self.hires_factor = hires_factor

    def solve(self, ell):
        logger.info('solving eigenvalue with nr = {} at ell = {}'.format(self.nr, ell))
        lamb_freq = np.sqrt(ell*(ell+1)/2)*self.nondim_S1
        self.solver = solve_dense(self.solver, ell, lamb_freq=lamb_freq, bruntN2=self.nondim_N2)
        self.ell = ell

        for sbsys in self.solver.subsystems:
            ss_m, ss_ell, r_couple = sbsys.group
            if ss_ell == ell and ss_m == 0:
                self.subsystem = sbsys
                break

    def setup_sparse_solve(self, ell):
        logger.info('setting up sparse solve for ell = {}'.format(ell))
        for sbsys in self.solver.subsystems:
            ss_m, ss_ell, r_couple = sbsys.group
            if ss_ell == ell and ss_m == 0:
                self.subsystem = sbsys
                break

        for subproblem in self.solver.subproblems:
            this_ell = subproblem.group[1]
            if this_ell != ell:
                continue

            sp = subproblem

            ell = subproblem.group[1]

            subsystems.build_subproblem_matrices(self.solver, [sp], ['M', 'L'])
            matrix_info(sp)

            self.A =  (sp.L_min @ sp.pre_right)
            self.B = -  (sp.M_min @ sp.pre_right)
            self.solver.eigenvalue_subproblem = sp
            break

    def solve_sparse(self, ell, eigenvalues):
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
        self.ell = ell
        N = 1
        full_evals = []
        full_evecs = []
        start_time = time.time()

        sp = self.solver.eigenvalue_subproblem

        for i, target in enumerate(eigenvalues):
            evalue, evec = scipy_sparse_eigs(A=self.A, B=self.B, N=N, target=target, matsolver=self.solver.matsolver)

            for j in range((sp.pre_right @ evec).shape[-1]):
                full_evals.append(evalue.ravel()[j])
                full_evecs.append((sp.pre_right @ evec)[:,j].ravel())
        end_time = time.time()
        logger.debug('sparse solve done in {:.2f} sec'.format(end_time - start_time))
        self.solver.eigenvalues = np.array(full_evals)
        self.solver.eigenvectors = np.swapaxes(np.array(full_evecs), 0, 1)
        logger.debug('shapes: {} / {}'.format(self.solver.eigenvalues.shape, self.solver.eigenvectors.shape))

        return self.solver

    def check_eigen(self, cutoff=1e-4, r_cz=1, cz_width=0.05, depth_cutoff=None, max_modes=None):
        """
        Compare eigenvalues and eigenvectors between a hi-res and lo-res solve.
        Only keep the solutions that match to within the specified cutoff between the two cases.
        """
        if not self.do_hires:
            logger.info("hires_scales = 1; skipping check_eigen()")
            return

        depths, smooth_oms, smooth_depths = calculate_optical_depths(self.solver, self.bases_keys, self.stitch_radii, self.r_outer, self.ncc_file, self.namespace, ell=self.ell)
        
        self.hires_EVP.setup_sparse_solve(self.ell)
        namespace = self.hires_EVP.problem.namespace

        scales = (1,1,self.hires_factor)
        dist   = self.hires_EVP.dist
        coords = self.hires_EVP.coords
        cz_KE_ops = []
        tot_KE_ops = []
        for bn, basis in self.hires_EVP.bases.items():
            namespace['cz_envelope_{}'.format(bn)] = cz_envelope = dist.Field(bases=basis)
            namespace['cz_envelope_{}'.format(bn)]['g'] = one_to_zero(namespace['r_{}'.format(bn)], r_cz, width=cz_width)
            namespace['evp_u_{}'.format(bn)] = u = dist.VectorField(coords, bases=basis)
            namespace['conj_u_{}'.format(bn)] = conj_u = dist.VectorField(coords, bases=basis)
            rho = namespace['rho0_{}'.format(bn)]
           
            cz_KE_ops.append(d3.integ(cz_envelope*rho*conj_u@u/2))
            tot_KE_ops.append(d3.integ(rho*conj_u@u/2))
        cz_KE_op = 0
        tot_KE_op = 0
        for i in range(len(cz_KE_ops)):
            cz_KE_op += cz_KE_ops[i]
            tot_KE_op += tot_KE_ops[i]

        logger.info('finding good eigenvalues with cutoff {}'.format(cutoff))

        good_values = []
        for i, v1 in enumerate(self.solver.eigenvalues):
#            if np.abs(v1.real) < 1e-3*np.abs(v1.imag):
#                logger.debug('skipping eigenvalue {}; damps very quickly'.format(v1))
#                continue
            if v1.imag > 0:
                logger.debug('skipping eigenvalue {}; spurious growth mode'.format(v1))
                continue
            if depth_cutoff is not None and depths[i] > depth_cutoff:
                logger.debug('skipping eigenvalue {}; diffusively dominated'.format(v1))
                continue

            #Check if Evalue is spurious or fully in the CZ:
            self.solver.set_state(i, self.subsystem)
            ef_u1, ef_u1_pieces = clean_eigvecs('u', self.bases_keys, self.namespace, scales=scales)
            for j, basis in enumerate(self.hires_EVP.bases):
                bn = self.bases_keys[j]
                u = namespace['evp_u_{}'.format(bn)]
                conj_u = namespace['conj_u_{}'.format(bn)]
                u['g'] = ef_u1_pieces[j]['g']
                conj_u['g'] = np.conj(u['g']) 
            mode_KE1 = tot_KE_op.evaluate()['g'].ravel()[0].real
            cz_KE1 = cz_KE_op.evaluate()['g'].ravel()[0].real
            cz_KE_frac = cz_KE1/mode_KE1
            if cz_KE_frac.real > 0.5:
                logger.debug('skipping eigenvalue {}; located in CZ'.format(v1))
                continue
            elif cz_KE_frac.real < 1e-12:
                logger.debug('skipping eigenvalue {}; spurious mode without evanescent tail'.format(v1))
                continue

            #solve hires EVP and compare
            self.hires_EVP.solver = self.hires_EVP.solve_sparse(self.ell, [v1,])
            v2 = self.hires_EVP.solver.eigenvalues[0]
            real_goodness = np.abs(v1.real - v2.real)/np.abs(v1.real).min()
            imag_goodness = np.abs(v1.imag - v2.imag)/np.abs(v1.imag).min()
            goodness = np.abs(v1 - v2)/np.abs(v1).min()

            logger.debug('Value: {:.3e} / Goodness: {:.3e}, Real: {:.3e}, Imag: {:.3e}'.format(v1, goodness, real_goodness, imag_goodness))

            if goodness < cutoff:
                self.hires_EVP.solver.set_state(0, self.hires_EVP.subsystem)
                ef_u2, ef_u2_pieces = clean_eigvecs('u', self.hires_EVP.bases_keys, self.hires_EVP.namespace)
                print([np.max(np.abs(ef_u2[ind,:])) for ind in range(3)])
                for j, basis in enumerate(self.hires_EVP.bases):
                    bn = self.bases_keys[j]
                    u = namespace['evp_u_{}'.format(bn)]
                    conj_u = namespace['conj_u_{}'.format(bn)]
                    u['g'] = ef_u2_pieces[j]['g']
                    conj_u['g'] = np.conj(u['g']) 
                mode_KE2 = tot_KE_op.evaluate()['g'].ravel()[0].real

                vector_diff = np.max(np.abs(mode_KE1/mode_KE2 - 1))
                if vector_diff < np.sqrt(cutoff):

                    logger.info('good evalue {} w/ vdiff {} and czfrac {}'.format(v1, vector_diff, cz_KE_frac.real))
#                    tots = []
#                    czs = []
#                    for i in range(len(cz_KE_ops)):
#                        czs.append(cz_KE_ops[i].evaluate()['g'].ravel()[0].real)
#                        tots.append(tot_KE_ops[i].evaluate()['g'].ravel()[0].real)
#                    logger.info('cz ops: {}, tot ops: {}'.format(czs, tots))
                    good_values.append(i)
                    if max_modes is not None and len(good_values) == max_modes:
                        logger.info('reached {} modes == max modes; breaking'.format(max_modes))
                        break
                else:
                    logger.debug('skipping eigenvalue {}; vector diff is {}'.format(v1, vector_diff))
                    continue

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
                if this_ell == self.ell and m == 0:
                    good[i,j] = True

        integ_energy_op = None
        rho_fields = []
        s1_surf = None
        enth_fluc_surf = None
        for i, bn in enumerate(self.bases_keys):
#            p, u = [self.namespace['{}_{}'.format(f, bn)] for f in ['p', 'u']]
#            self.namespace['pressure_{}'.format(bn)] = p + self.namespace['pomega_tilde_{}'.format(bn)]
            self.namespace['KE_{}'.format(bn)] = self.dist.Field(bases=self.bases[bn], name='KE_{}'.format(bn))
            rho_fields.append(self.namespace['rho0_{}'.format(bn)]['g'][0,0,:])

            if integ_energy_op is None:
                integ_energy_op = d3.integ(self.namespace['KE_{}'.format(bn)])
            else:
                integ_energy_op += d3.integ(self.namespace['KE_{}'.format(bn)])

            if i == len(self.bases_keys) - 1:
                s1_surf = self.namespace['s1_{}'.format(bn)](r=self.r_outer)
                enth_fluc_surf = self.namespace['enthalpy_fluc_{}'.format(bn)](r=self.r_outer)
        rho_full = np.concatenate(rho_fields, axis=-1)

        integ_energies = np.zeros_like(self.solver.eigenvalues, dtype=np.float64) 
        s1_amplitudes = []
        enth_amplitudes = []
        velocity_eigenfunctions = []
        velocity_eigenfunctions_pieces = []
        entropy_eigenfunctions = []
        ln_rho_eigenfunctions = []
        enthalpy_eigenfunctions = []
        full_velocity_eigenfunctions = []
        full_velocity_eigenfunctions_pieces = []
        full_entropy_eigenfunctions = []
        full_ln_rho_eigenfunctions = []
        full_enthalpy_eigenfunctions = []

        for i, e in enumerate(self.solver.eigenvalues):
            print('doing {}'.format(e))
            self.solver.set_state(i, self.subsystem)

#            #Get eigenvectors
#            for j, bn in enumerate(self.bases_keys):
#                self.namespace['pressure_field_{}'.format(bn)] = self.namespace['pressure_{}'.format(bn)].evaluate()

            ef_u, ef_u_pieces = clean_eigvecs('u', self.bases_keys, self.namespace, shift=False)
            ef_s1, ef_s1_pieces = clean_eigvecs('s1', self.bases_keys, self.namespace, shift=False)
            ef_ln_rho1, ef_ln_rho1_pieces = clean_eigvecs('ln_rho1', self.bases_keys, self.namespace, shift=False)
            ef_enth_fluc, ef_enth_fluc_pieces = clean_eigvecs('enthalpy_fluc', self.bases_keys, self.namespace, shift=False)

            #normalize & store eigenvectors
            
            ix = np.unravel_index(np.argmax(np.abs(ef_u[2,:])), ef_u[2,:].shape)
            shift = ef_u[2,:][ix]
            for data in [ef_u, ef_s1, ef_ln_rho1, ef_enth_fluc]:
                data[:] /= shift
            for piece_tuple in [ef_u_pieces, ef_s1_pieces, ef_ln_rho1_pieces, ef_enth_fluc_pieces]:
                for data in piece_tuple:
                    data['g'][:] /= shift

            print('u mags', [np.max(np.abs(ef_u[ind,:])) for ind in range(3)])
            print('s1 mags', [np.max(np.abs(ef_s1[:]))])

            vec_slices = (slice(None), slice(ix[0], ix[0]+1), slice(ix[1], ix[1]+1), slice(None))
            scalar_slices = vec_slices[1:]

            velocity_eigenfunctions.append(ef_u[vec_slices].squeeze())
            velocity_eigenfunctions_pieces.append([p['g'][vec_slices].squeeze() for p in ef_u_pieces])
            entropy_eigenfunctions.append(ef_s1[scalar_slices].squeeze())
            ln_rho_eigenfunctions.append(ef_ln_rho1[scalar_slices].squeeze())
            enthalpy_eigenfunctions.append(ef_enth_fluc[scalar_slices].squeeze())
            full_velocity_eigenfunctions.append(ef_u)
            full_velocity_eigenfunctions_pieces.append([np.copy(p['g']) for p in ef_u_pieces])
            full_entropy_eigenfunctions.append(ef_s1)
            full_ln_rho_eigenfunctions.append(ef_ln_rho1)
            full_enthalpy_eigenfunctions.append(ef_enth_fluc)

#            #Kinetic energy
            for j, bn in enumerate(self.bases_keys):
                rho = self.namespace['rho0_{}'.format(bn)]
                u_squared = np.sum(ef_u_pieces[j]['g']*np.conj(ef_u_pieces[j]['g']), axis=0)
                self.namespace['KE_{}'.format(bn)]['g'] = (rho['g']*u_squared.real/2)
            integ_energy = integ_energy_op.evaluate()
            integ_energies[i] = integ_energy['g'].min().real / 2 #factor of 2 accounts for spherical harmonic integration (we're treating the field like an ell = 0 one)

            #Surface entropy perturbations
            for j, bn in enumerate(self.bases_keys):
                self.namespace['s1_{}'.format(bn)]['g'] = ef_s1_pieces[j]['g']
            s1_surf_vals = s1_surf.evaluate()['g'][scalar_slices]
            enth_fluc_surf_vals = enth_fluc_surf.evaluate()['g'][scalar_slices]
            s1_amplitudes.append(np.copy(s1_surf_vals))
            enth_amplitudes.append(np.copy(s1_surf_vals))

        r = []
        bruntN2 = []
        for bn in self.bases_keys:
            r.append(self.namespace['r1_{}'.format(bn)])
            bruntN2.append(-self.namespace['g_{}'.format(bn)]['g'][2,0,0,:]*self.namespace['grad_s0_{}'.format(bn)]['g'][2,0,0,:]/self.namespace['Cp']['g'])
        r = np.concatenate(r, axis=-1)
        bruntN2 = np.concatenate(bruntN2, axis=-1)

        with h5py.File('{:s}/ell{:03d}_eigenvalues.h5'.format(self.out_dir, self.ell), 'w') as f:
            f['good_evalues'] = self.solver.eigenvalues
            f['good_omegas']  = self.solver.eigenvalues.real
            f['good_evalues_inv_day'] = self.solver.eigenvalues/self.tau_day
            f['good_omegas_inv_day']  = self.solver.eigenvalues.real/self.tau_day
            f['s1_amplitudes']  = s1_amplitudes
            f['enth_amplitudes']  = enth_amplitudes
            f['integ_energies'] = integ_energies
            f['velocity_eigenfunctions'] = np.array(velocity_eigenfunctions)
            f['entropy_eigenfunctions'] = np.array(entropy_eigenfunctions)
            f['ln_rho_eigenfunctions'] = np.array(ln_rho_eigenfunctions)
            f['enthalpy_fluc_eigenfunctions'] = np.array(enthalpy_eigenfunctions)
#            f['full_velocity_eigenfunctions'] = np.array(full_velocity_eigenfunctions)
#            f['full_entropy_eigenfunctions'] = np.array(full_entropy_eigenfunctions)
#            f['full_ln_rho_eigenfunctions'] = np.array(full_ln_rho_eigenfunctions)
#            f['full_enthalpy_fluc_eigenfunctions'] = np.array(full_enthalpy_eigenfunctions)
            for i, bn in enumerate(self.bases_keys):
                f['r_{}'.format(bn)] = self.namespace['r1_{}'.format(bn)]
                f['rho_{}'.format(bn)] = self.namespace['rho0_{}'.format(bn)]['g']
                for j in range(len(self.solver.eigenvalues)):
#                    f['pieces/velocity_eigenfunctions_piece_{}_{}'.format(j, bn)] = velocity_eigenfunctions_pieces[j][i]
                    f['pieces/full_velocity_eigenfunctions_piece_{}_{}'.format(j, bn)] = full_velocity_eigenfunctions_pieces[j][i]
            f['r'] = r
            f['rho_full'] = rho_full
            f['depths'] = np.array(depths)
            f['smooth_oms'] = smooth_oms
            f['smooth_depths'] = smooth_depths
            f['bruntN2'] = bruntN2

            #Pass through nondimensionalization

            if self.ncc_file is not None:
                with h5py.File(self.ncc_file, 'r') as nccf:
                    f['tau_nd'] = nccf['tau_nd'][()]
                    f['m_nd']   = nccf['m_nd'][()]
                    f['L_nd']   = nccf['L_nd'][()]   
                    f['T_nd']   = nccf['T_nd'][()]   
                    f['rho_nd'] = nccf['rho_nd'][()] 
                    f['s_nd']   = nccf['s_nd'][()]   

    def get_duals(self, ell=None, cleanup=False, max_cond=None, discard_cond_cutoff=None):
        if ell is not None:
            self.ell = ell
        full_velocity_eigenfunctions_pieces = []
        with h5py.File('{:s}/ell{:03d}_eigenvalues.h5'.format(self.out_dir, self.ell), 'r') as f:
            evalues = f['good_evalues'][()]
            velocity_eigenfunctions = f['velocity_eigenfunctions'][()]
            for i in range(len(f['pieces'].keys())):
                key = 'pieces/full_velocity_eigenfunctions_piece_{}'.format(i)
                these_pieces = []
                for j, bn in enumerate(self.bases_keys):
                    this_key = '{}_{}'.format(key, bn)
                    if this_key in f.keys():
                        these_pieces.append(f[this_key][()])
                if len(these_pieces) > 0:
                    full_velocity_eigenfunctions_pieces.append(these_pieces)

        #Calculate duals

        dist = self.dist
        work_fields = []
        conj_work_fields = []
        ip_elements = []
        int_field = None
        for bn, basis in self.bases.items():
            work_fields.append(dist.VectorField(self.coords, bases=basis))
            conj_work_fields.append(dist.VectorField(self.coords, bases=basis))
            r = self.namespace['r_{}'.format(bn)]
            dr = np.gradient(r, axis=-1)
            ip_elements.append(4*np.pi*r**2*dr*np.exp(self.namespace['ln_rho0_{}'.format(bn)]['g'][0,0,:]))

            if int_field is None:
                int_field = d3.integ(self.namespace['rho0_{}'.format(bn)]*conj_work_fields[-1]@work_fields[-1])
            else:
                int_field += d3.integ(self.namespace['rho0_{}'.format(bn)]*conj_work_fields[-1]@work_fields[-1])

        def IP_fast(velocity_list1, velocity_list2):
            """ Integrate the bra-ket of two eigenfunctions of velocity. """
            ip = 0
            for i, bn in enumerate(self.bases.keys()):
                ip += np.sum(ip_elements[i]*np.conj(velocity_list1[i])*velocity_list2[i])
            return ip

        def IP_slow(velocity_list1, velocity_list2):
            """ Integrate the bra-ket of two eigenfunctions of velocity. """
            for i, bn in enumerate(self.bases.keys()):
                velocity1 = velocity_list1[i]
                velocity2 = velocity_list2[i]
                conj_work_fields[i]['g'] = np.conj(velocity1)
                work_fields[i]['g'] = velocity2
            return int_field.evaluate()['g'].min()

        duals, discard = calculate_duals(velocity_eigenfunctions, self.bases, dist, IP=IP_fast, max_cond=max_cond, discard_cond_cutoff=discard_cond_cutoff)
        with h5py.File('{:s}/ell{:03d}_eigenvalues.h5'.format(self.out_dir, self.ell), 'r') as f:
            with h5py.File('{:s}/duals_ell{:03d}_eigenvalues.h5'.format(self.out_dir, self.ell), 'w') as df:
                for k in f.keys():
                    if k == 'pieces': continue
                    if k in ['r', 'rho_full', 'depths', 'smooth_oms', 'smooth_depths', 'bruntN2', 'tau_nd', 'm_nd', 'L_nd', 'T_nd', 'rho_nd', 's_nd']:
                        df.create_dataset(k, data=f[k])
                        continue
                    found = False
                    for i, bn in enumerate(self.bases_keys):
                        if k in ['r_{}'.format(bn), 'rho_{}'.format(bn)]:
                            df.create_dataset(k, data=f[k])
                            found = True
                    if found:
                        continue
                    df.create_dataset(k, data=f[k][:duals.shape[0]])
                df['velocity_duals'] = duals
                df['discard'] = discard
        if cleanup:
            #remove large pre-dual file.
            os.remove('{:s}/ell{:03d}_eigenvalues.h5'.format(self.out_dir, self.ell))

        gc.collect()
