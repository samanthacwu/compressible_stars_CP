"""
d3 script for eigenvalue problem of anelastic convection / waves in a massive star.

"""
import gc
import os
import sys
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

from d3_stars.simulations.anelastic_functions import make_bases, make_fields, fill_structure, get_anelastic_variables, set_anelastic_problem
from d3_stars.simulations.evp_functions import solve_sparse, solve_dense, combine_eigvecs, check_eigen, calculate_duals
from d3_stars.simulations.parser import parse_std_config

duals_only = False

if __name__ == '__main__':
    # Read in parameters and create output directory
    config, raw_config, star_dir, star_file = parse_std_config('controls.cfg')
    Lmax = config['lmax']
    ntheta = Lmax + 1
    nphi = 2*ntheta
    hires_factor = config['hires_factor']
    do_hires = hires_factor != 1

    out_dir = 'eigenvalues'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # Parameters
    ncc_cutoff = config['ncc_cutoff']
    resolutions = []
    resolutions_hi = []
    for nr in config['nr']:
        resolutions.append((nphi, ntheta, nr))
        resolutions_hi.append((nphi, ntheta, int(hires_factor*nr)))
    Re  = config['reynolds_target'] 
    Pr  = config['prandtl']
    Pe  = Pr*Re
    ncc_file  = star_file

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

    #Create bases
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
                                            sponge=sponge, do_rotation=do_rotation, scales=(1,1,1.5))

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

    logger.info('using ncc cutoff {:.2e}'.format(ncc_cutoff))
    solver = problem.build_solver(ncc_cutoff=ncc_cutoff)
    logger.info('solver built')

    if do_hires:
        coords_hi, dist_hi, bases_hi, bases_keys_hi = make_bases(resolutions_hi, stitch_radii, radius, dealias=dealias, dtype=np.complex128, mesh=None)
        variables_hi = make_fields(bases_hi, coords_hi, dist_hi, 
                                vec_fields=vec_fields, scalar_fields=scalar_fields, 
                                vec_taus=vec_taus, scalar_taus=scalar_taus, 
                                vec_nccs=vec_nccs, scalar_nccs=scalar_nccs,
                                sponge=sponge, do_rotation=do_rotation)


        variables_hi, timescales_hi = fill_structure(bases_hi, dist_hi, variables_hi, ncc_file, r_outer, Pe, 
                                                vec_fields=vec_fields, vec_nccs=vec_nccs, scalar_nccs=scalar_nccs,
                                                sponge=sponge, do_rotation=do_rotation, scales=(1, 1, 1)) #dealias / hires_factor

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

#    for bk in bases_keys:
#        plt.plot(variables['r_{}'.format(bk)][0,0,:], variables['rho_{}'.format(bk)]['g'][0,0,:])
#        plt.plot(variables_hi['r_{}'.format(bk)][0,0,:], variables_hi['rho_{}'.format(bk)]['g'][0,0,:])
#    plt.show()

    #ell = 1 solve
    for i in range(Lmax):
        ell = i + 1
        if not duals_only:
            logger.info('solving lores eigenvalue with nr = {}'.format(config['nr']))
            solve_dense(solver, ell)

            if do_hires:
                logger.info('solving hires eigenvalue with hires_factor = {}'.format(hires_factor))
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
            s1_surf = None
            for i, bn in enumerate(bases_keys):
                p, u = [variables['{}_{}'.format(f, bn)] for f in ['p', 'u']]
                variables['pomega_hat_{}'.format(bn)] = p - 0.5*d3.dot(u,u)
                variables['KE_{}'.format(bn)] = dist.Field(bases=bases[bn], name='KE_{}'.format(bn))
                rho_fields.append(variables['rho_{}'.format(bn)]['g'][0,0,:])

                if integ_energy_op is None:
                    integ_energy_op = d3.integ(variables['KE_{}'.format(bn)])
                else:
                    integ_energy_op += d3.integ(variables['KE_{}'.format(bn)])

                if i == len(bases_keys) - 1:
                    s1_surf = variables['s1_{}'.format(bn)](r=radius)
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
