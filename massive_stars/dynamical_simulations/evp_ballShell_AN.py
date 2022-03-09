"""
d3 script for eigenvalue problem of anelastic convection / waves in a massive star.

Usage:
    evp_ballShell_AN.py [options]
    evp_ballShell_AN.py <config> [options]

Options:
    --Re=<Re>            The Reynolds number of the numerical diffusivities [default: 5e1]
    --Pr=<Prandtl>       The Prandtl number  of the numerical diffusivities [default: 1]
    --L=<Lmax>           Angular resolution (Lmax   [default: 1]
    --nrB=<Nmax>          The ball radial degrees of freedom (Nmax+1)   [default: 64]
    --nrS=<Nmax>          The shell radial degrees of freedom (Nmax+1)   [default: 64]
    --nrB_hi=<Nmax>       The hires-ball radial degrees of freedom (Nmax+1)
    --nrS_hi=<Nmax>       The hires-shell radial degrees of freedom (Nmax+1)

    --label=<label>      A label to add to the end of the output directory

    --ncc_file=<f>      path to a .h5 file of ICCs, curated from a MESA model
    --ncc_file_hi=<f>   path to a .h5 file of ICCs, curated from a MESA model (for hires solve)
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

from anelastic_functions import make_bases, make_fields, fill_structure, get_anelastic_variables, set_anelastic_problem

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

        #Update solver
        solver.eigenvalues = values
        solver.eigenvectors = vectors
        return solver

def combine_eigvecs(field, ell_m_bool, bases, namespace, scales=None, shift=True):
    fields = []
    nr = []
    ef_base = []
    vector = False
    for i, bn in enumerate(bases.keys()):
        fields.append(namespace['{}_{}'.format(field, bn)])
        nr.append(namespace['resolutions'][i][-1])

        this_field = fields[-1]


        if len(this_field.tensorsig) == 1:
            vector = True

        if scales is not None:
            this_field.change_scales(scales[i])
        this_field['c']
        this_field.towards_grid_space()

        if vector:
            ef_base.append(this_field.data[:,ell_m_bool,:].squeeze())
        else:
            ef_base.append(this_field.data[ell_m_bool,:].squeeze())

    ef = np.concatenate(ef_base, axis=-1)

    if shift:
        if vector:
            divisor = np.max(np.abs(ef[2,:]))
        else:
            divisor = np.max(np.abs(ef))

        ef /= divisor

    if vector:
        #go from plus/minus to theta/phi
        ef_u = np.zeros_like(ef)
        ef_u[0,:] = (1j/np.sqrt(2))*(ef[1,:] - ef_u[0,:])
        ef_u[1,:] = ( 1/np.sqrt(2))*(ef[1,:] + ef_u[0,:])
        ef_u[2,:] = ef_u[2,:]
        ef[:] = ef_u[:]

    return ef, ef_base



def check_eigen(solver1, solver2, bases1, bases2, namespace1, namespace2, ell, cutoff=1e-2):
    """
    Compare eigenvalues and eigenvectors between a hi-res and lo-res solve.
    Only keep the solutions that match to within the specified cutoff between the two cases.
    """
    good_values1 = []
    good_values2 = []
    cutoff2 = np.sqrt(cutoff)

    scales = []
    for i, bn in enumerate(bases1.keys()):
        scales.append((1, 1, namespace2['resolutions_hi'][i][-1]/namespace1['resolutions'][i][-1]))

    needed_vars = ['ρ', 'u', 'r']
    ρB1, uB1, rB1 = [namespace1[v+'_B'] for v in needed_vars]
    ρS1, uS1, rS1 = [namespace1[v+'_S1'] for v in needed_vars]
    ρB2, uB2, rB2 = [namespace2[v+'_B'] for v in needed_vars]
    ρS2, uS2, rS2 = [namespace2[v+'_S1'] for v in needed_vars]

    ρ2 = np.concatenate((ρB2['g'][0,0,:].flatten(), ρS2['g'][0,0,:].flatten()))
    r1 = np.concatenate((rB1.flatten(), rS1.flatten()))
    r2 = np.concatenate((rB2.flatten(), rS2.flatten()))

    logger.info('solving lores eigenvalue with nr = ({}, {})'.format(nrB, nrS))
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
        for j, v2 in enumerate(solver2.eigenvalues):
            real_goodness = np.abs(v1.real - v2.real)/np.abs(v1.real).min()
            goodness = np.abs(v1 - v2)/np.abs(v1).min()
            if goodness < cutoff:# or (j == 0 and (i == 2 or i == 3)):# and (np.abs(v1.imag - v2.imag)/np.abs(v1.imag)).min() < 1e-1:
                print(v1/(2*np.pi), v2/(2*np.pi))
                
                for f in [uB1, uS1, uB2, uS2]:
                    f['c']
                    f.towards_grid_space()
#                print((np.abs(v1 - v2)/np.abs(v1)).min())
                solver1.set_state(i, subsystem1)
                solver2.set_state(j, subsystem2)

                ef_u1, ef_u1_pieces = combine_eigvecs('u', good, bases1, namespace1, scales=scales)
                ef_u2, ef_u2_pieces = combine_eigvecs('u', good, bases2, namespace2)

                #If mode KE is inside of the convection zone then it's a bad mode.
                mode_KE = ρ2*np.sum(ef_u2*np.conj(ef_u2), axis=0).real/2
                cz_KE = np.sum((mode_KE*4*np.pi*r2**2*rough_dr2)[r2 <= 1])
                tot_KE = np.sum((mode_KE*4*np.pi*r2**2*rough_dr2))
                cz_KE_frac = cz_KE/tot_KE
                vector_diff = np.max(np.abs(ef_u1 - ef_u2))
#                if vector_diff < cutoff2 and cz_KE_frac < 0.5:
                if vector_diff < cutoff2:
                    print('good evalue w/ vdiff', vector_diff, 'czfrac', cz_KE_frac.real)
                    good_values1.append(i)
                    good_values2.append(j)


    solver1.eigenvalues = solver1.eigenvalues[good_values1]
    solver2.eigenvalues = solver2.eigenvalues[good_values2]
    solver1.eigenvectors = solver1.eigenvectors[:, good_values1]
    solver2.eigenvectors = solver2.eigenvectors[:, good_values2]
    return solver1, solver2

#    def calculate_duals(velocity_listB, velocity_listS, rhoB, rhoS, work_fieldB, work_fieldS, coord):
#        """
#        Calculate the dual basis of the velocity eigenvectors.
#        """
#        int_field = d3.Integrate(rhoB*work_fieldB, coord) + d3.Integrate(rhoS*work_fieldS, coord)
#        def IP(velocities1, velocities2):
#            """ Integrate the bra-ket of two eigenfunctions of velocity. """
#            work_fieldB['g'] = np.sum(velocities1[0]*np.conj(velocities2[0]), axis=0)
#            work_fieldS['g'] = np.sum(velocities1[1]*np.conj(velocities2[1]), axis=0)
#            return int_field.evaluate()['g'].min()
#
#
#        velocity_listB = np.array(velocity_listB)
#        velocity_listS = np.array(velocity_listS)
#        n_modes = velocity_listB.shape[0]
#        IP_matrix = np.zeros((n_modes, n_modes), dtype=np.complex128)
#        for i in range(n_modes):
#            if i % 10 == 0: logger.info("duals {}/{}".format(i, n_modes))
#            for j in range(n_modes):
#                IP_matrix[i,j] = IP((velocity_listB[i], velocity_listS[i]), (velocity_listB[j], velocity_listS[j]))
#        
#        print('dual IP matrix cond: {:.3e}'.format(np.linalg.cond(IP_matrix)))
#        IP_inv = np.linalg.inv(IP_matrix)
#
#        vel_dualB = np.zeros_like(velocity_listB)
#        vel_dualS = np.zeros_like(velocity_listS)
#        for i in range(3):
#            vel_dualB[:,i,:] = np.einsum('ij,ik->kj', velocity_listB[:,i,:], np.conj(IP_inv))
#            vel_dualS[:,i,:] = np.einsum('ij,ik->kj', velocity_listS[:,i,:], np.conj(IP_inv))
#
#        return np.concatenate((vel_dualB, vel_dualS), axis=-1)



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
    nrS = int(args['--nrS'])
    resolutionB = (nphi, ntheta, nrB)
    resolutionS = (nphi, ntheta, nrS)
    Re  = float(args['--Re'])
    Pr  = 1
    Pe  = Pr*Re
    ncc_file  = args['--ncc_file']

    do_hires = (args['--nrB_hi'] is not None and args['--nrS_hi'] is not None)

    if do_hires:
        nrB_hi = int(args['--nrB_hi'])
        nrS_hi = int(args['--nrS_hi'])
        ncc_file_hi = args['--ncc_file_hi']
        resolutionB_hi = (nphi, ntheta, nrB_hi)
        resolutionS_hi = (nphi, ntheta, nrS_hi)
    else:
        nrB_hi = nrS_hi = ncc_file_hi = None

    out_dir = './' + sys.argv[0].split('.py')[0]
    if args['--ncc_file'] is None:
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

    if ncc_file is not None:
        with h5py.File(ncc_file, 'r') as f:
            Ri = f['r_inner'][()]
            Ro = f['r_outer'][()]
            tau_s = f['tau'][()]
            tau = tau_s/(60*60*24)
            N2_mesa = f['N2_mesa'][()]
            r_mesa = f['r_mesa'][()]
            L_mesa = f['L'][()]
    else:
        Ri = 1.1
        Ro = 1.5
        tau = 1

    sponge = False
    do_rotation = False

    logger.info('r_inner: {:.2f} / r_outer: {:.2f}'.format(Ri, Ro))

    #Create bases
    resolutions = (resolutionB, resolutionS)
    stitch_radii = (Ri,)
    radius = Ro
    coords, dist, bases, bases_keys = make_bases(resolutions, stitch_radii, radius, dealias=1, dtype=np.complex128, mesh=None)

    vec_fields = ['u',]
    scalar_fields = ['p', 's1', 'inv_T', 'H', 'ρ', 'T']
    vec_taus = ['tau_u']
    scalar_taus = ['tau_s']
    vec_nccs = ['grad_ln_ρ', 'grad_ln_T', 'grad_s0', 'grad_T', 'grad_inv_Pe_rad']
    scalar_nccs = ['ln_ρ', 'ln_T', 'inv_Pe_rad', 'sponge']
    variables = make_fields(bases, coords, dist, 
                            vec_fields=vec_fields, scalar_fields=scalar_fields, 
                            vec_taus=vec_taus, scalar_taus=scalar_taus, 
                            vec_nccs=vec_nccs, scalar_nccs=scalar_nccs,
                            sponge=sponge, do_rotation=do_rotation)


    variables, timescales = fill_structure(bases, dist, variables, ncc_file, Ro, Pe, 
                                            vec_fields=vec_fields, vec_nccs=vec_nccs, scalar_nccs=scalar_nccs,
                                            sponge=sponge, do_rotation=do_rotation)
    variables.update(locals())
    omega = dist.Field(name='omega')
    variables['dt'] = lambda A: -1j * omega * A

    prob_variables = get_anelastic_variables(bases, bases_keys, variables)

    problem = d3.EVP(prob_variables, eigenvalue=omega, namespace=variables)

    problem = set_anelastic_problem(problem, bases, bases_keys, stitch_radii=stitch_radii)
    logger.info('problem built')

    solver = problem.build_solver()
    logger.info('solver built')

    if do_hires:
        resolutions_hi = (resolutionB_hi, resolutionS_hi)
        coords_hi, dist_hi, bases_hi, bases_keys_hi = make_bases(resolutions_hi, stitch_radii, radius, dealias=1, dtype=np.complex128, mesh=None)
        variables_hi = make_fields(bases_hi, coords_hi, dist_hi, 
                                vec_fields=vec_fields, scalar_fields=scalar_fields, 
                                vec_taus=vec_taus, scalar_taus=scalar_taus, 
                                vec_nccs=vec_nccs, scalar_nccs=scalar_nccs,
                                sponge=sponge, do_rotation=do_rotation)


        variables_hi, timescales_hi = fill_structure(bases_hi, dist_hi, variables_hi, ncc_file_hi, Ro, Pe, 
                                                vec_fields=vec_fields, vec_nccs=vec_nccs, scalar_nccs=scalar_nccs,
                                                sponge=sponge, do_rotation=do_rotation)


        variables_hi.update(locals())
        omega_hi = dist_hi.Field(name='omega_hi')
        variables_hi['dt'] = lambda A: -1j * omega_hi * A
        prob_variables_hi = get_anelastic_variables(bases_hi, bases_keys_hi, variables_hi)
        problem_hi = d3.EVP(prob_variables_hi, eigenvalue=omega_hi, namespace=variables_hi)
        problem_hi = set_anelastic_problem(problem_hi, bases_hi, bases_keys_hi, stitch_radii=stitch_radii)
        logger.info('hires problem built')
        solver_hi = problem_hi.build_solver()
        logger.info('hires solver built')

    #ell = 1 solve
    for i in range(Lmax):
        ell = i + 1
        solve_dense(solver, ell)

        if do_hires:
            solve_dense(solver_hi, ell)
            print(variables['u_B']['g'].shape, variables_hi['u_B']['g'].shape, 'meh')
            solver, solver_hi = check_eigen(solver, solver_hi, bases, bases_hi, variables, variables_hi, ell, cutoff=1e-2)
    


#TODO address this
#        #Calculate 'optical depths' of each mode.
#        depths = []
#        for om in solver1.eigenvalues.real:
#            Lambda = np.sqrt(ell*(ell+1))
#            kr_cm = np.sqrt(N2_mesa)*Lambda/(r_mesa* (om/tau_s))
#            v_group = (om/tau_s) / kr_cm
#            inv_Pe = np.ones_like(r_mesa) / Pe
#            inv_Pe[r_mesa/L_mesa > 1.1] = interp1d(namespace1['rS'].flatten(), namespace1['inv_PeS']['g'][0,0,:], bounds_error=False, fill_value='extrapolate')(r_mesa[r_mesa/L_mesa > 1.1]/L_mesa)
#            k_rad = (L_mesa**2 / tau_s) * inv_Pe
#            gamma_rad = k_rad * kr_cm**2
#            depth_integrand = np.gradient(r_mesa) * gamma_rad/v_group
#
#            opt_depth = 0
#            for i, rv in enumerate(r_mesa):
#                if rv/L_mesa > 1.0 and rv/L_mesa < r_outer:
#                    opt_depth += depth_integrand[i]
#            depths.append(opt_depth)
#
        shape = list(variables['s1_B']['c'].shape[:2])
        good = np.zeros(shape, bool)
        for i in range(shape[0]):
            for j in range(shape[1]):
                grid_space = (False,False)
                elements = (np.array((i,)),np.array((j,)))
                m, this_ell = bases['B'].sphere_basis.elements_to_groups(grid_space, elements)
                if this_ell == ell and m == 1:
                    good[i,j] = True

        needed_fields = ['ρ', 'u', 'p', 's1']
        ρB, uB, pB, s1B = [variables[f + '_B'] for f in needed_fields]
        ρS, uS, pS, s1S = [variables[f + '_S1'] for f in needed_fields]
        for i, bn in enumerate(bases_keys):
            p, u = [variables['{}_{}'.format(f, bn)] for f in ['p', 'u']]
            variables['pomega_hat_{}'.format(bn)] = p - 0.5*d3.dot(u,u)
#
#        basisB, basisS = namespace1['basisB'], namespace1['basisS']
#        dist, coords = namespace1['dist'], namespace1['coords']
#        shell_ell, shell_m = namespace1['shell_ell'], namespace1['shell_m']
#
#        vol_int = lambda A: d3.Integrate(A, coords)
#        s1_surf = s1S(r=r_outer)
#        KEB  = dist.Field(bases=basisB, name="KEB")
#        KES  = dist.Field(bases=basisS, name="KES")
#        integ_energy_op = vol_int(KEB) + vol_int(KES)
#
#        integ_energies = np.zeros_like(   solver1.eigenvalues, dtype=np.float64) 
#        s1_amplitudes = np.zeros_like(solver1.eigenvalues, dtype=np.float64)  
        velocity_eigenfunctions = []
        velocity_eigenfunctions_pieces = []
        entropy_eigenfunctions = []
        wave_flux_eigenfunctions = []
#
#        subsystem = subsystem1
#        print('using subsystem ', subsystem.group, ' for eigenvectors')

        for sbsys in solver.subsystems:
            ss_m, ss_ell, r_couple = sbsys.group
            if ss_ell == ell and ss_m == 1:
                subsystem = sbsys
                break

        ρ_full = np.concatenate((ρB['g'], ρS['g']), axis=-1)
        for i, e in enumerate(solver.eigenvalues):
            solver.set_state(i, subsystem)

            #Get eigenvectors
            for i, bn in enumerate(bases_keys):
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
            wave_flux = ρ_full*ef_u[2,:]*np.conj(ef_pom).squeeze()
#            wave_fluxB = (ρB['g'][0,0,:]*ef_uB[2,:]*np.conj(ef_pomB)).squeeze()
#            wave_fluxS = (ρS['g'][0,0,:]*ef_uS[2,:]*np.conj(ef_pomS)).squeeze()
#            wave_flux_eig = np.concatenate((wave_fluxB, wave_fluxS), axis=-1)
            wave_flux_eigenfunctions.append(wave_flux)

#            #Kinetic energy
#            KES['g'] = (ρS['g'][0,0,:]*np.sum(ef_uS*np.conj(ef_uS), axis=0)).real/2
#            KEB['g'] = (ρB['g'][0,0,:]*np.sum(ef_uB*np.conj(ef_uB), axis=0)).real/2
#            integ_energy = integ_energy_op.evaluate()
#            integ_energies[i] = integ_energy['g'].min().real / 2 #factor of 2 accounts for spherical harmonic integration (we're treating the field like an ell = 0 one)
#
#            #Surface entropy perturbations
#            s1S['g'] = 0
#            s1S['c']
#            s1S['g'] = ef_s1S
#            s1_surf_vals = s1_surf.evaluate()['g'] / np.sqrt(2) #sqrt(2) accounts for spherical harmonic integration
#            s1_amplitudes[i] = np.abs(s1_surf_vals.max())
#
#        with h5py.File('{:s}/ell{:03d}_eigenvalues.h5'.format(out_dir, ell), 'w') as f:
#            f['good_evalues'] = solver1.eigenvalues
#            f['good_omegas']  = solver1.eigenvalues.real
#            f['good_evalues_inv_day'] = solver1.eigenvalues/tau
#            f['good_omegas_inv_day']  = solver1.eigenvalues.real/tau
#            f['s1_amplitudes']  = s1_amplitudes
#            f['integ_energies'] = integ_energies
#            f['wave_flux_eigenfunctions'] = np.array(wave_flux_eigenfunctions)
#            f['velocity_eigenfunctions'] = np.array(velocity_eigenfunctions)
#            f['entropy_eigenfunctions'] = np.array(entropy_eigenfunctions)
#            f['rB'] = namespace1['rB']
#            f['rS'] = namespace1['rS']
#            f['ρB'] = namespace1['ρB']['g']
#            f['ρS'] = namespace1['ρS']['g']
#            f['depths'] = np.array(depths)
#
#        #TODO: Fix dual calculation
#        work_fieldB = dist.Field(name='workB', bases=basisB)
#        work_fieldS = dist.Field(name='workB', bases=basisS)
#        velocity_duals = calculate_duals(velocity_eigenfunctionsB, velocity_eigenfunctionsS, ρB, ρS, work_fieldB, work_fieldS, coords)
#        with h5py.File('{:s}/duals_ell{:03d}_eigenvalues.h5'.format(out_dir, ell), 'w') as f:
#            f['good_evalues'] = solver1.eigenvalues
#            f['good_omegas']  = solver1.eigenvalues.real
#            f['good_evalues_inv_day'] = solver1.eigenvalues/tau
#            f['good_omegas_inv_day']  = solver1.eigenvalues.real/tau
#            f['s1_amplitudes']  = s1_amplitudes
#            f['integ_energies'] = integ_energies
#            f['wave_flux_eigenfunctions'] = np.array(wave_flux_eigenfunctions)
#            f['velocity_eigenfunctions'] = np.array(velocity_eigenfunctions)
#            f['entropy_eigenfunctions'] = np.array(entropy_eigenfunctions)
#            f['velocity_duals'] = velocity_duals
#            f['rB'] = namespace1['rB']
#            f['rS'] = namespace1['rS']
#            f['ρB'] = namespace1['ρB']['g']
#            f['ρS'] = namespace1['ρS']['g']
#            f['depths'] = np.array(depths)
#
#        gc.collect()
