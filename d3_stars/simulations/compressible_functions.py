from collections import OrderedDict

import h5py
import numpy as np
import dedalus.public as d3

import logging
logger = logging.getLogger(__name__)

import d3_stars.defaults.config as config

def make_bases(resolutions, stitch_radii, radius, dealias=3/2, dtype=np.float64, mesh=None):
    bases = OrderedDict()
    coords  = d3.SphericalCoordinates('phi', 'theta', 'r')
    dist    = d3.Distributor((coords,), mesh=mesh, dtype=dtype)
    bases_keys = ['B']
    for i, resolution in enumerate(resolutions):
        if i == 0:
            if len(resolutions) == 1:
                ball_radius = radius
            else:
                ball_radius = stitch_radii[i]
            bases['B']   = d3.BallBasis(coords, resolution, radius=ball_radius, dtype=dtype, dealias=dealias)
        else:
            if len(resolutions) == i+1:
                shell_radii = (stitch_radii[i-1], radius)
            else:
                shell_radii = (stitch_radii[i-1], stitch_radii[i])
            bases['S{}'.format(i)] = d3.ShellBasis(coords, resolution, radii=shell_radii, dtype=dtype, dealias=dealias)
            bases_keys += ['S{}'.format(i)]
    return coords, dist, bases, bases_keys

def make_fields(bases, coords, dist, vec_fields=[], scalar_fields=[], vec_taus=[], scalar_taus=[], vec_nccs=[], scalar_nccs=[], sponge=False, do_rotation=False, sponge_function=lambda r: r**2):
    variables = OrderedDict()
    variables['exp'] = np.exp
    variables['log'] = np.log
    for basis_number, bn in enumerate(bases.keys()):
        unit_vectors = ['ephi', 'etheta', 'er', 'ex', 'ey', 'ez']
        basis = bases[bn]
        phi, theta, r = basis.local_grids(basis.dealias)
        phi1, theta1, r1 = basis.local_grids((1,1,1))
        variables['phi_'+bn], variables['theta_'+bn], variables['r_'+bn] = phi, theta, r
        variables['phi1_'+bn], variables['theta1_'+bn], variables['r1_'+bn] = phi1, theta1, r1

        #Define problem fields
        for fn in vec_fields:
            key = '{}_{}'.format(fn, bn)
            logger.debug('creating vector field {}'.format(key))
            variables[key] = dist.VectorField(coords, name=key, bases=basis)
        for fn in scalar_fields:
            key = '{}_{}'.format(fn, bn)
            logger.debug('creating scalar field {}'.format(key))
            variables[key] = dist.Field(name=key, bases=basis)

        #Taus
        S2_basis = basis.S2_basis()
        tau_names = []
        if type(basis) == d3.BallBasis:
            tau_names += ['',]
        else:
            tau_names += ['1', '2']
        for name in tau_names:
            for fn in vec_taus:
                key = '{}{}_{}'.format(fn, name, bn)
                logger.debug('creating vector tau {}'.format(key))
                variables[key] = dist.VectorField(coords, name=key, bases=S2_basis)
            for fn in scalar_taus:
                key = '{}{}_{}'.format(fn, name, bn)
                logger.debug('creating scalar tau {}'.format(key))
                variables[key] = dist.Field(name=key, bases=S2_basis)
        
        #Define problem NCCs
        for fn in vec_nccs + ['er_LHS', 'etheta_LHS', 'ephi_LHS', 'rvec_LHS']:
            key = '{}_{}'.format(fn, bn)
            logger.debug('creating vector NCC {}'.format(key))
            variables[key] = dist.VectorField(coords, name=key, bases=basis.radial_basis)
            if fn == 'er_LHS':
                variables[key]['g'][2] = 1
            if fn == 'etheta_LHS':
                variables[key]['g'][1] = 1
            if fn == 'ephi_LHS':
                variables[key]['g'][0] = 1
            elif fn == 'rvec_LHS':
                variables[key]['g'][2] = r1
        for fn in scalar_nccs:
            key = '{}_{}'.format(fn, bn)
            logger.debug('creating scalar NCC {}'.format(key))
            variables[key] = dist.Field(name=key, bases=basis.radial_basis)

        #Define identity matrix
        key = 'I_matrix_{}'.format(bn)
        logger.debug('creating identity matrix NCC {}'.format(key))
        variables[key] = dist.TensorField(coords, name=key, bases=basis.radial_basis)
        for i in range(3):
            variables[key]['g'][i,i,:] = 1

        #Define unit vectors
        for fn in unit_vectors:
            key = '{}_{}'.format(fn, bn)
            logger.debug('creating unit vector field {}'.format(key))
            variables[key] = dist.VectorField(coords, name=key, bases=basis)

        if sponge:
            variables['sponge_{}'.format(bn)]['g'] = sponge_function(r1)

        variables['er'] = dist.VectorField(coords, name='er')
        variables['ephi'] = dist.VectorField(coords, name='ephi')
        variables['etheta'] = dist.VectorField(coords, name='etheta')
        variables['er']['g'][2] = 1
        variables['ephi']['g'][0] = 1
        variables['etheta']['g'][1] = 1

        variables['ephi_{}'.format(bn)]['g'][0,:] = 1 
        variables['etheta_{}'.format(bn)]['g'][1,:] = 1 
        variables['er_{}'.format(bn)]['g'][2,:] = 1 

        variables['ex_{}'.format(bn)]['g'][0] = -np.sin(phi1)
        variables['ex_{}'.format(bn)]['g'][1] = np.cos(theta1)*np.cos(phi1)
        variables['ex_{}'.format(bn)]['g'][2] = np.sin(theta1)*np.cos(phi1)

        variables['ey_{}'.format(bn)]['g'][0] = np.cos(phi1)
        variables['ey_{}'.format(bn)]['g'][1] = np.cos(theta1)*np.sin(phi1)
        variables['ey_{}'.format(bn)]['g'][2] = np.sin(theta1)*np.sin(phi1)

        variables['ez_{}'.format(bn)]['g'][0] = 0
        variables['ez_{}'.format(bn)]['g'][1] = -np.sin(theta1)
        variables['ez_{}'.format(bn)]['g'][2] =  np.cos(theta1)
        
        for k in ['ex', 'ey', 'ez']:
            variables['{}_{}'.format(k, bn)] = d3.Grid(variables['{}_{}'.format(k, bn)]).evaluate()

        u = variables['u_{}'.format(bn)]
        ln_rho1 = variables['ln_rho1_{}'.format(bn)]
        T1 = variables['T1_{}'.format(bn)]
        I_mat = variables['I_matrix_{}'.format(bn)]
        grad_ln_rho0 = variables['grad_ln_rho0_{}'.format(bn)]
        grad_T0 = variables['grad_T0_{}'.format(bn)]
        rho0 = variables['rho0_{}'.format(bn)]
        ln_rho0 = variables['ln_rho0_{}'.format(bn)]
        T0 = variables['T0_{}'.format(bn)]
        g_phi = variables['g_phi_{}'.format(bn)]
        nu_diff = variables['nu_diff_{}'.format(bn)]

        variables['gamma'] = gamma = dist.Field(name='gamma')
        variables['R_gas'] = R_gas = dist.Field(name='R_gas')
        variables['Cp'] = Cp = dist.Field(name='Cp')
        variables['Cv'] = Cv = dist.Field(name='Cv')

        g = variables['g_{}'.format(bn)]
        er_LHS = variables['er_LHS_{}'.format(bn)]
        ephi_LHS = variables['ephi_LHS_{}'.format(bn)]
        etheta_LHS = variables['etheta_LHS_{}'.format(bn)]

        # Lift operators for boundary conditions
        variables['grad_u_RHS_{}'.format(bn)] = grad_u_RHS = d3.grad(u)
        variables['div_u_RHS_{}'.format(bn)] = div_u_RHS = d3.div(u)
        if type(basis) == d3.BallBasis:
            lift_basis = basis.clone_with(k=0)
            variables['lift_{}'.format(bn)] = lift_fn = lambda A: d3.Lift(A, lift_basis, -1)
            variables['taus_lnrho_{}'.format(bn)] = 0
            variables['taus_u_{}'.format(bn)] = lift_fn(variables['tau_u_{}'.format(bn)])
            variables['taus_T_{}'.format(bn)] = lift_fn(variables['tau_T_{}'.format(bn)])
            variables['grad_u_{}'.format(bn)] = grad_u = grad_u_RHS
            variables['grad_T1_{}'.format(bn)] = grad_T1 = d3.grad(T1)
            variables['grad_ln_rho1_{}'.format(bn)] = grad_ln_rho1 = d3.grad(ln_rho1)
            variables['div_u_{}'.format(bn)] = div_u = div_u_RHS
        else:
            lift_basis = basis.clone_with(k=2)
            variables['lift_{}'.format(bn)] = lift_fn = lambda A, n: d3.Lift(A, lift_basis, n)
            variables['taus_lnrho_{}'.format(bn)] = 0 #lift_fn(variables['tau_rho1_{}'.format(bn)], -1)
#            variables['taus_u_{}'.format(bn)] = lift_fn(variables['tau_u1_{}'.format(bn)], -1) + er_LHS*lift_fn(variables['tau_rho1_{}'.format(bn)], -2) + ephi_LHS*lift_fn(variables['tau_div_u1_{}'.format(bn)], -2) + etheta_LHS*lift_fn(variables['tau_ur1_{}'.format(bn)], -2)
            variables['taus_u_{}'.format(bn)] = lift_fn(variables['tau_u1_{}'.format(bn)], -1) + lift_fn(variables['tau_u2_{}'.format(bn)], -2)
#            variables['taus_u_{}'.format(bn)] = lift_fn(variables['tau_u1_{}'.format(bn)], -1) + lift_fn(variables['tau_u2_{}'.format(bn)], -2) + er_LHS*lift_fn(variables['tau_rho1_{}'.format(bn)], -3)
            variables['taus_T_{}'.format(bn)] = lift_fn(variables['tau_T1_{}'.format(bn)], -1) + lift_fn(variables['tau_T2_{}'.format(bn)], -2)
            variables['grad_u_{}'.format(bn)] = grad_u = grad_u_RHS
            variables['grad_T1_{}'.format(bn)] = grad_T1 = d3.grad(T1)
            variables['grad_ln_rho1_{}'.format(bn)] = grad_ln_rho1 = d3.grad(ln_rho1)
            variables['div_u_{}'.format(bn)] = div_u = div_u_RHS

        #Stress matrices & viscous terms
        variables['E_{}'.format(bn)] = E = 0.5*(grad_u + d3.trans(grad_u))
        variables['E_RHS_{}'.format(bn)] = E_RHS = 0.5*(grad_u_RHS + d3.trans(grad_u_RHS))
        variables['sigma_{}'.format(bn)] = sigma = 2*(E - (1/3)*div_u*I_mat)
        variables['sigma_RHS_{}'.format(bn)] = sigma_RHS = 2*(E_RHS - (1/3)*div_u_RHS*I_mat)
        variables['visc_div_stress_L_{}'.format(bn)] = nu_diff*(d3.div(sigma) + d3.dot(sigma, grad_ln_rho0))# + d3.dot(sigma, d3.grad(nu_diff))
        variables['visc_div_stress_R_{}'.format(bn)] = nu_diff*(d3.dot(sigma, grad_ln_rho1))
        variables['VH_{}'.format(bn)] = 2*(nu_diff)*(d3.trace(d3.dot(E_RHS, E_RHS)) - (1/3)*div_u_RHS*div_u_RHS)
#        variables['VH_{}'.format(bn)] = 2*(nu_diff/Cv)*(d3.trace(d3.dot(E_RHS, E_RHS)) - (1/3)*div_u_RHS*div_u_RHS)


        #variables['div_rad_flux_{}'.format(bn)] = (1/Re)*d3.div(grad_s)
        chi_rad = variables['chi_rad_{}'.format(bn)]
        grad_chi_rad = variables['grad_chi_rad_{}'.format(bn)]
        variables['div_rad_flux_L_{}'.format(bn)] = gamma*(chi_rad*(d3.div(grad_T1) + d3.dot(grad_T1, grad_ln_rho0)) + d3.dot(grad_T1, grad_chi_rad))
        variables['div_rad_flux_R_{}'.format(bn)] = gamma*chi_rad*d3.dot(grad_T1, grad_ln_rho1)

        # Rotation and damping terms
        if do_rotation:
            ez = variables['ez_{}'.format(bn)]
            variables['rotation_term_{}'.format(bn)] = -2*Omega*d3.cross(ez, u)
        else:
            variables['rotation_term_{}'.format(bn)] = 0

        if sponge:
            variables['sponge_term_{}'.format(bn)] = u*variables['sponge_{}'.format(bn)]
        else:
            variables['sponge_term_{}'.format(bn)] = 0

        #output tasks
        variables['ones_{}'.format(bn)] = ones = dist.Field(name='ones_{}'.format(bn), bases=basis)
        ones['g']  = 1
        variables['r_vec_{}'.format(bn)] = r_vec = dist.VectorField(coords, name='r_vec_{}'.format(bn), bases=basis)
        variables['r_vals_{}'.format(bn)] = r_vals = dist.Field(name='r_vals_{}'.format(bn), bases=basis)
        r_vals['g'] = variables['r1_{}'.format(bn)]
        r_vec['g'][2] = variables['r1_{}'.format(bn)]

        er = variables['er_{}'.format(bn)]
        variables['P0_{}'.format(bn)] = P0 = R_gas*rho0*T0
        variables['ur_{}'.format(bn)] = d3.dot(er, u)
        variables['T_full_{}'.format(bn)]   = T_full = T0*ones + T1
        variables['rho_full_{}'.format(bn)] = rho_full = rho0*np.exp(ln_rho1)
        variables['rho_fluc_{}'.format(bn)] = rho_fluc = rho0*(np.exp(ln_rho1) - 1)
        variables['rho_fluc_drho0_{}'.format(bn)] = rho_fluc_drho0 = (np.exp(ln_rho1) - 1)
        variables['P_full_{}'.format(bn)] = rho_full = R_gas*rho_full*T_full
        variables['s_full_{}'.format(bn)] = s_full = Cp * ( (1/gamma)*np.log(T_full) - ((gamma-1)/gamma)*np.log(rho_full))
        variables['s0_{}'.format(bn)] = s0 = Cp * ( (1/gamma)*np.log(T0*ones) - ((gamma-1)/gamma)*ones*ln_rho0)
        variables['s1_{}'.format(bn)] = s1 = s_full - s0
        variables['momentum_{}'.format(bn)] = momentum = rho_full * u
        variables['u_squared_{}'.format(bn)] = u_squared = d3.dot(u,u)
        variables['KE_{}'.format(bn)] = KE = 0.5 * rho_full * variables['u_squared_{}'.format(bn)]
        variables['PE_{}'.format(bn)] = PE = rho_full * g_phi
        variables['IE_{}'.format(bn)] = IE = rho_full * Cv * T_full
        variables['PE0_{}'.format(bn)] = PE0 = rho0 * g_phi
        variables['IE0_{}'.format(bn)] = IE0 = rho0 * Cv * T0
        variables['PE1_{}'.format(bn)] = PE1 = PE - PE0
        variables['IE1_{}'.format(bn)] = IE1 = IE - IE0
        variables['TotE_{}'.format(bn)] = KE + PE + IE
        variables['FlucE_{}'.format(bn)] = KE + PE1 + IE1
        variables['Re_{}'.format(bn)] = np.sqrt(u_squared) / nu_diff
        variables['L_{}'.format(bn)] = d3.cross(r_vec, momentum)

        variables['F_cond_{}'.format(bn)] = F_cond = -rho_full * Cp * chi_rad * grad_T1
        variables['F_KE_{}'.format(bn)] = F_KE = u * KE
        variables['F_PE_{}'.format(bn)] = F_PE = u * PE
        variables['F_enth_{}'.format(bn)] = F_enth = momentum * Cp * T_full
        variables['F_visc_{}'.format(bn)] = F_visc = -nu_diff*d3.dot(momentum, sigma_RHS)
    return variables

def fill_structure(bases, dist, variables, ncc_file, radius, Pe, vec_fields=[], vec_nccs=[], scalar_nccs=[], sponge=False, do_rotation=False, scales=None):
    logger.info('using NCC file {}'.format(ncc_file))
    max_dt = None
    t_buoy = None
    t_rot = None
    logger.info('collecing nccs for {}'.format(bases.keys()))
    for basis_number, bn in enumerate(bases.keys()):
        basis = bases[bn]
        ncc_scales = scales
        if ncc_scales is None:
            ncc_scales = basis.dealias
        phi, theta, r = basis.local_grids(ncc_scales)
        phi1, theta1, r1 = basis.local_grids((1,1,1))
        # Load MESA NCC file or setup NCCs using polytrope
        a_vector = variables['{}_{}'.format(vec_fields[0], bn)]
        grid_slices  = dist.layouts[-1].slices(a_vector.domain, ncc_scales[-1])
        a_vector.change_scales(ncc_scales)
        local_vncc_size = variables['{}_{}'.format(vec_nccs[0], bn)]['g'].size
        if ncc_file is not None:
            logger.info('reading NCCs from {}'.format(ncc_file))
            for k in vec_nccs + scalar_nccs + ['H', 'rho0']:
                variables['{}_{}'.format(k, bn)].change_scales(ncc_scales)
            with h5py.File(ncc_file, 'r') as f:
                variables['Cp']['g'] = f['Cp'][()]
                variables['R_gas']['g'] = f['R_gas'][()]
                variables['gamma']['g'] = f['gamma1'][()]
                variables['Cv']['g'] = f['Cp'][()] - f['R_gas'][()]
                logger.info('using Cp: {}, Cv: {}, R_gas: {}, gamma: {}'.format(variables['Cp']['g'], variables['Cv']['g'], variables['R_gas']['g'], variables['gamma']['g']))
                for k in vec_nccs:
                    dist.comm_cart.Barrier()
                    if '{}_{}'.format(k, bn) not in f.keys():
                        logger.info('skipping {}_{}, not in file'.format(k, bn))
                        continue
                    if local_vncc_size > 0:
                        logger.info('reading {}_{}'.format(k, bn))
                        variables['{}_{}'.format(k, bn)]['g'] = f['{}_{}'.format(k, bn)][:,:1,:1,grid_slices[-1]]
                for k in scalar_nccs:
                    dist.comm_cart.Barrier()
                    if '{}_{}'.format(k, bn) not in f.keys():
                        logger.info('skipping {}_{}, not in file'.format(k, bn))
                        continue
                    logger.info('reading {}_{}'.format(k, bn))
                    variables['{}_{}'.format(k, bn)]['g'] = f['{}_{}'.format(k, bn)][:,:,grid_slices[-1]]
                variables['H_{}'.format(bn)]['g']         = f['H_{}'.format(bn)][:,:,grid_slices[-1]]
                variables['rho0_{}'.format(bn)]['g']       = np.exp(f['ln_rho0_{}'.format(bn)][:,:,grid_slices[-1]])[None,None,:]

#                #TODO: do this in star_builder
#                grad_ln_rho = (d3.grad(variables['rho_{}'.format(bn)])/variables['rho_{}'.format(bn)]).evaluate()
#                if local_vncc_size > 0:
#                    variables['grad_ln_rho_{}'.format(bn)]['g'] = grad_ln_rho['g'][:,:1,:1,:]

                if max_dt is None:
                    max_dt = f['max_dt'][()]

                if t_buoy is None:
                    t_buoy = f['tau_heat'][()]/f['tau_nd'][()]

                if t_rot is None:
                    if do_rotation:
                        sim_tau_sec = f['tau_nd'][()]
                        sim_tau_day = sim_tau_sec / (60*60*24)
                        Omega = sim_tau_day * dimensional_Omega 
                        t_rot = 1/(2*Omega)
                    else:
                        t_rot = np.inf

                if sponge:
                    f_brunt = f['tau_nd'][()]*np.sqrt(f['N2max_sim'][()])/(2*np.pi)
                    variables['sponge_{}'.format(bn)]['g'] *= f_brunt
            for k in vec_nccs + scalar_nccs + ['rho0']:
                variables['{}_{}'.format(k, bn)].change_scales((1,1,1))

        else:
            raise NotImplementedError("Must supply star file")

        if do_rotation:
            logger.info("Running with Coriolis Omega = {:.3e}".format(Omega))


    # Grid-lock some operators / define grad's
    for field in ['H']:
        variables['{}_{}'.format(field, bn)] = d3.Grid(variables['{}_{}'.format(field, bn)]).evaluate()

    return variables, (max_dt, t_buoy, t_rot)

def get_compressible_variables(bases, bases_keys, variables):
    problem_variables = []
    for field in ['ln_rho1', 'u', 'T1']:
        for basis_number, bn in enumerate(bases_keys):
            problem_variables.append(variables['{}_{}'.format(field, bn)])

    problem_taus = []
    for tau in ['tau_T']:
        for basis_number, bn in enumerate(bases_keys):
            if type(bases[bn]) == d3.BallBasis:
                problem_taus.append(variables['{}_{}'.format(tau, bn)])
            else:
                problem_taus.append(variables['{}1_{}'.format(tau, bn)])
                problem_taus.append(variables['{}2_{}'.format(tau, bn)])
    for tau in ['tau_u']:
        for basis_number, bn in enumerate(bases_keys):
            if type(bases[bn]) == d3.BallBasis:
                problem_taus.append(variables['{}_{}'.format(tau, bn)])
            else:
                problem_taus.append(variables['{}1_{}'.format(tau, bn)])
                problem_taus.append(variables['{}2_{}'.format(tau, bn)])

#    for tau in ['tau_rho', 'tau_div_u', 'tau_ur']:
##    for tau in ['tau_rho',]:
#        for basis_number, bn in enumerate(bases_keys):
#            if type(bases[bn]) != d3.BallBasis:
#                problem_taus.append(variables['{}1_{}'.format(tau, bn)])
##                problem_taus.append(variables['{}2_{}'.format(tau, bn)])
#
    return problem_variables + problem_taus

def set_compressible_problem(problem, bases, bases_keys, stitch_radii=[]):
    equations = OrderedDict()
    u_BCs = OrderedDict()
    T_BCs = OrderedDict()
    for basis_number, bn in enumerate(bases_keys):
        basis = bases[bn]

        #Standard Equations
        # Assumes background is in hse: -(grad T0 + T0 grad ln rho0) + gvec = 0.
        if config.numerics['equations'] == 'FC_HD':
            equations['continuity_{}'.format(bn)] = "dt(ln_rho1_{0}) + div_u_{0} + u_{0}@grad_ln_rho0_{0} + taus_lnrho_{0} = -u_{0}@grad(ln_rho1_{0})".format(bn)
            equations['momentum_{}'.format(bn)] = "dt(u_{0}) + R_gas*(grad(T1_{0}) + T1_{0}*grad_ln_rho0_{0} + T0_{0}*grad(ln_rho1_{0})) - visc_div_stress_L_{0} + sponge_term_{0} + taus_u_{0} = -u_{0}@grad(u_{0}) - R_gas*T1_{0}*grad(ln_rho1_{0}) + rotation_term_{0} + visc_div_stress_R_{0}".format(bn)
            equations['energy_{}'.format(bn)] = "dt(T1_{0}) + dot(u_{0}, grad_T0_{0}) + (gamma-1)*T0_{0}*div_u_{0} - div_rad_flux_L_{0} + taus_T_{0} = -dot(u_{0}, grad_T1_{0}) - (gamma-1)*T1_{0}*div(u_{0}) + (1/Cv)*((1/rho_full_{0})*H_{0} + VH_{0}) + div_rad_flux_R_{0}".format(bn)
#            equations['energy_{}'.format(bn)] = "dt(T1_{0}) + dot(u_{0}, grad_T0_{0}) + (gamma-1)*T0_{0}*div_u_{0} - div_rad_flux_L_{0} + taus_T_{0} = -dot(u_{0}, grad_T1_{0}) - (gamma-1)*T1_{0}*div(u_{0}) + (1/Cv)*(1/rho_full_{0})*H_{0} + VH_{0} + div_rad_flux_R_{0}".format(bn)
#            equations['energy_{}'.format(bn)] = "dt(T1_{0}) + dot(u_{0}, grad_T0_{0}) + (gamma-1)*T0_{0}*div_u_{0} - div_rad_flux_L_{0} + taus_T_{0} = 0 ".format(bn)
        elif config.numerics['equations'] == 'FC_HD_LinForce':
            equations['continuity_{}'.format(bn)] = "dt(ln_rho1_{0}) + div_u_{0} + u_{0}@grad_ln_rho0_{0} + taus_lnrho_{0} = 0".format(bn)
            equations['momentum_{}'.format(bn)] = "dt(u_{0}) + grad(T1_{0}) + T1_{0}*grad_ln_rho0_{0} + T0_{0}*grad(ln_rho1_{0}) - visc_div_stress_L_{0} + sponge_term_{0} + taus_u_{0} = F_{0}".format(bn)
            equations['energy_{}'.format(bn)] = "dt(T1_{0}) + dot(u_{0}, grad_T0_{0}) + (gamma-1)*T0_{0}*div(u_{0}) - div_rad_flux_L_{0} + taus_T_{0} = 0".format(bn)



        constant_gradP = "(grad(T1_{0}) + T0_{0}*grad(ln_rho1_{0}) + T1_{0}*grad_ln_rho0_{0})(r={2}) - (grad(T1_{1}) + T0_{1}*grad(ln_rho1_{1}) + T1_{1}*grad_ln_rho0_{1})(r={2}) = (T1_{1}*grad(ln_rho1_{1}))(r={2}) - (T1_{0}*grad_ln_rho1_{0})(r={2})"
        constant_rhoU = "u_{0}(r={2}) - u_{1}(r={2}) = -((rho_fluc_drho0_{0}*u_{0})(r={2}) - (rho_fluc_drho0_{1}*u_{1})(r={2})) "
        constant_U = "u_{0}(r={2}) - u_{1}(r={2}) = 0 "
        constant_rad_sigma = "radial(sigma_{0}(r={2}) - sigma_{1}(r={2})) = -radial((rho_fluc_drho0_{0}*sigma_RHS_{0})(r={2}) - (rho_fluc_drho0_{1}*sigma_RHS_{1})(r={2}))"
        constant_T = "T1_{0}(r={2}) - T1_{1}(r={2}) = 0"
        constant_rhoT = "T1_{0}(r={2}) - T1_{1}(r={2}) = -((rho_fluc_drho0_{0}*T1_{0})(r={2}) - (rho_fluc_drho0_{1}*T1_{1})(r={2}))"
        constant_rho_gradT = "radial(grad_T1_{0}(r={2}) - grad_T1_{1}(r={2})) = -radial((rho_fluc_drho0_{0}*grad_T1_{0})(r={2}) - (rho_fluc_drho0_{1}*grad_T1_{1})(r={2}))"

        constant_rad_ang_sigma = "angular(radial(sigma_{0}(r={2}) - sigma_{1}(r={2}))) = -angular(radial((rho_fluc_drho0_{0}*sigma_RHS_{0})(r={2}) - (rho_fluc_drho0_{1}*sigma_RHS_{1})(r={2})))"
#        constant_rad_ang_sigma = "angular(radial(sigma_{0}(r={2}) - sigma_{1}(r={2}))) = 0"
        constant_gradT = "radial((chi_rad_{0}*grad_T1_{0})(r={2}) - (chi_rad_{1}*grad_T1_{1})(r={2})) = 0"
        constant_P = "ln_rho1_{0}(r={2}) - ln_rho1_{1}(r={2}) = 0"
#        constant_P = "ln_rho1_{0}(r={2}) - ln_rho1_{1}(r={2}) = -((ln_rho0_{0}*ones_{0} + log(T_full_{0}))(r={2}) - (ln_rho0_{1}*ones_{1} + log(T_full_{1}))(r={2}))" 

        #Boundary conditions
        if type(basis) == d3.BallBasis:
            if basis_number == len(bases_keys) - 1:
                #No shell bases
                u_BCs['BC_u1_{}'.format(bn)] = "radial(u_{0}(r={1})) = 0".format(bn, 'radius')
                u_BCs['BC_u2_{}'.format(bn)] = "angular(radial(E_{0}(r={1}))) = 0".format(bn, 'radius')
                T_BCs['BC_T_{}'.format(bn)] = "radial(grad_T1_{0}(r={1})) = 0".format(bn, 'radius')
            else:
                shell_name = bases_keys[basis_number+1] 
                rval = stitch_radii[basis_number]
                u_BCs['BC_u1_{}'.format(bn)] = constant_U.format(bn, shell_name, rval)
#                u_BCs['BC_u1_{}'.format(bn)] = constant_rhoU.format(bn, shell_name, rval)
                T_BCs['BC_T_{}'.format(bn)] = constant_T.format(bn, shell_name, rval)
#                T_BCs['BC_T_{}'.format(bn)] = constant_rhoT.format(bn, shell_name, rval)
        else:
            #Stitch to basis below
            below_name = bases_keys[basis_number - 1]
            rval = stitch_radii[basis_number - 1]
#            u_BCs['BC_u1_vec_{}'.format(bn)] = constant_rad_sigma.format(bn, below_name, rval)
#            T_BCs['BC_T0_{}'.format(bn)] = constant_rho_gradT.format(bn, below_name, rval)
            u_BCs['BC_u1_vec_{}'.format(bn)] = constant_rad_ang_sigma.format(bn, below_name, rval)
            u_BCs['BC_u2_vec_{}'.format(bn)] = constant_P.format(bn, below_name, rval)
            T_BCs['BC_T0_{}'.format(bn)] = constant_gradT.format(bn, below_name, rval)

            #Add upper BCs
            if basis_number == len(bases_keys) - 1:
                #top of domain
                u_BCs['BC_u2_{}'.format(bn)] = "radial(u_{0}(r={1})) = 0".format(bn, 'radius')
                u_BCs['BC_u3_{}'.format(bn)] = "angular(radial(E_{0}(r={1}))) = 0".format(bn, 'radius')
                T_BCs['BC_T2_{}'.format(bn)] = "radial(grad_T1_{0}(r={1})) = 0".format(bn, 'radius')
            else:
                shn = bases_keys[basis_number+1] 
                rval = stitch_radii[basis_number]
                u_BCs['BC_u4_vec_{}'.format(bn)] = constant_U.format(bn, shn, rval)
#                u_BCs['BC_u4_vec_{}'.format(bn)] = constant_rhoU.format(bn, shn, rval)
                T_BCs['BC_T2_{}'.format(bn)] = constant_T.format(bn, shn, rval)
#                T_BCs['BC_T2_{}'.format(bn)] = constant_rhoT.format(bn, shn, rval)


    for bn, basis in bases.items():
        continuity = equations['continuity_{}'.format(bn)]
        logger.info('adding eqn "{}"'.format(continuity))
        problem.add_equation(continuity)

    for bn, basis in bases.items():
        momentum = equations['momentum_{}'.format(bn)]
        logger.info('adding eqn "{}"'.format(momentum))
        problem.add_equation(momentum)

    for bn, basis in bases.items():
        energy = equations['energy_{}'.format(bn)]
        logger.info('adding eqn "{}"'.format(energy))
        problem.add_equation(energy)

    for BC in u_BCs.values():
        logger.info('adding BC "{}"'.format(BC))
        problem.add_equation(BC)

    for BC in T_BCs.values():
        logger.info('adding BC "{}"'.format(BC))
        problem.add_equation(BC)

    return problem
