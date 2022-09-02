from collections import OrderedDict

import h5py
import numpy as np
import dedalus.public as d3
from dedalus.core.operators import convert 

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

def make_fields(bases, coords, dist, vec_fields=[], scalar_fields=[], vec_nccs=[], scalar_nccs=[], sponge=False, do_rotation=False, sponge_function=lambda r: r**2):
    vec_fields = ['u', ] + vec_fields
    scalar_fields = ['ln_rho1', 's1', 'Q', 'ones'] + scalar_fields
    vec_taus = ['tau_u']
    scalar_taus = ['tau_s']
    vec_nccs = ['grad_pom0', 'grad_ln_pom0', 'grad_ln_rho0', 'grad_s0', 'g', 'rvec', 'grad_nu_diff', 'grad_chi_rad'] + vec_nccs
    scalar_nccs = ['pom0', 'rho0', 'ln_rho0', 'g_phi', 'nu_diff', 'chi_rad', 's0'] + scalar_nccs
    sphere_unit_vectors = ['ephi', 'etheta', 'er']
    cartesian_unit_vectors = ['ex', 'ey', 'ez']
    if sponge:
        scalar_nccs += ['sponge']

    namespace = OrderedDict()
    namespace['exp'] = np.exp
    namespace['log'] = np.log
    namespace['Grid'] = d3.Grid
    namespace['dt'] = dt =  d3.TimeDerivative
    one = dist.Field(name='one')
    one['g'] = 1
    namespace['one'] = one = d3.Grid(one)

    for basis_number, bn in enumerate(bases.keys()):
        basis = bases[bn]
        phi, theta, r = basis.local_grids(basis.dealias)
        phi1, theta1, r1 = basis.local_grids((1,1,1))
        namespace['phi_'+bn], namespace['theta_'+bn], namespace['r_'+bn] = phi, theta, r
        namespace['phi1_'+bn], namespace['theta1_'+bn], namespace['r1_'+bn] = phi1, theta1, r1

        #Define problem fields
        for fn in vec_fields:
            key = '{}_{}'.format(fn, bn)
            logger.debug('creating vector field {}'.format(key))
            namespace[key] = dist.VectorField(coords, name=key, bases=basis)
        for fn in scalar_fields:
            key = '{}_{}'.format(fn, bn)
            logger.debug('creating scalar field {}'.format(key))
            namespace[key] = dist.Field(name=key, bases=basis)

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
                namespace[key] = dist.VectorField(coords, name=key, bases=S2_basis)
            for fn in scalar_taus:
                key = '{}{}_{}'.format(fn, name, bn)
                logger.debug('creating scalar tau {}'.format(key))
                namespace[key] = dist.Field(name=key, bases=S2_basis)
        
        #Define problem NCCs
        for fn in vec_nccs:
            key = '{}_{}'.format(fn, bn)
            logger.debug('creating vector NCC {}'.format(key))
            namespace[key] = dist.VectorField(coords, name=key, bases=basis.radial_basis)
        namespace['rvec_{}'.format(bn)]['g'][2] = r1
        for fn in scalar_nccs:
            key = '{}_{}'.format(fn, bn)
            logger.debug('creating scalar NCC {}'.format(key))
            namespace[key] = dist.Field(name=key, bases=basis.radial_basis)

        #Define identity matrix
        logger.debug('creating identity matrix')
        namespace['eye'] = dist.TensorField(coords, name='eye', bases=basis.radial_basis)
        for i in range(3):
            namespace['eye']['g'][i,i] = 1

        if sponge:
            namespace['sponge_{}'.format(bn)]['g'] = sponge_function(r1)

        #Define unit vectors
        for fn in sphere_unit_vectors:
            logger.debug('creating unit vector field {}'.format(key))
            namespace[fn] = dist.VectorField(coords, name=fn)
        namespace['er']['g'][2] = 1
        namespace['ephi']['g'][0] = 1
        namespace['etheta']['g'][1] = 1

        for fn in cartesian_unit_vectors:
            key = '{}_{}'.format(fn, bn)
            logger.debug('creating unit vector field {}'.format(key))
            namespace[key] = dist.VectorField(coords, name=key, bases=basis)


        namespace['ex_{}'.format(bn)]['g'][0] = -np.sin(phi1)
        namespace['ex_{}'.format(bn)]['g'][1] = np.cos(theta1)*np.cos(phi1)
        namespace['ex_{}'.format(bn)]['g'][2] = np.sin(theta1)*np.cos(phi1)

        namespace['ey_{}'.format(bn)]['g'][0] = np.cos(phi1)
        namespace['ey_{}'.format(bn)]['g'][1] = np.cos(theta1)*np.sin(phi1)
        namespace['ey_{}'.format(bn)]['g'][2] = np.sin(theta1)*np.sin(phi1)

        namespace['ez_{}'.format(bn)]['g'][0] = 0
        namespace['ez_{}'.format(bn)]['g'][1] = -np.sin(theta1)
        namespace['ez_{}'.format(bn)]['g'][2] =  np.cos(theta1)
        
        for k in ['ex', 'ey', 'ez']:
            namespace['{}_{}'.format(k, bn)] = d3.Grid(namespace['{}_{}'.format(k, bn)]).evaluate()

        u = namespace['u_{}'.format(bn)]
        ln_rho1 = namespace['ln_rho1_{}'.format(bn)]
        s1 = namespace['s1_{}'.format(bn)]
        s0 = namespace['s0_{}'.format(bn)]
        Q = namespace['Q_{}'.format(bn)]
        eye = namespace['eye']
        grad_ln_rho0 = namespace['grad_ln_rho0_{}'.format(bn)]
        grad_ln_pom0 = namespace['grad_ln_pom0_{}'.format(bn)]
        grad_pom0 = namespace['grad_pom0_{}'.format(bn)]
        rho0 = namespace['rho0_{}'.format(bn)]
        ln_rho0 = namespace['ln_rho0_{}'.format(bn)]
        pom0 = namespace['pom0_{}'.format(bn)]
        grad_s0 = namespace['grad_s0_{}'.format(bn)]
        g_phi = namespace['g_phi_{}'.format(bn)]
        gravity = namespace['g_{}'.format(bn)]
        nu_diff = namespace['nu_diff_{}'.format(bn)]
        grad_nu_diff = namespace['grad_nu_diff_{}'.format(bn)]
        chi_rad = namespace['chi_rad_{}'.format(bn)]
        grad_chi_rad = namespace['grad_chi_rad_{}'.format(bn)]
        ones = namespace['ones_{}'.format(bn)]
        ones['g'] = 1
        namespace['ones_{}'.format(bn)] = ones = d3.Grid(ones).evaluate()

        namespace['grid_nu_diff_{}'.format(bn)] = grid_nu_diff = d3.Grid(nu_diff)

        if bn == 'B':
            namespace['gamma'] = gamma = dist.Field(name='gamma')
            namespace['R_gas'] = R_gas = dist.Field(name='R_gas')
            namespace['Cp'] = Cp = dist.Field(name='Cp')
            namespace['Cv'] = Cv = dist.Field(name='Cv')
            namespace['grid_cp'] = grid_cp = d3.Grid(Cp) 
            namespace['grid_cp_div_R'] = grid_cp_div_R = d3.Grid(Cp/R_gas) #= gamma/(gamma-1)
            namespace['grid_R_div_cp'] = grid_R_div_cp = d3.Grid(R_gas/Cp) #= (gamma-1)/gamma
            namespace['grid_inv_cp'] = grid_inv_cp = d3.Grid(1/Cp)
            namespace['grid_R'] = grid_R = d3.Grid(R_gas)
            namespace['grid_gamma'] = grid_gamma = d3.Grid(gamma)
            namespace['grid_inv_gamma'] = grid_inv_gamma = d3.Grid(1/gamma)




        g = namespace['g_{}'.format(bn)]
        er = namespace['er']
        ephi = namespace['ephi']
        etheta = namespace['etheta']
        rvec = namespace['rvec_{}'.format(bn)]

        # Lift operators for boundary conditions
        namespace['grad_u_{}'.format(bn)] = grad_u = d3.grad(u)
        namespace['div_u_{}'.format(bn)] = div_u = d3.div(u)
        namespace['grad_s1_{}'.format(bn)] = grad_s1 = d3.grad(s1)
        namespace['grad_ln_rho1_{}'.format(bn)] = grad_ln_rho1 = d3.grad(ln_rho1)
        if type(basis) == d3.BallBasis:
            lift_basis = basis.derivative_basis(0)
            namespace['lift_{}'.format(bn)] = lift_fn = lambda A: d3.Lift(A, lift_basis, -1)
            namespace['taus_lnrho_{}'.format(bn)] = taus_lnrho = 0
            namespace['taus_u_{}'.format(bn)] = taus_u = lift_fn(namespace['tau_u_{}'.format(bn)])
            namespace['taus_s_{}'.format(bn)] = taus_s = lift_fn(namespace['tau_s_{}'.format(bn)])
        else:
            lift_basis = basis.derivative_basis(2)
            namespace['lift_{}'.format(bn)] = lift_fn = lambda A, n: d3.Lift(A, lift_basis, n)
            namespace['taus_lnrho_{}'.format(bn)] = taus_lnrho = (1/nu_diff)*rvec@lift_fn(namespace['tau_u2_{}'.format(bn)], -1)
            namespace['taus_u_{}'.format(bn)] = taus_u = lift_fn(namespace['tau_u1_{}'.format(bn)], -1) + lift_fn(namespace['tau_u2_{}'.format(bn)], -2)
            namespace['taus_s_{}'.format(bn)] = taus_s = lift_fn(namespace['tau_s1_{}'.format(bn)], -1) + lift_fn(namespace['tau_s2_{}'.format(bn)], -2)


        #These fields are coming to the grid:
        grid_u = d3.Grid(u)
        grid_ln_rho1 = d3.Grid(ln_rho1)
        grid_s1 = d3.Grid(s1)
        grid_grad_u = d3.Grid(grad_u)
        grid_grad_ln_rho1 = d3.Grid(grad_ln_rho1)
        grid_grad_s1 = d3.Grid(grad_s1)
        grid_lap_ln_rho1 = d3.Grid(d3.lap(ln_rho1))
        grid_lap_s1 = d3.Grid(d3.lap(s1))
        grid_div_u = d3.Grid(div_u)

        #reused background fields:
        grid_rho0 = d3.Grid(ones*rho0)
        grid_ln_rho0 = d3.Grid(ones*ln_rho0)
        grid_grad_ln_rho0 = d3.Grid(ones*grad_ln_rho0)
        grid_s0 = d3.Grid(ones*s0)
        grid_grad_s0 = d3.Grid(ones*grad_s0)
        grid_pom0 = d3.Grid(ones*pom0)
        grid_grad_pom0 = d3.Grid(ones*grad_pom0)
        grid_g = d3.Grid(g)
        grid_chi_rad = d3.Grid(chi_rad*ones)
        grid_grad_chi_rad = d3.Grid(ones*grad_chi_rad)
        namespace['inv_pom0_{}'.format(bn)] = inv_pom0 = (1/pom0)
        grid_inv_pom0 = d3.Grid(inv_pom0)

        neg_one = d3.Grid(-ones)


        lap_domain = d3.lap(s1).domain
        namespace['lap_C_{}'.format(bn)] = lap_C = lambda A: convert(A, lap_domain.bases)

        #Stress matrices & viscous terms
        namespace['E_{}'.format(bn)] = E = grad_u/2 + d3.trans(grad_u/2)
        namespace['sigma_{}'.format(bn)] = sigma = (E - div_u*eye/3)*2
        namespace['E_RHS_{}'.format(bn)] = E_RHS = (grid_grad_u + d3.trans(grid_grad_u))/2
        namespace['sigma_RHS_{}'.format(bn)] = sigma_RHS = (E_RHS - grid_div_u*d3.Grid(eye)/3)*2
        namespace['visc_div_stress_L_{}'.format(bn)] = visc_div_stress_L = nu_diff*(d3.div(sigma) + sigma@grad_ln_rho0) + sigma@grad_nu_diff
        namespace['visc_div_stress_L_RHS_{}'.format(bn)] = visc_div_stress_L_RHS = grid_nu_diff*(d3.div(sigma) + sigma_RHS@grid_grad_ln_rho0) + sigma_RHS@d3.Grid(grad_nu_diff)
        namespace['visc_div_stress_R_{}'.format(bn)] = visc_div_stress_R = grid_nu_diff*(sigma_RHS@grid_grad_ln_rho1)
        namespace['VH_{}'.format(bn)] = VH = (grid_nu_diff)*(d3.trace(E_RHS@E_RHS) - (1/3)*grid_div_u**2)*2

        #Thermodynamics: rho, pressure, s 
        namespace['rho_full_{}'.format(bn)] = rho_full = grid_rho0*np.exp(ln_rho1)
        namespace['rho_fluc_{}'.format(bn)] = rho_fluc = rho_full - grid_rho0
        namespace['ln_rho_full_{}'.format(bn)] = ln_rho_full = (grid_ln_rho0 + ln_rho1)
        namespace['grad_ln_rho_full_{}'.format(bn)] = grad_ln_rho_full = grid_grad_ln_rho0 + grid_grad_ln_rho1
        namespace['P0_{}'.format(bn)] = P0 = rho0*pom0
        namespace['s_full_{}'.format(bn)] = s_full = grid_s0 + grid_s1
        namespace['grad_s_full_{}'.format(bn)] = grad_s_full = grid_grad_s0 + grid_grad_s1
        #Linear Pomega = R * T
        namespace['pom1_over_pom0_{}'.format(bn)] = pom1_over_pom0 = gamma*(s1/Cp + ((gamma-1)/gamma)*ln_rho1)
        namespace['grad_pom1_over_pom0_{}'.format(bn)] = grad_pom1_over_pom0 = gamma*(grad_s1/Cp + ((gamma-1)/gamma)*grad_ln_rho1)
        namespace['pom1_{}'.format(bn)] = pom1 = pom0 * pom1_over_pom0
        namespace['grad_pom1_{}'.format(bn)] = grad_pom1 = grad_pom0*pom1_over_pom0 + pom0*grad_pom1_over_pom0

        #RHS terms
        namespace['pom1_over_pom0_RHS_{}'.format(bn)] = pom1_over_pom0_RHS = grid_gamma*(grid_s1*grid_inv_cp + grid_R_div_cp*grid_ln_rho1)
        namespace['grad_pom1_over_pom0_RHS_{}'.format(bn)] = grad_pom1_over_pom0_RHS = grid_gamma*(grad_s1*grid_inv_cp + grid_R_div_cp*grad_ln_rho1)
        namespace['lap_pom1_over_pom0_RHS_{}'.format(bn)] = lap_pom1_over_pom0_RHS = grid_gamma*(grid_lap_s1*grid_inv_cp + grid_R_div_cp*grid_lap_ln_rho1)
        namespace['grad_pom1_RHS_{}'.format(bn)] = grad_pom1_RHS = grid_grad_pom0*pom1_over_pom0_RHS + grid_pom0*grad_pom1_over_pom0_RHS

        #Full pomega subs
        namespace['pom_fluc_over_pom0_{}'.format(bn)] = pom_fluc_over_pom0 = np.exp(pom1_over_pom0_RHS) + neg_one 
        namespace['pom_fluc_{}'.format(bn)] = pom_fluc = grid_pom0*pom_fluc_over_pom0
        namespace['grad_pom_fluc_{}'.format(bn)] = grad_pom_fluc = grid_grad_pom0*pom_fluc_over_pom0 + (pom_fluc_over_pom0 + ones)*grid_pom0*grad_pom1_over_pom0_RHS
        namespace['lap_pom_fluc_{}'.format(bn)] = lap_pom_fluc = d3.lap(pom_fluc)

        #Nonlinear Pomega = R*T
        namespace['pom2_over_pom0_{}'.format(bn)] = pom2_over_pom0 = pom_fluc_over_pom0 - pom1_over_pom0_RHS
        namespace['pom2_{}'.format(bn)] = pom2 = grid_pom0*pom2_over_pom0
        namespace['grad_pom2_{}'.format(bn)] = grad_pom2 = d3.grad(pom2)

        namespace['grad_pom_full_{}'.format(bn)] = grad_pom_full = (grid_grad_pom0 + grad_pom_fluc)
        namespace['pom_full_{}'.format(bn)] = pom_full = (grid_pom0 + pom_fluc)
        namespace['inv_pom_full_{}'.format(bn)] = inv_pom_full = d3.Grid(1/pom_full)
        namespace['grad_pom2_over_pom0_{}'.format(bn)] = grad_pom2_over_pom0 = grad_pom1_over_pom0_RHS*pom_fluc_over_pom0

        #Equation of state goodness
        namespace['EOS_{}'.format(bn)]    = EOS = (s_full)*grid_inv_cp - ( grid_inv_gamma * (np.log(pom_full) - np.log(grid_R)) - grid_R_div_cp * ln_rho_full )
        namespace['EOS_bg_{}'.format(bn)] = EOS_bg = d3.Grid(ones*(s0/Cp - ( grid_inv_gamma * (np.log(pom0) - np.log(R_gas)) - ((gamma-1)/(gamma)) * ln_rho0)))
        namespace['EOS_goodness_{}'.format(bn)]    = EOS_good_ = np.sqrt(EOS**2)
        namespace['EOS_goodness_bg_{}'.format(bn)] = EOS_good_bg = d3.Grid(np.sqrt(EOS_bg**2))

        #Momentum thermo / hydrostatic terms:
        namespace['gradP0_div_rho0_{}'.format(bn)]         = gradP0_div_rho0 = gamma*pom0*(grad_ln_rho0 + grad_s0*grid_inv_cp)
        namespace['background_HSE_{}'.format(bn)]          = background_HSE = gradP0_div_rho0 - g
        namespace['linear_gradP_div_rho_{}'.format(bn)]    = linear_gradP_div_rho    = gamma*pom0*(grad_ln_rho1 + grad_s1/Cp) + g*pom1_over_pom0
        namespace['nonlinear_gradP_div_rho_{}'.format(bn)] = nonlinear_gradP_div_rho = grid_gamma*pom_fluc*(grid_grad_ln_rho1 + grid_grad_s1*grid_inv_cp) + grid_g*pom2_over_pom0

        #Thermal diffusion
        cp_times_chi_rad = d3.Grid(grid_cp * grid_chi_rad)
        cp_times_grad_chi_rad = d3.Grid(grid_cp * grid_grad_chi_rad)
        C_grad_ln_rho = d3.Grid(lap_C(grid_grad_ln_rho0)) + d3.Grid(lap_C(grid_grad_ln_rho1))
        namespace['F_cond_{}'.format(bn)] = F_cond = -1*chi_rad*rho_full*Cp*((grad_pom_fluc)/R_gas)
        namespace['div_rad_flux_L_{}'.format(bn)] = div_rad_flux_L = Cp * chi_rad * d3.div(grad_pom1_over_pom0) + Cp * inv_pom0 * (grad_pom1)@(chi_rad * grad_ln_rho0 + grad_chi_rad) 
        namespace['div_rad_flux_L_RHS_{}'.format(bn)] = div_rad_flux_L_RHS = cp_times_chi_rad * lap_pom1_over_pom0_RHS + d3.Grid(grid_cp * grid_inv_pom0) * (grad_pom1_RHS)@(d3.Grid(grid_chi_rad * grid_grad_ln_rho0 + grid_grad_chi_rad)) 

        namespace['full_div_rad_flux_pt1_{}'.format(bn)] = full_div_rad_flux_pt1 =   inv_pom_full*((grad_pom_fluc@(d3.Grid(cp_times_chi_rad*C_grad_ln_rho)+d3.Grid(lap_C(cp_times_grad_chi_rad))) ))
        namespace['full_div_rad_flux_pt2_{}'.format(bn)] = full_div_rad_flux_pt2 =   lap_pom_fluc*cp_times_chi_rad*inv_pom_full
        namespace['div_rad_flux_R_{}'.format(bn)] = div_rad_flux_R = d3.Grid(lap_C(full_div_rad_flux_pt1 - div_rad_flux_L_RHS)) + d3.Grid(lap_C(full_div_rad_flux_pt2))


        #Moar thermo
        namespace['P_full_{}'.format(bn)] = P_full = np.exp(grid_gamma*(s_full*grid_inv_cp + grid_ln_rho1 + grid_ln_rho0))
#        namespace['P_full_{}'.format(bn)] = P_full = rho_full*pom_full
        namespace['enthalpy_{}'.format(bn)] = enthalpy = grid_cp_div_R*P_full
        namespace['enthalpy_fluc_{}'.format(bn)] = enthalpy_fluc = enthalpy - d3.Grid(grid_cp_div_R*P0*ones)




        # Rotation and damping terms
        if do_rotation:
            ez = namespace['ez_{}'.format(bn)]
            namespace['rotation_term_{}'.format(bn)] = -2*Omega*d3.cross(ez, u)
        else:
            namespace['rotation_term_{}'.format(bn)] = 0

        if sponge:
            namespace['sponge_term_{}'.format(bn)] = u*namespace['sponge_{}'.format(bn)]
        else:
            namespace['sponge_term_{}'.format(bn)] = 0

        sponge_term = namespace['sponge_term_{}'.format(bn)]
        rotation_term = namespace['rotation_term_{}'.format(bn)]


        #The sum order matters here based on ball or shell...weird.
        energy_terms_1 = d3.Grid(lap_C(-grid_u@grid_grad_s1 + d3.Grid(grid_R/P_full)*d3.Grid(Q) + d3.Grid(grid_R*inv_pom_full)*VH - div_rad_flux_L_RHS + full_div_rad_flux_pt1))
        energy_terms_2 = d3.Grid(lap_C(full_div_rad_flux_pt2))
        namespace['energy_RHS_{}'.format(bn)] = energy_terms_1 + energy_terms_2

        #output tasks
        er = namespace['er']
        namespace['r_vals_{}'.format(bn)] = r_vals = d3.Grid(er@(ones*rvec)).evaluate()
        namespace['ur_{}'.format(bn)] = er@u
        namespace['momentum_{}'.format(bn)] = momentum = rho_full * u
        namespace['u_squared_{}'.format(bn)] = u_squared = u@u
        namespace['KE_{}'.format(bn)] = KE = rho_full * u_squared / 2
        namespace['PE_{}'.format(bn)] = PE = rho_full * d3.Grid(g_phi)
        namespace['IE_{}'.format(bn)] = IE = (P_full)*d3.Grid(Cv/R_gas)
        namespace['PE0_{}'.format(bn)] = PE0 = d3.Grid(rho0 * g_phi)
        namespace['IE0_{}'.format(bn)] = IE0 = d3.Grid(P0*(Cv/R_gas))
        namespace['PE1_{}'.format(bn)] = PE1 = PE + d3.Grid(-PE0*ones)
        namespace['IE1_{}'.format(bn)] = IE1 = IE + d3.Grid(-IE0*ones)
        namespace['TotE_{}'.format(bn)] = KE + PE + IE
        namespace['FlucE_{}'.format(bn)] = KE + PE1 + IE1
        namespace['Re_{}'.format(bn)] = np.sqrt(u_squared) * d3.Grid(1/nu_diff)
        namespace['Ma_{}'.format(bn)] = np.sqrt(u_squared) / np.sqrt(pom_full) 
        namespace['L_{}'.format(bn)] = d3.cross(rvec, momentum)

        #Fluxes
        namespace['F_KE_{}'.format(bn)] = F_KE = u * KE
        namespace['F_PE_{}'.format(bn)] = F_PE = u * PE
        namespace['F_enth_{}'.format(bn)] = F_enth = grid_cp_div_R * momentum * pom_full
        namespace['F_visc_{}'.format(bn)] = F_visc = d3.Grid(-nu_diff)*momentum@sigma_RHS

        #Waves
        namespace['N2_{}'.format(bn)] = N2 = grad_s_full@d3.Grid(-g/Cp)

        #Source terms
        namespace['energy_visc_heating_{}'.format(bn)] = energy_visc_heating = rho_full * VH
        namespace['rad_flux_production_{}'.format(bn)] = rad_flux_production = (P_full*d3.Grid(1/R_gas))*(div_rad_flux_L_RHS + div_rad_flux_R)
        namespace['Q_production_{}'.format(bn)] = Q_production = namespace['Q_{}'.format(bn)]
        namespace['momentum_gradP_{}'.format(bn)] = gradP_production = momentum @ (d3.Grid(-gradP0_div_rho0*ones) - linear_gradP_div_rho - nonlinear_gradP_div_rho) #does not include PE
        namespace['energy_PdivU_{}'.format(bn)] = energy_PdivU = -P_full*div_u
        namespace['momentum_flux_div_{}'.format(bn)] = momentum_fluxes = - d3.div(u*(KE + PE) + grid_nu_diff*momentum@sigma_RHS*-1)
        namespace['energy_flux_div_{}'.format(bn)] = energy_fluxes  = -d3.div(u*IE)

        namespace['momentum_visc_cooling_{}'.format(bn)] = momentum_visc_cooling = momentum @ (visc_div_stress_L_RHS + visc_div_stress_R)
        namespace['source_KE_{}'.format(bn)] = momentum_visc_cooling + momentum @ (-d3.grad(P_full)/rho_full) #g term turns into dt(PE) + div(u*PE); do not include here while trying to solve for dt(KE) + div(u*KE).
        namespace['source_IE_{}'.format(bn)] = d3.Grid(Q) + (P_full/grid_R)*(full_div_rad_flux_pt1 + full_div_rad_flux_pt2 + grid_R*inv_pom_full*VH) - P_full*div_u
        namespace['tot_source_{}'.format(bn)] = namespace['source_KE_{}'.format(bn)] + namespace['source_IE_{}'.format(bn)]
    return namespace

def fill_structure(bases, dist, namespace, ncc_file, radius, Pe, vec_fields=[], vec_nccs=[], scalar_nccs=[], sponge=False, do_rotation=False, scales=None):
    vec_fields = ['u', ] + vec_fields
    scalar_fields = ['ln_rho1', 's1', 'Q', 'ones']
    vec_taus = ['tau_u']
    scalar_taus = ['tau_s']
    vec_nccs = ['grad_pom0', 'grad_ln_pom0', 'grad_ln_rho0', 'grad_s0', 'g', 'rvec', 'grad_nu_diff', 'grad_chi_rad'] + vec_nccs
    scalar_nccs = ['pom0', 'rho0', 'ln_rho0', 'g_phi', 'nu_diff', 'chi_rad', 's0'] + scalar_nccs

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
        a_vector = namespace['{}_{}'.format(vec_fields[0], bn)]
        grid_slices  = dist.layouts[-1].slices(a_vector.domain, ncc_scales[-1])
        a_vector.change_scales(ncc_scales)
        local_vncc_size = namespace['{}_{}'.format(vec_nccs[0], bn)]['g'].size
        if ncc_file is not None:
            logger.info('reading NCCs from {}'.format(ncc_file))
            for k in vec_nccs + scalar_nccs + ['Q', 'rho0']:
                namespace['{}_{}'.format(k, bn)].change_scales(ncc_scales)
            with h5py.File(ncc_file, 'r') as f:
                namespace['Cp']['g'] = f['Cp'][()]
                namespace['R_gas']['g'] = f['R_gas'][()]
                namespace['gamma']['g'] = f['gamma1'][()]
                namespace['Cv']['g'] = f['Cp'][()] - f['R_gas'][()]
                logger.info('using Cp: {}, Cv: {}, R_gas: {}, gamma: {}'.format(namespace['Cp']['g'], namespace['Cv']['g'], namespace['R_gas']['g'], namespace['gamma']['g']))
                for k in vec_nccs:
                    dist.comm_cart.Barrier()
                    if '{}_{}'.format(k, bn) not in f.keys():
                        logger.info('skipping {}_{}, not in file'.format(k, bn))
                        continue
                    if local_vncc_size > 0:
                        logger.info('reading {}_{}'.format(k, bn))
                        namespace['{}_{}'.format(k, bn)]['g'] = f['{}_{}'.format(k, bn)][:,:1,:1,grid_slices[-1]]
                for k in scalar_nccs:
                    dist.comm_cart.Barrier()
                    if '{}_{}'.format(k, bn) not in f.keys():
                        logger.info('skipping {}_{}, not in file'.format(k, bn))
                        continue
                    logger.info('reading {}_{}'.format(k, bn))
                    namespace['{}_{}'.format(k, bn)]['g'] = f['{}_{}'.format(k, bn)][:,:,grid_slices[-1]]
                namespace['Q_{}'.format(bn)]['g']         = f['Q_{}'.format(bn)][:,:,grid_slices[-1]]
                namespace['rho0_{}'.format(bn)]['g']       = np.exp(f['ln_rho0_{}'.format(bn)][:,:,grid_slices[-1]])[None,None,:]


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
                    namespace['sponge_{}'.format(bn)]['g'] *= f_brunt
            for k in vec_nccs + scalar_nccs + ['rho0']:
                namespace['{}_{}'.format(k, bn)].change_scales((1,1,1))

        else:
            raise NotImplementedError("Must supply star file")

        if do_rotation:
            logger.info("Running with Coriolis Omega = {:.3e}".format(Omega))

    return namespace, (max_dt, t_buoy, t_rot)

def get_compressible_variables(bases, bases_keys, namespace):
    problem_variables = []
    for field in ['ln_rho1', 'u', 's1']:
        for basis_number, bn in enumerate(bases_keys):
            problem_variables.append(namespace['{}_{}'.format(field, bn)])

    problem_taus = []
    for tau in ['tau_s']:
        for basis_number, bn in enumerate(bases_keys):
            if type(bases[bn]) == d3.BallBasis:
                problem_taus.append(namespace['{}_{}'.format(tau, bn)])
            else:
                problem_taus.append(namespace['{}1_{}'.format(tau, bn)])
                problem_taus.append(namespace['{}2_{}'.format(tau, bn)])
    for tau in ['tau_u']:
        for basis_number, bn in enumerate(bases_keys):
            if type(bases[bn]) == d3.BallBasis:
                problem_taus.append(namespace['{}_{}'.format(tau, bn)])
            else:
                problem_taus.append(namespace['{}1_{}'.format(tau, bn)])
                problem_taus.append(namespace['{}2_{}'.format(tau, bn)])

    return problem_variables + problem_taus

def set_compressible_problem(problem, bases, bases_keys,  stitch_radii=[]):
    equations = OrderedDict()
    u_BCs = OrderedDict()
    T_BCs = OrderedDict()
    for basis_number, bn in enumerate(bases_keys):
        basis = bases[bn]

        #Standard Equations
        # Assumes background is in hse: -(grad T0 + T0 grad ln rho0) + gvec = 0.
        if config.numerics['equations'] == 'FC_HD':
            equations['continuity_{}'.format(bn)] = "dt(ln_rho1_{0}) + div_u_{0} + u_{0}@grad_ln_rho0_{0} + taus_lnrho_{0} = -(u_{0}@grad_ln_rho1_{0})".format(bn)
            equations['momentum_{}'.format(bn)] = "dt(u_{0}) + linear_gradP_div_rho_{0} - visc_div_stress_L_{0} + sponge_term_{0} + taus_u_{0} = (-(u_{0}@grad_u_{0}) - nonlinear_gradP_div_rho_{0} + visc_div_stress_R_{0})".format(bn)
            equations['energy_{}'.format(bn)] = "dt(s1_{0}) + u_{0}@grad_s0_{0} - div_rad_flux_L_{0} + taus_s_{0} = energy_RHS_{0}".format(bn)
        else:
            raise ValueError("Unknown equation choice, plesae use 'FC_HD'")

        constant_U = "u_{0}(r={2}) - u_{1}(r={2}) = 0 "
        constant_s = "s1_{0}(r={2}) - s1_{1}(r={2}) = 0"
        constant_gradT = "radial(grad_pom1_over_pom0_{0}(r={2}) - grad_pom1_over_pom0_{1}(r={2})) = 0"
#        constant_gradT = "radial(grad_pom1_{0}(r={2}) - grad_pom1_{1}(r={2})) = -radial(grad_pom2_{0}(r={2}) - grad_pom2_{1}(r={2}))"
        constant_ln_rho = "ln_rho1_{0}(r={2}) - ln_rho1_{1}(r={2}) = 0"
        constant_momentum_ang = "angular(radial(sigma_{0}(r={2}) - sigma_{1}(r={2}))) = 0"



        #Boundary conditions
        if type(basis) == d3.BallBasis:
            if basis_number == len(bases_keys) - 1:
                #No shell bases
                u_BCs['BC_u1_{}'.format(bn)] = "radial(u_{0}(r={1})) = 0".format(bn, 'radius')
                u_BCs['BC_u2_{}'.format(bn)] = "angular(radial(E_{0}(r={1}))) = 0".format(bn, 'radius')
#                T_BCs['BC_T_{}'.format(bn)] = "radial(grad_pom1_over_pom0_{0}(r={1})) = 0".format(bn, 'radius')
                T_BCs['BC_T_{}'.format(bn)] = "radial(grad_pom1_{0}(r={1})) = - radial(grad_pom2_{0}(r={1}))".format(bn, 'radius') #needed for energy conservation
            else:
                shell_name = bases_keys[basis_number+1] 
                rval = stitch_radii[basis_number]
                u_BCs['BC_u2_{}'.format(bn)] = constant_U.format(bn, shell_name, rval)
                T_BCs['BC_T1_{}'.format(bn)] = constant_s.format(bn, shell_name, rval)
        else:
            #Stitch to basis below
            below_name = bases_keys[basis_number - 1]
            rval = stitch_radii[basis_number - 1]
            u_BCs['BC_u1_vec_{}'.format(bn)] = constant_momentum_ang.format(bn, below_name, rval)
            u_BCs['BC_u2_vec_{}'.format(bn)] = constant_ln_rho.format(bn, below_name, rval)
            T_BCs['BC_T0_{}'.format(bn)] = constant_gradT.format(bn, below_name, rval)

            #Add upper BCs
            if basis_number != len(bases_keys) - 1:
                shn = bases_keys[basis_number+1] 
                rval = stitch_radii[basis_number]
                u_BCs['BC_u3_vec_{}'.format(bn)] = constant_U.format(bn, shn, rval)
                T_BCs['BC_T3_{}'.format(bn)] = constant_s.format(bn, shn, rval)
            else:
                #top of domain
                u_BCs['BC_u2_{}'.format(bn)] = "radial(u_{0}(r={1})) = 0".format(bn, 'radius')
                u_BCs['BC_u3_{}'.format(bn)] = "angular(radial(E_{0}(r={1}))) = 0".format(bn, 'radius')
#                T_BCs['BC_T2_{}'.format(bn)] = "radial(grad_pom1_over_pom0_{0}(r={1})) = 0".format(bn, 'radius')
                T_BCs['BC_T2_{}'.format(bn)] = "radial(grad_pom1_{0}(r={1})) = - radial(grad_pom2_{0}(r={1}))".format(bn, 'radius')#needed for energy conservation


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
