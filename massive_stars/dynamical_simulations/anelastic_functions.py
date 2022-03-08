from collections import OrderedDict

import h5py
import numpy as np
import dedalus.public as d3

import logging
logger = logging.getLogger(__name__)

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

def make_fields(bases, coords, dist, vec_fields=[], scalar_fields=[], vec_taus=[], scalar_taus=[], vec_nccs=[], scalar_nccs=[], sponge=False, do_rotation=False):
    variables = OrderedDict()
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
            logger.info('creating vector field {}'.format(key))
            variables[key] = dist.VectorField(coords, name=key, bases=basis)
        for fn in scalar_fields:
            key = '{}_{}'.format(fn, bn)
            logger.info('creating scalar field {}'.format(key))
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
                logger.info('creating vector tau {}'.format(key))
                variables[key] = dist.VectorField(coords, name=key, bases=S2_basis)
            for fn in scalar_taus:
                key = '{}{}_{}'.format(fn, name, bn)
                logger.info('creating scalar tau {}'.format(key))
                variables[key] = dist.Field(name=key, bases=S2_basis)
        
        #Define problem NCCs
        for fn in vec_nccs + ['er_LHS', 'rvec_LHS']:
            key = '{}_{}'.format(fn, bn)
            logger.info('creating vector NCC {}'.format(key))
            variables[key] = dist.VectorField(coords, name=key, bases=basis.radial_basis)
            if fn == 'er_LHS':
                variables[key]['g'][2] = 1
            elif fn == 'rvec_LHS':
                variables[key]['g'][2] = r1
        for fn in scalar_nccs:
            key = '{}_{}'.format(fn, bn)
            logger.info('creating scalar NCC {}'.format(key))
            variables[key] = dist.Field(name=key, bases=basis.radial_basis)

        #Define identity matrix
        key = 'I_matrix_{}'.format(bn)
        logger.info('creating identity matrix NCC {}'.format(key))
        variables[key] = dist.TensorField(coords, name=key, bases=basis.radial_basis)
        for i in range(3):
            variables[key]['g'][i,i,:] = 1

        #Define unit vectors
        for fn in unit_vectors:
            key = '{}_{}'.format(fn, bn)
            logger.info('creating unit vector field {}'.format(key))
            variables[key] = dist.VectorField(coords, name=key, bases=basis)

        if sponge:
            variables['sponge_{}'.format(bn)]['g'] = sponge_function(r1)

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
        s1 = variables['s1_{}'.format(bn)]
        I_mat = variables['I_matrix_{}'.format(bn)]
        grad_ln_rho = variables['grad_ln_ρ_{}'.format(bn)]
        grad_ln_T = variables['grad_ln_T_{}'.format(bn)]
        er_LHS = variables['er_LHS_{}'.format(bn)]

        # Lift operators for boundary conditions
        if type(basis) == d3.BallBasis:
            lift_basis = basis.clone_with(k=0)
            variables['lift_{}'.format(bn)] = lift_fn = lambda A: d3.Lift(A, lift_basis, -1)
            variables['taus_u_{}'.format(bn)] = lift_fn(variables['tau_u_{}'.format(bn)])
            variables['taus_s_{}'.format(bn)] = lift_fn(variables['tau_s_{}'.format(bn)])
            variables['grad_u_{}'.format(bn)] = grad_u = d3.grad(u)
            variables['grad_s1_{}'.format(bn)] = grad_s = d3.grad(s1)
            variables['div_u_RHS_{}'.format(bn)] = div_u_RHS = d3.div(u)
            variables['div_u_{}'.format(bn)] = div_u = div_u_RHS
        else:
            lift_basis = basis.clone_with(k=2)
            variables['lift_{}'.format(bn)] = lift_fn = lambda A, n: d3.Lift(A, lift_basis, n)
            variables['taus_u_{}'.format(bn)] = lift_fn(variables['tau_u1_{}'.format(bn)], -1) + lift_fn(variables['tau_u2_{}'.format(bn)], -2)
            variables['taus_s_{}'.format(bn)] = lift_fn(variables['tau_s1_{}'.format(bn)], -1) + lift_fn(variables['tau_s2_{}'.format(bn)], -2)
            variables['grad_u_{}'.format(bn)] = grad_u = d3.grad(u) 
            variables['grad_s1_{}'.format(bn)] = grad_s = d3.grad(s1)
            variables['div_u_RHS_{}'.format(bn)] = div_u_RHS = d3.div(u)
            variables['div_u_{}'.format(bn)] = div_u = div_u_RHS

        #Stress matrices & viscous terms (assumes uniform kinematic viscosity; so dynamic viscosity mu = const * rho)
        variables['E_{}'.format(bn)] = E = 0.5*(grad_u + d3.transpose(grad_u))
        variables['E_RHS_{}'.format(bn)] = E_RHS = 0.5*(d3.grad(u) + d3.transpose(d3.grad(u)))
        variables['sigma_{}'.format(bn)] = sigma = 2*(E - (1/3)*div_u*I_mat)
        variables['sigma_RHS_{}'.format(bn)] = sigma_RHS = 2*(E_RHS - (1/3)*d3.div(u)*I_mat)
        variables['visc_div_stress_{}'.format(bn)] = d3.div(sigma) + d3.dot(sigma, grad_ln_rho)
        variables['VH_{}'.format(bn)] = 2*(d3.trace(d3.dot(E_RHS, E_RHS)) - (1/3)*div_u_RHS*div_u_RHS)


        # Grid-lock some operators / define grad's
        for field in ['H', 'inv_T']:
            variables['{}_{}'.format(field, bn)] = d3.Grid(variables['{}_{}'.format(field, bn)]).evaluate()

        #variables['div_rad_flux_{}'.format(bn)] = (1/Re)*d3.div(grad_s)
        chi_rad = variables['inv_Pe_rad_{}'.format(bn)]
        grad_chi_rad = variables['grad_inv_Pe_rad_{}'.format(bn)]
        variables['div_rad_flux_{}'.format(bn)] = chi_rad*(d3.div(grad_s) + d3.dot(grad_s, (grad_ln_rho + grad_ln_T))) + d3.dot(grad_s, grad_chi_rad)

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
    return variables

def fill_structure(bases, dist, variables, ncc_file, radius, Pe, vec_fields=[], vec_nccs=[], scalar_nccs=[], sponge=False, do_rotation=False):
    max_dt = None
    t_buoy = None
    t_rot = None
    for basis_number, bn in enumerate(bases.keys()):
        basis = bases[bn]
        phi, theta, r = basis.local_grids(basis.dealias)
        phi1, theta1, r1 = basis.local_grids((1,1,1))
        # Load MESA NCC file or setup NCCs using polytrope
        a_vector = variables['{}_{}'.format(vec_fields[0], bn)]
        grid_slices  = dist.layouts[-1].slices(a_vector.domain, basis.dealias[-1])
        a_vector.change_scales(basis.dealias)
        local_vncc_size = variables['{}_{}'.format(vec_nccs[0], bn)]['g'].size
        if ncc_file is not None:
            logger.info('reading NCCs from {}'.format(ncc_file))
            for k in vec_nccs + scalar_nccs + ['H', 'ρ', 'T', 'inv_T']:
                variables['{}_{}'.format(k, bn)].change_scales(basis.dealias)
            with h5py.File(ncc_file, 'r') as f:
                for k in vec_nccs:
                    if '{}{}'.format(k, bn) not in f.keys():
                        logger.info('skipping {}{}, not in file'.format(k, bn))
                        continue
                    if local_vncc_size > 0:
                        logger.info('reading {}{}'.format(k, bn))
                        variables['{}_{}'.format(k, bn)]['g'] = f['{}{}'.format(k, bn)][:,0,0,grid_slices[-1]][:,None,None,:]
                for k in scalar_nccs:
                    if '{}{}'.format(k, bn) not in f.keys():
                        logger.info('skipping {}{}, not in file'.format(k, bn))
                        continue
                    logger.info('reading {}{}'.format(k, bn))
                    variables['{}_{}'.format(k, bn)]['g'] = f['{}{}'.format(k, bn)][:,:,grid_slices[-1]]
                variables['H_{}'.format(bn)]['g']         = f['H_eff{}'.format(bn)][:,:,grid_slices[-1]]
                variables['ρ_{}'.format(bn)]['g']         = np.exp(f['ln_ρ{}'.format(bn)][:,:,grid_slices[-1]])[None,None,:]
                variables['T_{}'.format(bn)]['g']         = f['T{}'.format(bn)][:,:,grid_slices[-1]][None,None,:]
                variables['inv_T_{}'.format(bn)]['g']     = 1/variables['T_{}'.format(bn)]['g']

                if max_dt is None:
                    max_dt = f['max_dt'][()]

                if t_buoy is None:
                    t_buoy = 1 #assume nondimensionalization on heating ~ buoyancy time

                if t_rot is None:
                    if do_rotation:
                        sim_tau_sec = f['tau_nd'][()]
                        sim_tau_day = sim_tau_sec / (60*60*24)
                        Omega = sim_tau_day * dimensional_Omega 
                        t_rot = 1/(2*Omega)
                    else:
                        t_rot = np.inf

                if sponge:
                    f_brunt = f['tau_nd'][()]*np.sqrt(f['N2max_shell'][()])/(2*np.pi)
                    variables['sponge_{}'.format(bn)]['g'] *= f_brunt

        else:
            logger.info("Using polytropic initial conditions")
            from scipy.interpolate import interp1d
            with h5py.File('benchmark/poly_nOuter1.6.h5', 'r') as f:
                T_func = interp1d(f['r'][()], f['T'][()])
                ρ_func = interp1d(f['r'][()], f['ρ'][()])
                grad_s0_func = interp1d(f['r'][()], f['grad_s0'][()])
                H_func   = interp1d(f['r'][()], f['H_eff'][()])
            max_grad_s0 = grad_s0_func(radius)
            if max_dt is None:
                max_dt = 2/np.sqrt(max_grad_s0)
            if t_buoy is None:
                t_buoy      = 1
            if t_rot is None:
                if do_rotation:
                    Omega = dimensional_Omega 
                    t_rot = 1/(2*Omega)
                else:
                    t_rot = np.inf

            variables['T_{}'.format(bn)]['g'] = T_func(r1)
            variables['ρ_{}'.format(bn)]['g'] = ρ_func(r1)
            variables['H_{}'.format(bn)]['g'] = H_func(r)
            variables['inv_T_{}'.format(bn)]['g'] = 1/T_func(r)
            
            grad_ln_ρ_full = (d3.grad(variables['ρ_{}'.format(bn)])/variables['ρ_{}'.format(bn)]).evaluate()
            grad_T_full = d3.grad(variables['T_{}'.format(bn)]).evaluate()
            grad_ln_T_full = (grad_T_full/variables['T_{}'.format(bn)]).evaluate()
            if local_vncc_size > 0:
                variables['grad_s0_{}'.format(bn)].change_scales(1)
                print(variables['grad_s0_{}'.format(bn)]['g'].shape, 'grad_s0_{}'.format(bn))
                variables['grad_s0_{}'.format(bn)]['g'][2]  = grad_s0_func(r1)
                for f in ['grad_ln_ρ', 'grad_ln_T', 'grad_T']: variables['{}_{}'.format(f, bn)].change_scales(basis.dealias)
                variables['grad_ln_ρ_{}'.format(bn)]['g']   = grad_ln_ρ_full['g'][:,0,0,None,None,:]
                variables['grad_ln_T_{}'.format(bn)]['g']   = grad_ln_T_full['g'][:,0,0,None,None,:]
                variables['grad_T_{}'.format(bn)]['g']      = grad_T_full['g'][:,0,0,None,None,:]
                variables['grad_inv_Pe_rad_{}'.format(bn)]['g'] = 0
            variables['ln_T_{}'.format(bn)]['g']   = np.log(T_func(r1))
            variables['ln_ρ_{}'.format(bn)]['g']   = np.log(ρ_func(r1))
            variables['inv_Pe_rad_{}'.format(bn)]['g'] = 1/Pe

        if do_rotation:
            logger.info("Running with Coriolis Omega = {:.3e}".format(Omega))

        return variables, (max_dt, t_buoy, t_rot)

def get_anelastic_variables(bases, bases_keys, variables):
    problem_variables = []
    problem_taus = []
    equations = OrderedDict()
    u_BCs = OrderedDict()
    s_BCs = OrderedDict()
    for basis_number, bn in enumerate(bases_keys):
        basis = bases[bn]
        phi, theta, r = basis.local_grids(basis.dealias)
        phi1, theta1, r1 = basis.local_grids((1,1,1))

        for field in ['p', 'u', 's1']:
            problem_variables.append(variables['{}_{}'.format(field, bn)])
        for tau in ['tau_u', 'tau_s']:
            if type(basis) == d3.BallBasis:
                problem_taus.append(variables['{}_{}'.format(tau, bn)])
            else:
                problem_taus.append(variables['{}1_{}'.format(tau, bn)])
                problem_taus.append(variables['{}2_{}'.format(tau, bn)])

    return problem_variables + problem_taus

def set_anelastic_problem(problem, bases, bases_keys, stitch_radii=[]):
    equations = OrderedDict()
    u_BCs = OrderedDict()
    s_BCs = OrderedDict()
    for basis_number, bn in enumerate(bases_keys):
        basis = bases[bn]

        #Standard Equations
        equations['continuity_{}'.format(bn)] = "div_u_{0} + dot(u_{0}, grad_ln_ρ_{0}) = 0".format(bn)
        equations['momentum_{}'.format(bn)] = "dt(u_{0}) + grad(p_{0}) + grad_T_{0}*s1_{0} - (1/Re)*visc_div_stress_{0} + sponge_term_{0} + taus_u_{0} = cross(u_{0}, curl(u_{0})) + rotation_term_{0}".format(bn)
        equations['energy_{}'.format(bn)] = "dt(s1_{0}) + dot(u_{0}, grad_s0_{0}) - div_rad_flux_{0} + taus_s_{0} = - dot(u_{0}, grad_s1_{0}) + H_{0} + (1/Re)*inv_T_{0}*VH_{0}".format(bn)

        #Boundary conditions
        if type(basis) == d3.BallBasis:
            if basis_number == len(bases_keys) - 1:
                #No shell bases
                u_BCs['BC_u1_{}'.format(bn)] = "radial(u_{0}(r={1})) = 0".format(bn, 'radius')
                u_BCs['BC_u2_{}'.format(bn)] = "angular(radial(E_{0}(r={1}))) = 0".format(bn, 'radius')
                s_BCs['BC_s_{}'.format(bn)] = "radial(grad_s1_{0}(r={1})) = 0".format(bn, 'radius')
            else:
                shell_name = bases_keys[basis_number+1] 
                rval = stitch_radii[basis_number]
                u_BCs['BC_u_{}'.format(bn)] = "u_{0}(r={2}) - u_{1}(r={2}) = 0".format(bn, shell_name, rval)
                s_BCs['BC_s_{}'.format(bn)] = "s1_{0}(r={2}) - s1_{1}(r={2}) = 0".format(bn, shell_name, rval)
        else:
            #Stitch to basis below
            below_name = bases_keys[basis_number - 1]
            rval = stitch_radii[basis_number - 1]
            u_BCs['BC_u1_{}'.format(bn)] = "p_{0}(r={2}) - p_{1}(r={2}) = 0".format(bn, below_name, rval)
            u_BCs['BC_u2_{}'.format(bn)] = "angular(radial(sigma_{0}(r={2}) - sigma_{1}(r={2}))) = 0".format(bn, below_name, rval)
            s_BCs['BC_s1_{}'.format(bn)] = "radial(grad(s1_{0})(r={2}) - grad(s1_{1})(r={2})) = 0".format(bn, below_name, rval)

            #Add upper BCs
            if basis_number == len(bases_keys) - 1:
                #top of domain
                u_BCs['BC_u3_{}'.format(bn)] = "radial(u_{0}(r={1})) = 0".format(bn, 'radius')
                u_BCs['BC_u4_{}'.format(bn)] = "angular(radial(E_{0}(r={1}))) = 0".format(bn, 'radius')
                s_BCs['BC_s2_{}'.format(bn)] = "radial(grad_s1_{0}(r={1})) = 0".format(bn, 'radius')
            else:
                shn = bases_keys[basis_number+1] 
                rval = stitch_radii[basis_number]
                u_BCs['BC_u3_{}'.format(bn)] = "u_{0}(r={2}) - u_{1}(r={2}) = 0".format(bn, shn, rval)
                s_BCs['BC_s2_{}'.format(bn)] = "s1_{0}(r={2}) - s1_{1}(r={2}) = 0".format(bn, shn, rval)


    for bn, basis in bases.items():
        continuity = equations['continuity_{}'.format(bn)]
        continuity_ell0 = "p_{} = 0".format(bn)
        logger.info('adding eqn "{}" for ntheta != 0'.format(continuity))
        logger.info('adding eqn "{}" for ntheta == 0'.format(continuity_ell0))
        problem.add_equation(continuity, condition="ntheta != 0")
        problem.add_equation(continuity_ell0, condition="ntheta == 0")

        momentum = equations['momentum_{}'.format(bn)]
        momentum_ell0 = "u_{} = 0".format(bn)
        logger.info('adding eqn "{}" for ntheta != 0'.format(momentum))
        logger.info('adding eqn "{}" for ntheta == 0'.format(momentum_ell0))
        problem.add_equation(momentum, condition="ntheta != 0")
        problem.add_equation(momentum_ell0, condition="ntheta == 0")

        energy = equations['energy_{}'.format(bn)]
        logger.info('adding eqn "{}"'.format(energy))
        problem.add_equation(energy)

    for BC in s_BCs.values():
        logger.info('adding BC "{}"'.format(BC))
        problem.add_equation(BC)

    for BC in u_BCs.values():
        logger.info('adding BC "{}" for ntheta != 0'.format(BC))
        problem.add_equation(BC, condition="ntheta != 0")

    for bn, basis in bases.items():
        if type(basis) == d3.BallBasis:
            BC = 'tau_u_{} = 0'.format(bn)
            logger.info('adding BC "{}" for ntheta == 0'.format(BC))
            problem.add_equation(BC, condition="ntheta == 0")
        else:
            for i in [1, 2]:
                BC = 'tau_u{}_{} = 0'.format(i, bn)
                logger.info('adding BC "{}" for ntheta == 0'.format(BC))
                problem.add_equation(BC, condition="ntheta == 0")
    return problem
