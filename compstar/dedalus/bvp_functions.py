
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3

from scipy.interpolate import interp1d
from ..tools.general import one_to_zero
import logging
logger = logging.getLogger(__name__)

interp_kwargs = {'fill_value' : 'extrapolate', 'bounds_error' : False}

#first one is not used right now. will use combo of HSE_solve_CZ + HSE_solve_RZ + HSE_EOS_solve for smoothed quantities

def HSE_solve(coords, dist, bases, g_phi_func, grad_ln_rho_func, ln_rho_func, N2_func, Fconv_func, r_stitch=[], r_outer=1, low_nr=16, \
              R=1, gamma=5/3, G=1, nondim_radius=1, ncc_cutoff=1e-9, tolerance=1e-9, HSE_tolerance = 1e-4):
    """
    Solves for hydrostatic equilibrium in a calorically perfect ideal gas.
    The solution for density, entropy, and gravity is found given a specified function of N^2 and grad ln rho.
    The heating term associated with a convective luminosity is also found given a specified function of the convective flux, Fconv.

    Arguments
    ---------
    coords : Dedalus CoordinateSystem object
        The coordinate system in which the solution is found.
    dist : Dedalus Distributor object
        The distributor object associated with the bases; should NOT be in parallel.
    bases : dict
        A dictionary of Dedalus bases, with keys 'B', 'S1', 'S2', etc. for the Ball basis, first Shell basis, second Shell basis, etc.
    grad_ln_rho_func : function
        A function of radius that returns the gradient of the log of density. Input r should be nondimensionalized.
    N2_func : function
        A function of radius that returns the nondimensionalized Brunt-Vaisala frequency squared. Input r should be nondimensionalized.
    Fconv_func : function
        A function of radius that returns the nondimensionalized convective flux. Input r should be nondimensionalized.
    r_stitch : list
        A list of radii at which to stitch together the solutions from different bases. 
        The first element should be the radius of the outer boundary of the BallBasis.
        If there is only one basis, r_stitch should be an empty list.
    r_outer : float
        The radius of the outer boundary of the simulation domain.
    low_nr : int
        The number of radial points in the low resolution domain; used to set up background fields for solve. #TODO: make this by-basis.
    R : float
        The nondimensional value of the gas constant divided by the mean molecular weight.
    gamma : float
        The adiabatic index of the gas.
    nondim_radius : float
        The radius where thermodynamics are nondimensionalized.
    ncc_cutoff : float
        The NCC floor for the solver. See Dedalus.core.solvers.SolverBase
    tolerance : float
        The tolerance for perturbation norm of the newton iteration.
    HSE_tolerance : float
        The tolerance for hydrostatic equilibrium of the BVP solve.
    
    Returns
    -------
    atmosphere : dict
        A dictionary of interpolated functions which return atmospheric quantities as a function of nondimensional radius.
    """
    # Parameters
    namespace = dict()
    namespace['G'] = G
    namespace['R'] = R
    namespace['Cp'] = Cp = R*gamma/(gamma-1)
    namespace['gamma'] = gamma
    namespace['log'] = np.log
    namespace['exp'] = np.exp

    #Loop over bases, set up fields and operators.
    for k, basis in bases.items():
        namespace['basis_{}'.format(k)] = basis
        namespace['S2_basis_{}'.format(k)] = S2_basis = basis.S2_basis()

        # Make problem variables and taus.
        namespace['g_phi_{}'.format(k)] = g_phi = dist.Field(name='g_phi', bases=basis)
        namespace['Q_{}'.format(k)] = Q = dist.Field(name='Q', bases=basis)
        namespace['s_{}'.format(k)] = s = dist.Field(name='s', bases=basis)
        namespace['g_{}'.format(k)] = g = dist.VectorField(coords, name='g', bases=basis)
        namespace['ln_rho_{}'.format(k)] = ln_rho = dist.Field(name='ln_rho', bases=basis)
        namespace['grad_ln_rho_{}'.format(k)] = grad_ln_rho = dist.VectorField(coords, name='grad_ln_rho', bases=basis)
        namespace['tau_s_{}'.format(k)] = tau_s = dist.Field(name='tau_s', bases=S2_basis)
        namespace['tau_rho_{}'.format(k)] = tau_rho = dist.Field(name='tau_rho', bases=S2_basis)
        namespace['tau_g_phi_{}'.format(k)] = tau_g_phi = dist.Field(name='tau_g_phi', bases=S2_basis)
        namespace['tau_g_phi_1_{}'.format(k)] = tau_g_phi_1 = dist.Field(name='tau_g_phi_1', bases=S2_basis)
        namespace['tau_g_phi_2_{}'.format(k)] = tau_g_phi_2 = dist.Field(name='tau_g_phi_2', bases=S2_basis)

        # Set up some fundamental grid data
        low_scales = low_nr/basis.radial_basis.radial_size 
        phi, theta, r = dist.local_grids(basis)
        phi_de, theta_de, r_de = dist.local_grids(basis, scales=basis.dealias)
        phi_low, theta_low, r_low = dist.local_grids(basis, scales=(1,1,low_scales))
        namespace['r_de_{}'.format(k)] = r_de
        namespace['r_vec_{}'.format(k)] = r_vec = dist.VectorField(coords, bases=basis.radial_basis)
        r_vec['g'][2] = r
        namespace['r_squared_{}'.format(k)] = r_squared = dist.Field(bases=basis.radial_basis)
        r_squared['g'] = r**2       

        # Make lift operators for BCs
        if k == 'B':
            namespace['lift_{}'.format(k)] = lift = lambda A: d3.Lift(A, basis, -1)
        else:
            namespace['lift_{}'.format(k)] = lift = lambda A: d3.Lift(A, basis.derivative_basis(2), -1)
            namespace['lift2_{}'.format(k)] = lift2 = lambda A: d3.Lift(A, basis.derivative_basis(2), -2)

        # Make a field of ones for converting NCCs to full fields.
        namespace['ones_{}'.format(k)] = ones = dist.Field(bases=basis, name='ones')
        ones['g'] = 1

        #make a field of 4piG*ones
        namespace['four_pi_G_{}'.format(k)] = four_pi_G = dist.Field(bases=basis, name='four_pi_G')
        four_pi_G['g'] = 4*np.pi*G*ones['g']

        #Make a field that smooths at the edge of the ball basis.
        namespace['edge_smoothing_{}'.format(k)] = edge_smooth = dist.Field(bases=basis, name='edge_smooth')
        edge_smooth['g'] = one_to_zero(r, 0.95*bases['B'].radius, width=0.03*bases['B'].radius)

        # Get a high-resolution N^2 in the ball; low-resolution elsewhere where it transitions more gradually.
        namespace['N2_{}'.format(k)] = N2 = dist.Field(bases=basis, name='N2')
        if k == 'B':
            N2['g'] = N2_func(r)
        else:
            N2.change_scales(low_scales)
            N2['g'] = N2_func(r_low)

        #Set grad ln rho. Not needed in this version
        # grad_ln_rho.change_scales(low_scales)
        # grad_ln_rho['g'][2] = grad_ln_rho_func(r_low)
        # grad_ln_rho['g'][2] = grad_ln_rho_func(r)
            
        # Set ln rho initial guess.
        namespace['ln_rho_in_{}'.format(k)] = ln_rho_in = dist.Field(bases=basis, name='ln_rho_in')
        ln_rho_in['g'] = ln_rho_func(r)

        # Set the convective flux.
        namespace['Fconv_{}'.format(k)] = Fconv   = dist.VectorField(coords, name='Fconv', bases=basis)
        Fconv['g'][2] = Fconv_func(r)

        # set initial guess for g_phi
        namespace['g_phi_in_{}'.format(k)] = g_phi_in = dist.Field(bases=basis, name='g_phi_in')
        g_phi_in['g'] = g_phi_func(r)

        # Create important operations from the fields.
        namespace['ln_pomega_LHS_{}'.format(k)] = ln_pomega_LHS = gamma*(s/Cp + ((gamma-1)/gamma)*ln_rho*ones)
        namespace['ln_pomega_{}'.format(k)] = ln_pomega = ln_pomega_LHS + np.log(R)
        namespace['pomega_{}'.format(k)] = pomega = np.exp(ln_pomega)
        namespace['P_{}'.format(k)] = P = pomega*np.exp(ln_rho)
        namespace['HSE_{}'.format(k)] = HSE = gamma*pomega*(d3.grad(ones*ln_rho) + d3.grad(s)/Cp) + d3.grad(g_phi)*ones
        #gamma*pomega*(d3.grad(ones*ln_rho) + d3.grad(s)/Cp) - g*ones
        namespace['N2_op_{}'.format(k)] = N2_op = d3.grad(g_phi)@d3.grad(s)/Cp
        namespace['rho_{}'.format(k)] = rho = np.exp(ln_rho*ones)
        namespace['grad_ln_rho_{}'.format(k)] = grad_ln_rho = d3.grad(ln_rho)
        namespace['T_{}'.format(k)] = T = pomega/R
        namespace['ln_T_{}'.format(k)] = ln_T = ln_pomega - np.log(R)
        namespace['grad_pomega_{}'.format(k)] = d3.grad(pomega)
        namespace['grad_ln_pomega_{}'.format(k)] = d3.grad(ln_pomega)
        namespace['grad_s_{}'.format(k)] = grad_s = d3.grad(s)
        namespace['r_vec_g_{}'.format(k)] = r_vec@g
        namespace['g_op_{}'.format(k)] = gamma * pomega * (grad_s/Cp + grad_ln_rho)
        namespace['s0_{}'.format(k)] = Cp * ((1/gamma)*(ln_pomega + ln_rho) - ln_rho) #s with an offset so s0 = cp * (1/gamma * lnP - ln_rho)

    namespace['pi'] = np.pi
    locals().update(namespace)
    
    # Solve for poisson equation, HSE, heating, and EOS simultaneously.
    variables, taus = [], []
    for k, basis in bases.items():
        variables += [namespace['g_phi_{}'.format(k)],namespace['s_{}'.format(k)],namespace['ln_rho_{}'.format(k)],namespace['Q_{}'.format(k)], ]#,namespace['ln_rho_{}'.format(k)],]
        
        taus += [ namespace['tau_g_phi_1_{}'.format(k)],namespace['tau_s_{}'.format(k)], namespace['tau_g_phi_{}'.format(k)]] #namespace['tau_g_phi_{}'.format(k)],namespace['tau_s_{}'.format(k)],
        if k != 'B':
            taus += [namespace['tau_g_phi_2_{}'.format(k)],]

    print('variables',len(variables),'taus',len(taus))
    problem = d3.NLBVP(variables + taus, namespace=locals())
    count_eqn = 0
    for k, basis in bases.items():
    
        # Set a decent initial guess for s.
        # namespace['s_{}'.format(k)].change_scales(basis.dealias)
        # namespace['s_{}'.format(k)]['g'] = -(R*namespace['ln_rho_in_{}'.format(k)]).evaluate()['g']
        # Set a decent initial guess for ln_rho.
        namespace['ln_rho_{}'.format(k)].change_scales(basis.dealias)
        namespace['ln_rho_in_{}'.format(k)].change_scales(basis.dealias)
        namespace['ln_rho_{}'.format(k)]['g'] = np.copy(namespace['ln_rho_in_{}'.format(k)]['g'])
       
        # set initial guess for g_phi
        namespace['g_phi_{}'.format(k)].change_scales(basis.dealias)
        namespace['g_phi_in_{}'.format(k)].change_scales(basis.dealias)
        namespace['g_phi_{}'.format(k)]['g'] = (namespace['g_phi_in_{}'.format(k)]).evaluate()['g']
        #Set the equations: poisson
        if k != 'B':
            problem.add_equation("lap(g_phi_{0}) - four_pi_G_{0}*exp(ln_rho_{0}*ones_{0}) + lift_{0}(tau_g_phi_1_{0}) + lift2_{0}(tau_g_phi_2_{0}) = 0".format(k))
            count_eqn+=1
        elif k == 'B':
            problem.add_equation("lap(g_phi_{0}) - four_pi_G_{0}*exp(ln_rho_{0}*ones_{0})  + lift_{0}(tau_g_phi_1_{0}) = 0".format(k))
            count_eqn+=1
        
        #Set the equations: hydrostatic equilibrium
        problem.add_equation("-grad(g_phi_{0}) + r_vec_{0}*lift_{0}(tau_g_phi_{0}) = g_op_{0} ".format(k))
        # problem.add_equation("grad(ln_rho_{0})@(grad(s_{0})/Cp) + lift_{0}(tau_s_{0}) = -N2_{0}/(gamma*pomega_{0}) - grad(s_{0})@grad(s_{0}) / Cp**2".format(k))
        problem.add_equation("grad(g_phi_{0})@grad(s_{0}) / Cp + lift_{0}(tau_s_{0}) = N2_{0}".format(k))
        
        count_eqn+=2
        #Set equation for heating
        problem.add_equation("Q_{0} = edge_smoothing_{0}*div(Fconv_{0})".format(k))
        count_eqn+=1
    
    #Set the boundary conditions.
    iter = 0
    for k, basis in bases.items():
        if k != 'B':
            k_old = list(bases.keys())[iter-1]
            r_s = r_stitch[iter-1]
            problem.add_equation("ln_rho_{0}(r={2}) - ln_rho_{1}(r={2}) = 0".format(k, k_old, r_s))
            problem.add_equation("s_{0}(r={2}) - s_{1}(r={2}) = 0".format(k, k_old, r_s))
            problem.add_equation("g_phi_{0}(r={2}) - g_phi_{1}(r={2}) = 0".format(k, k_old, r_s))
            problem.add_equation("grad(g_phi_{0})(r={2}) - grad(g_phi_{1})(r={2}) = 0".format(k, k_old, r_s))
            count_eqn+=4
        iter += 1
        if iter == len(bases.items()):
            # problem.add_equation("g_phi_{0}(r=r_outer) = g_phi_in_{0}(r=r_outer)".format(k))
            problem.add_equation("g_phi_{0}(r=r_outer) = 0".format(k))
            count_eqn+=1
    
    problem.add_equation("ln_pomega_LHS_B(r=nondim_radius) = 0")
    problem.add_equation("ln_rho_B(r=nondim_radius) = 0")
    count_eqn+=2
    print('number of eqns',count_eqn)

    #Solve with tolerances on pert_norm and hydrostatic equilibrium.
    solver = problem.build_solver(ncc_cutoff=ncc_cutoff)
    pert_norm = np.inf
    while pert_norm > tolerance or HSE_err > HSE_tolerance:
        HSE_err = 0
        solver.newton_iteration(damping=1)
        pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in solver.perturbations)
        logger.info(f'Perturbation norm: {pert_norm:.3e}')
        for k, basis in bases.items():
            this_HSE = np.max(np.abs(namespace['HSE_{}'.format(k)].evaluate()['g']))
            logger.info('HSE in {}:{:.3e}'.format(k, this_HSE))
            if this_HSE > HSE_err:
                HSE_err = this_HSE
    # Stitch together the fields for creation of interpolators that span the full simulation domain.
    #Need: grad_pom0, grad_ln_pom0, grad_ln_rho0, grad_s0, g, pom0, rho0, ln_rho0, g_phi
    stitch_fields = OrderedDict()
    fields = ['grad_pomega', 'grad_ln_pomega', 'grad_ln_rho', 'grad_s', 'g', 'pomega', 'rho', 'ln_rho', 'g_phi', 'r_vec', 'HSE', 'N2_op', 'Q', 's0']
    for f in fields:
        stitch_fields[f] = []
    
    for k, basis in bases.items():
        for f in fields:
            stitch_fields[f] += [np.copy(namespace['{}_{}'.format(f, k)].evaluate()['g'])]

    if len(stitch_fields['r_vec']) == 1:
        for f in fields:
            stitch_fields[f] = stitch_fields[f][0]
    else:
        for f in fields:
            stitch_fields[f] = np.concatenate(stitch_fields[f], axis=-1)

    grad_pom = stitch_fields['grad_pomega'][2,:].ravel()
    grad_ln_pom = stitch_fields['grad_ln_pomega'][2,:].ravel()
    grad_ln_rho = stitch_fields['grad_ln_rho'][2,:].ravel()
    grad_s = stitch_fields['grad_s'][2,:].ravel()
    g = stitch_fields['g'][2,:].ravel()
    HSE = stitch_fields['HSE'][2,:].ravel()
    r = stitch_fields['r_vec'][2,:].ravel()

    pom = stitch_fields['pomega'].ravel()
    rho = stitch_fields['rho'].ravel()
    ln_rho = stitch_fields['ln_rho'].ravel()
    g_phi = stitch_fields['g_phi'].ravel()
    N2 = stitch_fields['N2_op'].ravel()
    Q = stitch_fields['Q'].ravel()
    s0 = stitch_fields['s0'].ravel()


    #Plot the results.
    fig = plt.figure()
    ax1 = fig.add_subplot(4,2,1)
    ax2 = fig.add_subplot(4,2,2)
    ax3 = fig.add_subplot(4,2,3)
    ax4 = fig.add_subplot(4,2,4)
    ax5 = fig.add_subplot(4,2,5)
    ax6 = fig.add_subplot(4,2,6)
    ax7 = fig.add_subplot(4,2,7)
    ax8 = fig.add_subplot(4,2,8)
    ax1.plot(r, grad_pom, label='grad pomega')
    ax1.legend()
    ax2.plot(r, grad_ln_rho, label='grad ln rho')
    ax2.legend()
    ax3.plot(r, pom/R, label='pomega/R')
    ax3.plot(r, rho, label='rho')
    ax3.legend()
    ax4.plot(r, HSE, label='HSE')
    ax4.legend()
    ax5.plot(r, g, label='g')
    ax5.legend()
    ax6.plot(r, g_phi, label='g_phi')
    ax6.legend()
    ax7.plot(r, N2, label=r'$N^2$')
    ax7.plot(r, -N2, label=r'$-N^2$')
    ax7.plot(r, (N2_func(r)), label=r'$N^2$ goal', ls='--')
    ax7.set_yscale('log')
    yticks = (np.max(np.abs(N2.ravel()[r.ravel() < 0.5])), np.max(N2_func(r).ravel()))
    ax7.set_yticks(yticks)
    ax7.set_yticklabels(['{:.1e}'.format(n) for n in yticks])
    ax7.legend()
    ax8.plot(r, grad_s, label='grad s')
    ax8.set_yscale('log')
    ax8.legend()
    fig.savefig('stratification.png', bbox_inches='tight', dpi=300)

    #Create interpolators for the atmosphere.
    atmosphere = dict()
    atmosphere['grad_pomega'] = interp1d(r, grad_pom, **interp_kwargs)
    atmosphere['grad_ln_pomega'] = interp1d(r, grad_ln_pom, **interp_kwargs)
    atmosphere['grad_ln_rho'] = interp1d(r, grad_ln_rho, **interp_kwargs)
    atmosphere['grad_s'] = interp1d(r, grad_s, **interp_kwargs)
    atmosphere['g'] = interp1d(r, g, **interp_kwargs)
    atmosphere['pomega'] = interp1d(r, pom, **interp_kwargs)
    atmosphere['rho'] = interp1d(r, rho, **interp_kwargs)
    atmosphere['ln_rho'] = interp1d(r, ln_rho, **interp_kwargs)
    atmosphere['g_phi'] = interp1d(r, g_phi, **interp_kwargs)
    atmosphere['N2'] = interp1d(r, N2, **interp_kwargs)
    atmosphere['Q'] = interp1d(r, Q, **interp_kwargs)
    atmosphere['s0'] = interp1d(r, s0, **interp_kwargs)
    return atmosphere

# Solve HSE in CZ only
#No N2func because N2=0 in CZ exactly. 
def HSE_solve_CZ(coords, dist, bases, g_phi_func, ln_rho_func,  Fconv_func, r_stitch=[], r_outer=1, low_nr=16, \
              R=1, gamma=5/3, G=1, nondim_radius=1, ncc_cutoff=1e-9, tolerance=1e-9, HSE_tolerance = 1e-4):
    """
    Solves for hydrostatic equilibrium in a calorically perfect ideal gas.
    The solution for density, entropy, and gravity is found given a specified function of N^2 and grad ln rho.
    The heating term associated with a convective luminosity is also found given a specified function of the convective flux, Fconv.

    Arguments
    ---------
    coords : Dedalus CoordinateSystem object
        The coordinate system in which the solution is found.
    dist : Dedalus Distributor object
        The distributor object associated with the bases; should NOT be in parallel.
    bases : dict
        A dictionary of Dedalus bases, with keys 'B', 'S1', 'S2', etc. for the Ball basis, first Shell basis, second Shell basis, etc.
    grad_ln_rho_func : function
        A function of radius that returns the gradient of the log of density. Input r should be nondimensionalized.
    Fconv_func : function
        A function of radius that returns the nondimensionalized convective flux. Input r should be nondimensionalized.
    r_stitch : list
        A list of radii at which to stitch together the solutions from different bases. 
        The first element should be the radius of the outer boundary of the BallBasis.
        If there is only one basis, r_stitch should be an empty list.
    r_outer : float
        The radius of the outer boundary of the simulation domain.
    low_nr : int
        The number of radial points in the low resolution domain; used to set up background fields for solve. #TODO: make this by-basis.
    R : float
        The nondimensional value of the gas constant divided by the mean molecular weight.
    gamma : float
        The adiabatic index of the gas.
    nondim_radius : float
        The radius where thermodynamics are nondimensionalized.
    ncc_cutoff : float
        The NCC floor for the solver. See Dedalus.core.solvers.SolverBase
    tolerance : float
        The tolerance for perturbation norm of the newton iteration.
    HSE_tolerance : float
        The tolerance for hydrostatic equilibrium of the BVP solve.
    
    Returns
    -------
    atmosphere : dict
        A dictionary of interpolated functions which return atmospheric quantities as a function of nondimensional radius.
    """
    # Parameters
    namespace = dict()
    namespace['G'] = G
    namespace['R'] = R
    namespace['Cp'] = Cp = R*gamma/(gamma-1)
    namespace['gamma'] = gamma
    namespace['log'] = np.log
    namespace['exp'] = np.exp

    #Loop over bases, set up fields and operators.
    for k, basis in bases.items():
        namespace['basis_{}'.format(k)] = basis
        namespace['S2_basis_{}'.format(k)] = S2_basis = basis.S2_basis()

        # Make problem variables and taus.
        namespace['g_phi_{}'.format(k)] = g_phi = dist.Field(name='g_phi', bases=basis)
        namespace['Q_{}'.format(k)] = Q = dist.Field(name='Q', bases=basis)
        namespace['s_{}'.format(k)] = s = dist.Field(name='s', bases=basis)
        namespace['g_{}'.format(k)] = g = dist.VectorField(coords, name='g', bases=basis)
        namespace['ln_rho_{}'.format(k)] = ln_rho = dist.Field(name='ln_rho', bases=basis)
        namespace['grad_ln_rho_{}'.format(k)] = grad_ln_rho = dist.VectorField(coords, name='grad_ln_rho', bases=basis)
        namespace['tau_s_{}'.format(k)] = tau_s = dist.Field(name='tau_s', bases=S2_basis)
        namespace['tau_rho_{}'.format(k)] = tau_rho = dist.Field(name='tau_rho', bases=S2_basis)
        namespace['tau_g_phi_{}'.format(k)] = tau_g_phi = dist.Field(name='tau_g_phi', bases=S2_basis)
        namespace['tau_g_phi_1_{}'.format(k)] = tau_g_phi_1 = dist.Field(name='tau_g_phi_1', bases=S2_basis)
        namespace['tau_g_phi_2_{}'.format(k)] = tau_g_phi_2 = dist.Field(name='tau_g_phi_2', bases=S2_basis)

        # Set up some fundamental grid data
        low_scales = low_nr/basis.radial_basis.radial_size 
        phi, theta, r = dist.local_grids(basis)
        phi_de, theta_de, r_de = dist.local_grids(basis, scales=basis.dealias)
        phi_low, theta_low, r_low = dist.local_grids(basis, scales=(1,1,low_scales))
        namespace['r_de_{}'.format(k)] = r_de
        namespace['r_vec_{}'.format(k)] = r_vec = dist.VectorField(coords, bases=basis.radial_basis)
        r_vec['g'][2] = r
        namespace['r_squared_{}'.format(k)] = r_squared = dist.Field(bases=basis.radial_basis)
        r_squared['g'] = r**2       

        # Make lift operators for BCs
        if k == 'B':
            namespace['lift_{}'.format(k)] = lift = lambda A: d3.Lift(A, basis, -1)
        else:
            namespace['lift_{}'.format(k)] = lift = lambda A: d3.Lift(A, basis.derivative_basis(2), -1)
            namespace['lift2_{}'.format(k)] = lift2 = lambda A: d3.Lift(A, basis.derivative_basis(2), -2)

        # Make a field of ones for converting NCCs to full fields.
        namespace['ones_{}'.format(k)] = ones = dist.Field(bases=basis, name='ones')
        ones['g'] = 1

        #make a field of 4piG*ones
        namespace['four_pi_G_{}'.format(k)] = four_pi_G = dist.Field(bases=basis, name='four_pi_G')
        four_pi_G['g'] = 4*np.pi*G*ones['g']

        #Make a field that smooths at the edge of the ball basis.
        namespace['edge_smoothing_{}'.format(k)] = edge_smooth = dist.Field(bases=basis, name='edge_smooth')
        edge_smooth['g'] = one_to_zero(r, 0.95*bases['B'].radius, width=0.03*bases['B'].radius)

        
        #Set grad ln rho.
        # grad_ln_rho.change_scales(low_scales)
        # grad_ln_rho['g'][2] = grad_ln_rho_func(r_low)
        # grad_ln_rho['g'][2] = grad_ln_rho_func(r)
            
            
        # Set ln rho initial guess.
        namespace['ln_rho_in_{}'.format(k)] = ln_rho_in = dist.Field(bases=basis, name='ln_rho_in')
        ln_rho_in['g'] = ln_rho_func(r)

        # Set the convective flux.
        namespace['Fconv_{}'.format(k)] = Fconv   = dist.VectorField(coords, name='Fconv', bases=basis)
        Fconv['g'][2] = Fconv_func(r)

        # set initial guess for g_phi
        namespace['g_phi_in_{}'.format(k)] = g_phi_in = dist.Field(bases=basis, name='g_phi_in')
        g_phi_in['g'] = g_phi_func(r)


        # Create important operations from the fields.
        namespace['ln_pomega_LHS_{}'.format(k)] = ln_pomega_LHS = gamma*(s/Cp + ((gamma-1)/gamma)*ln_rho*ones)
        namespace['ln_pomega_{}'.format(k)] = ln_pomega = ln_pomega_LHS + np.log(R)
        namespace['pomega_{}'.format(k)] = pomega = np.exp(ln_pomega)
        namespace['P_{}'.format(k)] = P = pomega*np.exp(ln_rho)
        namespace['HSE_{}'.format(k)] = HSE = gamma*pomega*(d3.grad(ones*ln_rho) + d3.grad(s)/Cp) + d3.grad(g_phi)*ones
        #gamma*pomega*(d3.grad(ones*ln_rho) + d3.grad(s)/Cp) - g*ones
        namespace['N2_op_{}'.format(k)] = N2_op = d3.grad(g_phi)@d3.grad(s)/Cp
        namespace['rho_{}'.format(k)] = rho = np.exp(ln_rho*ones)
        namespace['grad_ln_rho_{}'.format(k)] = grad_ln_rho = d3.grad(ln_rho)
        namespace['T_{}'.format(k)] = T = pomega/R
        namespace['ln_T_{}'.format(k)] = ln_T = ln_pomega - np.log(R)
        namespace['grad_pomega_{}'.format(k)] = d3.grad(pomega)
        namespace['grad_ln_pomega_{}'.format(k)] = d3.grad(ln_pomega)
        namespace['grad_s_{}'.format(k)] = grad_s = d3.grad(s)
        namespace['r_vec_g_{}'.format(k)] = r_vec@g
        namespace['g_op_{}'.format(k)] = gamma * pomega * grad_ln_rho #for grad_s=0 case in CZ
        # namespace['g_op_{}'.format(k)] = gamma * pomega * (grad_s/Cp + grad_ln_rho)
        namespace['s0_{}'.format(k)] = Cp * ((1/gamma)*(ln_pomega + ln_rho) - ln_rho) #s with an offset so s0 = cp * (1/gamma * lnP - ln_rho)

    namespace['pi'] = np.pi
    locals().update(namespace)

    # Solve for poisson equation given ln_rho.
    variables, taus = [], []
    for k, basis in bases.items():
        variables += [namespace['g_phi_{}'.format(k)],namespace['s_{}'.format(k)],namespace['ln_rho_{}'.format(k)],namespace['Q_{}'.format(k)], ]#,namespace['ln_rho_{}'.format(k)],]
        
        taus += [ namespace['tau_g_phi_1_{}'.format(k)],namespace['tau_s_{}'.format(k)], namespace['tau_g_phi_{}'.format(k)]] #namespace['tau_g_phi_{}'.format(k)],namespace['tau_s_{}'.format(k)],
        if k != 'B':
            taus += [namespace['tau_g_phi_2_{}'.format(k)],]
        # variables += [namespace['s_{}'.format(k)],]
        
        # taus += [namespace['tau_s_{}'.format(k)], ]

    print('variables',len(variables),'taus',len(taus))
    problem = d3.NLBVP(variables + taus, namespace=locals())
    count_eqn = 0
    for k, basis in bases.items():
        # Set a decent initial guess for ln_rho.
        namespace['ln_rho_{}'.format(k)].change_scales(basis.dealias)
        namespace['ln_rho_in_{}'.format(k)].change_scales(basis.dealias)
        namespace['ln_rho_{}'.format(k)]['g'] = np.copy(namespace['ln_rho_in_{}'.format(k)]['g'])
       
        # set initial guess for g_phi
        namespace['g_phi_{}'.format(k)].change_scales(basis.dealias)
        namespace['g_phi_in_{}'.format(k)].change_scales(basis.dealias)
        namespace['g_phi_{}'.format(k)]['g'] = (namespace['g_phi_in_{}'.format(k)]).evaluate()['g']
        #Set the equations: poisson
        if k != 'B':
            problem.add_equation("lap(g_phi_{0}) - four_pi_G_{0}*exp(ln_rho_{0}*ones_{0}) + lift_{0}(tau_g_phi_1_{0}) + lift2_{0}(tau_g_phi_2_{0}) = 0".format(k))
            count_eqn+=1
        elif k == 'B':
            problem.add_equation("lap(g_phi_{0}) - four_pi_G_{0}*exp(ln_rho_{0}*ones_{0})  + lift_{0}(tau_g_phi_1_{0}) = 0".format(k))
            count_eqn+=1
        
        #Set the equations: hydrostatic equilibrium
        problem.add_equation("-grad(g_phi_{0}) + r_vec_{0}*lift_{0}(tau_g_phi_{0}) = g_op_{0} ".format(k))
        # problem.add_equation("grad(ln_rho_{0})@(grad(s_{0})/Cp) + lift_{0}(tau_s_{0}) = -N2_{0}/(gamma*pomega_{0}) - grad(s_{0})@grad(s_{0}) / Cp**2".format(k))
        problem.add_equation("grad(s_{0}) + r_vec_{0}*lift_{0}(tau_s_{0}) = 0".format(k))
        count_eqn+=2
        #Set equation for heating
        problem.add_equation("Q_{0} = edge_smoothing_{0}*div(Fconv_{0})".format(k))
        count_eqn+=1
    
    #Set the boundary conditions.
    iter = 0
    for k, basis in bases.items():
        if k != 'B':
            k_old = list(bases.keys())[iter-1]
            r_s = r_stitch[iter-1]
            problem.add_equation("ln_rho_{0}(r={2}) - ln_rho_{1}(r={2}) = 0".format(k, k_old, r_s))
            problem.add_equation("s_{0}(r={2}) - s_{1}(r={2}) = 0".format(k, k_old, r_s))
            problem.add_equation("g_phi_{0}(r={2}) - g_phi_{1}(r={2}) = 0".format(k, k_old, r_s))
            problem.add_equation("grad(g_phi_{0})(r={2}) - grad(g_phi_{1})(r={2}) = 0".format(k, k_old, r_s))
            count_eqn+=4
        iter += 1
        if iter == len(bases.items()):
            # problem.add_equation("g_phi_{0}(r=r_outer) = g_phi_in_{0}(r=r_outer)".format(k))
            problem.add_equation("g_phi_{0}(r=r_outer) = 0".format(k))
            count_eqn+=1
    
    problem.add_equation("ln_pomega_LHS_B(r=nondim_radius) = 0")
    problem.add_equation("ln_rho_B(r=nondim_radius) = 0")
    count_eqn+=2
    print('number of eqns',count_eqn)

    #Solve with tolerances on pert_norm and hydrostatic equilibrium.
    solver = problem.build_solver(ncc_cutoff=ncc_cutoff)
    pert_norm = np.inf
    while pert_norm > tolerance or HSE_err > HSE_tolerance:
        HSE_err = 0
        solver.newton_iteration(damping=1)
        pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in solver.perturbations)
        logger.info(f'Perturbation norm: {pert_norm:.3e}')
        for k, basis in bases.items():
            this_HSE = np.max(np.abs(namespace['HSE_{}'.format(k)].evaluate()['g']))
            logger.info('HSE in {}:{:.3e}'.format(k, this_HSE))
            if this_HSE > HSE_err:
                HSE_err = this_HSE
    
    # get g
    for k, basis in bases.items():
        namespace['g_phi_{}'.format(k)].change_scales(basis.dealias)
        namespace['g_phi_in_{}'.format(k)].change_scales(basis.dealias)
        namespace['g_{}'.format(k)] = -d3.grad(namespace['g_phi_{}'.format(k)])
    # Stitch together the fields for creation of interpolators that span the full simulation domain.
    #Need: grad_pom0, grad_ln_pom0, grad_ln_rho0, grad_s0, g, pom0, rho0, ln_rho0, g_phi
    stitch_fields = OrderedDict()
    fields = ['grad_pomega', 'grad_ln_pomega', 'grad_ln_rho', 'grad_s', 'g', 'pomega', 'rho', 'ln_rho', 'g_phi', 'r_vec', 'HSE', 'N2_op', 'Q', 's0']
    for f in fields:
        stitch_fields[f] = []
    
    for k, basis in bases.items():
        for f in fields:
            stitch_fields[f] += [np.copy(namespace['{}_{}'.format(f, k)].evaluate()['g'])]

    if len(stitch_fields['r_vec']) == 1:
        for f in fields:
            stitch_fields[f] = stitch_fields[f][0]
    else:
        for f in fields:
            stitch_fields[f] = np.concatenate(stitch_fields[f], axis=-1)

    grad_pom = stitch_fields['grad_pomega'][2,:].ravel()
    grad_ln_pom = stitch_fields['grad_ln_pomega'][2,:].ravel()
    grad_ln_rho = stitch_fields['grad_ln_rho'][2,:].ravel()
    grad_s = stitch_fields['grad_s'][2,:].ravel()
    g = stitch_fields['g'][2,:].ravel()
    HSE = stitch_fields['HSE'][2,:].ravel()
    r = stitch_fields['r_vec'][2,:].ravel()

    pom = stitch_fields['pomega'].ravel()
    rho = stitch_fields['rho'].ravel()
    ln_rho = stitch_fields['ln_rho'].ravel()
    g_phi = stitch_fields['g_phi'].ravel()
    N2 = stitch_fields['N2_op'].ravel() #this should be zero given grad_s = 0 in CZ
    Q = stitch_fields['Q'].ravel()
    s0 = stitch_fields['s0'].ravel()

    #Plot the results.
    fig = plt.figure()
    ax1 = fig.add_subplot(4,2,1)
    ax2 = fig.add_subplot(4,2,2)
    ax3 = fig.add_subplot(4,2,3)
    ax4 = fig.add_subplot(4,2,4)
    ax5 = fig.add_subplot(4,2,5)
    ax6 = fig.add_subplot(4,2,6)
    ax7 = fig.add_subplot(4,2,7)
    ax8 = fig.add_subplot(4,2,8)
    ax1.plot(r, grad_pom, label='grad pomega')
    ax1.legend()
    ax2.plot(r, grad_ln_rho, label='grad ln rho')
    ax2.legend()
    ax3.plot(r, pom/R, label='pomega/R')
    ax3.plot(r, rho, label='rho')
    ax3.legend()
    ax4.plot(r, HSE, label='HSE')
    ax4.legend()
    ax5.plot(r, g, label='g')
    ax5.legend()
    ax6.plot(r, g_phi, label='g_phi')
    ax6.legend()
    ax7.plot(r, N2, label=r'$N^2$')
    ax7.plot(r, -N2, label=r'$-N^2$')
    # ax7.plot(r, (N2_func(r)), label=r'$N^2$ goal', ls='--')
    ax7.set_yscale('log')
    # yticks = (np.max(np.abs(N2.ravel()[r.ravel() < 0.5])), np.max(N2_func(r).ravel()))
    # ax7.set_yticks(yticks)
    # ax7.set_yticklabels(['{:.1e}'.format(n) for n in yticks])
    ax7.legend()
    ax8.plot(r, grad_s, label='grad s')
    ax8.set_yscale('log')
    ax8.legend()
    plt.subplots_adjust(hspace=0.5,wspace=0.25)
    fig.savefig('stratification_CZonly.png', bbox_inches='tight', dpi=300)
    plt.close('all')

    for k, basis in bases.items():
        this_HSE = np.max(np.abs(namespace['HSE_{}'.format(k)].evaluate()['g']))
        print('this HSE',this_HSE)
    

    #Create interpolators for the atmosphere.
    atmosphere = dict()
    atmosphere['grad_pomega'] = interp1d(r, grad_pom, **interp_kwargs)
    atmosphere['grad_ln_pomega'] = interp1d(r, grad_ln_pom, **interp_kwargs)
    atmosphere['grad_ln_rho'] = interp1d(r, grad_ln_rho, **interp_kwargs)
    atmosphere['grad_s'] = interp1d(r, grad_s, **interp_kwargs)
    atmosphere['g'] = interp1d(r, g, **interp_kwargs)
    atmosphere['pomega'] = interp1d(r, pom, **interp_kwargs)
    atmosphere['rho'] = interp1d(r, rho, **interp_kwargs)
    atmosphere['ln_rho'] = interp1d(r, ln_rho, **interp_kwargs)
    atmosphere['g_phi'] = interp1d(r, g_phi, **interp_kwargs)
    atmosphere['N2'] = interp1d(r, N2, **interp_kwargs)
    atmosphere['Q'] = interp1d(r, Q, **interp_kwargs)
    atmosphere['s0'] = interp1d(r, s0, **interp_kwargs)

    # Create dictionary to pass to RZ solve
    quantities_CZ = dict()
    quantities_CZ['Q'] = Q
    quantities_CZ['s0'] = s0
    quantities_CZ['grad_s'] = grad_s
    quantities_CZ['g_phi'] = g_phi
    quantities_CZ['g'] = g
    quantities_CZ['ln_rho'] = ln_rho
    quantities_CZ['grad_ln_rho'] = grad_ln_rho
    quantities_CZ['rho'] = rho
    quantities_CZ['pomega'] = pom
    quantities_CZ['grad_pomega'] = grad_pom
    quantities_CZ['grad_ln_pomega'] = grad_ln_pom
    quantities_CZ['r_de'] = r
    return atmosphere, quantities_CZ

#given atmo from HSE solved above, then input quantities_CZ to this one
#e.g.: 
    #     ln_rho_func = interpolations['ln_rho0']
    #     g_phi_func = interpolations['g_phi']
    #     atmo_test_CZ, quantities_CZ=HSE_solve_CZ(c, d, bases, g_phi_func,ln_rho_func, F_conv_func,
    #             r_outer=r_bound_nd[-1], r_stitch=stitch_radii, \
    #             R=nondim_R_gas, gamma=nondim_gamma1, G=nondim_G, nondim_radius=1,tolerance=1e-5, HSE_tolerance = 1e-4)
# then
    #     L_rad_sim = dmr.Luminosity.cgs.value - (L_conv_sim*(r/L_nd)**2 * (4*np.pi)).cgs.value
    #     L_rad_sim/= (r/L_nd)**2 * (4*np.pi)
    #     F_rad_func = interp1d(r/L_nd, L_rad_sim/lum_nd, **interp_kwargs)
    #     def chi_rad_func(rho,T):
    #         Cp = nondim_cp*s_nd.cgs.value
    #         # opacity =  opacity_func_in(rho,T)
    #         opacity = 3.68e22*(1-z_frac_out)*(1+x_frac_out)*(rho*rho_nd.cgs.value)*(T*T_nd.cgs.value)**(-7./2.)*gff_out + ye_out*(0.2*(1+x_frac_out))
    #         chinum= (16 * constants.sigma_sb.cgs.value * (T*T_nd.cgs.value)**3 ) 
    #         chiden= (3 * (rho*rho_nd.cgs.value)**2 * Cp * opacity)
    #         chi = (chinum/chiden)* (tau_nd / L_nd**2).cgs.value
    #         return chi
    #     #assuming rho, T are nondimensionalized inputs, but will return dimensionalized rad_diff so must non-dimensionalize

    #     value_to_use = 1.1 #1.2
    #     for k, basis in bases.items():
    #         phi, theta, r_basis = d.local_grids(basis)
    #         # print(r)
    #         if r_basis[0][0][0] > 1.1:
    #             r_transition=r_basis[0][0][np.where((np.abs(r_basis[0][0]-value_to_use)/value_to_use < 0.01))[0]][0]
    #     N2_func = interp1d(r_nd, tau_nd**2 * smooth_N2, **interp_kwargs) #just for plotting

    #     atmo_test_RZ=HSE_solve_RZ(c, d, bases, quantities_CZ, r_transition, chi_rad_func, F_rad_func, N2_func,
    #                 r_outer=r_bound_nd[-1], r_stitch=stitch_radii, \
    #                 R=nondim_R_gas, gamma=nondim_gamma1, G=nondim_G, nondim_radius=1,tolerance=1e-5, HSE_tolerance = 1e-4)
def HSE_solve_RZ(coords, dist, bases, quantities_CZ, r_transition, chi_rad_func, Frad_func, N2_func, r_stitch=[], r_outer=1, low_nr=16, \
              R=1, gamma=5/3, G=1, nondim_radius=1, ncc_cutoff=1e-9, tolerance=1e-9, HSE_tolerance = 1e-4):
    """
    Solves for hydrostatic equilibrium in a calorically perfect ideal gas.
    The solution for density, entropy, and gravity is found given a specified function of N^2 and grad ln rho.
    The heating term associated with a convective luminosity is also found given a specified function of the convective flux, Fconv.
    
    Arguments
    ---------
    coords : Dedalus CoordinateSystem object
        The coordinate system in which the solution is found.
    dist : Dedalus Distributor object
        The distributor object associated with the bases; should NOT be in parallel.
    bases : dict
        A dictionary of Dedalus bases, with keys 'B', 'S1', 'S2', etc. for the Ball basis, first Shell basis, second Shell basis, etc.
    grad_ln_rho_func : function
        A function of radius that returns the gradient of the log of density. Input r should be nondimensionalized.
    N2_func : function
        A function of radius that returns the nondimensionalized Brunt-Vaisala frequency squared. Input r should be nondimensionalized. Just to compare answer to MESA smoothed N2
    atmo : dict
        A dictionary of Dedalus fields that represent the initial guess for the quantities and their values at fixed inner point r_transition.
    r_transition : float
        The radius at which to start integrating the radiative zone solution.
    r_stitch : list
        A list of radii at which to stitch together the solutions from different bases. 
        The first element should be the radius of the outer boundary of the BallBasis.
        If there is only one basis, r_stitch should be an empty list.
    r_outer : float
        The radius of the outer boundary of the simulation domain.
    low_nr : int
        The number of radial points in the low resolution domain; used to set up background fields for solve. #TODO: make this by-basis.
    R : float
        The nondimensional value of the gas constant divided by the mean molecular weight.
    gamma : float
        The adiabatic index of the gas.
    nondim_radius : float
        The radius where thermodynamics are nondimensionalized.
    ncc_cutoff : float
        The NCC floor for the solver. See Dedalus.core.solvers.SolverBase
    tolerance : float
        The tolerance for perturbation norm of the newton iteration.
    HSE_tolerance : float
        The tolerance for hydrostatic equilibrium of the BVP solve.
    
    Returns
    -------
    atmosphere : dict
        A dictionary of interpolated functions which return atmospheric quantities as a function of nondimensional radius.
    """
    # Parameters
    namespace = dict()
    namespace['G'] = G
    namespace['R'] = R
    namespace['Cp'] = Cp = R*gamma/(gamma-1)
    namespace['gamma'] = gamma
    namespace['log'] = np.log
    namespace['exp'] = np.exp
    #Loop over bases, set up fields and operators.
    for k, basis in bases.items():
        namespace['basis_{}'.format(k)] = basis
        namespace['S2_basis_{}'.format(k)] = S2_basis = basis.S2_basis()

        # Make problem variables and taus.
        namespace['g_phi_{}'.format(k)] = g_phi = dist.Field(name='g_phi', bases=basis)
        namespace['Q_{}'.format(k)] = Q = dist.Field(name='Q', bases=basis)
        namespace['s_{}'.format(k)] = s = dist.Field(name='s', bases=basis)
        namespace['g_{}'.format(k)] = g = dist.VectorField(coords, name='g', bases=basis)
        namespace['ln_rho_{}'.format(k)] = ln_rho = dist.Field(name='ln_rho', bases=basis)
        namespace['grad_ln_rho_{}'.format(k)] = grad_ln_rho = dist.VectorField(coords, name='grad_ln_rho', bases=basis)
        namespace['pomega_{}'.format(k)] = pomega = dist.Field(name='pomega', bases=basis)
        namespace['tau_s_{}'.format(k)] = tau_s = dist.Field(name='tau_s', bases=S2_basis)
        namespace['tau_rho_{}'.format(k)] = tau_rho = dist.Field(name='tau_rho', bases=S2_basis)
        namespace['tau_pomega_{}'.format(k)] = tau_rho = dist.Field(name='tau_rho', bases=S2_basis)
        namespace['tau_g_phi_{}'.format(k)] = tau_g_phi = dist.Field(name='tau_g_phi', bases=S2_basis)
        namespace['tau_g_phi_1_{}'.format(k)] = tau_g_phi_1 = dist.Field(name='tau_g_phi_1', bases=S2_basis)
        namespace['tau_g_phi_2_{}'.format(k)] = tau_g_phi_2 = dist.Field(name='tau_g_phi_2', bases=S2_basis)

        # Set up some fundamental grid data
        low_scales = low_nr/basis.radial_basis.radial_size 
        phi, theta, r = dist.local_grids(basis)
        phi_de, theta_de, r_de = dist.local_grids(basis, scales=basis.dealias)
        phi_low, theta_low, r_low = dist.local_grids(basis, scales=(1,1,low_scales))
        namespace['r_de_{}'.format(k)] = r_de
        namespace['r_vec_{}'.format(k)] = r_vec = dist.VectorField(coords, bases=basis.radial_basis)
        r_vec['g'][2] = r
        namespace['r_squared_{}'.format(k)] = r_squared = dist.Field(bases=basis.radial_basis)
        r_squared['g'] = r**2       

        # Make lift operators for BCs
        if k == 'B':
            namespace['lift_{}'.format(k)] = lift = lambda A: d3.Lift(A, basis, -1)
        else:
            namespace['lift_{}'.format(k)] = lift = lambda A: d3.Lift(A, basis.derivative_basis(2), -1)
            namespace['lift2_{}'.format(k)] = lift2 = lambda A: d3.Lift(A, basis.derivative_basis(2), -2)

        # Make a field of ones for converting NCCs to full fields.
        namespace['ones_{}'.format(k)] = ones = dist.Field(bases=basis, name='ones')
        ones['g'] = 1

        #make a field of 4piG*ones
        namespace['four_pi_G_{}'.format(k)] = four_pi_G = dist.Field(bases=basis, name='four_pi_G')
        four_pi_G['g'] = 4*np.pi*G*ones['g']

        #Make a field that smooths at the edge of the ball basis.
        namespace['edge_smoothing_{}'.format(k)] = edge_smooth = dist.Field(bases=basis, name='edge_smooth')
        edge_smooth['g'] = one_to_zero(r, 0.95*bases['B'].radius, width=0.03*bases['B'].radius)

        # Get a high-resolution N^2 in the ball; low-resolution elsewhere where it transitions more gradually.
        namespace['N2_{}'.format(k)] = N2 = dist.Field(bases=basis, name='N2')
        if k == 'B':
            N2['g'] = N2_func(r)
        else:
            N2.change_scales(low_scales)
            N2['g'] = N2_func(r_low)
        # not used in the integration of equations, but keep N2_func for comparison at the end

        # Set the radiative flux.
        namespace['Frad_{}'.format(k)] = Frad = dist.VectorField(coords, name='Frad', bases=basis)
        Frad['g'][2] = Frad_func(r)

        # Set ln rho initial guess.
        namespace['ln_rho_in_{}'.format(k)] = ln_rho_in = dist.Field(bases=basis, name='ln_rho_in')
        namespace['grad_ln_rho_in_{}'.format(k)] = grad_ln_rho_in = dist.VectorField(coords,bases=basis, name='grad_ln_rho_in')
        ln_rho_in.change_scales(basis.dealias)
        grad_ln_rho_in.change_scales(basis.dealias)
        if k == 'B':
            dealias_length = int(basis.shape[-1]*basis.dealias[-1])
            ln_rho_in['g'] = quantities_CZ['ln_rho'][:dealias_length]
            grad_ln_rho_in['g'][2] = quantities_CZ['grad_ln_rho'][:dealias_length]
        elif k == 'S1':
            ln_rho_in['g'] = quantities_CZ['ln_rho'][dealias_length:]
            namespace['ln_rho_r_transition'] = namespace['ln_rho_in_{}'.format(k)](r=r_transition)
            grad_ln_rho_in['g'][2] = quantities_CZ['grad_ln_rho'][dealias_length:]
            namespace['grad_ln_rho_r_transition'] = namespace['grad_ln_rho_in_{}'.format(k)](r=r_transition)
        # ln_rho_in['g'] = ln_rho_func(r)

        # set initial guess for g_phi
        namespace['g_phi_in_{}'.format(k)] = g_phi_in = dist.Field(bases=basis, name='g_phi_in')
        g_phi_in.change_scales(basis.dealias)
        if k == 'B':
            g_phi_in['g'] = quantities_CZ['g_phi'][:dealias_length]
        elif k == 'S1':
            g_phi_in['g'] = quantities_CZ['g_phi'][dealias_length:]
            namespace['g_phi_r_transition'] = namespace['g_phi_in_{}'.format(k)](r=r_transition)

        # set initial value for g
        namespace['g_in_{}'.format(k)] = g_in = dist.VectorField(coords,bases=basis, name='g_in')
        g_in.change_scales(basis.dealias)
        if k == 'B':
            g_in['g'][2] = quantities_CZ['g'][:dealias_length]
        elif k == 'S1':
            g_in['g'][2] = quantities_CZ['g'][dealias_length:]
            namespace['g_r_transition'] = namespace['g_in_{}'.format(k)](r=r_transition)
        
        # set initial value for pomega
        namespace['pomega_in_{}'.format(k)] = pomega_in = dist.Field(bases=basis, name='pomega_in')
        namespace['grad_pomega_in_{}'.format(k)] = grad_pomega_in = dist.VectorField(coords,bases=basis, name='grad_pomega_in')
        pomega_in.change_scales(basis.dealias)
        grad_pomega_in.change_scales(basis.dealias)
        if k == 'B':
            pomega_in['g'] = quantities_CZ['pomega'][:dealias_length]
            grad_pomega_in['g'][2] = quantities_CZ['grad_pomega'][:dealias_length]
        elif k == 'S1':
            pomega_in['g'] = quantities_CZ['pomega'][dealias_length:]
            namespace['pomega_r_transition'] = namespace['pomega_in_{}'.format(k)](r=r_transition)
            grad_pomega_in['g'][2] = quantities_CZ['grad_pomega'][dealias_length:]
            namespace['grad_pomega_r_transition'] = namespace['grad_pomega_in_{}'.format(k)](r=r_transition)

        # set initial value for s0
        namespace['s0_in_{}'.format(k)] = s0_in = dist.Field(bases=basis, name='s0_in')
        s0_in.change_scales(basis.dealias)
        if k == 'B':
            s0_in['g'] = quantities_CZ['s0'][:dealias_length]
        elif k == 'S1':
            s0_in['g'] = quantities_CZ['s0'][dealias_length:]
            namespace['s0_r_transition'] = namespace['s0_in_{}'.format(k)](r=r_transition)

        # Set value of Q (just comes from before)
        Q.change_scales(basis.dealias)
        if k == 'B':
            Q['g'] = quantities_CZ['Q'][:dealias_length]
        elif k == 'S1':
            Q['g'] = quantities_CZ['Q'][dealias_length:]
            
        namespace['chi_rad_{}'.format(k)] = chi_rad = lambda rho,pomega: chi_rad_func(rho, pomega/R)
        # Create important operations from the fields.
        namespace['ln_pomega_LHS_{}'.format(k)] = ln_pomega_LHS = gamma*(s/Cp + ((gamma-1)/gamma)*ln_rho*ones)
        namespace['ln_pomega_{}'.format(k)] = ln_pomega = ln_pomega_LHS + np.log(R)
        # namespace['pomega_{}'.format(k)] = pomega = np.exp(ln_pomega) #this goes into the equations
        namespace['P_{}'.format(k)] = P = pomega*np.exp(ln_rho)
        namespace['HSE_{}'.format(k)] = HSE = gamma*pomega*(d3.grad(ones*ln_rho) + d3.grad(s)/Cp) + d3.grad(g_phi)*ones
        #gamma*pomega*(d3.grad(ones*ln_rho) + d3.grad(s)/Cp) - g*ones
        namespace['N2_op_{}'.format(k)] = N2_op = d3.grad(g_phi)@d3.grad(s)/Cp
        namespace['rho_{}'.format(k)] = rho = np.exp(ln_rho*ones)
        namespace['grad_ln_rho_{}'.format(k)] = grad_ln_rho = d3.grad(ln_rho)
        namespace['T_{}'.format(k)] = T = pomega/R
        namespace['ln_T_{}'.format(k)] = ln_T = np.log(pomega) - np.log(R) # ln_pomega - np.log(R)
        namespace['grad_pomega_{}'.format(k)] = grad_pomega = d3.grad(pomega)
        namespace['grad_ln_pomega_{}'.format(k)] = d3.grad(np.log(pomega)) #d3.grad(ln_pomega)
        namespace['grad_s_{}'.format(k)] = grad_s = d3.grad(s)
        namespace['r_vec_g_{}'.format(k)] = r_vec@g
        namespace['g_op_{}'.format(k)] = gamma * pomega * (d3.grad(s)/Cp + d3.grad(ln_rho))
        namespace['s0_{}'.format(k)] = Cp * ((1/gamma)*(np.log(pomega) + ln_rho) - ln_rho) #s with an offset so s0 = cp * (1/gamma * lnP - ln_rho)
        namespace['Frad_op_{}'.format(k)] = Frad*R/(rho * Cp) # = - grad (pomega)*chi_rad
    namespace['pi'] = np.pi
    
    locals().update(namespace)

    for k, basis in bases.items():
        phi, theta, r = dist.local_grids(basis)
        namespace['ones_{}'.format(k)] = ones = dist.Field(bases=basis, name='ones')
        ones['g'] = 1
        
        namespace['HSE_in_{}'.format(k)] = HSE_in = gamma*namespace['pomega_in_{}'.format(k)]*(d3.grad(ones*namespace['ln_rho_in_{}'.format(k)]) + d3.grad(namespace['s0_in_{}'.format(k)])/Cp) + d3.grad(namespace['g_phi_in_{}'.format(k)])*ones
        this_HSE = np.max(np.abs(namespace['HSE_in_{}'.format(k)].evaluate()['g']))
        print('this HSE',this_HSE)

    print('values at r_transition',
          'pomega',namespace['pomega_r_transition'].evaluate()['g'],
          'grad_pomega',namespace['grad_pomega_r_transition'].evaluate()['g'][2],
          's0',namespace['s0_r_transition'].evaluate()['g'],
          'ln_rho',namespace['ln_rho_r_transition'].evaluate()['g'],
          'grad_ln_rho',namespace['grad_ln_rho_r_transition'].evaluate()['g'][2],
          'g_phi',namespace['g_phi_r_transition'].evaluate()['g'],
          'g',namespace['g_r_transition'].evaluate()['g'][2])
    
    
    # Solve for radiative temperature gradient, HSE, and EOS in rad zone only (e.g. only in S1)
    variables, taus = [], []
    for k, basis in bases.items():
        # set values of the fields up until the transition point
        if k == 'B':
            # phi, theta, r = dist.local_grids(basis)
            # phi_de, theta_de, r_de = dist.local_grids(basis, scales=basis.dealias)
            namespace['g_phi_{}'.format(k)].change_scales(basis.dealias)
            namespace['ln_rho_{}'.format(k)].change_scales(basis.dealias)
            namespace['s_{}'.format(k)].change_scales(basis.dealias)
            namespace['pomega_{}'.format(k)].change_scales(basis.dealias)

            namespace['g_phi_{}'.format(k)]['g'] = namespace['g_phi_in_{}'.format(k)]['g']
            namespace['ln_rho_{}'.format(k)]['g']  = namespace['ln_rho_in_{}'.format(k)]['g']
            namespace['pomega_{}'.format(k)]['g']  = namespace['pomega_in_{}'.format(k)]['g']
            namespace['s_{}'.format(k)]['g'] = 0
            #ln pomega comes from these
            
        if k != 'B':
            variables += [namespace['g_phi_{}'.format(k)], namespace['s_{}'.format(k)],
                          namespace['ln_rho_{}'.format(k)], namespace['pomega_{}'.format(k)] ]
            
            taus += [ namespace['tau_g_phi_1_{}'.format(k)],  namespace['tau_g_phi_{}'.format(k)], 
                     namespace['tau_g_phi_2_{}'.format(k)], namespace['tau_pomega_{}'.format(k)]] 

    print('variables',len(variables),'taus',len(taus))
    problem = d3.NLBVP(variables + taus, namespace=locals())
    count_eqn = 0
    for k, basis in bases.items():
        if k != 'B':
            # Set a decent initial guess for ln_rho.
            namespace['ln_rho_{}'.format(k)].change_scales(basis.dealias)
            namespace['ln_rho_in_{}'.format(k)].change_scales(basis.dealias)
            namespace['ln_rho_{}'.format(k)]['g'] = np.copy(namespace['ln_rho_in_{}'.format(k)]['g'])
        
            # set initial guess for g_phi
            namespace['g_phi_{}'.format(k)].change_scales(basis.dealias)
            namespace['g_phi_in_{}'.format(k)].change_scales(basis.dealias)
            namespace['g_phi_{}'.format(k)]['g'] = np.copy(namespace['g_phi_in_{}'.format(k)].evaluate()['g'])

            # set initial guess for pomega
            namespace['pomega_{}'.format(k)].change_scales(basis.dealias)
            namespace['pomega_in_{}'.format(k)].change_scales(basis.dealias)
            namespace['pomega_{}'.format(k)]['g'] = np.copy(namespace['pomega_in_{}'.format(k)].evaluate()['g'])
            #Set the equations: poisson
            problem.add_equation("lap(g_phi_{0}) - four_pi_G_{0}*exp(ln_rho_{0}*ones_{0}) + lift_{0}(tau_g_phi_1_{0}) + lift2_{0}(tau_g_phi_2_{0}) = 0".format(k))
            count_eqn+=1
            #Set the equations: hydrostatic equilibrium
            problem.add_equation("-grad(g_phi_{0}) + r_vec_{0}*lift_{0}(tau_g_phi_{0}) = g_op_{0} ".format(k))
            # this has grad_s in it
            #Set equation for radiative temperature gradient
            # problem.add_equation("grad(s_{0}) + r_vec_{0}*lift_{0}(tau_s_{0}) = 0 ".format(k))
            problem.add_equation(" -grad(pomega_{0}) + r_vec_{0}*lift_{0}(tau_pomega_{0}) = Frad_op_{0}/chi_rad_{0}(exp(ln_rho_{0}*ones_{0}),pomega_{0})".format(k))  #Frad_op_{0}
            problem.add_equation("pomega_{0} = exp(ln_pomega_{0})".format(k))
            count_eqn+=3
    
    #Set the boundary conditions.
    iter = 0
    for k, basis in bases.items():
        if k != 'B':
            k_old = list(bases.keys())[iter-1]
            r_s = r_stitch[iter-1]
        iter += 1
    #fix them to the values at the transition point 
    problem.add_equation("pomega_S1(r=r_transition) = pomega_r_transition")
    problem.add_equation("ln_rho_S1(r=r_transition) = ln_rho_r_transition")
    problem.add_equation("g_phi_S1(r=r_transition) = g_phi_r_transition") 
    problem.add_equation(" - grad(g_phi_S1)(r=r_transition) = g_r_transition")
    count_eqn+=4
    print('number of eqns',count_eqn)

    #Solve with tolerances on pert_norm and hydrostatic equilibrium.
    solver = problem.build_solver(ncc_cutoff=ncc_cutoff)
    pert_norm = np.inf
    while pert_norm > tolerance or HSE_err > HSE_tolerance:
        HSE_err = 0
        solver.newton_iteration(damping=1)
        pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in solver.perturbations)
        logger.info(f'Perturbation norm: {pert_norm:.3e}')
        for k, basis in bases.items():
            this_HSE = np.max(np.abs(namespace['HSE_{}'.format(k)].evaluate()['g']))
            logger.info('HSE in {}:{:.3e}'.format(k, this_HSE))
            if this_HSE > HSE_err:
                HSE_err = this_HSE
    
    #get g
    for k, basis in bases.items():
        namespace['g_phi_{}'.format(k)].change_scales(basis.dealias)
        namespace['g_phi_in_{}'.format(k)].change_scales(basis.dealias)
        namespace['g_{}'.format(k)] = -d3.grad(namespace['g_phi_{}'.format(k)])
    # now, when stitch together, make sure to combine the one for the CZ with the one for the RZ
    # Stitch together the fields for creation of interpolators that span the full simulation domain.
    # Need: grad_pom0, grad_ln_pom0, grad_ln_rho0, grad_s0, g, pom0, rho0, ln_rho0, g_phi
    stitch_fields = OrderedDict()
    fields = ['grad_pomega', 'grad_ln_pomega', 'grad_ln_rho', 'grad_s', 'g', 'pomega', 'rho', 'ln_rho', 'g_phi', 'r_vec', 'HSE', 'N2_op', 'Q', 's0']
    for f in fields:
        stitch_fields[f] = []
    
    for k, basis in bases.items():
        for f in fields:
            if f == 'r_vec':
                namespace['{}_{}'.format(f, k)].change_scales(basis.dealias)
                # stitch_fields[f] += [np.copy(namespace['{}_{}'.format(f, k)].evaluate()['g'][2])]
            stitch_fields[f] += [np.copy(namespace['{}_{}'.format(f, k)].evaluate()['g'])]

    if len(stitch_fields['r_vec']) == 1:
        for f in fields:
            stitch_fields[f] = stitch_fields[f][0]
    else:
        for f in fields:
            stitch_fields[f] = np.concatenate(stitch_fields[f], axis=-1)

    grad_pom = stitch_fields['grad_pomega'][2,:].ravel()
    grad_ln_pom = stitch_fields['grad_ln_pomega'][2,:].ravel()
    grad_ln_rho = stitch_fields['grad_ln_rho'][2,:].ravel()
    grad_s = stitch_fields['grad_s'][2,:].ravel()
    g = stitch_fields['g'][2,:].ravel()
    HSE = stitch_fields['HSE'][2,:].ravel()
    r = stitch_fields['r_vec'][2,:].ravel()

    pom = stitch_fields['pomega'].ravel()
    rho = stitch_fields['rho'].ravel()
    ln_rho = stitch_fields['ln_rho'].ravel()
    g_phi = stitch_fields['g_phi'].ravel()
    N2 = stitch_fields['N2_op'].ravel()
    Q = stitch_fields['Q'].ravel()
    s0 = stitch_fields['s0'].ravel()

    #Plot the results.
    fig = plt.figure(figsize=(8,12))
    ax1 = fig.add_subplot(4,2,1)
    ax2 = fig.add_subplot(4,2,2)
    ax3 = fig.add_subplot(4,2,3)
    ax4 = fig.add_subplot(4,2,4)
    ax5 = fig.add_subplot(4,2,5)
    ax6 = fig.add_subplot(4,2,6)
    ax7 = fig.add_subplot(4,2,7)
    ax8 = fig.add_subplot(4,2,8)
    ax1.plot(r, grad_pom, label='grad pomega')
    ax1.legend()
    ax2.plot(r, grad_ln_rho, label='grad ln rho')
    ax2.legend()
    ax3.plot(r, pom/R, label='pomega/R')
    ax3.plot(r, rho, label='rho')
    ax3.legend()
    ax4.plot(r, HSE, label='HSE')
    ax4.legend()
    ax5.plot(r, g, label='g')
    ax5.legend()
    ax6.plot(r, g_phi, label='g_phi')
    ax6.legend()
    ax7.plot(r, N2, label=r'$N^2$')
    ax7.plot(r, -N2, label=r'$-N^2$')
    ax7.plot(r, (N2_func(r)), label=r'$N^2$ goal', ls='--')
    
    yticks = (np.max(np.abs(N2.ravel()[r.ravel() < 0.5])), np.max(N2_func(r).ravel()))
    print(yticks)
    ax7.set_yticks(yticks)
    ax7.set_yticklabels(['{:.1e}'.format(n) for n in yticks])
    ax7.set_yscale('log')
    ax7.legend()
    ax8.plot(r, grad_s, label='grad s')
    ax8.set_yscale('log')
    ax8.legend()
    plt.subplots_adjust(hspace=0.4,wspace=0.25)
    fig.savefig('stratification_CZplusRZ.png', bbox_inches='tight', dpi=300)
    plt.close('all')
    #Create interpolators for the atmosphere.
    atmosphere = dict()
    atmosphere['grad_pomega'] = interp1d(r, grad_pom, **interp_kwargs)
    atmosphere['grad_ln_pomega'] = interp1d(r, grad_ln_pom, **interp_kwargs)
    atmosphere['grad_ln_rho'] = interp1d(r, grad_ln_rho, **interp_kwargs)
    atmosphere['grad_s'] = interp1d(r, grad_s, **interp_kwargs)
    atmosphere['g'] = interp1d(r, g, **interp_kwargs)
    atmosphere['pomega'] = interp1d(r, pom, **interp_kwargs)
    atmosphere['rho'] = interp1d(r, rho, **interp_kwargs)
    atmosphere['ln_rho'] = interp1d(r, ln_rho, **interp_kwargs)
    atmosphere['g_phi'] = interp1d(r, g_phi, **interp_kwargs)
    atmosphere['N2'] = interp1d(r, N2, **interp_kwargs)
    atmosphere['Q'] = interp1d(r, Q, **interp_kwargs)
    atmosphere['s0'] = interp1d(r, s0, **interp_kwargs)
    return atmosphere

#after smoothing, call this function to make sure smoothed functions are consistent with EOS and HSE given smoothed grad_s profile.
def HSE_EOS_solve(coords, dist, bases, grad_s_smooth_func, g_func, ln_rho_func_in, pomega_func_in, s0_const, r_stitch=[], r_outer=1, low_nr=16, \
              R=1, gamma=5/3, G=1, nondim_radius=1, ncc_cutoff=1e-9, tolerance=1e-9, HSE_tolerance = 1e-4):
    
    # Parameters
    namespace = dict()
    namespace['G'] = G
    namespace['R'] = R
    namespace['Cp'] = Cp = R*gamma/(gamma-1)
    namespace['gamma'] = gamma
    namespace['log'] = np.log
    namespace['exp'] = np.exp

    #Loop over bases, set up fields and operators.
    for k, basis in bases.items():
        namespace['basis_{}'.format(k)] = basis
        namespace['S2_basis_{}'.format(k)] = S2_basis = basis.S2_basis()

        # Make problem variables and taus.
        
        namespace['s_{}'.format(k)] = s = dist.Field(name='s', bases=basis)
        namespace['pomega_{}'.format(k)] = pomega = dist.Field( name='pomega', bases=basis)
        namespace['ln_rho_{}'.format(k)] = ln_rho = dist.Field(name='ln_rho', bases=basis)
        namespace['grad_ln_rho_{}'.format(k)] = grad_ln_rho = dist.VectorField(coords, name='grad_ln_rho', bases=basis)
        namespace['tau_s_{}'.format(k)] = tau_pomega = dist.Field(name='tau_pomega', bases=S2_basis)
        namespace['tau_rho_{}'.format(k)] = tau_rho = dist.Field(name='tau_rho', bases=S2_basis)

        # Set up some fundamental grid data
        low_scales = low_nr/basis.radial_basis.radial_size 
        phi, theta, r = dist.local_grids(basis)
        phi_de, theta_de, r_de = dist.local_grids(basis, scales=basis.dealias)
        phi_low, theta_low, r_low = dist.local_grids(basis, scales=(1,1,low_scales))
        namespace['r_de_{}'.format(k)] = r_de
        namespace['r_vec_{}'.format(k)] = r_vec = dist.VectorField(coords, bases=basis.radial_basis)
        r_vec['g'][2] = r
        namespace['r_squared_{}'.format(k)] = r_squared = dist.Field(bases=basis.radial_basis)
        r_squared['g'] = r**2       

        # Make lift operators for BCs
        if k == 'B':
            namespace['lift_{}'.format(k)] = lift = lambda A: d3.Lift(A, basis, -1)
        else:
            namespace['lift_{}'.format(k)] = lift = lambda A: d3.Lift(A, basis.derivative_basis(2), -1)

        # Make a field of ones for converting NCCs to full fields.
        namespace['ones_{}'.format(k)] = ones = dist.Field(bases=basis, name='ones')
        ones['g'] = 1

        #Make a field that smooths at the edge of the ball basis.
        namespace['edge_smoothing_{}'.format(k)] = edge_smooth = dist.Field(bases=basis, name='edge_smooth')
        edge_smooth['g'] = one_to_zero(r, 0.95*bases['B'].radius, width=0.03*bases['B'].radius)

        # Set s0, g inputs.
        namespace['grad_s0_in_{}'.format(k)] = grad_s0_in = dist.VectorField(coords,bases=basis, name='grad_s0_in')
        grad_s0_in.change_scales(basis.dealias)
        grad_s0_in['g'][2] = grad_s_smooth_func(r_de)

        namespace['g_in_{}'.format(k)] = g_in = dist.VectorField(coords,bases=basis, name='g_in')
        g_in.change_scales(basis.dealias)
        g_in['g'][2] = g_func(r_de)

        #set initial guesses for ln_rho, pomega
        namespace['ln_rho_in_{}'.format(k)] = ln_rho_in = dist.Field(bases=basis, name='ln_rho_in')
        ln_rho_in.change_scales(basis.dealias)
        ln_rho_in['g'] = ln_rho_func_in(r_de)

        namespace['pomega_in_{}'.format(k)] = pomega_in = dist.Field(bases=basis, name='pomega_in')
        pomega_in.change_scales(basis.dealias)
        pomega_in['g'] = pomega_func_in(r_de)


        # Create important operations from the fields.
        namespace['P_{}'.format(k)] = P = pomega*np.exp(ln_rho)
        namespace['HSE_{}'.format(k)] = HSE = gamma*pomega*(grad_ln_rho + grad_s0_in/Cp) - g_in*ones
        namespace['rho_{}'.format(k)] = rho = np.exp(ln_rho*ones)
        namespace['T_{}'.format(k)] = T = pomega/R
        namespace['grad_pomega_{}'.format(k)] = d3.grad(pomega)
        namespace['grad_ln_pomega_{}'.format(k)] = d3.grad(pomega)/pomega
        namespace['r_vec_g_{}'.format(k)] = r_vec@g_in
        namespace['g_op_{}'.format(k)] = gamma * pomega * (grad_s0_in/Cp + grad_ln_rho)
        namespace['s0_op_{}'.format(k)] = Cp * ((1/gamma)*(np.log(pomega) + ln_rho) - ln_rho) #s with an offset so s0 = cp * (1/gamma * lnP - ln_rho)

    namespace['pi'] = np.pi
    locals().update(namespace)
    
    #Solve for s0.
    variables, taus = [], []
    for k, basis in bases.items():
        variables += [namespace['s_{}'.format(k)],]
        taus      += [namespace['tau_s_{}'.format(k)],]

    problem = d3.NLBVP(variables + taus, namespace=locals())
    for k, basis in bases.items():
        #Equation is just definitional.
        problem.add_equation("grad(s_{0}) - grad_s0_in_{0} + r_vec_{0}*lift_{0}(tau_s_{0}) = 0".format(k))
    
    #Set boundary conditions.
    iter = 0
    for k, basis in bases.items():
        if k != 'B':
            k_old = list(bases.keys())[iter-1]
            r_s = r_stitch[iter-1]
            problem.add_equation("s_{0}(r={2}) - s_{1}(r={2}) = 0".format(k, k_old, r_s))
        iter += 1
    problem.add_equation("s_B(r=0) = {0}".format(s0_const))

    solver = problem.build_solver(ncc_cutoff=ncc_cutoff)
    pert_norm = np.inf
    while pert_norm > tolerance:
        solver.newton_iteration(damping=1)
        pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in solver.perturbations)
        logger.info(f'Perturbation norm: {pert_norm:.3e}')
    logger.info('s0 found')

    # Solve for HSE/EOS simultaneously given s.
    variables, taus = [], []
    for k, basis in bases.items():
        variables += [namespace['pomega_{}'.format(k)],namespace['grad_ln_rho_{}'.format(k)],namespace['ln_rho_{}'.format(k)]]
        
        taus += [namespace['tau_rho_{}'.format(k)]] 

    print('variables',len(variables),'taus',len(taus))
    problem = d3.NLBVP(variables + taus, namespace=locals())
    count_eqn = 0
    for k, basis in bases.items():
        #set decent initial guesses for ln rho and pomega
        namespace['ln_rho_{}'.format(k)].change_scales(basis.dealias)
        namespace['pomega_{}'.format(k)].change_scales(basis.dealias)
        namespace['ln_rho_in_{}'.format(k)].change_scales(basis.dealias)
        namespace['pomega_in_{}'.format(k)].change_scales(basis.dealias)

        namespace['ln_rho_{}'.format(k)]['g'] = np.copy(namespace['ln_rho_in_{}'.format(k)]['g'])
        namespace['pomega_{}'.format(k)]['g'] = np.copy(namespace['pomega_in_{}'.format(k)]['g'])

    
        #Set the equations: hydrostatic equilibrium
        problem.add_equation("g_in_{0} = g_op_{0} ".format(k))
        problem.add_equation("s_{0} = s0_op_{0}".format(k))
        problem.add_equation("grad(ln_rho_{0}) - grad_ln_rho_{0} + r_vec_{0}*lift_{0}(tau_rho_{0}) = 0".format(k))
        #above is just definition of ln_rho/grad_ln rho to recover ln_rho
        
        count_eqn+=3

    
    #Set the boundary conditions.
    iter = 0
    for k, basis in bases.items():
        if k != 'B':
            k_old = list(bases.keys())[iter-1]
            r_s = r_stitch[iter-1]
            problem.add_equation("ln_rho_{0}(r={2}) - ln_rho_{1}(r={2}) = 0".format(k, k_old, r_s))
            # problem.add_equation("pomega_{0}(r={2}) - pomega_{1}(r={2}) = 0".format(k, k_old, r_s))
            count_eqn+=1
        iter += 1
    problem.add_equation("ln_rho_B(r=nondim_radius) = 0")
    count_eqn+=1
    print('number of eqns',count_eqn)

    #Solve with tolerances on pert_norm and hydrostatic equilibrium.
    solver = problem.build_solver(ncc_cutoff=ncc_cutoff)
    pert_norm = np.inf
    while pert_norm > tolerance or HSE_err > HSE_tolerance:
        HSE_err = 0
        solver.newton_iteration(damping=1)
        pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in solver.perturbations)
        logger.info(f'Perturbation norm: {pert_norm:.3e}')
        for k, basis in bases.items():
            this_HSE = np.max(np.abs(namespace['HSE_{}'.format(k)].evaluate()['g']))
            logger.info('HSE in {}:{:.3e}'.format(k, this_HSE))
            if this_HSE > HSE_err:
                HSE_err = this_HSE

    # plt.figure()

    # Stitch together the fields for creation of interpolators that span the full simulation domain.
    #Need: grad_pom0, grad_ln_pom0, grad_ln_rho0, grad_s0, g, pom0, rho0, ln_rho0, g_phi
    stitch_fields = OrderedDict()
    fields = ['grad_pomega', 'grad_ln_pomega', 'grad_ln_rho', 'pomega', 'rho', 'ln_rho', 's','grad_s0_in','r_vec','HSE','g_in']
    for f in fields:
        stitch_fields[f] = []
    
    for k, basis in bases.items():
        for f in fields:
            stitch_fields[f] += [np.copy(namespace['{}_{}'.format(f, k)].evaluate()['g'])]

    if len(stitch_fields['r_vec']) == 1:
        for f in fields:
            stitch_fields[f] = stitch_fields[f][0]
    else:
        for f in fields:
            stitch_fields[f] = np.concatenate(stitch_fields[f], axis=-1)

    grad_pom = stitch_fields['grad_pomega'][2,:].ravel()
    grad_ln_pom = stitch_fields['grad_ln_pomega'][2,:].ravel()
    grad_ln_rho = stitch_fields['grad_ln_rho'][2,:].ravel()
    grad_s = stitch_fields['grad_s0_in'][2,:].ravel()
    g = stitch_fields['g_in'][2,:].ravel()
    r = stitch_fields['r_vec'][2,:].ravel()

    pom = stitch_fields['pomega'].ravel()
    rho = stitch_fields['rho'].ravel()
    ln_rho = stitch_fields['ln_rho'].ravel()
    s0 = stitch_fields['s'].ravel()
    HSE = stitch_fields['HSE'][2,:].ravel()

    #Plot the results.
    fig = plt.figure(figsize=(8,12))
    ax1 = fig.add_subplot(4,2,1)
    ax2 = fig.add_subplot(4,2,2)
    ax3 = fig.add_subplot(4,2,3)
    ax4 = fig.add_subplot(4,2,4)
    ax5 = fig.add_subplot(4,2,5)
    ax6 = fig.add_subplot(4,2,6)
    ax1.plot(r, grad_pom, label='grad pomega')
    ax1.legend()
    ax2.plot(r, grad_ln_rho, label='grad ln rho')
    ax2.legend()
    ax3.plot(r, pom/R, label='pomega/R')
    ax3.plot(r, rho, label='rho')
    ax3.legend()
    ax4.plot(r, HSE, label='HSE')
    ax4.legend()
    ax5.plot(r, g, label='g')
    ax5.legend()

    ax6.plot(r, grad_s, label='grad s')
    ax6.set_yscale('log')
    ax6.legend()
    plt.subplots_adjust(hspace=0.5,wspace=0.25)
    fig.savefig('stratification_HSE_EOS_smoothed.png', bbox_inches='tight', dpi=300)
    plt.close('all')
    #Create interpolators for the atmosphere.
    atmosphere = dict()
    atmosphere['grad_pomega'] = interp1d(r, grad_pom, **interp_kwargs)
    atmosphere['grad_ln_pomega'] = interp1d(r, grad_ln_pom, **interp_kwargs)
    atmosphere['grad_ln_rho'] = interp1d(r, grad_ln_rho, **interp_kwargs)
    atmosphere['grad_s'] = interp1d(r, grad_s, **interp_kwargs)
    atmosphere['pomega'] = interp1d(r, pom, **interp_kwargs)
    atmosphere['rho'] = interp1d(r, rho, **interp_kwargs)
    atmosphere['ln_rho'] = interp1d(r, ln_rho, **interp_kwargs)
    atmosphere['s0'] = interp1d(r, s0, **interp_kwargs)
    return atmosphere