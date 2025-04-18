import os
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import h5py
import matplotlib.pyplot as plt
import dedalus.public as d3

from astropy import units as u
from astropy import constants
from scipy.interpolate import interp1d

import compstar 
from .compressible_functions import make_bases
from .parser import name_star
from .bvp_functions import HSE_solve_CZ, HSE_solve_RZ, HSE_EOS_solve
from compstar.tools.mesa import DimensionalMesaReader, find_core_cz_radius, adjust_opacity, opacity_func
from compstar.tools.general import one_to_zero, zero_to_one
import compstar.defaults.config as config

import logging
logger = logging.getLogger(__name__)

interp_kwargs = {'fill_value' : 'extrapolate', 'bounds_error' : False, 'kind': 'cubic'}

### Function definitions
def plot_ncc_figure(rvals, mesa_func, dedalus_vals, ylabel="", fig_name="", out_dir='.', 
                    zero_line=False, log=False, r_int=None, ylim=None, axhline=None, Ns=None, ncc_cutoff=1e-6):
    """ 
    Plots a figure which compares a dedalus field and the MESA profile that the Dedalus field is based on. 

    Parameters
    ----------
    rvals : list of arrays
        The radial values of the dedalus field
    mesa_func : function
        A function which takes a radius and returns the corresponding MESA value
    dedalus_vals : list of arrays
        The dedalus field values
    
    ylabel : str
        The label for the y-axis
    fig_name : str
        The name of the figure (will be saved as a png)
    out_dir : str
        The directory to save the figure in
    zero_line : bool
        Whether to plot a horizontal line at y=0
    log : bool
        Whether to plot the y-axis on a log scale
    r_int : list of floats
        The radii to plot vertical lines at
    ylim : list of floats
        The limits of the y-axis
    axhline : float
        A value to plot a horizontal line at
    Ns : list of int
        The number of coefficients used in the dedalus field expansion
    ncc_cutoff : float
        The NCC cutoff used.
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    if zero_line:
        ax1.axhline(0, c='k', lw=0.5)

    if axhline is not None:
        ax1.axhline(axhline, c='k')

    first = True
    for r, y in zip(rvals, dedalus_vals):
        mesa_y = mesa_func(r)
        if first:
            ax1.plot(r, mesa_y, label='mesa', c='k', lw=3)
            ax1.plot(r, y, label='dedalus', c='red')
            first = False
        else:
            ax1.plot(r, mesa_y, c='k', lw=3)
            ax1.plot(r, y, c='red')

        diff = np.abs(1 - mesa_y/y)
        ax2.plot(r, diff)

    ax1.legend(loc='best')
    ax1.set_xlabel('radius/L', labelpad=-3)
    ax1.set_ylabel('{}'.format(ylabel))
    ax1.xaxis.set_ticks_position('top')
    ax1.xaxis.set_label_position('top')
    if log:
        ax1.set_yscale('log')
    if ylim is not None:
        ax1.set_ylim(ylim)

    ax2.axhline(1e-1, c='k', lw=0.5)
    ax2.axhline(1e-2, c='k', lw=0.5)
    ax2.axhline(1e-3, c='k', lw=0.5)
    ax2.set_ylabel('abs(1 - mesa/dedalus)')
    ax2.set_xlabel('radius/L')
    ax2.set_yscale('log')

    ax2.set_ylim(1e-4, 1)
    fig.suptitle('coeff bandwidth = {}; cutoff = {:e}'.format(Ns, ncc_cutoff))
    if r_int is not None:
        for ax in [ax1, ax2]:
            for rval in r_int:
                ax.axvline(rval, c='k')
    fig.savefig('{:s}/{}.png'.format(out_dir, fig_name), bbox_inches='tight', dpi=200)

def make_NCC(basis, coords, dist, interp_func, Nmax=32, vector=False, grid_only=False, ncc_cutoff=1e-6):
    """
    Given a function which returns some MESA profile as a function of nondimensional radius,
    this function returns a dedalus field which is the Dedalus expansion of that function.

    Arguments
    ---------
    basis : dedalus basis
        The dedalus basis to use for the field
    coords : dedalus coordinates
        The dedalus coordinates to use for the field
    dist : dedalus distributor
        The dedalus distributor to use for the field
    interp_func : function
        A function which takes a radius and returns the corresponding (nondimensional) MESA value
    Nmax : int
        The maximum number of coefficients to use in the dedalus expansion
    vector : bool
        Whether the field is a vector field; if False, the field is a scalar field
    grid_only : bool
        If True, this field will never be transformed into coefficient space (i.e. it will always be in grid space)
    ncc_cutoff : float
        The NCC cutoff used.
    """
    if grid_only:
        scales = basis.dealias
    else:
        scales = basis.dealias
        scales_small = (1, 1, Nmax/basis.radial_basis.radial_size)
    rvals = basis.global_grid_radius(dist,scale=scales[2])
    if vector:
        this_field = dist.VectorField(coords, bases=basis)
        this_field.change_scales(scales)
        this_field['g'][2] = interp_func(rvals)
    else:
        this_field = dist.Field(bases=basis)
        this_field.change_scales(scales)
        this_field['g'] = interp_func(rvals)
    if not grid_only:
        this_field.change_scales(scales_small)
        this_field['g']
        this_field['c'][np.abs(this_field['c']) < ncc_cutoff] = 0
        this_field.change_scales(basis.dealias)
    return this_field

def build_nccs(plot_nccs=False, grad_s_transition_default=0.03, bg_CZ_RZ_transition_default=1.01, reapply_grad_s_filter=False):
    """
    This function builds the NCCs for the star, then saves them to a file.
    TODO: This function is a bit of a mess, and should be cleaned up. It should be turned into a class.
    TODO: Generalize this function; it is currently specific to the massive star case.

    Arguments
    ---------
    plot_nccs : bool
        Whether to plot the NCCs
    grad_s_transition_default : float
        The default value for how far the entropy gradient transition should be away from the BallBasis outer boundary
    reapply_grad_s_filter : bool
        Whether to reapply the zero-to-one filter after expanding the entropy gradient field.
    """
    # Read in parameters and create output directory
    out_dir, out_file = name_star()
    ncc_dict = config.nccs
    fluct_dict = config.initial_flucts

    # Find the path to the MESA profile file
    package_path = Path(compstar.__file__).resolve().parent
    stock_path = package_path.joinpath('stock_models')
    mesa_file_path = None
    if os.path.exists(config.star['path']):
        mesa_file_path = config.star['path']
    else:
        stock_file_path = stock_path.joinpath(config.star['path'])
        if os.path.exists(stock_file_path):
            mesa_file_path = str(stock_file_path)
        else:
            raise ValueError("Cannot find MESA profile file in {} or {}".format(config.star['path'], stock_file_path))

    # Read in the MESA profile
    reader = DimensionalMesaReader(mesa_file_path)
    dmr = SimpleNamespace(**reader.structure) # Turns the dictionary into a namespace so that fields can be accessed as attributes
    #make some commonly-used variables local.
    r, mass, rho, T = dmr.r, dmr.mass, dmr.rho, dmr.T
    N2, g, cp = dmr.N2, dmr.g, dmr.cp

    ### CORE CONVECTION LOGIC - lots of stuff here needs to be generalized for other types of stars.
    core_cz_radius = find_core_cz_radius(mesa_file_path, dimensionless=False)
    opacity_adjusted, gff_out, z_frac_out, x_frac_out,ye_out = adjust_opacity(mesa_file_path, dimensionless=False)

    ### Recalculate k_rad and rad_diff using adjusted opacity. might need to redo after the HSE solve since rho, T could change
    rad_diff        = (16 * constants.sigma_sb.cgs * T**3 / (3 * rho**2 * cp * opacity_adjusted)).cgs
    k_rad    = rad_cond = rad_diff*(rho * cp)

    # Get some rough MLT values.
    mlt_u = ((dmr.Luminosity / (4 * np.pi * r**2 * rho) )**(1/3)).cgs
    avg_core_u = np.sum((4*np.pi*r**2*np.gradient(r)*mlt_u)[r < core_cz_radius]) / (4*np.pi*core_cz_radius**3 / 3)
    avg_core_ma = np.sum((4*np.pi*r**2*np.gradient(r)*mlt_u/dmr.csound)[r < core_cz_radius]) / (4*np.pi*core_cz_radius**3 / 3)
    logger.info('avg core velocity: {:.3e} / ma: {:.3e}'.format(avg_core_u, avg_core_ma))

    # Specify fraction of total star to simulate
    r_bounds = list(config.star['r_bounds'])
    r_bools = []
    for i, rb in enumerate(r_bounds):
        if type(rb) == str:
            if 'R' in rb:
                r_bounds[i] = float(rb.replace('R', ''))*dmr.R_star
            elif 'L' in rb:
                r_bounds[i] = float(rb.replace('L', ''))*core_cz_radius
            else:
                try:
                    r_bounds[i] = float(r_bounds[i]) * u.cm
                except:
                    raise ValueError("index {} ('{}') of r_bounds is poorly specified".format(i, rb))
            r_bounds[i] = core_cz_radius*np.around(r_bounds[i]/core_cz_radius, decimals=2)
    for i, rb in enumerate(r_bounds):
        if i < len(r_bounds) - 1:
            r_bools.append((r > r_bounds[i])*(r <= r_bounds[i+1]))
    logger.info('fraction of FULL star simulated: {:.2f}, up to r={:.3e}'.format(r_bounds[-1]/dmr.R_star, r_bounds[-1]))
    sim_bool      = (r > r_bounds[0])*(r <= r_bounds[-1])
    logger.info('fraction of stellar mass simulated: {:.7f}'.format(mass[sim_bool][-1]/mass[-1]))

    #Get N2 info
    N2max_sim = N2[sim_bool].max()
    shell_points = np.sum(sim_bool*(r > core_cz_radius))
    N2plateau = np.median(N2[r > core_cz_radius][int(shell_points*0.25):int(shell_points*0.75)])
    f_brunt = np.sqrt(N2max_sim)/(2*np.pi)
 
    #Nondimensionalization
    L_CZ    = core_cz_radius
    m_core  = rho[0] * L_CZ**3
    T_core  = T[0]
    H0      = (rho*dmr.eps_nuc)[0]
    tau_heat  = ((H0*L_CZ/m_core)**(-1/3)).cgs #heating timescale
    L_nd    = L_CZ
    m_nd    = rho[r==L_nd][0] * L_nd**3 #mass at core cz boundary
    T_nd    = T[r==L_nd][0] #temp at core cz boundary
    tau_nd  = (1/f_brunt).cgs #timescale of max N^2
    rho_nd  = m_nd/L_nd**3
    u_nd    = L_nd/tau_nd
    s_nd    = L_nd**2 / tau_nd**2 / T_nd
    H_nd    = (m_nd / L_nd) * tau_nd**-3
    s_motions    = L_nd**2 / tau_heat**2 / T[0]
    lum_nd  = L_nd**2 * m_nd / (tau_nd**2) / tau_nd
    nondim_R_gas = (dmr.R_gas / s_nd).cgs.value
    nondim_gamma1 = (dmr.gamma1[0]).value
    nondim_cp = nondim_R_gas * nondim_gamma1 / (nondim_gamma1 - 1)
    nondim_G = (constants.G.cgs / (L_nd**3 / m_nd / tau_nd**2)).cgs
    u_heat_nd = (L_nd/tau_heat) / u_nd
    Ma2_r0 = ((u_nd*(tau_nd/tau_heat))**2 / ((dmr.gamma1[0]-1)*cp[0]*T[0])).cgs
    logger.info('Nondimensionalization: L_nd = {:.2e}, T_nd = {:.2e}, m_nd = {:.2e}, tau_nd = {:.2e}'.format(L_nd, T_nd, m_nd, tau_nd))
    logger.info('Thermo: Cp/s_nd: {:.2e}, R_gas/s_nd: {:.2e}, gamma1: {:.4f}'.format(nondim_cp, nondim_R_gas, nondim_gamma1))
    logger.info('m_nd/M_\odot: {:.3f}'.format((m_nd/constants.M_sun).cgs))
    logger.info('estimated mach number: {:.3e} / t_heat: {:.3e}'.format(np.sqrt(Ma2_r0), tau_heat))

    #Gravitational potential, set to -1 at r = R_star
    g_phi = np.cumsum(g*np.gradient(r))  #gvec = -grad phi; 
    g_phi -= g_phi[-1] - u_nd**2 #set g_phi = -1 at r = dmr.R_star
    
    #construct diffusivity profiles which will be used in simulation.
    rad_diff_nd = rad_diff * (tau_nd / L_nd**2)
    # rad_diff_nd = dmr.rad_diff * (tau_nd / L_nd**2)
    rad_diff_cutoff = (1/(config.numerics['prandtl']*config.numerics['reynolds_target'])) * ((L_CZ**2/tau_heat) / (L_nd**2/tau_nd))
    sim_rad_diff = np.copy(rad_diff_nd) + rad_diff_cutoff
    sim_nu_diff = config.numerics['prandtl']*rad_diff_cutoff*np.ones_like(sim_rad_diff)
    Re_shift = ((L_nd**2/tau_nd) / (L_CZ**2/tau_heat))

    logger.info('u_heat_nd: {:.3e}'.format(u_heat_nd))
    logger.info('rad_diff cutoff: {:.3e}'.format(rad_diff_cutoff))
    logger.info('rad_diff cutoff: {:.3e}'.format(rad_diff_cutoff * (L_nd**2/tau_nd)))
    
    #MESA radial values at simulation joints & across full star in simulation units
    r_bound_nd = [(rb/L_nd).value for rb in r_bounds]
    r_nd = (r/L_nd).cgs
    
  
    ### Make dedalus domain and bases
    resolutions = [(1, 1, nr) for nr in config.star['nr']]
    stitch_radii = r_bound_nd[1:-1]
    dtype=np.float64
    mesh=None
    dealias = config.numerics['N_dealias']
    c, d, bases, bases_keys = make_bases(resolutions, stitch_radii, r_bound_nd[-1], dealias=(1,1,dealias), dtype=dtype, mesh=mesh)
    dedalus_r = OrderedDict()
    for bn in bases.keys():
        phi, theta, r_vals = bases[bn].global_grids(dist=d,scales=(1, 1, dealias))
        dedalus_r[bn] = r_vals

    # Construct convective flux function which determines how convection is driven
    if config.star['smooth_h']:
        #smooth CZ-RZ transition
        L_conv_sim = np.copy(dmr.L_conv)
        L_conv_sim *= one_to_zero(r, 0.9*core_cz_radius, width=0.05*core_cz_radius)
        L_conv_sim *= one_to_zero(r, 0.95*core_cz_radius, width=0.05*core_cz_radius)
        L_conv_sim /= (r/L_nd)**2 * (4*np.pi)
        F_conv_func = interp1d(r/L_nd, L_conv_sim/lum_nd, **interp_kwargs)
    else:
        raise NotImplementedError("must use smooth_h")

    # Get some timestepping & wave frequency info
    f_nyq = 2*tau_nd*np.sqrt(N2max_sim)/(2*np.pi)
    nyq_dt   = (1/f_nyq) 
    kepler_tau     = 30*60*u.s
    max_dt_kepler  = kepler_tau/tau_nd
    max_dt = max_dt_kepler
    logger.info('needed nyq_dt is {} s / {} % of a nondimensional time (Kepler 30 min is {} %) '.format(nyq_dt*tau_nd, nyq_dt*100, max_dt_kepler*100))
 
    #Create interpolations of the various fields that may be used in the problem
    interpolations = OrderedDict()
    interpolations['ln_rho0'] = interp1d(r_nd, np.log(rho/rho_nd), **interp_kwargs)
    interpolations['ln_T0'] = interp1d(r_nd, np.log(T/T_nd), **interp_kwargs)
    interpolations['grad_ln_rho0'] = interp1d(r_nd, dmr.dlogrhodr*L_nd, **interp_kwargs)
    interpolations['grad_ln_T0'] = interp1d(r_nd, dmr.dlogTdr*L_nd, **interp_kwargs)
    interpolations['T0'] = interp1d(r_nd, T/T_nd, **interp_kwargs)
    interpolations['nu_diff'] = interp1d(r_nd, sim_nu_diff, **interp_kwargs)
    interpolations['chi_rad'] = interp1d(r_nd, sim_rad_diff, **interp_kwargs)
    interpolations['grad_chi_rad'] = interp1d(r_nd, np.gradient(rad_diff_nd, r_nd), **interp_kwargs)
    interpolations['g'] = interp1d(r_nd, -g * (tau_nd**2/L_nd), **interp_kwargs)
    interpolations['g_phi'] = interp1d(r_nd, g_phi * (tau_nd**2 / L_nd**2), **interp_kwargs)

    # construct N2 function 
    # TODO: I think some of this logic is happening inside the BVP; make sure it's all together.
    ### More core convection zone logic here
    grad_s_width = grad_s_transition_default
    grad_s_width *= (L_CZ/L_nd).value
    grad_s_transition_point = r_bound_nd[1] - grad_s_width
    logger.info('using default grad s transition point = {}'.format(grad_s_transition_point))
    logger.info('using default grad s width = {}'.format(grad_s_width))
 
    #Build a nice function for our basis in the ball
    #have N^2 = A*r^2 + B; grad_N2 = 2 * A * r, so A = (grad_N2) / (2 * r_stitch) & B = stitch_value - A*r_stitch^2
    stitch_point = 1
    stitch_point = bases['B'].radius
    stitch_value = np.interp(stitch_point, r/L_nd, N2)
    grad_N2_stitch = np.gradient(N2, r)[r/L_nd < stitch_point][-1]
    A = grad_N2_stitch / (2*bases['B'].radius * L_nd)
    B = stitch_value - A* (bases['B'].radius * L_nd)**2
    smooth_N2 = np.copy(N2)
    smooth_N2[r/L_nd < stitch_point] = A*(r[r/L_nd < stitch_point])**2 + B
    smooth_N2 *= zero_to_one(r/L_nd, grad_s_transition_point, width=grad_s_width)

    # Solve for hydrostatic equilibrium for background
    N2_func = interp1d(r_nd, tau_nd**2 * smooth_N2, **interp_kwargs)
    # grad_ln_rho_func = interpolations['grad_ln_rho0']
    ln_rho_func = interpolations['ln_rho0']
    g_phi_func = interpolations['g_phi']

    L_rad_sim = dmr.Luminosity.cgs.value - (L_conv_sim*(r/L_nd)**2 * (4*np.pi)).cgs.value
    L_rad_sim/= (r/L_nd)**2 * (4*np.pi)
    F_rad_func = interp1d(r/L_nd, L_rad_sim/lum_nd, **interp_kwargs)

    # opacity_func_in = lambda rho, T: opacity_func(rho*rho_nd.cgs.value,T*T_nd.cgs.value,gff=gff_out,z_frac=z_frac_out,x_frac=x_frac_out,ye=ye_out,dimensionless=True)[0]

    #assuming rho, T are nondimensionalized inputs, but opacity_func returns a dimensionalized opacity 
    # chi_rad_func = lambda rho,T: (16 * constants.sigma_sb.cgs.value * (T*T_nd.cgs.value)**3 / (3 * (rho*rho_nd.cgs.value)**2 * cp.cgs.value * opacity_func_in(rho,T).cgs.value)) * (tau_nd / L_nd**2).cgs.value
    def chi_rad_func(rho,T):
        Cp = nondim_cp*s_nd.cgs.value
        opacity = 3.68e22*(1-z_frac_out)*(1+x_frac_out)*(rho*rho_nd.cgs.value)*(T*T_nd.cgs.value)**(-7./2.)*gff_out + ye_out*(0.2*(1+x_frac_out))
        chinum= (16 * constants.sigma_sb.cgs.value * (T*T_nd.cgs.value)**3 ) 
        chiden= (3 * (rho*rho_nd.cgs.value)**2 * Cp * opacity)
        chi = (chinum/chiden)* (tau_nd / L_nd**2).cgs.value
        return chi #nondimensionalized
    
    # will have to call HSE_solve_CZ and HSE_solve_RZ instead
    # atmo = HSE_solve(c, d, bases, g_phi_func, grad_ln_rho_func, ln_rho_func, N2_func, F_conv_func,
    #           r_outer=r_bound_nd[-1], r_stitch=stitch_radii, \
    #           R=nondim_R_gas, gamma=nondim_gamma1, G=nondim_G, nondim_radius=1)

    logger.info('starting HSE_solve_CZ')
    stitch_radii2 = [bg_CZ_RZ_transition_default] #set transition point for the CZ + RZ solve for background quantities
    resolutions2 = [(1,1,128),(1,1,64)] #set resolution at which to solve for these background quantities
    logger.info('transitioning background solve at {}'.format(stitch_radii2[0]))
    logger.info('using resolutions B: {}, S: {}'.format(resolutions2[0][-1],resolutions2[1][-1]))
    c2, d2, bases2, bases_keys2 = make_bases(resolutions2, stitch_radii2, 
                                            r_bound_nd[-1], dealias=(1,1,dealias), dtype=dtype, mesh=mesh)

    atmo_test_CZ, quantities_CZ = HSE_solve_CZ(c2, d2, bases2, g_phi_func,ln_rho_func, F_conv_func,
              r_outer=r_bound_nd[-1], r_stitch=stitch_radii2, \
              R=nondim_R_gas, gamma=nondim_gamma1, G=nondim_G, nondim_radius=1,tolerance=1e-6, HSE_tolerance = 1e-5)

    value_to_use = bg_CZ_RZ_transition_default
    for k, basis in bases2.items():
        phi, theta, r_basis = d2.local_grids(basis)
        if r_basis[0][0][0] > bg_CZ_RZ_transition_default-0.01:
            r_transition=r_basis[0][0][np.where((np.abs(r_basis[0][0]-value_to_use)/value_to_use < 0.01))[0]][0]
    logger.info('transition point for atmo_test_RZ: {}'.format(r_transition))
    logger.info('starting HSE_solve_RZ')
    atmo_test_RZ=HSE_solve_RZ(c2, d2, bases2, quantities_CZ, r_transition, chi_rad_func, F_rad_func, N2_func,
                r_outer=r_bound_nd[-1], r_stitch=stitch_radii2, \
                R=nondim_R_gas, gamma=nondim_gamma1, G=nondim_G, nondim_radius=1,tolerance=1e-5, HSE_tolerance = 1e-4)

    # smooth grad_s from atmo_test_RZ, then recalculate grad_ln_rho and pomega from HSE and EOS simultaneously
    grad_s_smooth = np.copy(atmo_test_RZ['grad_s'](r_nd))
    grad_s_smooth*=zero_to_one(r_nd.cgs.value,1.03*stitch_radii2[0],width=0.03*stitch_radii2[0])
    grad_s_smooth*=zero_to_one(r_nd.cgs.value,1.035*stitch_radii2[0],width=0.03*stitch_radii2[0])
    grad_s_smooth_func = interp1d(r_nd, grad_s_smooth)
    logger.info('transition point for HSE_EOS_solve: {}'.format(r_transition))
    logger.info('starting HSE_EOS_solve')
    atmo_test_HSE_EOS=HSE_EOS_solve(c2, d2, bases2, grad_s_smooth_func, 
              atmo_test_RZ['g'], atmo_test_RZ['ln_rho'], atmo_test_RZ['pomega'], atmo_test_RZ['s0'](r_nd)[0], 
              r_outer=r_bound_nd[-1], r_stitch=stitch_radii2, \
              R=nondim_R_gas, gamma=nondim_gamma1, G=nondim_G, nondim_radius=1,tolerance=1e-5, HSE_tolerance = 1e-5)

    interpolations['ln_rho0'] = atmo_test_HSE_EOS['ln_rho']
    interpolations['grad_ln_rho0'] = atmo_test_HSE_EOS['grad_ln_rho']
    interpolations['Q'] = atmo_test_RZ['Q']
    interpolations['g'] = atmo_test_RZ['g']
    interpolations['g_phi'] = atmo_test_RZ['g_phi']
    interpolations['grad_s0'] = atmo_test_HSE_EOS['grad_s']
    interpolations['s0'] = atmo_test_HSE_EOS['s0']
    interpolations['pom0'] = atmo_test_HSE_EOS['pomega']
    interpolations['grad_ln_pom0'] = atmo_test_HSE_EOS['grad_ln_pomega']

    # use opacity_func to calculate rad_diff = chi
    ### Recalculate rad_diff using rho, T now smoothed

    opacity_smooth = opacity_func((np.exp(interpolations['ln_rho0'](r_nd))*rho_nd.cgs.value),((interpolations['pom0'](r_nd)/nondim_R_gas)*T_nd.cgs.value),gff_out,z_frac_out,x_frac_out,ye_out,dimensionless=False)
    rad_diff       = (16 * constants.sigma_sb.cgs * ((interpolations['pom0'](r_nd)/nondim_R_gas)*T_nd.cgs.value)**3 / (3 * (np.exp(interpolations['ln_rho0'](r_nd))*rho_nd.cgs.value)**2 * cp * opacity_smooth)).cgs
    rad_diff_nd = rad_diff * (tau_nd / L_nd**2)
    # print(rad_diff_nd, rad_diff_cutoff)
    sim_rad_diff = rad_diff_nd.cgs.value + rad_diff_cutoff
    sim_nu_diff = config.numerics['prandtl']*rad_diff_cutoff*np.ones_like(sim_rad_diff)
    # print('sim_rad_diff',sim_rad_diff)
    interpolations['kappa_rad'] = interp1d(r_nd, np.exp(interpolations['ln_rho0'](r_nd))*nondim_cp*sim_rad_diff, **interp_kwargs)
    interpolations['grad_kappa_rad'] = interp1d(r_nd, np.gradient(interpolations['kappa_rad'](r_nd), r_nd), **interp_kwargs)
    interpolations['chi_rad'] = interp1d(r_nd, sim_rad_diff, **interp_kwargs)
    interpolations['grad_chi_rad'] = interp1d(r_nd, np.gradient(rad_diff_nd, r_nd), **interp_kwargs)
    interpolations['nu_diff'] = interp1d(r_nd, sim_nu_diff, **interp_kwargs)

    plt.figure()
    
    for bn, basis in bases.items():
        global_r = basis.global_grid_radius(d,scale=basis.dealias[2])
        # print(global_r)
        HSE_pre_construct = (nondim_gamma1*interpolations['pom0'](global_r)*(interpolations['grad_ln_rho0'](global_r) + interpolations['grad_s0'](global_r) / nondim_cp) - interpolations['g'](global_r))

        plt.plot(global_r[0,0,:], np.abs(HSE_pre_construct)[0,0,:], color='black')
        
    # plt.xscale('log')
    plt.yscale('log')
    plt.savefig('star/HSE_pre_construct.png')
    ## Construct Dedalus NCCs
    for ncc in ncc_dict.keys():
        for i, bn in enumerate(bases.keys()):
            ncc_dict[ncc]['Nmax_{}'.format(bn)] = ncc_dict[ncc]['nr_max'][i]
            ncc_dict[ncc]['field_{}'.format(bn)] = None
        if ncc in interpolations.keys():
            ncc_dict[ncc]['interp_func'] = interpolations[ncc]
        else:
            ncc_dict[ncc]['interp_func'] = None

    # plt.figure()
    #Loop over bases, then loop over the NCCs that need to be built for each basis
    for bn, basis in bases.items():
        # rvals = dedalus_r[bn]
        for ncc in ncc_dict.keys():
            interp_func = ncc_dict[ncc]['interp_func']
            #If we have an interpolation function, build the NCC from the interpolator, 
            # unless we're using the Dedalus gradient of another field.
            if interp_func is not None and not ncc_dict[ncc]['from_grad']:
                Nmax = ncc_dict[ncc]['Nmax_{}'.format(bn)]
                vector = ncc_dict[ncc]['vector']
                grid_only = ncc_dict[ncc]['grid_only']
                ncc_dict[ncc]['field_{}'.format(bn)] = make_NCC(basis, c, d, interp_func, Nmax=Nmax, vector=vector, grid_only=grid_only, ncc_cutoff=config.numerics['ncc_cutoff'])
                # if ncc=='grad_s0':
                #     print('ncc',ncc)
                #     rvals = basis.global_grid_radius(d,scale=basis.dealias[2])
                #     plt.plot(rvals[0,0,:],ncc_dict[ncc]['field_{}'.format(bn)]['g'][2][0,0,:])
                #     plt.yscale('log')
                    
                if ncc_dict[ncc]['get_grad']: #If another NCC needs the gradient of this one, build it
                    name = ncc_dict[ncc]['grad_name']
                    logger.info('getting {}'.format(name))
                    grad_field = d3.grad(ncc_dict[ncc]['field_{}'.format(bn)]).evaluate()
                    grad_field.change_scales((1,1,(Nmax+1)/resolutions[bases_keys == bn][2]))
                    grad_field.change_scales(basis.dealias)
                    ncc_dict[name]['field_{}'.format(bn)] = grad_field
                    ncc_dict[name]['Nmax_{}'.format(bn)] = Nmax+1
                if ncc_dict[ncc]['get_inverse']: #If another NCC needs the inverse of this one, build it
                    name = 'inv_{}'.format(ncc)
                    inv_func = lambda r: 1/interp_func(r)
                    ncc_dict[name]['field_{}'.format(bn)] = make_NCC(basis, c, d, inv_func, Nmax=Nmax, vector=vector, grid_only=grid_only, ncc_cutoff=config.numerics['ncc_cutoff'])
                    ncc_dict[name]['Nmax_{}'.format(bn)] = Nmax
    # plt.savefig('grad_s0_test.png')
        # Special case for gravity; we build the NCC from the potential, then take the gradient, which is -g.
        # if 'neg_g' in ncc_dict.keys():
        #     if 'g' not in ncc_dict.keys():
        #         ncc_dict['g'] = OrderedDict()
        #     name = 'g'
        #     ncc_dict['g']['field_{}'.format(bn)] = (-ncc_dict['neg_g']['field_{}'.format(bn)]).evaluate()
        #     ncc_dict['g']['vector'] = True
        #     ncc_dict['g']['interp_func'] = interpolations['g']
        #     ncc_dict['g']['Nmax_{}'.format(bn)] = ncc_dict['neg_g']['Nmax_{}'.format(bn)]
        #     ncc_dict['g']['from_grad'] = True 
    
    HSEs = []
    rs = []
    for bn in bases_keys:
        rs.append(dedalus_r[bn].ravel())
        grad_ln_rho0 = ncc_dict['grad_ln_rho0']['field_{}'.format(bn)]
        grad_ln_pom0 = ncc_dict['grad_ln_pom0']['field_{}'.format(bn)]
        pom0 = ncc_dict['pom0']['field_{}'.format(bn)]
        ln_rho0 = ncc_dict['ln_rho0']['field_{}'.format(bn)]
        gvec = ncc_dict['g']['field_{}'.format(bn)]
        grad_s0 = ncc_dict['grad_s0']['field_{}'.format(bn)]
        s0 = ncc_dict['s0']['field_{}'.format(bn)]
        pom0 = ncc_dict['pom0']['field_{}'.format(bn)]
        HSE = (nondim_gamma1*pom0*(grad_ln_rho0 + grad_s0 / nondim_cp) - gvec).evaluate()
        HSEs.append(HSE['g'][2,:])
    r_dedalus_pre_filter = np.concatenate(rs, axis=-1)
    HSE_dedalus_pre_filter = np.concatenate(HSEs, axis=-1).ravel()
    plt.figure()
    plt.axhline(s_motions/nondim_cp / s_nd, c='k')
    plt.plot(r_dedalus_pre_filter, np.abs(HSE_dedalus_pre_filter),label='dedalus')
    plt.yscale('log')
    plt.xlabel('r')
    plt.ylabel("HSE")
    plt.legend()
    plt.savefig('star/HSE_goodness_pre_filter.png')

    #Force more zeros in the CZ if requested. Uses all available coefficients in the expansion of grad s0.
    if reapply_grad_s_filter:
        for bn, basis in bases.items():
            ncc_dict['grad_s0']['field_{}'.format(bn)]['g'] *= zero_to_one(dedalus_r[bn], grad_s_transition_point-5*grad_s_width, width=grad_s_width)
            ncc_dict['grad_s0']['field_{}'.format(bn)]['c'] *= 1
            ncc_dict['grad_s0']['field_{}'.format(bn)]['g']

    #reset ln_rho and ln_T interpolations for nice plots
    # interpolations['ln_rho0'] = interp1d(r_nd, np.log(rho/rho_nd), **interp_kwargs)
    # interpolations['ln_T0'] = interp1d(r_nd, np.log(T/T_nd), **interp_kwargs)

    #Fixup heating term to make simulation energy-neutral.       
    integral = 0
    for bn in bases.keys():
        integral += d3.integ(ncc_dict['Q']['field_{}'.format(bn)])
    C = integral.evaluate()['g']
    vol = (4/3) * np.pi * (r_bound_nd[-1])**3
    adj = C / vol
    logger.info('adjusting dLdt for energy conservation; subtracting {} from H'.format(adj))
    for bn in bases.keys():
        ncc_dict['Q']['field_{}'.format(bn)]['g'] -= adj 

    if plot_nccs:
        #Plot the NCCs
        for ncc in ncc_dict.keys():
            if ncc_dict[ncc]['interp_func'] is None:
                continue
            axhline = None
            log = False
            ylim = None
            rvals = []
            dedalus_yvals = []
            nvals = []
            for bn, basis in bases.items():
                rvals.append(dedalus_r[bn].ravel())
                nvals.append(ncc_dict[ncc]['Nmax_{}'.format(bn)])
                if ncc_dict[ncc]['vector']:
                    dedalus_yvals.append(np.copy(ncc_dict[ncc]['field_{}'.format(bn)]['g'][2,0,0,:]))
                else:
                    dedalus_yvals.append(np.copy(ncc_dict[ncc]['field_{}'.format(bn)]['g'][0,0,:]))
    
            interp_func = ncc_dict[ncc]['interp_func']
            if ncc in ['T', 'grad_T', 'chi_rad', 'grad_chi_rad', 'grad_s0', 'kappa_rad', 'grad_kappa_rad']:
                log = True
            if ncc == 'grad_s0': 
                axhline = (s_motions / s_nd)
            elif ncc in ['chi_rad', 'grad_chi_rad']:
                if ncc == 'chi_rad':
                    interp_func = interp1d(r_nd, (L_nd**2/tau_nd).value*rad_diff_nd, **interp_kwargs)
                    for ind in range(len(dedalus_yvals)):
                        dedalus_yvals[ind] *= (L_nd**2/tau_nd).value
                axhline = rad_diff_cutoff*(L_nd**2/tau_nd).value
    
            if ncc == 'grad_s0':
                interp_func = interp1d(r_nd, (L_nd/s_nd) * dmr.grad_s, **interp_kwargs)
            elif ncc in ['ln_T0', 'ln_rho0', 'grad_s0']:
                interp_func = interpolations[ncc]
    
            if ncc in ['grad_T', 'grad_kappa_rad']:
                interp_func = lambda r: -ncc_dict[ncc]['interp_func'](r)
                ylabel='-{}'.format(ncc)
                for i in range(len(dedalus_yvals)):
                    dedalus_yvals[i] *= -1
            elif ncc == 'chi_rad':
                ylabel = 'radiative diffusivity (cm^2/s)'
            else:
                ylabel = ncc

            ### this doesn't really plot the "Mesa Vals", just interpolated vs. on dedalus grid values
            plot_ncc_figure(rvals, interp_func, dedalus_yvals, Ns=nvals, \
                        ylabel=ylabel, fig_name=ncc, out_dir=out_dir, log=log, ylim=ylim, \
                        r_int=stitch_radii, axhline=axhline, ncc_cutoff=config.numerics['ncc_cutoff'])

    

    with h5py.File('{:s}'.format(out_file), 'w') as f:
        # Save output fields.
        # slicing preserves dimensionality
        for bn, basis in bases.items():
            f['r_{}'.format(bn)] = dedalus_r[bn]
            for ncc in ncc_dict.keys():
                this_field = ncc_dict[ncc]['field_{}'.format(bn)]
                if ncc_dict[ncc]['vector']:
                    f['{}_{}'.format(ncc, bn)] = this_field['g'][:, :1,:1,:]
                    f['{}_{}'.format(ncc, bn)].attrs['rscale_{}'.format(bn)] = ncc_dict[ncc]['Nmax_{}'.format(bn)]/resolutions[bases_keys == bn][2]
                else:
                    f['{}_{}'.format(ncc, bn)] = this_field['g'][:1,:1,:]
                    f['{}_{}'.format(ncc, bn)].attrs['rscale_{}'.format(bn)] = ncc_dict[ncc]['Nmax_{}'.format(bn)]/resolutions[bases_keys == bn][2]
    
        f['Cp'] = nondim_cp
        f['R_gas'] = nondim_R_gas
        f['gamma1'] = nondim_gamma1

        #Save properties of the star, with units.
        f['L_nd']   = L_nd
        f['L_nd'].attrs['units'] = str(L_nd.unit)
        f['rho_nd']  = rho_nd
        f['rho_nd'].attrs['units']  = str(rho_nd.unit)
        f['T_nd']  = T_nd
        f['T_nd'].attrs['units']  = str(T_nd.unit)
        f['tau_heat'] = tau_heat
        f['tau_heat'].attrs['units'] = str(tau_heat.unit)
        f['tau_nd'] = tau_nd 
        f['tau_nd'].attrs['units'] = str(tau_nd.unit)
        f['m_nd'] = m_nd 
        f['m_nd'].attrs['units'] = str(m_nd.unit)
        f['s_nd'] = s_nd
        f['s_nd'].attrs['units'] = str(s_nd.unit)
        f['P_r0']  = dmr.P[0]
        f['P_r0'].attrs['units']  = str(dmr.P[0].unit)
        f['H_nd']  = H_nd
        f['H_nd'].attrs['units']  = str(H_nd.unit)
        f['H0']  = H0
        f['H0'].attrs['units']  = str(H0.unit)
        f['N2max_sim'] = N2max_sim
        f['N2max_sim'].attrs['units'] = str(N2max_sim.unit)
        f['N2plateau'] = N2plateau
        f['N2plateau'].attrs['units'] = str(N2plateau.unit)
        f['cp_surf'] = cp[sim_bool][-1]
        f['cp_surf'].attrs['units'] = str(cp[sim_bool][-1].unit)
        f['r_mesa'] = r
        f['r_mesa'].attrs['units'] = str(r.unit)
        f['N2_mesa'] = N2
        f['N2_mesa'].attrs['units'] = str(N2.unit)
        f['S1_mesa'] = dmr.lamb_freq(1)
        f['S1_mesa'].attrs['units'] = str(dmr.lamb_freq(1).unit)
        f['g_mesa'] = g 
        f['g_mesa'].attrs['units'] = str(g.unit)
        f['cp_mesa'] = cp
        f['cp_mesa'].attrs['units'] = str(cp.unit)

        #TODO: put sim lum back
        f['lum_r_vals'] = lum_r_vals = np.linspace(r_bound_nd[0], r_bound_nd[-1], 1000)
        f['sim_lum'] = (4*np.pi*lum_r_vals**2)*F_conv_func(lum_r_vals)
        f['r_stitch']   = stitch_radii
        f['Re_shift'] = Re_shift
        f['r_outer']   = r_bound_nd[-1] 
        f['max_dt'] = max_dt
        f['Ma2_r0'] = Ma2_r0
        for k in ['r_stitch', 'r_outer', 'max_dt', 'Ma2_r0', 'Re_shift', 'lum_r_vals', 'sim_lum',\
                    'Cp', 'R_gas', 'gamma1']:
            f[k].attrs['units'] = 'dimensionless'
    logger.info('finished saving NCCs to {}'.format(out_file))
    logger.info('We recommend looking at the plots in {}/ to make sure the non-constant coefficients look reasonable'.format(out_dir))

    #Make some plots of stratification, hydrostatic equilibrium, etc.
    logger.info('Making final plots...')
    plt.figure()
    N2s = []
    HSEs = []
    EOSs = []
    grad_s0s = []
    grad_ln_rho0s = []
    grad_ln_pom0s = []
    rs = []
    for bn in bases_keys:
        rs.append(dedalus_r[bn].ravel())
        grad_ln_rho0 = ncc_dict['grad_ln_rho0']['field_{}'.format(bn)]
        grad_ln_pom0 = ncc_dict['grad_ln_pom0']['field_{}'.format(bn)]
        pom0 = ncc_dict['pom0']['field_{}'.format(bn)]
        ln_rho0 = ncc_dict['ln_rho0']['field_{}'.format(bn)]
        gvec = ncc_dict['g']['field_{}'.format(bn)]
        grad_s0 = ncc_dict['grad_s0']['field_{}'.format(bn)]
        s0 = ncc_dict['s0']['field_{}'.format(bn)]
        pom0 = ncc_dict['pom0']['field_{}'.format(bn)]
        HSE = (nondim_gamma1*pom0*(grad_ln_rho0 + grad_s0 / nondim_cp) - gvec).evaluate()
        EOS = s0/nondim_cp - ( (1/nondim_gamma1) * (np.log(pom0) - np.log(nondim_R_gas)) - ((nondim_gamma1-1)/nondim_gamma1) * ln_rho0 )
        N2_val = -gvec['g'][2,:] * grad_s0['g'][2,:] / nondim_cp 
        N2s.append(N2_val)
        HSEs.append(HSE['g'][2,:])
        EOSs.append(EOS.evaluate()['g'])
        grad_ln_rho0s.append(grad_ln_rho0['g'][2,:])
        grad_ln_pom0s.append(grad_ln_pom0['g'][2,:])
    r_dedalus = np.concatenate(rs, axis=-1)
    N2_dedalus = np.concatenate(N2s, axis=-1).ravel()
    HSE_dedalus = np.concatenate(HSEs, axis=-1).ravel()
    EOS_dedalus = np.concatenate(EOSs, axis=-1).ravel()
    grad_ln_rho0_dedalus = np.concatenate(grad_ln_rho0s, axis=-1).ravel()
    grad_ln_pom0_dedalus = np.concatenate(grad_ln_pom0s, axis=-1).ravel()
    plt.plot(r_nd, tau_nd**2*N2, label='mesa', c='k')
    plt.plot(r_nd, -tau_nd**2*N2, c='k', ls='--')
#    plt.plot(r_nd, atmo['N2'](r_nd), label='atmosphere', c='b')
#    plt.plot(r_nd, -atmo['N2'](r_nd), c='b', ls='--')
    plt.plot(r_dedalus, N2_dedalus, label='dedalus', c='g')
    plt.plot(r_dedalus, -N2_dedalus, ls='--', c='g')
    plt.legend()
    plt.ylabel(r'$N^2$')
    plt.xlabel('r')
    plt.yscale('log')
    plt.ylim(1e-17,)
    plt.savefig('star/N2_goodness.png')
#    plt.show()

    plt.figure()
    plt.axhline(s_motions/nondim_cp / s_nd, c='k')
    for bn, basis in bases.items():
        global_r = basis.global_grid_radius(d,scale=basis.dealias[2])
        # print(global_r)
        HSE_pre_construct = (nondim_gamma1*interpolations['pom0'](global_r)*(interpolations['grad_ln_rho0'](global_r) + interpolations['grad_s0'](global_r) / nondim_cp) - interpolations['g'](global_r))
        if bn == 'B':
            label='Pre build_nccs'
        else:
            label=None
        plt.plot(global_r[0,0,:], np.abs(HSE_pre_construct)[0,0,:], color='black',label=label)
    plt.plot(r_dedalus, np.abs(HSE_dedalus),label='dedalus')
    plt.yscale('log')
    plt.xlabel('r')
    plt.ylabel("HSE")
    plt.legend()
    plt.savefig('star/HSE_goodness.png')

    plt.figure()
    plt.axhline(s_motions/nondim_cp / s_nd, c='k')
    plt.plot(r_dedalus, np.abs(EOS_dedalus),label='dedalus')
    plt.yscale('log')
    plt.xlabel('r')
    plt.ylabel("EOS")
    plt.legend()
    plt.savefig('star/EOS_goodness.png')


    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    plt.plot(r_dedalus, grad_ln_rho0_dedalus)
    plt.xlabel('r')
    plt.ylabel("grad_ln_rho0")
    ax2 = fig.add_subplot(2,1,2)
    plt.plot(r_dedalus, grad_ln_pom0_dedalus)
    plt.xlabel('r')
    plt.ylabel("grad_ln_pom0")
    plt.savefig('star/ln_thermo_goodness.png')
#    plt.show()
    plt.close('all')
    #### save and plot fluctuations

    # define fluctuations as RZ (full answer) - HSE_EOS (smoothed background)
    delta_ln_rho = atmo_test_RZ['ln_rho'](r_nd)-atmo_test_HSE_EOS['ln_rho'](r_nd)
    delta_s = atmo_test_RZ['s0'](r_nd)-atmo_test_HSE_EOS['s0'](r_nd)
    interpolations['s1'] = interp1d(r_nd, delta_s, **interp_kwargs)
    interpolations['ln_rho1'] = interp1d(r_nd, delta_ln_rho, **interp_kwargs)

    ## Construct Dedalus NCCs
    for fluct in fluct_dict.keys():        
        for i, bn in enumerate(bases.keys()):
            fluct_dict[fluct]['Nmax_{}'.format(bn)] = fluct_dict[fluct]['nr_max'][i]
            fluct_dict[fluct]['field_{}'.format(bn)] = None
        if fluct in interpolations.keys():
            fluct_dict[fluct]['interp_func'] = interpolations[fluct]
        else:
            fluct_dict[fluct]['interp_func'] = None

    #Loop over bases, then loop over the flucts that need to be built for each basis
    for bn, basis in bases.items():
        rvals = dedalus_r[bn]
        for fluct in fluct_dict.keys():
            interp_func = fluct_dict[fluct]['interp_func']
            #If we have an interpolation function, build the fluct from the interpolator, 
            # unless we're using the Dedalus gradient of another field.
            if interp_func is not None and not fluct_dict[fluct]['from_grad']:
                Nmax = fluct_dict[fluct]['Nmax_{}'.format(bn)]
                vector = fluct_dict[fluct]['vector']
                grid_only = fluct_dict[fluct]['grid_only']
                fluct_dict[fluct]['field_{}'.format(bn)] = make_NCC(basis, c, d, interp_func, Nmax=Nmax, vector=vector, grid_only=grid_only, ncc_cutoff=config.numerics['ncc_cutoff'])
    if plot_nccs:
        #Plot the initial fluctuations too
        for fluct in fluct_dict.keys():
            if fluct_dict[fluct]['interp_func'] is None:
                continue
            axhline = None
            log = False
            ylim = None
            rvals = []
            dedalus_yvals = []
            nvals = []
            for bn, basis in bases.items():
                rvals.append(dedalus_r[bn].ravel())
                nvals.append(fluct_dict[fluct]['Nmax_{}'.format(bn)])
                if fluct_dict[fluct]['vector']:
                    dedalus_yvals.append(np.copy(fluct_dict[fluct]['field_{}'.format(bn)]['g'][2,0,0,:]))
                else:
                    dedalus_yvals.append(np.copy(fluct_dict[fluct]['field_{}'.format(bn)]['g'][0,0,:]))
    
            interp_func = fluct_dict[fluct]['interp_func']
            
            if fluct in ['ln_rho1', 's1']:
                interp_func = interpolations[fluct]
                ylabel = fluct
            plot_ncc_figure(rvals, interp_func, dedalus_yvals, Ns=nvals, \
                        ylabel=ylabel, fig_name=fluct, out_dir=out_dir, log=log, ylim=ylim, \
                        r_int=stitch_radii, axhline=axhline, ncc_cutoff=config.numerics['ncc_cutoff'])

    fluct_out_file = out_file.replace('star_','star_fluct_')
    with h5py.File('{:s}'.format(fluct_out_file), 'w') as f:
        # Save output fields.
        # slicing preserves dimensionality
        for bn, basis in bases.items():
            f['r_{}'.format(bn)] = dedalus_r[bn]
            for fluct in fluct_dict.keys():
                this_field = fluct_dict[fluct]['field_{}'.format(bn)]
                if fluct_dict[fluct]['vector']:
                    f['{}_{}'.format(fluct, bn)] = this_field['g'][:, :1,:1,:]
                    f['{}_{}'.format(fluct, bn)].attrs['rscale_{}'.format(bn)] = fluct_dict[fluct]['Nmax_{}'.format(bn)]/resolutions[bases_keys == bn][2]
                else:
                    f['{}_{}'.format(fluct, bn)] = this_field['g'][:1,:1,:]
                    f['{}_{}'.format(fluct, bn)].attrs['rscale_{}'.format(bn)] = fluct_dict[fluct]['Nmax_{}'.format(bn)]/resolutions[bases_keys == bn][2]
                    ###
    logger.info('finished saving initial fluctuations to {}'.format(fluct_out_file))
    logger.info('We recommend looking at the plots in {}/ to make sure the non-constant coefficients look reasonable'.format(out_dir))