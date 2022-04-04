"""
Turns a MESA .data file of a massive star into a .h5 file of NCCs for a d3 run.
There is a ball basis, and two shell bases.

Usage:
    build_star.py [options]

Options:
    --no_plot         If flagged, don't output plots
"""
import os, sys
from collections import OrderedDict
from pathlib import Path
from configparser import ConfigParser

import numpy as np
import h5py
from mpi4py import MPI
import mesa_reader as mr
import matplotlib.pyplot as plt
import dedalus.public as d3
from dedalus.core import field
from docopt import docopt

from astropy import units as u
from astropy import constants
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
import scipy.integrate as si
from numpy.polynomial import Chebyshev as Pfit

#Import parent directory and anelastic_functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import d3_stars
from d3_stars.simulations.anelastic_functions import make_bases
from d3_stars.simulations.star_builder import one_to_zero, zero_to_one, plot_ncc_figure, make_NCC

args = docopt(__doc__)
PLOT = not(args['--no_plot'])

if __name__ == '__main__':
    # Read in parameters and create output directory
    config = OrderedDict()
    raw_config = OrderedDict()
    config_file = 'controls.cfg'
    config_file = Path(config_file)
    config_p = ConfigParser()
    config_p.read(str(config_file))
    print(config_p)
    for n, v in config_p.items('star'):
        if n.lower() == 'nr':
            config[n] = [int(n) for n in v.split(',')]
        elif n.lower() == 'r_bounds':
            config[n] = [n.replace(' ', '') for n in v.split(',')]
        elif v.lower() == 'true':
            config[n] = True
        elif v.lower() == 'false':
            config[n] = False
        else:
            config[n] = v
        raw_config[n] = v

    for n, v in config_p.items('numerics'):
        config[n] = v
        raw_config[n] = v

    for k in ['reynolds_target', 'prandtl', 'ncc_cutoff', 'n_dealias', 'l_dealias']:
        config[k] = float(config[k])

    if float(config['r_bounds'][0].lower()) != 0:
        raise ValueError("The inner basis must currently be a BallBasis; set the first value of r_bounds to zero.")


    out_dir  = 'star' 
    out_file = '{:s}/star_'.format(out_dir)
    out_file += (len(config['nr'])*"{}+").format(*tuple(config['nr']))[:-1]
    out_file += '_bounds{}-{}'.format(config['r_bounds'][0], config['r_bounds'][-1])
    out_file += '_Re{}_de{}_cutoff{}.h5'.format(raw_config['reynolds_target'], raw_config['n_dealias'], raw_config['ncc_cutoff'])
    print('saving output to {}'.format(out_file))
    if not os.path.exists('{:s}'.format(out_dir)):
        os.mkdir('{:s}'.format(out_dir))


    package_path = Path(d3_stars.__file__).resolve().parent
    stock_path = package_path.joinpath('stock_models')
    sys.path.append(stock_path)

    #TODO: figure out how to make MESA the file path w.r.t. stock model path w/o supplying full path here 
    p = mr.MesaData(str(stock_path.joinpath(config['path'])))
    mass           = (p.mass[::-1] * u.M_sun).cgs
    r              = (p.radius[::-1] * u.R_sun).cgs
    rho            = 10**p.logRho[::-1] * u.g / u.cm**3
    P              = p.pressure[::-1] * u.g / u.cm / u.s**2
    T              = p.temperature[::-1] * u.K
    nablaT         = p.gradT[::-1] #dlnT/dlnP
    nablaT_ad      = p.grada[::-1]
    chiRho         = p.chiRho[::-1]
    chiT           = p.chiT[::-1]
    cp             = p.cp[::-1]  * u.erg / u.K / u.g
    opacity        = p.opacity[::-1] * (u.cm**2 / u.g)
    Luminosity     = (p.luminosity[::-1] * u.L_sun).cgs
    conv_L_div_L   = p.lum_conv_div_L[::-1]
    csound         = p.csound[::-1] * u.cm / u.s
    N2             = p.brunt_N2[::-1] / u.s**2
    N2_structure   = p.brunt_N2_structure_term[::-1] / u.s**2
    N2_composition = p.brunt_N2_composition_term[::-1] / u.s**2
    eps_nuc        = p.eps_nuc[::-1] * u.erg / u.g / u.s
    lamb_freq = lambda ell : np.sqrt(ell*(ell + 1)) * csound/r
    
    R_star = (p.photosphere_r * u.R_sun).cgs
    
    #Put all MESA fields into cgs and calculate secondary MESA fields
    g               = constants.G.cgs*mass/r**2
    dlogPdr         = -rho*g/P
    gamma1          = dlogPdr/(-g/csound**2)
    dlogrhodr       = dlogPdr*(chiT/chiRho)*(nablaT_ad - nablaT) - g/csound**2
    dlogTdr         = dlogPdr*(nablaT)
    grad_s          = cp*N2/g #entropy gradient, for NCC, includes composition terms
    L_conv          = conv_L_div_L*Luminosity
    dTdr            = (T)*dlogTdr
    
    #True calculation of rad_diff, rad_cond
    #rad_diff        = (16 * constants.sigma_sb.cgs * T**3 / (3 * rho**2 * cp * opacity)).cgs
    #rad_cond        = rho*cp*rad_diff
    
    #Calculate k_rad using luminosities and smooth things.
    k_rad = rad_cond = -(Luminosity - L_conv)/(4*np.pi*r**2*dTdr)
    rad_diff        = k_rad / (rho * cp)
    
    
    ### Split up the domain
    # Find edge of core cz
    cz_bool = (L_conv.value > 1)*(mass < 0.9*mass[-1]) #rudimentary but works
    core_index  = np.argmin(np.abs(mass - mass[cz_bool][-1]))
    core_cz_radius = r[core_index]
    
    # Specify fraction of total star to simulate
    r_bounds = config['r_bounds']
    r_bools = []
    for i, rb in enumerate(r_bounds):
        if 'R' in rb:
            r_bounds[i] = float(rb.replace('R', ''))*R_star
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
    print('fraction of FULL star simulated: {}, up to r={:.3e}'.format(r_bounds[-1]/R_star, r_bounds[-1]))

    #Set things up to slice out the star appropriately
    sim_bool      = (r > r_bounds[0])*(r <= r_bounds[-1])
    
    # Calculate heating function
    # Goal: H_eff= np.gradient(L_conv,r, edge_order=1)/(4*np.pi*r**2) # Heating, for ncc, H = rho*eps - portion carried by radiation
    # (1/4pir^2) dL_conv/dr = rho * eps + (1/r^2)d/dr (r^2 k_rad dT/dr) -> chain rule
    eo=2
    H_eff = (1/(4*np.pi*r**2))*np.gradient(Luminosity, r, edge_order=eo) + 2*k_rad*dTdr/r + dTdr*np.gradient(k_rad, r, edge_order=eo) + k_rad*np.gradient(dTdr, r, edge_order=eo)
    H_eff_secondary = rho*eps_nuc + 2*k_rad*dTdr/r + dTdr*np.gradient(k_rad, r, edge_order=eo) + k_rad*np.gradient(dTdr, r, edge_order=eo)
    H_eff[:2] = H_eff_secondary[:2]
    
    sim_H_eff = np.copy(H_eff)
    L_conv_sim = np.zeros_like(L_conv)
    L_eps = np.zeros_like(Luminosity)
    for i in range(L_conv_sim.shape[0]):
        L_conv_sim[i] = np.trapz((4*np.pi*r**2*sim_H_eff)[:1+i], r[:1+i])
        L_eps[i] = np.trapz((4*np.pi*r**2*rho*eps_nuc)[:i+1], r[:i+1])
    L_excess = L_conv_sim[-5] - Luminosity[-5]
    
    #construct internal heating field
    if config['smooth_h']:
        #smooth CZ-RZ transition
        L_conv_sim *= one_to_zero(r, 0.95*core_cz_radius, width=0.15*core_cz_radius)
        L_conv_sim *= one_to_zero(r, 0.95*core_cz_radius, width=0.05*core_cz_radius)
    
        transition_region = (r > 0.5*core_cz_radius)
        sim_H_eff[transition_region] = ((1/(4*np.pi*r**2))*np.gradient(L_conv_sim, r, edge_order=eo))[transition_region]
    
    #    plt.figure()
    #    plt.axhline(0, c='k')
    #    plt.plot(r, L_conv)
    #    plt.plot(r, L_conv_sim, c='k', ls='--')
    #    plt.figure()
    #    plt.plot(r, H_eff)
    #    plt.plot(r, sim_H_eff, ls='--', c='k')
    #    plt.show()
    else:
        sim_H_eff = H_eff
    
    #Get N2 info
    N2max_sim = N2[sim_bool].max()
    shell_points = np.sum(sim_bool*(r > core_cz_radius))
    N2plateau = np.median(N2[r > core_cz_radius][int(shell_points*0.25):int(shell_points*0.75)])
    f_brunt = np.sqrt(N2max_sim)/(2*np.pi)
    
    
    #Nondimensionalization
    L_CZ    = core_cz_radius
    m_core  = rho[0] * L_CZ**3
    T_core  = T[0]
    H0      = (rho*eps_nuc)[0]
    tau_heat  = ((H0*L_CZ/m_core)**(-1/3)).cgs #heating timescale
    L_nd    = L_CZ
    #m_nd    = m_core
    #T_nd    = T_core
    #tau_nd  = tau_heat
    #L_nd    = r_S2_MESA - r_ball_MESA
    m_nd    = rho[r==L_nd][0] * L_nd**3 #mass at core cz boundary
    T_nd    = T[r==L_nd][0] #temp at core cz boundary
    tau_nd  = (1/f_brunt).cgs #timescale of max N^2
    rho_nd  = m_nd/L_nd**3
    u_nd    = L_nd/tau_nd
    s_nd    = L_nd**2 / tau_nd**2 / T_nd
    #H_nd    = H0
    H_nd    = (m_nd / L_nd) * tau_nd**-3
    s_motions    = L_nd**2 / tau_heat**2 / T[0]
    rad_diff_nd = inv_Pe_rad = rad_diff * (tau_nd / L_nd**2)
    rad_diff_cutoff = (1/(config['prandtl']*config['reynolds_target'])) * ((L_CZ**2/tau_heat) / (L_nd**2/tau_nd))
    Re_shift = ((L_nd**2/tau_nd) / (L_CZ**2/tau_heat))
    print('Nondimensionalization: L_nd = {:.2e}, T_nd = {:.2e}, m_nd = {:.2e}, tau_nd = {:.2e}'.format(L_nd, T_nd, m_nd, tau_nd))
    print('m_nd/M_\odot: {:.3f}'.format((m_nd/constants.M_sun).cgs))
    
    #Central values
    rho_r0    = rho[0]
    P_r0      = P[0]
    T_r0      = T[0]
    cp_r0     = cp[0]
    gamma1_r0  = gamma1[0]
    Ma2_r0 = ((u_nd*(tau_nd/tau_heat))**2 / ((gamma1_r0-1)*cp_r0*T_r0)).cgs
    print('estimated mach number: {:.3e}'.format(np.sqrt(Ma2_r0)))
    
    cp_surf = cp[sim_bool][-1]
    
    #MESA radial values at simulation joints & across full star in simulation units
    r_bound_nd = [(rb/L_nd).value for rb in r_bounds]
    r_nd = (r/L_nd).cgs
    
    ### entropy gradient
    grad_s_transition_point = 1.05
    grad_s_width = 0.05
    grad_s_center =  grad_s_transition_point - 0.5*grad_s_width
    grad_s_width *= (L_CZ/L_nd).value
    grad_s_center *= (L_CZ/L_nd).value
    
    #Build a nice function for our basis in the ball
    grad_s_smooth = np.copy(grad_s)
    flat_value  = np.interp(grad_s_transition_point, r/L_nd, grad_s)
    grad_s_smooth[r/L_nd < grad_s_transition_point] = flat_value
    
    # Get some timestepping & wave frequency info
    f_nyq = 2*tau_nd*np.sqrt(N2max_sim)/(2*np.pi)
    nyq_dt   = (1/f_nyq) 
    kepler_tau     = 30*60*u.s
    max_dt_kepler  = kepler_tau/tau_nd
    max_dt = max_dt_kepler
    print('needed nyq_dt is {} s / {} % of a nondimensional time (Kepler 30 min is {} %) '.format(nyq_dt*tau_nd, nyq_dt*100, max_dt_kepler*100))
    
    ### Make dedalus domain and bases
    resolutions = [(8, 4, nr) for nr in config['nr']]
    stitch_radii = r_bound_nd[1:-1]
    print(resolutions, stitch_radii)
    dtype=np.float64
    mesh=None
    dealias = config['n_dealias']
    c, d, bases, bases_keys = make_bases(resolutions, stitch_radii, r_bound_nd[-1], dealias=(1,1,dealias), dtype=dtype, mesh=mesh)
    dedalus_r = OrderedDict()
    for bn in bases.keys():
        phi, theta, r_vals = bases[bn].global_grids((1, 1, dealias))
        dedalus_r[bn] = r_vals
    
    #construct rad_diff_profile
    #sim_rad_diff = np.copy(rad_diff_nd)
    #diff_transition = r_nd[sim_rad_diff > 1/config['reynolds_target']][0]
    #sim_rad_diff[:] = (1/config['reynolds_target'])*one_to_zero(r_nd, diff_transition*1.05, width=0.02*diff_transition)\
    #                + rad_diff_nd*zero_to_one(r_nd, diff_transition*0.95, width=0.1*diff_transition)
    sim_rad_diff = np.copy(rad_diff_nd) + rad_diff_cutoff #1/config['reynolds_target']
    
    ncc_dict = OrderedDict()
    for ncc in ['ln_rho', 'grad_ln_rho', 'ln_T', 'grad_ln_T', 'T', 'grad_T', 'H', 'grad_s0', 'chi_rad', 'grad_chi_rad', 'nu_diff']:
        ncc_dict[ncc] = OrderedDict()
        for bn in bases.keys():
            ncc_dict[ncc]['Nmax_{}'.format(bn)] = 32
            ncc_dict[ncc]['field_{}'.format(bn)] = None
        ncc_dict[ncc]['Nmax_S2'.format(bn)] = 10
        ncc_dict[ncc]['vector'] = False
        ncc_dict[ncc]['grid_only'] = False 
    
    ncc_dict['ln_rho']['interp_func'] = interp1d(r_nd, np.log(rho/rho_nd))
    ncc_dict['ln_T']['interp_func'] = interp1d(r_nd, np.log(T/T_nd))
    ncc_dict['grad_ln_rho']['interp_func'] = interp1d(r_nd, dlogTdr*L_nd)
    ncc_dict['grad_ln_T']['interp_func'] = interp1d(r_nd, dlogTdr*L_nd)
    ncc_dict['T']['interp_func'] = interp1d(r_nd, T/T_nd)
    ncc_dict['grad_T']['interp_func'] = interp1d(r_nd, (L_nd/T_nd)*dTdr)
    ncc_dict['H']['interp_func'] = interp1d(r_nd, ( sim_H_eff/(rho*T) ) * (rho_nd*T_nd/H_nd))
    ncc_dict['grad_s0']['interp_func'] = interp1d(r_nd, (L_nd/s_nd) * grad_s_smooth)
    
    ncc_dict['nu_diff']['interp_func'] = interp1d(r_nd, config['prandtl']*rad_diff_cutoff*np.ones_like(r_nd))
    ncc_dict['chi_rad']['interp_func'] = interp1d(r_nd, sim_rad_diff)
    ncc_dict['grad_chi_rad']['interp_func'] = interp1d(r_nd, np.gradient(rad_diff_nd, r_nd))
    
    ncc_dict['grad_ln_rho']['vector'] = True
    ncc_dict['grad_ln_T']['vector'] = True
    ncc_dict['grad_T']['vector'] = True
    ncc_dict['grad_s0']['vector'] = True
    ncc_dict['grad_chi_rad']['vector'] = True
    
    ncc_dict['grad_s0']['Nmax_B'] = 10
    ncc_dict['ln_T']['Nmax_B'] = 16
    ncc_dict['grad_ln_T']['Nmax_B'] = 17
    ncc_dict['H']['Nmax_B'] = 60
    ncc_dict['H']['Nmax_S1'] = 2
    ncc_dict['H']['Nmax_S2'] = 2
    
    ncc_dict['chi_rad']['Nmax_B'] = 1
    ncc_dict['chi_rad']['Nmax_S1'] = 20
    ncc_dict['chi_rad']['Nmax_S2'] = 10
    
    ncc_dict['nu_diff']['Nmax_B'] = 1
    ncc_dict['nu_diff']['Nmax_S1'] = 1
    ncc_dict['nu_diff']['Nmax_S2'] = 1
    
    ncc_dict['H']['grid_only'] = True
    
    
    for bn, basis in bases.items():
        rvals = dedalus_r[bn]
        for ncc in ncc_dict.keys():
            interp_func = ncc_dict[ncc]['interp_func']
            Nmax = ncc_dict[ncc]['Nmax_{}'.format(bn)]
            vector = ncc_dict[ncc]['vector']
            grid_only = ncc_dict[ncc]['grid_only']
            ncc_dict[ncc]['field_{}'.format(bn)] = make_NCC(basis, c, d, interp_func, Nmax=Nmax, vector=vector, grid_only=grid_only, ncc_cutoff=config['ncc_cutoff'])
    #        if ncc == 'grad_T':
    #            print(ncc_dict[ncc]['field_{}'.format(bn)]['g'][2,0,0,:])
    
        #Evaluate for grad chi rad
        ncc_dict['grad_chi_rad']['field_{}'.format(bn)]['g'] = d3.grad(ncc_dict['chi_rad']['field_{}'.format(bn)]).evaluate()['g']
    
    #Further post-process work to make grad_S nice in the ball
    NmaxB_after = 60 - 1
    ncc_dict['grad_s0']['field_B']['g'][2] *= zero_to_one(dedalus_r['B'], grad_s_center, width=grad_s_width)
    ncc_dict['grad_s0']['field_B']['c'][:,:,:,NmaxB_after:] = 0
    ncc_dict['grad_s0']['field_B']['c'][np.abs(ncc_dict['grad_s0']['field_B']['c']) < config['ncc_cutoff']] = 0
    
    #Post-processing for grad chi rad - doesn't work great...
    diff_transition = r_nd[sim_rad_diff > rad_diff_cutoff][0].value
    gradPe_cutoff = dict()
    gradPe_cutoff['S1'] = 32
    gradPe_cutoff['S2'] = 15
    for bn in bases.keys():
        if bn == 'B': continue
        ncc_dict['grad_chi_rad']['field_{}'.format(bn)]['g'][2,] *= zero_to_one(dedalus_r[bn], diff_transition, width=(r_bound_nd[-1]-r_bound_nd[-2])/10)
        ncc_dict['grad_chi_rad']['field_{}'.format(bn)]['c'][:,:,:,gradPe_cutoff[bn]:] = 0
        ncc_dict['grad_chi_rad']['field_{}'.format(bn)]['c'][np.abs(ncc_dict['grad_chi_rad']['field_{}'.format(bn)]['c']) < config['ncc_cutoff']] = 0
    
    if PLOT:
        for ncc in ncc_dict.keys():
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
    
            if ncc in ['T', 'grad_T', 'chi_rad', 'grad_chi_rad', 'grad_s0']:
                print('log scale for ncc {}'.format(ncc))
                log = True
            if ncc == 'grad_s0': 
                axhline = s_motions / s_nd
            elif ncc in ['chi_rad', 'grad_chi_rad']:
                axhline = rad_diff_cutoff
    
            interp_func = ncc_dict[ncc]['interp_func']
            if ncc == 'H':
                interp_func = interp1d(r_nd, ( one_to_zero(r_nd, 1.5*r_bound_nd[1], width=0.05*r_bound_nd[1])*H_eff/(rho*T) ) * (rho_nd*T_nd/H_nd) )
            elif ncc == 'grad_s0':
                interp_func = interp1d(r_nd, (L_nd/s_nd) * grad_s)
    
            if ncc == 'grad_T':
                interp_func = lambda r: -ncc_dict[ncc]['interp_func'](r)
                ylabel='-{}'.format(ncc)
                for i in range(len(dedalus_yvals)):
                    dedalus_yvals[i] *= -1
            else:
                ylabel = ncc
    
            plot_ncc_figure(rvals, interp_func, dedalus_yvals, nvals, \
                        ylabel=ylabel, fig_name=ncc, out_dir=out_dir, log=log, ylim=ylim, \
                        r_int=stitch_radii, axhline=axhline, ncc_cutoff=config['ncc_cutoff'])
    
    
    with h5py.File('{:s}'.format(out_file), 'w') as f:
        # Save output fields.
        # slicing preserves dimensionality
        for bn, basis in bases.items():
            f['r_{}'.format(bn)] = dedalus_r[bn]
            for ncc in ncc_dict.keys():
                this_field = ncc_dict[ncc]['field_{}'.format(bn)]
                if ncc_dict[ncc]['vector']:
                    f['{}_{}'.format(ncc, bn)] = this_field['g'][:, :1,:1,:]
                else:
                    f['{}_{}'.format(ncc, bn)] = this_field['g'][:1,:1,:]
    
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
        f['P_r0']  = P_r0
        f['P_r0'].attrs['units']  = str(P_r0.unit)
        f['H_nd']  = H_nd
        f['H_nd'].attrs['units']  = str(H_nd.unit)
        f['H0']  = H0
        f['H0'].attrs['units']  = str(H0.unit)
        f['N2max_sim'] = N2max_sim
        f['N2max_sim'].attrs['units'] = str(N2max_sim.unit)
        f['N2plateau'] = N2plateau
        f['N2plateau'].attrs['units'] = str(N2plateau.unit)
        f['cp_surf'] = cp_surf
        f['cp_surf'].attrs['units'] = str(cp_surf.unit)
        f['r_mesa'] = r
        f['r_mesa'].attrs['units'] = str(r.unit)
        f['N2_mesa'] = N2
        f['N2_mesa'].attrs['units'] = str(N2.unit)
        f['S1_mesa'] = lamb_freq(1)
        f['S1_mesa'].attrs['units'] = str(lamb_freq(1).unit)
        f['g_mesa'] = g 
        f['g_mesa'].attrs['units'] = str(g.unit)
        f['cp_mesa'] = cp
        f['cp_mesa'].attrs['units'] = str(cp.unit)
    
        f['r_stitch']   = stitch_radii
        f['Re_shift'] = Re_shift
        f['r_outer']   = r_bound_nd[-1] 
        f['max_dt'] = max_dt
        f['Ma2_r0'] = Ma2_r0
        for k in ['r_stitch', 'r_outer', 'max_dt', 'Ma2_r0', 'Re_shift']:
            f[k].attrs['units'] = 'dimensionless'
    print('finished saving NCCs to {}'.format(out_file))
