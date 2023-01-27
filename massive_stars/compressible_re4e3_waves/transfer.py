"""
Calculate transfer function to get surface response of convective forcing.
Outputs a function which, when multiplied by sqrt(wave flux), gives you the surface response.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy import interpolate
from pathlib import Path
from scipy.interpolate import interp1d
from configparser import ConfigParser

import d3_stars
from d3_stars.defaults import config
from d3_stars.simulations.parser import name_star
from d3_stars.simulations.star_builder import find_core_cz_radius
from d3_stars.simulations.evp_functions import calculate_refined_transfer

#Grab relevant information about the simulation stratification.
out_dir, out_file = name_star()
with h5py.File(out_file, 'r') as f:
    L_nd = f['L_nd'][()]
    rs = []
    rhos = []
    chi_rads = []
    N2s = []
    for bk in ['B', 'S1', 'S2']:
        rs.append(f['r_{}'.format(bk)][()])
        rhos.append(np.exp(f['ln_rho0_{}'.format(bk)][()]))
        chi_rads.append(f['chi_rad_{}'.format(bk)][()])
        #N^2 = -g[2] * grad_s[2] / cp
        N2s.append(-f['g_{}'.format(bk)][2,:]*f['grad_s0_{}'.format(bk)][2,:]/f['Cp'][()])
    rs = np.concatenate(rs, axis=-1)
    rhos = np.concatenate(rhos, axis=-1)
    chi_rads = np.concatenate(chi_rads, axis=-1)
    N2s = np.concatenate(N2s, axis=-1)
rho = interpolate.interp1d(rs.flatten(), rhos.flatten())
chi_rad = interpolate.interp1d(rs.flatten(), chi_rads.flatten())
N2 = interpolate.interp1d(rs.flatten(), N2s.flatten())


def eff(gam, om, dt):
    # combined, for convenience
    B = gam + 1j*om
#    B = gam + 1j*om

    # adding term by term - see overleaf for disucussion of which ones
    # are important in the regime of omega >> gamma

    # O(1)
    numerical = B
#    numerical = 2*B
    # O(dt**2)
    numerical += (1/3)*(B**3)*(dt**2)
    # O(dt**3)
    numerical += (1/4)*(B**4)*(dt**3)
    # and so on
    numerical += (7/60)*(B**5)*(dt**4)
    numerical += (7/72)*(B**6)*(dt**5)
    numerical += (241/2520)*(B**7)*(dt**6)
    numerical += (211/2880)*(B**8)*(dt**7)
    return numerical.real, numerical.imag

timestep = 0.09




# Generalized logic for getting forcing radius.
package_path = Path(d3_stars.__file__).resolve().parent
stock_path = package_path.joinpath('stock_models')
if os.path.exists(config.star['path']):
    mesa_file_path = config.star['path']
else:
    stock_file_path = stock_path.joinpath(config.star['path'])
    if os.path.exists(stock_file_path):
        mesa_file_path = str(stock_file_path)
    else:
        raise ValueError("Cannot find MESA profile file in {} or {}".format(config.star['path'], stock_file_path))
core_cz_radius = find_core_cz_radius(mesa_file_path)
forcing_radius = 1.02 * core_cz_radius / L_nd


#Calculate transfer functions
Lmax = config.eigenvalue['Lmax']
ell_list = np.arange(1, Lmax+1)
eig_dir = 'eigenvalues'
for ell in ell_list:
    plt.figure()
    print("ell = %i" % ell)

    xmin = 1e99
    xmax = -1e99

    transfers = []
    oms = []
    depth_list = [2,]
    depth_end  = depth_list[1:] + [1e-10]
    for j, d_filter in enumerate(depth_list):
        d_end = depth_end[j]
        #Read in eigenfunction values.
        #Require: eigenvalues, horizontal duals, transfer surface (s1), optical depths

        with h5py.File('{:s}/duals_ell{:03d}_eigenvalues.h5'.format(eig_dir, ell), 'r') as f:
            velocity_duals = f['velocity_duals'][()]
            values = f['good_evalues'][()]
            s1_amplitudes = f['s1_amplitudes'][()].squeeze()
            depths = f['depths'][()]

            eff_evalues = []
            for ev in values:
                gamma_eff, omega_eff = eff(-ev.imag, np.abs(ev.real), timestep)
                if ev.real < 0:
                    eff_evalues.append(-omega_eff - 1j*gamma_eff)
                else:
                    eff_evalues.append(omega_eff - 1j*gamma_eff)
            values = np.array(eff_evalues)
 

            rs = []
            for bk in ['B', 'S1', 'S2']:
                rs.append(f['r_{}'.format(bk)][()].flatten())
            r = np.concatenate(rs)
            smooth_oms = f['smooth_oms'][()]
            smooth_depths = f['smooth_depths'][()]
            depthfunc = interp1d(smooth_oms, smooth_depths, bounds_error=False, fill_value='extrapolate')

        #Pick out eigenvalues that have less optical depth than a given cutoff
        print('depth cutoff: {}, end: {}'.format(d_filter, d_end))
        good = (depths < d_filter)*(values.real > 0)*(depths > 1e-10)
        values = values[good]
        s1_amplitudes = s1_amplitudes[good]
        velocity_duals = velocity_duals[good]

        #Construct frequency grid for evaluation
        om0 = np.min(np.abs(values.real))*0.95
        om1 = np.max(values.real)*1.05
        if om0 < xmin: xmin = om0
        if om1 > xmax: xmax = om1
#        if j == 0:
#            om0/= 10**(1)
#        stitch_om = np.abs(values[depths[good] <= 1][-1].real)
        om = np.exp( np.linspace(np.log(om0), np.log(om1), num=5000, endpoint=True) )

        #Get forcing radius and dual basis evaluated there.
        r0 = forcing_radius
        r1 = forcing_radius * (1.05)
        r_range = np.linspace(r0, r1, num=100, endpoint=True)
        uphi_dual_interp = interpolate.interp1d(r, velocity_duals[:,0,:], axis=-1)(r_range)

        #Calculate and store transfer function
        om, T = calculate_refined_transfer(om, values, uphi_dual_interp, s1_amplitudes, r_range, rho(r_range), ell)
        oms.append(om)
        transfers.append(T)
        plt.loglog(om, T)
    om_new = np.logspace(np.log10(om0)-0.3, np.log10(om0), 100)
    T_damp = T[0]*np.exp(-depthfunc(om_new)+2)
    oms.append(om_new)
    transfers.append(T_damp)

#    plt.loglog(om_new, T_damp)
#    plt.loglog(om, 1000*np.exp(-depthfunc(om)))
#    plt.axvline(om[depthfunc(om) > 1].max())
#    plt.axvline(om[depthfunc(om) > 3].max())
#    plt.show()


    #right now we have a transfer function for each optical depth filter
    # We want to just get one transfer function value at each om
    # use the minimum T value from all filters we've done.
    good_om = np.sort(np.concatenate(oms))
    good_T = np.zeros_like(good_om)

    from scipy.interpolate import interp1d
    interps = []
    for om, T in zip(oms, transfers):
        interps.append(interp1d(om, T, bounds_error=False, fill_value=np.inf))

    for i, omega in enumerate(good_om):
        vals = []
        for f in interps:
            vals.append(f(omega))
        good_T[i] = np.min(vals)

    #Do WKB at low frequency -- exponentially attenuate by exp(-om/om_{tau=1}) [assume transfer does everything right up to tau=1]
#    good_T *= np.exp(-depthfunc(good_om))
#    good_T[good_om <= stitch_om] *= np.exp(-depthfunc(good_om[good_om <= stitch_om]) + depthfunc(stitch_om))


    # Right now the transfer function gets us from ur (near RCB) -> surface. We want wave luminosity (near RCB) -> surface.
    chi_rad_u = chi_rad(forcing_radius)
    brunt_N2_u = N2s.max()
#    brunt_N2_u = N2(forcing_radius)
    rho_u = rho(forcing_radius)

    #k_r is eqn 13 of Lecoanet+2015 PRE 91, 063016.
    k_h = np.sqrt(ell*(ell+1))/forcing_radius 
    k_r_low_diss  = -np.sqrt(brunt_N2_u/good_om**2 - 1).real*k_h
    k_r_high_diss = (((-1)**(3/4) / np.sqrt(2))\
                          *np.sqrt(-2*1j*k_h**2 - (good_om/chi_rad_u) + np.sqrt((good_om)**3 + 4*1j*k_h**2*chi_rad_u*brunt_N2_u)/(chi_rad_u*np.sqrt(good_om)) )).real
    k_r_err = np.abs(1 - k_r_low_diss/k_r_high_diss)

    k_r = np.copy(k_r_high_diss)
    if np.sum(k_r_err < 1e-3) > 0:
        om_switch = good_om.ravel()[k_r_err < 1e-3][0]
        k_r[good_om.ravel() >= om_switch] = k_r_low_diss[good_om.ravel() >= om_switch]
#
    k2 = k_r**2 + k_h**2
    root_lum_to_ur = np.sqrt(1/(4*np.pi*forcing_radius**2*rho_u))*np.sqrt(np.array(1/(-(good_om + 1j*chi_rad_u*k2)*k_r / k_h**2).real, dtype=np.complex128))

#    plt.loglog(good_om/(2*np.pi), good_T)
#    plt.loglog(good_om/(2*np.pi), (root_lum_to_ur*good_T).real, color='orange')
#    plt.loglog(good_om/(2*np.pi), (root_lum_to_ur*good_T).imag, color='orange', ls='--')
#    plt.axvline(stitch_om/(2*np.pi))
#    plt.xlabel('frequency')
#    plt.ylabel('T')
#    plt.show()
#
    with h5py.File('{:s}/transfer_ell{:03d}_eigenvalues.h5'.format(eig_dir, ell), 'w') as f:
        f['om'] = good_om
        f['transfer_ur'] = good_T
        f['transfer_root_lum'] = root_lum_to_ur*good_T 


