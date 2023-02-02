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

import mesa_reader as mr
import d3_stars
from d3_stars.defaults import config
from d3_stars.simulations.parser import name_star
from d3_stars.simulations.star_builder import find_core_cz_radius
from d3_stars.simulations.evp_functions import calculate_refined_transfer


#get info about mesa background
mesa_LOG = 'LOGS/profile47.data'
Rsun_to_cm = 6.957e10
sigma_SB = 5.67037442e-05 #cgs
p = mr.MesaData(mesa_LOG)
r = p.radius[::-1]*Rsun_to_cm #in cm
bruntN2 = p.brunt_N2[::-1] #rad^2/s^2
T       = p.temperature[::-1] #K
rhos     = 10**(p.logRho[::-1]) #g/cm^3
opacity = p.opacity[::-1] #cm^2 / g
cp      = p.cp[::-1] #erg / K / g
chi_rads = 16 * sigma_SB * T**3 / (3 * rhos**2 * cp * opacity)
rho = interpolate.interp1d(r.flatten(), rhos.flatten())
chi_rad = interpolate.interp1d(r.flatten(), chi_rads.flatten())
N2 = interpolate.interp1d(r.flatten(), bruntN2.flatten())
#plt.semilogy(r, bruntN2)
#plt.show()


# Generalized logic for getting forcing radius.
#package_path = Path(d3_stars.__file__).resolve().parent
#stock_path = package_path.joinpath('stock_models')
#if os.path.exists(config.star['path']):
#    mesa_file_path = config.star['path']
#else:
#    stock_file_path = stock_path.joinpath(config.star['path'])
#    if os.path.exists(stock_file_path):
#        mesa_file_path = str(stock_file_path)
#    else:
#        raise ValueError("Cannot find MESA profile file in {} or {}".format(config.star['path'], stock_file_path))
core_cz_radius = find_core_cz_radius(mesa_LOG)
forcing_radius = 1.02 * core_cz_radius


#Calculate transfer functions
Lmax = 10
ell_list = np.arange(1, Lmax+1)
eig_dir = 'gyre_output'
for ell in ell_list:
    plt.figure()
    print("ell = %i" % ell)

    xmin = 1e99
    xmax = -1e99

    transfers = []
    oms = []
    depth_list = [1e10, 1, 0.3, 0.1,]
    for j, d_filter in enumerate(depth_list):
        #Read in eigenfunction values.
        #Require: eigenvalues, horizontal duals, transfer surface (s1), optical depths

        with h5py.File('{:s}/ell{:03d}_eigenvalues.h5'.format(eig_dir, ell), 'r') as f:
            r = f['r'][()]
            uh_duals = f['uh_dual'][()]
            raw_values = 2*np.pi*f['dimensional_freqs'][()] #TODO: check if this is right
#            print(raw_values)
            Lum_amplitudes = f['L_top'][()].squeeze()
            depths = f['depths'][()]

            smooth_oms = f['smooth_oms'][()]
            smooth_depths = f['smooth_depths'][()]
            depthfunc = interp1d(smooth_oms, smooth_depths, bounds_error=False, fill_value='extrapolate')

        #Pick out eigenvalues that have less optical depth than a given cutoff
        print('depth cutoff: {}'.format(d_filter))
        good = (depths < d_filter)*(raw_values.real > 0)#*(depths > d_filter/15)
        values = raw_values[good]
        Lum_amplitudes = Lum_amplitudes[good]
        uh_duals = uh_duals[good]

        #Construct frequency grid for evaluation
        om1 = np.max(values.real)*1.05
        if j == 0:
            om0 = np.min(np.abs(values.real))*0.95
        else:
            om0 = np.min(np.abs(values.real))*0.95
        if om0 < xmin: xmin = om0
#        if om1 > xmax: xmax = om1
##            om0/= 10**(1)
#            stitch_om = smooth_oms[smooth_depths <= 10][0]
#            print('stitch', stitch_om)
        om = np.exp( np.linspace(np.log(om0), np.log(om1), num=5000, endpoint=True) )

        #Get forcing radius and dual basis evaluated there.
        r0 = forcing_radius
        r1 = forcing_radius * (1.05)
        r_range = np.linspace(r0, r1, num=100, endpoint=True)
        uh_dual_interp = interpolate.interp1d(r, uh_duals[:,:], axis=-1)(r_range)

        #Calculate and store transfer function
        om, T = calculate_refined_transfer(om, values, uh_dual_interp, Lum_amplitudes, r_range, rho(r_range), ell)
        oms.append(om)
        transfers.append(T)
#        plt.loglog(om/(2*np.pi), T)
#    for v in raw_values:
#        plt.axvline(v.real)
#    plt.show()

#    om_new = np.logspace(np.log10(om0)-0.3, np.log10(om0), 100)
#    T_damp = T[0]*np.exp(-depthfunc(om_new)+3)
#    oms.append(om_new)
#    transfers.append(T_damp)




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
#    wkb = np.exp(-depthfunc(good_om/(2*np.pi)))
#    wkb *= good_T[good_om > stitch_om][0] / np.exp(-depthfunc(good_om[good_om > stitch_om][0]/(2*np.pi)))
#    plt.loglog(good_om, good_T)
#    plt.loglog(good_om, wkb)
#    plt.axvline(stitch_om)
#    plt.show()
#    good_T *= np.exp(-depthfunc(good_om))
#    good_T[good_om <= stitch_om] *= np.exp(-depthfunc(good_om[good_om <= stitch_om]) + depthfunc(stitch_om))


    # Right now the transfer function gets us from ur (near RCB) -> surface. We want wave luminosity (near RCB) -> surface.
    chi_rad_u = chi_rad(forcing_radius)
    brunt_N2_u = N2(0.5*r.max())
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
    transfer_root_lum = root_lum_to_ur*good_T 


#    plt.loglog(good_om, depthfunc(good_om))
#    plt.loglog(good_om, good_om**(-4)/1e20)
#    plt.loglog(good_om, good_om**(-1/4))
#    plt.show()
#    plt.figure()

#    plt.loglog(good_om/(2*np.pi), good_T)
    plt.loglog(good_om/(2*np.pi),transfer_root_lum.real, color='black', label='transfer')
    plt.loglog(good_om/(2*np.pi),transfer_root_lum.imag, color='black', ls='--')
#    plt.loglog(good_om/(2*np.pi), np.exp(1-depthfunc(good_om)), c='blue', label='WKB envelope')
#    plt.loglog(good_om/(2*np.pi), 1e-3*(good_om/1e-5/2/np.pi)**(-3.75), label='Forcing shape')
#    plt.ylim(1e-10, 1)
    plt.xlabel('frequency')
    plt.ylabel('T')
    plt.show()
#
    with h5py.File('{:s}/transfer_ell{:03d}_eigenvalues.h5'.format(eig_dir, ell), 'w') as f:
        f['om'] = good_om
        f['transfer_ur'] = good_T
        f['transfer_root_lum'] = root_lum_to_ur*good_T 


