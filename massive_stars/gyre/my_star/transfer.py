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

from astropy import constants
import astropy.units as u

import mesa_reader as mr
import d3_stars
from d3_stars.defaults import config
from d3_stars.simulations.parser import name_star
from d3_stars.simulations.star_builder import find_core_cz_radius
from d3_stars.simulations.evp_functions import calculate_refined_transfer

if __name__ == "__main__":
    plot = True

    mesa_LOG = 'LOGS/profile47.data'
    p = mr.MesaData(mesa_LOG)
    mass           = (p.mass[::-1] * u.M_sun).cgs
    r              = (p.radius[::-1] * u.R_sun).cgs
    rho            = 10**p.logRho[::-1] * u.g / u.cm**3
    P              = p.pressure[::-1] * u.g / u.cm / u.s**2
    g               = constants.G.cgs*mass/r**2
    T               = p.temperature[::-1] * u.K
    opacity        = p.opacity[::-1] * (u.cm**2 / u.g)
    cp             = p.cp[::-1]  * u.erg / u.K / u.g
    csound         = p.csound[::-1] * u.cm / u.s
    dlogPdr         = -rho*g/P
    bruntN2         = p.brunt_N2[::-1] / u.s**2
    gamma1          = dlogPdr/(-g/csound**2)
    chi_rad = 16 * constants.sigma_sb.cgs * T**3 / (3 * rho**2 * cp * opacity)
    gamma = gamma1[0]

    #get info about mesa background
    rho_func = interpolate.interp1d(r.flatten(), rho.flatten())
    chi_rad_func = interpolate.interp1d(r.flatten(), chi_rad.flatten())
    N2_func = interpolate.interp1d(r.flatten(), bruntN2.flatten())
    N2_max = N2_func(r.max()/2).max()
#    print('N2 vals', N2_max, N2(r.max()/2))

    core_cz_radius = find_core_cz_radius(mesa_LOG)*u.cm
    r0 = 0.99 * core_cz_radius
    r1 = 1.01 * core_cz_radius


    #Calculate transfer functions
    Lmax = 1
    ell_list = np.arange(1, Lmax+1)
    eig_dir = 'gyre_output'
    for ell in ell_list:
        plt.figure()
        print("ell = %i" % ell)

        #Read in eigenfunction values.
        #Require: eigenvalues, horizontal duals, transfer surface (s1), optical depths
        with h5py.File('{:s}/ell{:03d}_eigenvalues.h5'.format(eig_dir, ell), 'r') as f:
            r = f['r'][()]
            uh_duals = f['u_h_dual'][()]
            values = 2*np.pi*f['freq'][()]
   #         lum_amplitudes = f['dF_mumags_Red'][()].squeeze()
   #        lum_amplitudes = -2.5/np.log(10)*1e6*f['delta_L_dL_top'][()].squeeze()
            lum_amplitudes = -2.5*1e6*f['delta_L_dL_top'][()].squeeze() #need to retrieve old form using this

            uh_duals = uh_duals[values.real > 0,:]
            lum_amplitudes = lum_amplitudes[values.real > 0]
            values = values[values.real > 0]
#            print(uh_duals.shape, lum_amplitudes.shape, values.shape)

        #Construct frequency grid for evaluation
        om0 = np.min(np.abs(values.real))*0.95
        om1 = np.max(values.real)*2
        om = np.logspace(np.log10(om0), np.log10(om1), num=1000, endpoint=True) 

        #Get forcing radius and dual basis evaluated there.
        r_range = np.linspace(r0.value, r1.value, num=50, endpoint=True)
        uh_dual_interp = interpolate.interp1d(r, uh_duals[:,:], axis=-1)(r_range)
#        for i in range(values.size):
#            plt.plot(r, uh_duals[i,:])
#            plt.plot(r_range, uh_dual_interp[i,:])
#            plt.show()
#        print(uh_dual_interp.shape, om.shape, values.shape, uh_dual_interp.shape)

        #Calculate and store transfer function
        good_om, good_T = calculate_refined_transfer(om, values, uh_dual_interp, lum_amplitudes, r_range, ell, rho_func, chi_rad_func, N2_max, gamma, plot=True)

        if plot:
            plt.loglog(good_om/(2*np.pi),good_T.real, color='black', label='transfer')
            plt.loglog(good_om/(2*np.pi),good_T.imag, color='black', ls='--')
            plt.xlabel('frequency')
            plt.ylabel('T')
            plt.show()
    #
        with h5py.File('{:s}/transfer_ell{:03d}_eigenvalues.h5'.format(eig_dir, ell), 'w') as f:
            f['om'] = good_om
            f['transfer_root_lum'] = good_T 


