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

plot = False

mesa_LOG = 'LOGS/profile47.data'
p = mr.MesaData(mesa_LOG)
mass           = (p.mass[::-1] * u.M_sun).cgs
r              = (p.radius[::-1] * u.R_sun).cgs
rho            = 10**p.logRho[::-1] * u.g / u.cm**3
P              = p.pressure[::-1] * u.g / u.cm / u.s**2
g               = constants.G.cgs*mass/r**2
csound         = p.csound[::-1] * u.cm / u.s
dlogPdr         = -rho*g/P
gamma1          = dlogPdr/(-g/csound**2)
gamma = gamma1[0]



#get info about mesa background
Rsun_to_cm = 6.957e10
sigma_SB = 5.67037442e-05 #cgs
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

core_cz_radius = find_core_cz_radius(mesa_LOG)
forcing_radius = 1.02 * core_cz_radius


#Calculate transfer functions
Lmax = 10
ell_list = np.arange(1, Lmax+1)
eig_dir = 'gyre_output'
for ell in ell_list:
    plt.figure()
    print("ell = %i" % ell)

    #Read in eigenfunction values.
    #Require: eigenvalues, horizontal duals, transfer surface (s1), optical depths
    with h5py.File('{:s}/ell{:03d}_eigenvalues.h5'.format(eig_dir, ell), 'r') as f:
        r = f['r'][()]
        uh_duals = f['uh_dual'][()]
        values = 2*np.pi*f['dimensional_freqs'][()] #TODO: check if this is right
        Lum_amplitudes = f['L_top'][()].squeeze()
        depths = f['depths'][()]

        smooth_oms = f['smooth_oms'][()]
        smooth_depths = f['smooth_depths'][()]
        depthfunc = interp1d(smooth_oms, smooth_depths, bounds_error=False, fill_value='extrapolate')

    #Construct frequency grid for evaluation
    om0 = np.min(np.abs(values.real))*0.95
    om1 = np.max(values.real)*1.05
    om = np.logspace(np.log10(om0), np.log10(om1), num=5000, endpoint=True) 

    #Get forcing radius and dual basis evaluated there.
    r0 = forcing_radius
    r1 = forcing_radius * (1.05)
    r_range = np.linspace(r0, r1, num=100, endpoint=True)
    uh_dual_interp = interpolate.interp1d(r, uh_duals[:,:], axis=-1)(r_range)

    #Calculate and store transfer function
    good_om, good_T = calculate_refined_transfer(om, values, uh_dual_interp, Lum_amplitudes, r_range, ell, rho, chi_rad, N2(0.5*r.max()), gamma)

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


