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
Lsun = 3.839e33 #erg/s
p = mr.MesaData(mesa_LOG)
r = p.radius[::-1]*Rsun_to_cm #in cm
bruntN2 = p.brunt_N2[::-1] #rad^2/s^2
T       = p.temperature[::-1] #K
rhos     = 10**(p.logRho[::-1]) #g/cm^3
opacity = p.opacity[::-1] #cm^2 / g
cp      = p.cp[::-1] #erg / K / g
Lum     = p.photosphere_L * Lsun #erg/s
chi_rads = 16 * sigma_SB * T**3 / (3 * rhos**2 * cp * opacity)
rho = interpolate.interp1d(r.flatten(), rhos.flatten())
chi_rad = interpolate.interp1d(r.flatten(), chi_rads.flatten())
N2 = interpolate.interp1d(r.flatten(), bruntN2.flatten())
#plt.semilogy(r, bruntN2)
#plt.show()


#Calculate transfer functions
Lmax = 3
ell_list = np.arange(1, Lmax+1)
eig_dir = 'gyre_output'
plot_freqs = np.logspace(-6, -4, 10000)
total_signal = np.zeros_like(plot_freqs)

wave_luminosity = lambda f, l: 1e-15*f**(-7.5)*l**3
for ell in ell_list:
    plt.figure()
    print("ell = %i" % ell)


    with h5py.File('{:s}/transfer_ell{:03d}_eigenvalues.h5'.format(eig_dir, ell), 'r') as f:
        om = f['om'][()]
        transfer_root_lum = f['transfer_root_lum'][()].real

    print(om)
    deltaL_d_L = transfer_root_lum*np.sqrt(wave_luminosity(om/(2*np.pi), ell))/Lum
    total_signal += 10**(interp1d(np.log10(om/(2*np.pi)), np.log10(deltaL_d_L))(np.log10(plot_freqs))) / ell
    plt.loglog(om/(2*np.pi), deltaL_d_L)
    plt.ylim(1e-13, 1e-2)
    plt.ylabel(r'$\delta L / L_*$')
    plt.xlabel(r'frequency ($\mu$Hz)')
    plt.xlim(1e-6, 1e-4)
    plt.show()


plt.loglog(plot_freqs, total_signal)
plt.ylim(1e-13, 1e-2)
plt.ylabel(r'$\delta L / L_*$')
plt.xlabel(r'frequency ($\mu$Hz)')
plt.xlim(1e-6, 1e-4)
plt.show()
