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


#Appendix information from https://www.aanda.org/articles/aa/pdf/2020/08/aa38224-20.pdf
#B dwarfs: HD 36960, HD 37042, HD 43112, HD 35912, HD 48977
star_names = [ 'HD 36960', 'HD 37042', 'HD 43112', 'HD 35912', 'HD 48977']
alpha0 = [30.868, 57.494, 6.269, 241.602, 311.431] #mumag
nu_char = [1.83618, 1.20717, 0.76982, 2.08064, 1.46731] #d-1
gamma = [2.46009, 1.84494, 1.16679, 3.01021, 2.33096]
Cw = [3.814, 9.116, 3.770, 5.396, 4.565] #mumaga
log10Teff = [4.46, 4.47, 4.41, 4.26, 4.25]
log10LdLsol = [3.31, 3.06, 2.95, 2.44, 2.61]


#NOTE to compare to MESA model we need to calculate 'ell' not bolometric luminosity.
# from matteo:
#ell_sun=(5777)**4.0/(274*100)  
#ell = (10**logt)**4.0/(10**logg)
#ell=np.log10(ell/ell_sun)  
#This gives ell ~ 3.12 for the 15 M_sol LMC model. Similar to the stars we're looking at.

#MESA history for getting ell.
mesa_history = 'LOGS/history.data'
history = mr.MesaData(mesa_history)
mn = history.model_number
log_g = history.log_g
log_Teff = history.log_Teff

ell_sun=(5777)**4.0/(274*100)  
ell = (10**log_Teff)**4.0/(10**log_g)
ell=np.log10(ell/ell_sun)



#get info about mesa background
mesa_LOG = 'LOGS/profile47.data'
Rsun_to_cm = 6.957e10
sigma_SB = 5.67037442e-05 #cgs
Lsun = 3.839e33 #erg/s
p = mr.MesaData(mesa_LOG)
this_model = p.model_number
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

star_log_Teff = log_Teff[mn == this_model]
star_log_Ell  = ell[mn == this_model]
#plt.semilogy(r, bruntN2)
#plt.show()


#Calculate transfer functions
Lmax = 1
ell_list = np.arange(1, Lmax+1)
eig_dir = 'gyre_output'
plot_freqs = np.logspace(-7, -4, 10000)
total_signal = np.zeros_like(plot_freqs)

for ell in ell_list:
    print("ell = %i" % ell)


    with h5py.File('{:s}/transfer_ell{:03d}_eigenvalues.h5'.format(eig_dir, ell), 'r') as f:
        om = f['om'][()]
        transfer_root_lum = f['transfer_root_lum'][()].real
    plt.loglog(om/(2*np.pi), transfer_root_lum, label='ell={}'.format(ell))

plt.figure()
wave_luminosity = lambda f, l: 1e-15*f**(-7.5)*l**3
#wave_luminosity = lambda f, l: 1e-19*f**(-8)*np.sqrt(l*(l+1))**5
for ell in ell_list:
    print("ell = %i" % ell)


    with h5py.File('{:s}/transfer_ell{:03d}_eigenvalues.h5'.format(eig_dir, ell), 'r') as f:
        om = f['om'][()]
        transfer_root_lum = f['transfer_root_lum'][()].real

    print(om)
    micromag = transfer_root_lum*np.sqrt(wave_luminosity(om/(2*np.pi), ell))
    total_signal += 10**(interp1d(np.log10(om/(2*np.pi)), np.log10(micromag), bounds_error=False, fill_value=-10000)(np.log10(plot_freqs)))
    plt.loglog(om/(2*np.pi), micromag, label='ell={}'.format(ell))
    print(micromag)
    plt.ylabel(r'$\delta L / L_*$')
    plt.xlabel(r'frequency (Hz)')
    plt.xlim(1e-6, 1e-4)
plt.legend()
plt.savefig('obs_ell_contributions.png', bbox_inches='tight')

plot_freqs *= 60*60*24 #1/s -> 1/day

fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(2,3,1)
ax2 = fig.add_subplot(2,3,2)
ax3 = fig.add_subplot(2,3,3)
ax4 = fig.add_subplot(2,3,4)
ax5 = fig.add_subplot(2,3,5)
ax6 = fig.add_subplot(2,3,6)

plt.subplots_adjust(hspace=0.5, wspace=0.7)


log10Teff = [4.46, 4.47, 4.41, 4.26, 4.25]
log10LdLsol = [3.31, 3.06, 2.95, 2.44, 2.61]
obs_axs = [ax2, ax3, ax4, ax5, ax6]
ax1.scatter(star_log_Teff, star_log_Ell, c='k', marker='*')
ax1.set_xlim(4.6, 4.1)
ax1.set_ylim(2.4, 3.4)
ax1.set_ylabel(r'$\mathrm{log}_{10}(\mathcal{L}/\mathcal{L}_{\odot})$')
ax1.set_xlabel(r'$\mathrm{log}_{10}(T_{\rm eff})$')

from palettable.colorbrewer.qualitative import Dark2_5
for i in range(len(star_names)):
    color = Dark2_5.mpl_colors[i]
    ax1.scatter(log10Teff[i], log10LdLsol[i], color=color)
    #focus on proper subplot
    plt.axes(obs_axs[i])
    plt.fill_between([3e-2, 1e-1], 1e-20, 1e10, color='grey', alpha=0.5)
    plt.fill_between([1e1, 3e1], 1e-20, 1e10, color='grey', alpha=0.5)
    alphanu = (alpha0[i] / (1 + (plot_freqs/nu_char[i])**gamma[i]) + Cw[i]) * 1e-6 #mags
    plt.loglog(plot_freqs, alphanu, color=color)
    plt.loglog(plot_freqs, total_signal, lw=2, c='k')#, label=r'15 $M_{\odot}$ LMC sim')
    plt.loglog(plot_freqs, total_signal + Cw[i], c='grey')#, label='sim + white noise')
    obs_axs[i].text(0.98, 0.88, star_names[i], ha='right', transform=obs_axs[i].transAxes, color=color)
    plt.ylim(1e-1, 3e1)
    plt.ylabel(r'$\Delta m$')
    plt.xlabel(r'frequency (1/day)')
    plt.xlim(3e-2, 1e1)
plt.savefig('obs_prediction.png', bbox_inches='tight', dpi=300)
plt.show()
