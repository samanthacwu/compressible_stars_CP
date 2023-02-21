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

from read_mist_models import EEP
import mesa_reader as mr
import d3_stars
from d3_stars.defaults import config
from d3_stars.simulations.parser import name_star
from d3_stars.simulations.star_builder import find_core_cz_radius
from d3_stars.simulations.evp_functions import calculate_refined_transfer
from matplotlib.patches import ConnectionPatch


#Appendix information from https://www.aanda.org/articles/aa/pdf/2020/08/aa38224-20.pdf
star_names   = [ 'HD36960', 'HD37042', 'HD43112', 'HD34816', 'HD46328', 'HD50707']
alpha0       = [30.868, 57.494, 6.269, 61.383, 54.479, 132.11] #mumag
nu_char      = [1.83618, 1.20717, 0.76982, 0.66173, 1.08885, 4.27661] #d-1
gamma        = [2.46009, 1.84494, 1.16679, 1.34255, 1.34896, 1.75206]
Cw           = [3.814, 9.116, 3.770, 1.445, 3.490, 2.889] #mumag
log10Teff    = [4.46, 4.47, 4.41, 4.46, 4.40, 4.38]
log10LdLsol  = [3.31, 3.06, 2.95, 3.22, 3.28, 3.31]

Bdwarf_star_names   = star_names[:3]
Bsubgiant_star_names   = star_names[3:]

#NOTE to compare to MESA model we need to calculate 'ell' not bolometric luminosity.
# from matteo:
#ell_sun=(5777)**4.0/(274*100)  
#ell = (10**logt)**4.0/(10**logg)
#ell=np.log10(ell/ell_sun)  
sLum_sun=(5777)**4.0/(274*100)  


#Calculate transfer functions
Lmax = 16
ell_list = np.arange(1, Lmax+1)
eig_dir = 'gyre_output'


plt.figure()
star_dirs = ['3msol', '40msol', '15msol']
luminosity_amplitudes = [3e-15, 3e-10, 3e-11]
#star_dirs = ['15msol', '15msol', '40msol']
#luminosity_amplitudes = [3e-11, 3e-11, 3e-10]
obs_length_days = 365
obs_length_sec  = obs_length_days*24*60*60
obs_cadence = 30*60 #30 min
df = 1/obs_length_sec
N_data = int(obs_length_sec/obs_cadence)
freqs = np.arange(N_data)*df

signals = []
specLums = []
logTeffs = []
for i, sdir in enumerate(star_dirs):
    wave_luminosity = lambda f, l: luminosity_amplitudes[i]*f**(-6.5)*np.sqrt(l*(l+1))**4
    transfer_oms = []
    transfer_signal = []

    #MESA history for getting ell.
    mesa_history = '{}/LOGS/history.data'.format(sdir)
    history = mr.MesaData(mesa_history)
    mn = history.model_number
    log_g = history.log_g
    log_Teff = history.log_Teff

    sLum = (10**log_Teff)**4.0/(10**log_g)
    sLum=np.log10(sLum/sLum_sun)
    specLums.append(sLum[-1])
    logTeffs.append(log_Teff[-1])


    plt.figure()
    for ell in ell_list:
        print("ell = %i" % ell)


        with h5py.File('{:s}/{:s}/ell{:03d}_eigenvalues.h5'.format(sdir, eig_dir, ell), 'r') as f:
            om = f['transfer_om'][()]
            transfer_root_lum = f['transfer_root_lum'][()].real
        micromag = transfer_root_lum*np.sqrt(wave_luminosity(om/(2*np.pi), ell))

        plt.loglog(om/(2*np.pi), micromag, label='ell={}'.format(ell))
        plt.ylabel(r'$\delta L / L_*$')
        plt.xlabel(r'frequency (Hz)')
        plt.xlim(3e-7, 1e-4)

        transfer_oms.append(om[np.isfinite(micromag)])
        transfer_signal.append(micromag[np.isfinite(micromag)])



    total_signal = np.zeros_like(freqs)
    for i in range(freqs.size-1):
        for oms, signal in zip(transfer_oms, transfer_signal):
            good = (2*np.pi*freqs[i+1] >= oms)*(2*np.pi*freqs[i] < oms)
            if np.sum(good) > 0:
                total_signal[i] += np.max(signal[good])
            elif 2*np.pi*freqs[i] > oms.min():
                total_signal[i] += signal[np.argmin(np.abs(2*np.pi*freqs[i] - oms))]
    plt.loglog(freqs, total_signal, c='k')
    plt.legend()
    plt.savefig('obs_ell_contributions_{}.png'.format(sdir), bbox_inches='tight')
    signals.append(total_signal)

freqs *= 60*60*24 #1/s -> 1/day

#### MAKE PAPER FIGURE
fig = plt.figure(figsize=(8,3))
ax1 = fig.add_axes((0,    0, 0.27, 1))
ax2 = fig.add_axes((0.40, 0, 0.27, 1))
ax3 = fig.add_axes((0.73,  0, 0.27, 1))
con1 = ConnectionPatch(xyA=(1e1,1e0), xyB=(3e-2,1e0), coordsA='data', coordsB='data', axesA=ax2, axesB=ax3, color='k', lw=0.5)
con2 = ConnectionPatch(xyA=(1e1,1e-4), xyB=(3e-2,1e-4), coordsA='data', coordsB='data', axesA=ax2, axesB=ax3, color='k', lw=0.5)
ax2.add_artist(con1)
ax2.add_artist(con2)

plt.subplots_adjust(hspace=0.5, wspace=0.7)


#Reads in non-rotating mist models.
#Data: https://waps.cfa.harvard.edu/MIST/model_grids.html (v/vcrit = 0; [Fe/H] = 0) EEP tracks.
#Read script: https://github.com/jieunchoi/MIST_codes/blob/master/scripts/read_mist_models.py
for mass in ['00300', '00500', '01000', '02000', '04000']:
    model = EEP('mist/MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.0_EEPS/{}M.track.eep'.format(mass), verbose=True)
    mass = model.minit
    center_h1 = model.eeps['center_h1']
    good = (center_h1 < center_h1[0]*0.999)*(center_h1 > 0.02)
    log_Teff = model.eeps['log_Teff']
    log_g = model.eeps['log_g']
    ell = (10**log_Teff)**4.0/(10**log_g)
    ell = np.log10(ell/sLum_sun)
    ax1.plot(log_Teff[good], ell[good], label=mass, c='grey', lw=1, zorder=0)
#    ax1.text(0.01+log_Teff[good][0], -0.1+ell[good][0], '{:d}'.format(int(mass))+r'$M_{\odot}$', ha='right')
    ax1.text(0.02+log_Teff[good][0], -0.1+ell[good][0], '{:d}'.format(int(mass)), ha='right', size=10, color='grey')

ax1.scatter(4.715, 1.4, c='white', marker='*', s=100, zorder=1, edgecolors='k', linewidths=0.5)
ax1.text(4.715 - 0.04, 1.4, 'simulations', c='k', ha='left', va='center')
ax1.set_xlim(4.75, 4.0)
ax1.set_ylim(1.3, 4.0)
ax1.set_ylabel(r'$\mathrm{log}_{10}(\mathcal{L}/\mathcal{L}_{\odot})$')
ax1.set_xlabel(r'$\mathrm{log}_{10}(T_{\rm eff})$')

plt.axes(ax2)
plt.fill_between([3e-2, 1e-1], 1e-20, 1e10, color='grey', alpha=0.5)
plt.fill_between([1e1, 3e1], 1e-20, 1e10, color='grey', alpha=0.5)
ax2.text(0.12, 0.1, 'Predicted wave signal', ha='left')
ax2.text(0.2, 2e2, 'Observed red noise', ha='left', va='center')

from palettable.colorbrewer.qualitative import Dark2_5 as cmap
for i in range(len(star_names)):
    if star_names[i] in Bdwarf_star_names:
        color = cmap.mpl_colors[3]
        if star_names[i] == Bdwarf_star_names[0]:
            ax1.text(0.99, 0.99, 'B dwarf stars', ha='right', va='top', transform=ax1.transAxes, color=color)
    elif star_names[i] in Bsubgiant_star_names:
        color = cmap.mpl_colors[4]
        if star_names[i] == Bsubgiant_star_names[0]:
            ax1.text(0.99, 0.95, 'B subgiant stars', ha='right', va='top', transform=ax1.transAxes, color=color)
    ax1.scatter(log10Teff[i], log10LdLsol[i], color=color, zorder=1, s=20, edgecolors='k', linewidths=0.5)
    alphanu = (alpha0[i] / (1 + (freqs/nu_char[i])**gamma[i]) + Cw[i]) #mags
    plt.loglog(freqs, alphanu, color=color)
#    ax1.text(log10Teff[i]*0.995, log10LdLsol[i], star_names[i], ha='left', color=color)
#    ax1.text(0.99, 1.03-0.04*(i+1), star_names[i], ha='right', va='top', transform=ax1.transAxes, color=color)
    plt.ylim(1e-4, 3e2)
    plt.ylabel(r'$\Delta m$ ($\mu$mag)')
    plt.xlabel(r'frequency (d$^{-1}$)')
    plt.xlim(3e-2, 1e1)

for i in range(3):
    star_log_Teff, star_log_Ell  = logTeffs[i], specLums[i]
    ax1.scatter(star_log_Teff, star_log_Ell, c=cmap.mpl_colors[i], marker='*', s=100, zorder=1, edgecolors='k', linewidths=0.5)
    ax3.loglog(freqs, signals[i], color=cmap.mpl_colors[i], lw=1)#, label=r'15 $M_{\odot}$ LMC sim')
    if i == 2:
        plt.loglog(freqs, signals[i], lw=1, c=cmap.mpl_colors[i])#, label=r'15 $M_{\odot}$ LMC sim')
ax3.text(5.5e-2, 9e-2, r'40 $M_{\odot}$', color=cmap.mpl_colors[1], ha='center', va='center', size=8)
ax3.text(9e-2, 3.7e-2, r'15 $M_{\odot}$', color=cmap.mpl_colors[2], ha='center', va='center', size=8)
ax3.text(1.4e-1, 2e-3, r'3 $M_{\odot}$', color=cmap.mpl_colors[0], ha='center', va='center', size=8)
ax3.set_ylim(1e-4, 1e0)
ax3.set_xlim(3e-2, 1e1)
ax3.set_xlabel(r'frequency (d$^{-1}$)')

plt.savefig('obs_prediction.png', bbox_inches='tight', dpi=300)
