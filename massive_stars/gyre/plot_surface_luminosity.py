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
from palettable.colorbrewer.qualitative import Dark2_5 as cmap
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'


#Appendix information from https://www.aanda.org/articles/aa/pdf/2020/08/aa38224-20.pdf
star_names   = [ 'HD36960', 'HD37042', 'HD43112', 'HD34816', 'HD46328', 'HD50707']
alpha0       = [30.868, 57.494, 6.269, 61.383, 54.479, 132.11] #mumag
nu_char      = [1.83618, 1.20717, 0.76982, 0.66173, 1.08885, 4.27661] #d-1
gamma        = [2.46009, 1.84494, 1.16679, 1.34255, 1.34896, 1.75206]
Cw           = [3.814, 9.116, 3.770, 1.445, 3.490, 2.889] #mumag
log10Teff    = [4.46, 4.47, 4.41, 4.46, 4.40, 4.38]
log10LdLsol  = [3.31, 3.06, 2.95, 3.22, 3.28, 3.31]

#NOTE to compare to MESA model we need to calculate 'ell' not bolometric luminosity.
# from matteo:
#ell_sun=(5777)**4.0/(274*100)  
#ell = (10**logt)**4.0/(10**logg)
#ell=np.log10(ell/ell_sun)  
sLum_sun=(5777)**4.0/(274*100)  

output_file = 'magnitude_spectra.h5'
out_f = h5py.File(output_file, 'r')
freqs = out_f['frequencies'][()]


#Calculate transfer functions
eig_dir = 'gyre_output'


plt.figure()
star_dirs = ['3msol', '40msol', '15msol']
Lmax = [16,16,16]

signals = []
specLums = []
logTeffs = []
for i, sdir in enumerate(star_dirs):

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



    signals.append(np.sum(out_f['{}_magnitude_cube'.format(sdir)][:Lmax[i],:], axis=0))
freqs *= 60*60*24 #1/s -> 1/day

#### MAKE PAPER FIGURE
fig = plt.figure(figsize=(8,3))
ax1 = fig.add_axes((0.00,  0, 0.27, 1))
ax2 = fig.add_axes((0.33,  0, 0.27, 1))
ax3 = fig.add_axes((0.73,  0, 0.27, 1))

plt.subplots_adjust(hspace=0.5, wspace=0.7)


#Reads in non-rotating mist models.
#Data: https://waps.cfa.harvard.edu/MIST/model_grids.html (v/vcrit = 0; [Fe/H] = 0) EEP tracks.
#Read script: https://github.com/jieunchoi/MIST_codes/blob/master/scripts/read_mist_models.py
zamsT = []
zamsL = []
for mass in ['00300', '00500', '01000', '02000', '04000']:
    model = EEP('mist/MIST_v1.2_feh_p0.00_afe_p0.0_vvcrit0.0_EEPS/{}M.track.eep'.format(mass), verbose=True)
    mass = model.minit
    center_h1 = model.eeps['center_h1']
    good = (center_h1 < center_h1[0]*0.999)*(center_h1 > 0.02)
    log_Teff = model.eeps['log_Teff']
    log_g = model.eeps['log_g']
    ell = (10**log_Teff)**4.0/(10**log_g)
    ell = np.log10(ell/sLum_sun)
    zamsT.append(log_Teff[good][0])
    zamsL.append(ell[good][0])
    ax3.plot(log_Teff[good], ell[good], label=mass, c='grey', lw=1, zorder=0)
#    ax3.text(0.01+log_Teff[good][0], -0.1+ell[good][0], '{:d}'.format(int(mass))+r'$M_{\odot}$', ha='right')
    ax3.text(0.02+log_Teff[good][0], -0.1+ell[good][0], '{:d}'.format(int(mass)), ha='right', size=10, color='grey')

#make colormap based on main sequence distance for stars
#ax3.plot(zamsT, zamsL, c='grey', zorder=0, lw=1)
denseT = np.linspace(4.2, 4.7, 200)
denseL = interp1d(zamsT, zamsL, bounds_error=False, fill_value='extrapolate')(denseT)
distance = []
for T, L in zip(log10Teff, log10LdLsol):
    distance.append(np.min(np.sqrt((denseT - T)**2 + (denseL - L)**2)))
abs_star_distance = np.array(100*np.array(distance)/np.max(distance) - 1, dtype=int)
black = (0, 0, 0)
pink_rgbs = []
for i in range(3):
    pink_rgbs.append(np.linspace(cmap.mpl_colors[3][i], black[i], 150)[:100])
star_colors = []
for i in range(len(star_names)):
    ind = abs_star_distance[i]
    color = []
    for j in range(3):
        color.append(pink_rgbs[j][ind])
    star_colors.append(color)

#make legend & set clims, etc.
ax3.scatter(0.05, 0.04, c='white', marker='*', s=100, zorder=1, edgecolors='k', linewidths=0.5, transform=ax3.transAxes)
ax3.scatter(0.05, 0.09, c='white', marker='o', s=20, zorder=1, edgecolors='k',  linewidths=0.5, transform=ax3.transAxes)
ax3.text(0.10, 0.0375, 'Simulations', c='k', ha='left', va='center', transform=ax3.transAxes)
ax3.text(0.10, 0.09, 'Observed stars', ha='left', va='center', transform=ax3.transAxes, color='k')#cmap.mpl_colors[3])
ax3.set_xlim(4.75, 4.0)
ax3.set_ylim(1.3, 4.0)
ax3.set_ylabel(r'$\log_{10}\, \mathscr{L} / \mathscr{L}_\odot$')
ax3.set_xlabel(r'$\log_{10}\, $T$_{\rm eff}/$K')
#ax3.set_ylabel(r'$\mathrm{log}_{10}(\mathcal{L}/\mathcal{L}_{\odot})$')
#ax3.set_xlabel(r'$\mathrm{log}_{10}(T_{\rm eff})$')

plt.axes(ax1)
#plt.fill_between([3e-2, 1e-1], 1e-20, 1e10, color='grey', alpha=0.5)
#plt.fill_between([1e1, 3e1], 1e-20, 1e10, color='grey', alpha=0.5)
ax1.text(0.12, 3e-2, 'Predicted wave signal', ha='left')
ax1.text(0.2, 2e2, 'Observed red noise', ha='left', va='center')

for i in range(len(star_names)):
#    if star_names[i] in Bdwarf_star_names:
    color = star_colors[i]
#    elif star_names[i] in Bsubgiant_star_names:
#        color = cmap.mpl_colors[4]
#        if star_names[i] == Bsubgiant_star_names[0]:
#            ax3.text(0.99, 0.95, 'B subgiant stars', ha='right', va='top', transform=ax3.transAxes, color=color)
    ax3.scatter(log10Teff[i], log10LdLsol[i], color=color, zorder=1, s=20, edgecolors='k', linewidths=0.5)
    alphanu = (alpha0[i] / (1 + (freqs/nu_char[i])**gamma[i]) + Cw[i]) #mags
    plt.loglog(freqs, alphanu, color=color)
#    ax3.text(log10Teff[i]*0.995, log10LdLsol[i], star_names[i], ha='left', color=color)
#    ax3.text(0.99, 1.03-0.04*(i+1), star_names[i], ha='right', va='top', transform=ax3.transAxes, color=color)
    plt.ylim(1e-4, 3e2)
    plt.ylabel(r'$\Delta m$ ($\mu$mag)')
    plt.xlabel(r'frequency (d$^{-1}$)')

min_plot_freq = 1e-3
for i in range(3):
    star_log_Teff, star_log_Ell  = logTeffs[i], specLums[i]
    ax3.scatter(star_log_Teff, star_log_Ell, c=cmap.mpl_colors[i], marker='*', s=100, zorder=1, edgecolors='k', linewidths=0.5)
    good = freqs >= min_plot_freq
    ax2.loglog(freqs[good], signals[i][good], color=cmap.mpl_colors[i], lw=1)#, label=r'15 $M_{\odot}$ LMC sim')
    if i == 2:
        #plot on middle panel
        ax1.loglog(freqs[good], signals[i][good], color=cmap.mpl_colors[i], lw=1)#, label=r'15 $M_{\odot}$ LMC sim')

#con2 = ConnectionPatch(xyA=(1e1,1e-4), xyB=(4e-2,1e-4), coordsA='data', coordsB='data', axesA=ax1, axesB=ax2, color='grey', lw=0.5)
#ax1.add_artist(con2)
con1 = ConnectionPatch(xyA=(1e1,1.3e-1), xyB=(4e-2,1.3e-1), coordsA='data', coordsB='data', axesA=ax1, axesB=ax2, color='grey', lw=1)
ax1.add_artist(con1)
ax1.plot([7e0,1e1],[1.3e-1,1.3e-1], c='grey', lw=1)

ax2.text(6.5e-2, 6e-2, r'40 $M_{\odot}$', color=cmap.mpl_colors[1], ha='center', va='center', size=8)
ax2.text(9e-2, 5.5e-3, r'15 $M_{\odot}$', color=cmap.mpl_colors[2], ha='center', va='center', size=8)
ax2.text(1.4e-1, 2.8e-4, r'3 $M_{\odot}$', color=cmap.mpl_colors[0], ha='center', va='center', size=8)
ax2.set_ylim(1e-4, 1.3e-1)
ax2.set_xlabel(r'frequency (d$^{-1}$)')
for ax in [ax1, ax2]:
    ax.set_xlim(4e-2, 1e1)

plt.savefig('obs_prediction.png', bbox_inches='tight', dpi=300)
plt.savefig('obs_prediction.pdf', bbox_inches='tight', dpi=300)
