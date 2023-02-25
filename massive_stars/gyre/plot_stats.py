import numpy as np
import mesa_reader as mr
import matplotlib as mpl
import matplotlib.pyplot as plt
from palettable.colorbrewer.qualitative import Dark2_5 as cmap
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
sLum_sun=(5777)**4.0/(274*100)

#Get simulation details.
star_dirs = ['3msol', '40msol', '15msol']
sim_mass = [3, 40, 15]
sim_alpha = [3.5e-3, 9e-2, 6e-2]
sim_nuchar = [1.8, 0.55, 1.1]
sim_specLums = []
sim_logTeffs = []
for i, sdir in enumerate(star_dirs):
    #MESA history for getting ell.
    mesa_history = '{}/LOGS/history.data'.format(sdir)
    history = mr.MesaData(mesa_history)
    mn = history.model_number
    log_g = history.log_g
    log_Teff = history.log_Teff

    sLum = (10**log_Teff)**4.0/(10**log_g)
    sLum=np.log10(sLum/sLum_sun)
    sim_specLums.append(sLum[-1])
    sim_logTeffs.append(log_Teff[-1])

fig = plt.figure(figsize=(7.5, 3))
ax1 = fig.add_axes([0.02, 0.02, 0.43, 0.8])
ax2 = fig.add_axes([0.55, 0.02, 0.43, 0.8])
cax = fig.add_axes([0.25, 0.9, 0.50, 0.05])
axs = [ax1, ax2]

norm = mpl.colors.Normalize(vmin=4, vmax=4.7)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.viridis)

#TODO: need 2022 bowman data.
data1 = np.genfromtxt('bowman_a1.csv', delimiter=',', skip_header=1, dtype=str)
data2 = np.genfromtxt('bowman_a2.csv', delimiter=',', skip_header=1, dtype=str)
log10Teff = np.array(data1[:,3], dtype=np.float64)
log10Ell  = np.array(data1[:,4], dtype=np.float64)
alpha0    = np.array(data2[:,2], dtype=np.float64)
nuchar    = np.array(data2[:,4], dtype=np.float64)

for ax in axs:
    ax.set_xlabel(r'$\log_{10}\,\mathscr{L}/\mathscr{L}_{\odot}$')
ax1.set_ylabel(r'$\alpha_{0}$ ($\mu$mag)')
ax2.set_ylabel(r'$\nu_{\rm char}$ (d$^{-1}$)')

for j in range(log10Teff.size):
    ax1.scatter(log10Ell[j], alpha0[j], color=sm.to_rgba(log10Teff[j]), marker='o', edgecolors='k', linewidths=0.5)
    ax2.scatter(log10Ell[j], nuchar[j], color=sm.to_rgba(log10Teff[j]), marker='o', edgecolors='k', linewidths=0.5)
ax1.set_yscale('log')

for i in range(len(sim_mass)):
    ax1.scatter(sim_specLums[i], sim_alpha[i],  color=sm.to_rgba(sim_logTeffs[i]), marker='*', s=200, edgecolors='k', linewidths=1)
    ax2.scatter(sim_specLums[i], sim_nuchar[i], color=sm.to_rgba(sim_logTeffs[i]), marker='*', s=200, edgecolors='k', linewidths=1)
#    ax1.scatter(sim_specLums[i], sim_alpha[i],  color=cmap.mpl_colors[i], marker='*', s=200, edgecolors='k', linewidths=1)
#    ax2.scatter(sim_specLums[i], sim_nuchar[i], color=cmap.mpl_colors[i], marker='*', s=200, edgecolors='k', linewidths=1)

cb = plt.colorbar(sm, cax=cax, orientation='horizontal')
cax.text(-0.02, 0.5, r'$\log_{10} T_{\rm eff}$', ha='right', va='center', transform=cax.transAxes)
cax.invert_xaxis()
cb.set_ticks((4, 4.2, 4.4, 4.6))

fig.savefig('scatter_plots.png', dpi=300, bbox_inches='tight')

