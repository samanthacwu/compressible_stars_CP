import h5py
import numpy as np
import matplotlib as mpl 
import matplotlib.pyplot as plt 
from palettable.colorbrewer.qualitative import Accent_5 as cmap
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'

from d3_stars.simulations.star_builder import DimensionalMesaReader

mesa_files = [  'gyre/3msol/LOGS/profile43.data',\
                'gyre/15msol/LOGS/profile47.data',\
                'gyre/40msol/LOGS/profile53.data',\
                'gyre/15msol/LOGS/profile47.data',\
              ]

structures = [DimensionalMesaReader(f) for f in mesa_files]

star_files = [ 'other_stars/msol3_twoRcore_re1e4_damping/star/star_256+128_bounds0-2L_Re1.30e+04_de1.5_cutoff1.0e-12.h5',\
               'twoRcore_re3e4_damping/star/star_512+192_bounds0-2L_Re3.00e+04_de1.5_cutoff1.0e-10.h5',\
               'other_stars/msol40_twoRcore_re1e4_damping/star/star_256+128_bounds0-2L_Re7.80e+03_de1.5_cutoff1.0e-10.h5',\
               'compressible_re1e4_waves/star/star_256+192+64_bounds0-0.93R_Re1.00e+04_de1.5_cutoff1.0e-10.h5',\
             ]

star_handles = [h5py.File(f, 'r') for f in star_files]


def get_dedalus_stratification(i, basis_list):
    rs = []
    lnrhos = []
    N2s = []
    grad_s = []
    chi_rad = []
    for bases in basis_list:
        rs.append(star_handles[i]['r_{}'.format(bases)][()]*star_handles[i]['L_nd'][()])
        lnrhos.append(star_handles[i]['ln_rho0_{}'.format(bases)][()])
        N2s.append(-star_handles[i]['g_{}'.format(bases)][2,:]*star_handles[i]['grad_s0_{}'.format(bases)][2,:]/star_handles[i]['Cp'][()] *(star_handles[i]['tau_nd'][()])**(-2))
        grad_s.append(star_handles[i]['grad_s0_{}'.format(bases)][2,:]*star_handles[i]['s_nd'][()]/star_handles[i]['L_nd'][()])
        chi_rad.append(star_handles[i]['kappa_rad_{}'.format(bases)]/(np.exp(star_handles[i]['ln_rho0_{}'.format(bases)][()])*star_handles[i]['Cp'][()]) * star_handles[i]['L_nd'][()]**2 / star_handles[i]['tau_nd'][()] )
    r = np.concatenate([a.ravel() for a in rs])
    ln_rho = np.concatenate([a.ravel() for a in lnrhos])
    rho = np.exp(ln_rho)*star_handles[i]['rho_nd'][()]
    grad_s = np.concatenate([a.ravel() for a in grad_s])
    N2 = np.concatenate([a.ravel() for a in N2s])
    chi_rad = np.concatenate([a.ravel() for a in chi_rad])

    return r, ln_rho, rho, grad_s, N2, chi_rad


fig = plt.figure(figsize=(7.5, 6))
ax1_1 = fig.add_axes([0.100, 0.760, 0.280, 0.220])
ax1_2 = fig.add_axes([0.100, 0.540, 0.280, 0.220])
ax1_3 = fig.add_axes([0.100, 0.320, 0.280, 0.220])
ax1_4 = fig.add_axes([0.100, 0.100, 0.280, 0.220])

ax2_1 = fig.add_axes([0.400, 0.760, 0.280, 0.220])
ax2_2 = fig.add_axes([0.400, 0.540, 0.280, 0.220])
ax2_3 = fig.add_axes([0.400, 0.320, 0.280, 0.220])
ax2_4 = fig.add_axes([0.400, 0.100, 0.280, 0.220])

ax3_1 = fig.add_axes([0.70, 0.760, 0.280, 0.220])
ax3_2 = fig.add_axes([0.70, 0.540, 0.280, 0.220])
ax3_3 = fig.add_axes([0.70, 0.320, 0.280, 0.220])
ax3_4 = fig.add_axes([0.70, 0.100, 0.280, 0.220])

row1_axs = [ax1_1, ax2_1, ax3_1]
row2_axs = [ax1_2, ax2_2, ax3_2]
row3_axs = [ax1_3, ax2_3, ax3_3]
row4_axs = [ax1_4, ax2_4, ax3_4]

axs = row1_axs + row2_axs + row3_axs + row4_axs

ax1_1.set_ylabel(r'$\rho$ (g cm$^{-3}$)')
ax1_2.set_ylabel(r'$N^2$ (s$^{-2}$)')
ax1_3.set_ylabel(r'$\nabla s$ ()')
ax1_4.set_ylabel(r'$\chi_{\rm rad}$ (cm$^{2}$ s$^{-1}$)')

for ax in axs:
    ax.set_yscale('log')


for i in range(len(mesa_files)):
    if i < 3: 
        row1_axs[i].plot(structures[i].structure['r'], structures[i].structure['rho'],      c='k', lw=2, label='MESA')    
        row2_axs[i].plot(structures[i].structure['r'], structures[i].structure['N2'],       c='k', lw=2)    
        row3_axs[i].plot(structures[i].structure['r'], structures[i].structure['grad_s'],   c='k', lw=2)    
        row4_axs[i].plot(structures[i].structure['r'], structures[i].structure['rad_diff'], c='k', lw=2)

        r, ln_rho, rho, grad_s, N2, chi_rad = get_dedalus_stratification(i, ['B', 'S1'])

        row1_axs[i].plot(r, rho,     c=cmap.mpl_colors[0], zorder=10, label='WG sim')    
        row2_axs[i].plot(r, N2,      c=cmap.mpl_colors[0], zorder=10)    
        row3_axs[i].plot(r, grad_s,  c=cmap.mpl_colors[0], zorder=10)    
        row4_axs[i].plot(r, chi_rad, c=cmap.mpl_colors[0], zorder=10)
    else:
        r, ln_rho, rho, grad_s, N2, chi_rad = get_dedalus_stratification(i, ['B', 'S1', 'S2'])

        row1_axs[1].plot(r, rho,     c=cmap.mpl_colors[2], label='WP sim')    
        row2_axs[1].plot(r, N2,      c=cmap.mpl_colors[2])    
        row3_axs[1].plot(r, grad_s,  c=cmap.mpl_colors[2])    
        row4_axs[1].plot(r, chi_rad, c=cmap.mpl_colors[2])


for ax in [ax1_1, ax2_1, ax3_1]:
    ax.set_ylim(1e-10, 1e2)
    ax.set_yticks((1e-8, 1e-6, 1e-4, 1e-2, 1, 1e2))
for ax in [ax1_2, ax2_2, ax3_2]:
    ax.set_ylim(1e-10, 1e-4)
    ax.set_yticks((1e-9, 1e-8, 1e-7, 1e-6, 1e-5, ))
for ax in [ax1_3, ax2_3, ax3_3]:
    ax.set_ylim(1e-8, 1e1)
    ax.set_yticks(( 1e-7, 1e-5, 1e-3, 1e-1 ))
for ax in [ax1_4, ax2_4, ax3_4]:
    ax.set_ylim(1e5, 1e18)
    ax.set_yticks(( 1e6, 1e8, 1e10, 1e12, 1e14, 1e16 ))

for ax in axs[:-3]:
    ax.set_xticklabels(())
for ax in axs[-3:]:
    ax.set_xlabel('radius (cm)')

for i, ax in enumerate(axs):
    if i % 3 == 0: 
        if i < 3:
            ax.text(0.98, 0.92, r'$3\,M_{\odot}$',  ha='right', va='center', transform=ax.transAxes)
        else:
            ax.text(0.02, 0.92, r'$3\,M_{\odot}$',  ha='left', va='center', transform=ax.transAxes)
        continue
    elif (i + 1) % 3 == 0:
        if i < 3:
            ax.text(0.98, 0.92, r'$40\,M_{\odot}$',  ha='right', va='center', transform=ax.transAxes)
        else:
            ax.text(0.02, 0.92, r'$40\,M_{\odot}$',  ha='left', va='center', transform=ax.transAxes)
    elif (i + 2) % 3 == 0:
        if i < 3:
            ax.text(0.98, 0.92, r'$15\,M_{\odot}$',  ha='right', va='center', transform=ax.transAxes)
        else:
            ax.text(0.02, 0.92, r'$15\,M_{\odot}$',  ha='left', va='center', transform=ax.transAxes)
    ax.set_yticklabels(())

row1_axs[1].legend(loc='lower left')




[h.close() for h in star_handles]
fig.savefig('mesa_dedalus_stratification.png', dpi=300)
fig.savefig('mesa_dedalus_stratification.pdf', dpi=300)
