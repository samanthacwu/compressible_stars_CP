import numpy as np
import matplotlib.pyplot as plt

def mass_temp_relation(temp):
    """ 
        log Teff = −0.170(026) × (log M)2 + 0.888(037) × log M + 3.671(010) 
        Table 5 https://academic.oup.com/mnras/article/479/4/5491/5056185
    """
    A = -0.170
    B = 0.888
    C = 3.671 - np.log10(temp)
    root1 = (-B + np.sqrt(B**2 - 4*A*C) ) / (2 * A)
    root2 = (-B - np.sqrt(B**2 - 4*A*C) ) / (2 * A)
    return 10**root1, 10**root2

sim_mass = [3, 15, 40]
sim_alpha = [2.5e-3, 5e-2, 8e-2]
sim_nuchar = np.array([2e-5, 1.5e-5, 7e-6])*24*60*60 #Hz; where it has dropped by factor of 2.
sim_nudamp = np.array([1.4e-6, 9e-7, 5.5e-7])*24*60*60 #Hz; where it has dropped by factor of 2.

fig = plt.figure(figsize=(7.5, 3))
ax1 = fig.add_axes([0.02, 0.02, 0.25, 0.8])
ax2 = fig.add_axes([0.37, 0.02, 0.25, 0.8])
ax3 = fig.add_axes([0.75, 0.02, 0.25, 0.8])
axs = [ax1, ax2, ax3]

data1 = np.genfromtxt('bowman_a1.csv', delimiter=',', skip_header=1, dtype=str)
data2 = np.genfromtxt('bowman_a2.csv', delimiter=',', skip_header=1, dtype=str)

log10Teff = np.array(data1[:,3], dtype=np.float64)
log10Ell  = np.array(data1[:,4], dtype=np.float64)
alpha0    = np.array(data2[:,2], dtype=np.float64)
nuchar    = np.array(data2[:,4], dtype=np.float64)

mass, badmass = mass_temp_relation(10**log10Teff)
print(mass)

for ax in axs:
    ax.set_xlabel(r'$M/M_*$')
ax1.set_ylabel(r'$\alpha_{0}$ ($\mu$mag)')
ax2.set_ylabel(r'$\nu_{\rm char}$ (d$^{-1}$)')
ax3.set_ylabel(r'$\nu_{\rm damp}$ (d$^{-1}$)')

ax1.loglog(mass, alpha0, lw=0, color='k', marker='o')
ax2.plot(mass, nuchar, lw=0,   color='k', marker='o')
ax1.plot(sim_mass, sim_alpha,  color='orange', lw=0, marker='*')
ax2.plot(sim_mass, sim_nuchar, color='orange', lw=0, marker='*')
ax3.plot(sim_mass, sim_nudamp, color='orange', lw=0, marker='*')

fig.savefig('scatter_plots.png', dpi=300, bbox_inches='tight')

