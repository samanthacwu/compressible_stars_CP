import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import mesa_reader as mr
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'



mesa_LOG = '../d3_stars/stock_models/zams_15Msol/LOGS/profile47.data'
p = mr.MesaData(mesa_LOG)
Lstar = p.header_data['photosphere_L']*p.header_data['lsun']
print('stellar lum: {:.2e}'.format(Lstar))

rv = '1.25'
hz_to_invday = 24*60*60

with h5py.File('twoRcore_re3e4_damping/wave_flux/wave_luminosities.h5', 'r') as f:
    freqs_512 = f['cgs_freqs'][()]
    ells_512  = f['ells'][()].ravel()
#    power = f['vel_power(r=1.1)'][0,:]
    lum_512 = f['cgs_wave_luminosity(r={})'.format(rv)][0,:]

with h5py.File('twoRcore_re2e4_damping/wave_flux/wave_luminosities.h5', 'r') as f:
    freqs_384 = f['cgs_freqs'][()]
    ells_384  = f['ells'][()].ravel()
#    power = f['vel_power(r=1.1)'][0,:]
    lum_384 = f['cgs_wave_luminosity(r={})'.format(rv)][0,:]

with h5py.File('twoRcore_re1e4_damping/wave_flux/wave_luminosities.h5', 'r') as f:
    freqs_256 = f['cgs_freqs'][()]
    ells_256  = f['ells'][()].ravel()
#    power = f['vel_power(r=1.1)'][0,:]
    lum_256 = f['cgs_wave_luminosity(r={})'.format(rv)][0,:]

with h5py.File('twoRcore_re2e3_damping/wave_flux/wave_luminosities.h5', 'r') as f:
    freqs_128_low = f['cgs_freqs'][()]
    ells_128_low  = f['ells'][()].ravel()
    lum_128_low = f['cgs_wave_luminosity(r={})'.format(rv)][0,:]

with h5py.File('twoRcore_re4e3_damping/wave_flux/wave_luminosities.h5', 'r') as f:
    freqs_128 = f['cgs_freqs'][()]
    ells_128  = f['ells'][()].ravel()
    lum_128 = f['cgs_wave_luminosity(r={})'.format(rv)][0,:]

with h5py.File('twoRcore_re1e3_damping/wave_flux/wave_luminosities.h5', 'r') as f:
    freqs_96 = f['cgs_freqs'][()]
    ells_96  = f['ells'][()].ravel()
    lum_96 = f['cgs_wave_luminosity(r={})'.format(rv)][0,:]


freqs_512 *= hz_to_invday
freqs_384 *= hz_to_invday
freqs_256 *= hz_to_invday
freqs_128 *= hz_to_invday
freqs_128_low *= hz_to_invday
freqs_96 *= hz_to_invday




from palettable.colorbrewer.sequential import RdPu_7
cmap = RdPu_7.mpl_colors[1:]
for freq in np.array([3e-6, 5e-6, 1e-5])*hz_to_invday:
    print('f = {}'.format(freq))
    plt.loglog(ells_96, lum_96[freqs_96 > freq, :][0,:],                 color=cmap[0], label=r'Re $\sim$ 200')
    plt.loglog(ells_128_low, lum_128_low[freqs_128_low > freq, :][0,:],  color=cmap[1], label=r'Re $\sim$ 400')
    plt.loglog(ells_128, lum_128[freqs_128 > freq, :][0,:],              color=cmap[2], label=r'Re $\sim$ 800')
    plt.loglog(ells_256, lum_256[freqs_256 > freq, :][0,:],              color=cmap[3], label=r'Re $\sim$ 2000')
    plt.loglog(ells_384, lum_384[freqs_384 > freq, :][0,:],              color=cmap[4], label=r'Re $\sim$ 4000')
    plt.loglog(ells_512, lum_512[freqs_512 > freq, :][0,:],              color=cmap[5], label=r'Re $\sim$ 6000')
    kh = np.sqrt(ells_512*(ells_512+1))
    plt.loglog(ells_512, 3e-11*(freq/hz_to_invday)**(-6.5)*kh**4, c='k', label=r'$(3\times10^{-11})(f/Hz)^{-6.5}\left[\sqrt{\ell(\ell+1)}\,\right]^{4}$')
    plt.xlim(1e-2, 1e1)
    plt.ylabel('wave luminosity')
    plt.xlabel(r'$\ell$')
    plt.legend()
    plt.title(r'$f = $' + '{}'.format(freq))
    plt.savefig('wave_luminosity_comparison/freq{:0.2e}.png'.format(freq))
    plt.clf()

for ell in range(1, 4):
    print('ell = {}'.format(ell))
    plt.loglog(freqs_96,      np.abs(lum_96[:, ells_96 == ell]),           color=cmap[0], label=r'Re $\sim$ 200')
    plt.loglog(freqs_128_low, np.abs(lum_128_low[:, ells_128_low == ell]), color=cmap[1], label=r'Re $\sim$ 400')
    plt.loglog(freqs_128,     np.abs(lum_128[:, ells_128 == ell]),         color=cmap[2], label=r'Re $\sim$ 800')
    plt.loglog(freqs_256,     np.abs(lum_256[:, ells_256 == ell]),         color=cmap[3], label=r'Re $\sim$ 2000')
    plt.loglog(freqs_384,     np.abs(lum_384[:, ells_384 == ell]),         color=cmap[4], label=r'Re $\sim$ 4000')
    plt.loglog(freqs_512,     np.abs(lum_512[:, ells_512 == ell]),         color=cmap[5], label=r'Re $\sim$ 6000')
#    plt.loglog(freqs_256, 2.14e-28*freqs_256**(-6.5)*ell**2, c='k')
    kh = np.sqrt(ell*(ell+1))
    plt.loglog(freqs_256, 3e-11*(freqs_256/hz_to_invday)**(-6.5)*kh**4, c='k', label=r'$(3\times10^{-11})(f/Hz)^{-6.5}\left[\sqrt{\ell(\ell+1)}\,\right]^{4}$')
    plt.xlabel('freq')
    plt.ylabel('wave luminosity')
    plt.xlim(1e-2, 1e1)
    plt.ylim(1e10, 1e30)
    plt.legend()
    plt.title(r'$\ell = $' + '{}'.format(ell))
    plt.savefig('wave_luminosity_comparison/ell{:03d}.png'.format(ell))
    plt.clf()
#    plt.show()



lum_512     /= Lstar
lum_384     /= Lstar
lum_256     /= Lstar
lum_128     /= Lstar
lum_128_low /= Lstar
lum_96      /= Lstar

fig = plt.figure(figsize=(7.5, 2.5))
ax1 = fig.add_axes([0, 0, 0.45, 0.88])
ax2 = fig.add_axes([0.55, 0, 0.49, 0.88])
cax = fig.add_axes([0.25, 0.93, 0.50, 0.07])

bounds = [100, 200, 400, 800, 2000, 4000, 6000]
norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=6)
listed_cmap = mpl.colors.ListedColormap(cmap)
sm = mpl.cm.ScalarMappable(norm=norm, cmap=listed_cmap)

ell = 1
freq = 0.8
ax1.loglog(freqs_96,      np.abs(lum_96[:, ells_96 == ell]),           color=cmap[0], label=r'Re $\sim$ 200',  zorder=1, lw=0.5)
ax1.loglog(freqs_128_low, np.abs(lum_128_low[:, ells_128_low == ell]), color=cmap[1], label=r'Re $\sim$ 400',  zorder=2, lw=0.5)
ax1.loglog(freqs_128,     np.abs(lum_128[:, ells_128 == ell]),         color=cmap[2], label=r'Re $\sim$ 800',  zorder=3, lw=0.5)
ax1.loglog(freqs_256,     np.abs(lum_256[:, ells_256 == ell]),         color=cmap[3], label=r'Re $\sim$ 2000', zorder=4, lw=0.5)
ax1.loglog(freqs_384,     np.abs(lum_384[:, ells_384 == ell]),         color=cmap[4], label=r'Re $\sim$ 4000', zorder=5, lw=0.5)
ax1.loglog(freqs_512,     np.abs(lum_512[:, ells_512 == ell]),         color=cmap[5], label=r'Re $\sim$ 6000', zorder=6, lw=0.5)
kh = np.sqrt(ell*(ell+1))
ax1.loglog(freqs_256, (1/Lstar)*3e-11*(freqs_256/hz_to_invday)**(-6.5)*kh**4, c='k', label=r'$(3\times10^{-11})(f/Hz)^{-6.5}\left[\sqrt{\ell(\ell+1)}\,\right]^{4}$', zorder=7)
ax1.set_ylim(1e-29, 1e-8)
ax1.set_xlim(1e-2, 2e1)
ax1.text(0.15, 2e28/Lstar, r'$f^{-6.5}$', rotation=0)
ax1.set_xlabel(r'$f$ (d$^{-1}$)')
ax1.set_ylabel(r'$L_{w}/L_{*}$')
#ax1.set_ylabel(r'$L_{\rm wave}/L_{*}$ ($L_{*} = 7.45 \times 10^{37}\,$ erg$\,$s$^{-1}$)')

ax2.loglog(ells_96, lum_96[freqs_96 > freq, :][0,:],                 color=cmap[0], label=r'Re $\sim$ 200',  zorder=1, lw=1)
ax2.loglog(ells_128_low, lum_128_low[freqs_128_low > freq, :][0,:],  color=cmap[1], label=r'Re $\sim$ 400',  zorder=2, lw=1)
ax2.loglog(ells_128, lum_128[freqs_128 > freq, :][0,:],              color=cmap[2], label=r'Re $\sim$ 800',  zorder=3, lw=1)
ax2.loglog(ells_256, lum_256[freqs_256 > freq, :][0,:],              color=cmap[3], label=r'Re $\sim$ 2000', zorder=4, lw=1)
ax2.loglog(ells_384, lum_384[freqs_384 > freq, :][0,:],              color=cmap[4], label=r'Re $\sim$ 4000', zorder=5, lw=1)
ax2.loglog(ells_512, lum_512[freqs_512 > freq, :][0,:],              color=cmap[5], label=r'Re $\sim$ 6000', zorder=6, lw=1)
kh = np.sqrt(ells_384*(ells_384+1))
ax2.loglog(ells_384, (1/Lstar)*3e-11*(freq/hz_to_invday)**(-6.5)*kh**4, c='k', label=r'$(3\times10^{-11})(f/Hz)^{-6.5}\left[\sqrt{\ell(\ell+1)}\,\right]^{4}$', zorder=7)
ax2.set_xlim(1, 100)
ax2.set_ylim(1e-29, 1e-8)
ax2.text(15, 3e27/Lstar, r'$k_h^4=[\ell(\ell+1)]^2$', rotation=0,ha='right')
ax2.set_xlabel(r'$\ell$')
ax2.set_ylabel(r'$L_{w}/L_{*}$')

for i, Re in enumerate([200, 400, 800, 2000, 4000, 6000]):
    if Re >= 1000:
        color='lightgrey'
    else:
        color='k'
    cax.text((1/12)+i/6, 0.4, Re, ha='center', va='center', transform=cax.transAxes, color=color)

cb = plt.colorbar(sm, cax=cax, orientation='horizontal')
cax.text(-0.02, 0.5, 'Re', ha='right', va='center', transform=cax.transAxes)
cb.set_ticks(())

ax1.text(0.99, 0.98, r'$\ell = 1$', ha='right', va='top', transform=ax1.transAxes)
ax2.text(0.01, 0.02, r'$f = 0.8$ (d$^{-1}$)', ha='left', va='bottom', transform=ax2.transAxes)

plt.savefig('wave_luminosity_comparison/turbulence_waveflux_variation.png', dpi=300, bbox_inches='tight')
plt.savefig('wave_luminosity_comparison/turbulence_waveflux_variation.pdf', dpi=300, bbox_inches='tight')
