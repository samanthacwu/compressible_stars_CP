import h5py
import numpy as np
import matplotlib.pyplot as plt

rv = '1.25'

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



#Plot up power of CZ velocities.
#ells, freqs = np.meshgrid(ells_256, freqs_256)
#print(ells.shape, freqs.shape, np.log10(power).max())
#plt.pcolormesh(freqs, ells, np.log10(power), vmin=-40, vmax=-6, cmap='tab20b')
#plt.axhline(128)
#plt.xscale('log')
#plt.yscale('log')
#plt.ylim(1, 256)
#plt.xlim(1e-4, 1)
#plt.show()

#power_v_ell = np.sum(power, axis=0)
#plt.loglog(ells_256, power_v_ell)
#plt.loglog(ells_256, power_v_ell[1]*ells_256**(-5/3))
#plt.axvline(128)
#plt.show()
#

from palettable.colorbrewer.sequential import RdPu_5
for freq in [3e-6, 5e-6, 1e-5]:
    print('f = {}'.format(freq))
    plt.loglog(ells_96, lum_96[freqs_96 > freq, :][0,:],                 color=RdPu_5.mpl_colors[0], label=r'Re $\sim$ 200')
    plt.loglog(ells_128_low, lum_128_low[freqs_128_low > freq, :][0,:],  color=RdPu_5.mpl_colors[1], label=r'Re $\sim$ 400')
    plt.loglog(ells_128, lum_128[freqs_128 > freq, :][0,:],              color=RdPu_5.mpl_colors[2], label=r'Re $\sim$ 800')
    plt.loglog(ells_256, lum_256[freqs_256 > freq, :][0,:],              color=RdPu_5.mpl_colors[3], label=r'Re $\sim$ 2000')
    plt.loglog(ells_384, lum_384[freqs_384 > freq, :][0,:],              color=RdPu_5.mpl_colors[4], label=r'Re $\sim$ 4000')
    kh = np.sqrt(ells_384*(ells_384+1))
    plt.loglog(ells_384, 3e-11*freq**(-6.5)*kh**4, c='k', label=r'$(3\times10^{-11})f^{-6.5}\left[\sqrt{\ell(\ell+1)}\,\right]^{4}$')
    plt.ylabel('wave luminosity')
    plt.xlabel(r'$\ell$')
    plt.legend()
    plt.title(r'$f = $' + '{}'.format(freq))
    plt.savefig('wave_luminosity_comparison/freq{:0.2e}.png'.format(freq))
    plt.clf()

for ell in range(1, 4):
    print('ell = {}'.format(ell))
    plt.loglog(freqs_96,      np.abs(lum_96[:, ells_96 == ell]),           color=RdPu_5.mpl_colors[0], label=r'Re $\sim$ 200')
    plt.loglog(freqs_128_low, np.abs(lum_128_low[:, ells_128_low == ell]), color=RdPu_5.mpl_colors[1], label=r'Re $\sim$ 400')
    plt.loglog(freqs_128,     np.abs(lum_128[:, ells_128 == ell]),         color=RdPu_5.mpl_colors[2], label=r'Re $\sim$ 800')
    plt.loglog(freqs_256,     np.abs(lum_256[:, ells_256 == ell]),         color=RdPu_5.mpl_colors[3], label=r'Re $\sim$ 2000')
    plt.loglog(freqs_384,     np.abs(lum_384[:, ells_384 == ell]),         color=RdPu_5.mpl_colors[4], label=r'Re $\sim$ 4000')
#    plt.loglog(freqs_256, 2.14e-28*freqs_256**(-6.5)*ell**2, c='k')
    kh = np.sqrt(ell*(ell+1))
    plt.loglog(freqs_256, 3e-11*freqs_256**(-6.5)*kh**4, c='k', label=r'$(3\times10^{-11})f^{-6.5}\left[\sqrt{\ell(\ell+1)}\,\right]^{4}$')
    plt.xlabel('freq')
    plt.ylabel('wave luminosity')
    plt.xlim(1e-6, 1e-4)
    plt.ylim(1e10, 1e30)
    plt.legend()
    plt.title(r'$\ell = $' + '{}'.format(ell))
    plt.axvline(0.2)
    plt.axvline(0.5)
    plt.savefig('wave_luminosity_comparison/ell{:03d}.png'.format(ell))
    plt.clf()
#    plt.show()
