import h5py
import numpy as np
import matplotlib.pyplot as plt

rv = '1.25'

hz_to_invday = 24*60*60

with h5py.File('twoRcore_re1e4_damping/wave_flux/wave_luminosities.h5', 'r') as f:
    freqs_nonrot = f['cgs_freqs'][()]*hz_to_invday
    ells_nonrot  = f['ells'][()].ravel()
#    power = f['vel_power(r=1.1)'][0,:]
    lum_nonrot = f['cgs_wave_luminosity(r={})'.format(rv)][0,:]

with h5py.File('other_stars/msol40_twoRcore_re1e4_damping/wave_flux/wave_luminosities.h5', 'r') as f:
    freqs_msol40 = f['cgs_freqs'][()]*hz_to_invday
    ells_msol40  = f['ells'][()].ravel()
    lum_msol40 = f['cgs_wave_luminosity(r={})'.format(rv)][0,:]

with h5py.File('other_stars/msol3_twoRcore_re1e4_damping/wave_flux/wave_luminosities.h5', 'r') as f:
    freqs_msol3 = f['cgs_freqs'][()]*hz_to_invday
    ells_msol3  = f['ells'][()].ravel()
    lum_msol3 = f['cgs_wave_luminosity(r={})'.format(rv)][0,:]



#Plot up power of CZ velocities.
#ells, freqs = np.meshgrid(ells_nonrot, freqs_nonrot)
#print(ells.shape, freqs.shape, np.log10(power).max())
#plt.pcolormesh(freqs, ells, np.log10(power), vmin=-40, vmax=-6, cmap='tab20b')
#plt.xscale('log')
#plt.yscale('log')
#plt.ylim(1, nonrot)
#plt.xlim(1e-4, 1)
#plt.show()

#power_v_ell = np.sum(power, axis=0)
#plt.loglog(ells_nonrot, power_v_ell)
#plt.loglog(ells_nonrot, power_v_ell[1]*ells_nonrot**(-5/3))
#plt.show()
#

from palettable.colorbrewer.sequential import Oranges_7_r 
for freq in np.array([3e-6, 5e-6, 1e-5])*hz_to_invday:
    print('f = {}'.format(freq))
    plt.loglog(ells_msol3, lum_msol3[freqs_msol3 > freq, :][0,:],                 color=Oranges_7_r.mpl_colors[3], label=r'3 $M_{\odot}$')
    plt.loglog(ells_nonrot, lum_nonrot[freqs_nonrot > freq, :][0,:],              color=Oranges_7_r.mpl_colors[1], label=r'15 $M_{\odot}$')
    plt.loglog(ells_msol40, lum_msol40[freqs_msol40 > freq, :][0,:],  color=Oranges_7_r.mpl_colors[0], label=r'40 $M_{\odot}$')
    kh = np.sqrt(ells_nonrot*(ells_nonrot+1))
    plt.loglog(ells_nonrot, 3e-15*(freq/hz_to_invday)**(-6.5)*kh**4, c='lightgrey', label=r'$(3\times10^{-15})f^{-6.5}\left[\sqrt{\ell(\ell+1)}\,\right]^{4}$')
    plt.loglog(ells_nonrot, 3e-11*(freq/hz_to_invday)**(-6.5)*kh**4, c='grey', label=r'$(3\times10^{-11})f^{-6.5}\left[\sqrt{\ell(\ell+1)}\,\right]^{4}$')
    plt.loglog(ells_nonrot, 3e-10*(freq/hz_to_invday)**(-6.5)*kh**4, c='k', label=r'$(3\times10^{-10})f^{-6.5}\left[\sqrt{\ell(\ell+1)}\,\right]^{4}$')
    plt.ylabel('wave luminosity')
    plt.xlabel(r'$\ell$')
    plt.legend()
    plt.title(r'$f = $' + '{}'.format(freq))
    plt.savefig('wave_luminosity_comparison/mass_freq{:0.2e}.png'.format(freq))
    plt.clf()

for ell in range(1, 4):
    print('ell = {}'.format(ell))
    plt.loglog(freqs_msol3,      np.abs(lum_msol3[:, ells_msol3 == ell]),           color=Oranges_7_r.mpl_colors[3], label=r'3 $M_{\odot}$')
    plt.loglog(freqs_nonrot,     np.abs(lum_nonrot[:, ells_nonrot == ell]),         color=Oranges_7_r.mpl_colors[1], label=r'$15 M_{\odot}$')
    plt.loglog(freqs_msol40, np.abs(lum_msol40[:, ells_msol40 == ell]), color=Oranges_7_r.mpl_colors[0], label=r'40 $M_{\odot}$')
#    plt.loglog(freqs_nonrot, 2.14e-28*freqs_nonrot**(-6.5)*ell**2, c='k')
    kh = np.sqrt(ell*(ell+1))
    plt.loglog(freqs_nonrot, 3e-15*(freqs_nonrot/hz_to_invday)**(-6.5)*kh**4, c='lightgrey', label=r'$(3\times10^{-15})(f/Hz)^{-6.5}\left[\sqrt{\ell(\ell+1)}\,\right]^{4}$')
    plt.loglog(freqs_nonrot, 3e-11*(freqs_nonrot/hz_to_invday)**(-6.5)*kh**4, c='grey', label=r'$(3\times10^{-11})(f/Hz)^{-6.5}\left[\sqrt{\ell(\ell+1)}\,\right]^{4}$')
    plt.loglog(freqs_nonrot, 3e-10*(freqs_nonrot/hz_to_invday)**(-6.5)*kh**4, c='k', label=r'$(3\times10^{-10})(f/Hz)^{-6.5}\left[\sqrt{\ell(\ell+1)}\,\right]^{4}$')
    plt.xlabel('freq (1/day)')
    plt.ylabel('wave luminosity')
    plt.xlim(1e-2, 1e1)
    plt.ylim(1e12, 1e32)
    plt.legend()
    plt.title(r'$\ell = $' + '{}'.format(ell))
    plt.savefig('wave_luminosity_comparison/mass_ell{:03d}.png'.format(ell))
    plt.clf()
#    plt.show()
