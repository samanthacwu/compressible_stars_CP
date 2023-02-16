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

with h5py.File('rot10day_twoRcore_re1e4_damping/wave_flux/wave_luminosities.h5', 'r') as f:
    freqs_10day = f['cgs_freqs'][()]*hz_to_invday
    ells_10day  = f['ells'][()].ravel()
    lum_10day = f['cgs_wave_luminosity(r={})'.format(rv)][0,:]

with h5py.File('rot3day_twoRcore_re1e4_damping/wave_flux/wave_luminosities.h5', 'r') as f:
    freqs_3day = f['cgs_freqs'][()]*hz_to_invday
    ells_3day  = f['ells'][()].ravel()
    lum_3day = f['cgs_wave_luminosity(r={})'.format(rv)][0,:]

with h5py.File('rot1day_twoRcore_re1e4_damping/wave_flux/wave_luminosities.h5', 'r') as f:
    freqs_1day = f['cgs_freqs'][()]*hz_to_invday
    ells_1day  = f['ells'][()].ravel()
    lum_1day = f['cgs_wave_luminosity(r={})'.format(rv)][0,:]



#Plot up power of CZ velocities.
#ells, freqs = np.meshgrid(ells_nonrot, freqs_nonrot)
#print(ells.shape, freqs.shape, np.log10(power).max())
#plt.pcolormesh(freqs, ells, np.log10(power), vmin=-40, vmax=-6, cmap='tab20b')
#plt.axhline(3day)
#plt.xscale('log')
#plt.yscale('log')
#plt.ylim(1, nonrot)
#plt.xlim(1e-4, 1)
#plt.show()

#power_v_ell = np.sum(power, axis=0)
#plt.loglog(ells_nonrot, power_v_ell)
#plt.loglog(ells_nonrot, power_v_ell[1]*ells_nonrot**(-5/3))
#plt.axvline(3day)
#plt.show()
#

from palettable.colorbrewer.sequential import YlGnBu_7_r 
for freq in np.array([3e-6, 5e-6, 1e-5])*hz_to_invday:
    print('f = {}'.format(freq))
    plt.loglog(ells_1day, lum_1day[freqs_1day > freq, :][0,:],                 color=YlGnBu_7_r.mpl_colors[3], label=r'P = 1 day')
    plt.loglog(ells_10day, lum_10day[freqs_10day > freq, :][0,:],  color=YlGnBu_7_r.mpl_colors[1], label=r'P = 10 days')
    plt.loglog(ells_3day, lum_3day[freqs_3day > freq, :][0,:],              color=YlGnBu_7_r.mpl_colors[2], label=r'P = 3 days')
    plt.loglog(ells_nonrot, lum_nonrot[freqs_nonrot > freq, :][0,:],              color=YlGnBu_7_r.mpl_colors[0], label=r'Nonrotating')
    kh = np.sqrt(ells_nonrot*(ells_nonrot+1))
    plt.loglog(ells_nonrot, 3e-11*(freq/hz_to_invday)**(-6.5)*kh**4, c='k', label=r'$(3\times10^{-11})f^{-6.5}\left[\sqrt{\ell(\ell+1)}\,\right]^{4}$')
    plt.ylabel('wave luminosity')
    plt.xlabel(r'$\ell$')
    plt.legend()
    plt.title(r'$f = $' + '{}'.format(freq))
    plt.savefig('wave_luminosity_comparison/rotating_freq{:0.2e}.png'.format(freq))
    plt.clf()

for ell in range(1, 4):
    print('ell = {}'.format(ell))
    plt.loglog(freqs_1day,      np.abs(lum_1day[:, ells_1day == ell]),           color=YlGnBu_7_r.mpl_colors[3], label=r'P = 1 day')
    plt.loglog(freqs_10day, np.abs(lum_10day[:, ells_10day == ell]), color=YlGnBu_7_r.mpl_colors[1], label=r'P = 10 days')
    plt.loglog(freqs_3day,     np.abs(lum_3day[:, ells_3day == ell]),         color=YlGnBu_7_r.mpl_colors[2], label=r'P = 3 days')
    plt.loglog(freqs_nonrot,     np.abs(lum_nonrot[:, ells_nonrot == ell]),         color=YlGnBu_7_r.mpl_colors[0], label=r'Nonrotating')
#    plt.loglog(freqs_nonrot, 2.14e-28*freqs_nonrot**(-6.5)*ell**2, c='k')
    kh = np.sqrt(ell*(ell+1))
    plt.loglog(freqs_nonrot, 3e-11*(freqs_nonrot/hz_to_invday)**(-6.5)*kh**4, c='k', label=r'$(3\times10^{-11})(f/Hz)^{-6.5}\left[\sqrt{\ell(\ell+1)}\,\right]^{4}$')
    plt.xlabel('freq (1/day)')
    plt.ylabel('wave luminosity')
    plt.xlim(1e-2, 1e1)
    plt.ylim(1e20, 1e35)
    plt.legend()
    plt.title(r'$\ell = $' + '{}'.format(ell))
    plt.savefig('wave_luminosity_comparison/rotating_ell{:03d}.png'.format(ell))
    plt.clf()
#    plt.show()
