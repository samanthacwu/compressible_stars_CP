import h5py
import numpy as np
import matplotlib.pyplot as plt

with h5py.File('twoRcore_re1e4_damping/wave_flux/wave_luminosities.h5', 'r') as f:
    freqs_256 = f['cgs_freqs'][()]
    ells_256  = f['ells'][()].ravel()
#    power = f['vel_power(r=1.1)'][0,:]
    lum_256 = f['cgs_wave_luminosity(r=1.25)'][0,:]

with h5py.File('twoRcore_re4e3_damping/wave_flux/wave_luminosities.h5', 'r') as f:
    freqs_128 = f['cgs_freqs'][()]
    ells_128  = f['ells'][()].ravel()
    lum_128 = f['cgs_wave_luminosity(r=1.25)'][0,:]

with h5py.File('twoRcore_re1e3_damping/wave_flux/wave_luminosities.h5', 'r') as f:
    freqs_96 = f['cgs_freqs'][()]
    ells_96  = f['ells'][()].ravel()
    lum_96 = f['cgs_wave_luminosity(r=1.25)'][0,:]



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
#for freq in [0.2, 0.3, 0.4, 0.5]:
#    plt.loglog(ells_128, lum_r1_128[freqs_128 > freq, :][0,:], label='128 (r=1.1)')
#    plt.loglog(ells_128, lum_128[freqs_128 > freq, :][0,:], label='128 (r=1.25)')
#    plt.loglog(ells_256, lum_r1_256[freqs_256 > freq, :][0,:], label='256 (r=1.1)')
#    plt.loglog(ells_256, lum_256[freqs_256 > freq, :][0,:], label='256 (r=1.25)')
#    plt.loglog(ells_256, 2.14e-28*freq**(-6.5)*ells_256**2, c='k')
#    plt.ylabel('wave luminosity')
#    plt.xlabel(r'$\ell$')
#    plt.legend()
#    plt.title(r'$f = $' + '{}'.format(freq))
#    plt.show()
#
for ell in range(1, 10):
    plt.loglog(freqs_96, lum_96[:, ells_96 == ell], label='96')
    plt.loglog(freqs_128, lum_128[:, ells_128 == ell], label='128')
    plt.loglog(freqs_256, lum_256[:, ells_256 == ell], label='256')
#    plt.loglog(freqs_256, 2.14e-28*freqs_256**(-6.5)*ell**2, c='k')
    plt.loglog(freqs_256, 1e-15*freqs_256**(-7.5)*ell**3, c='k')
    plt.xlabel('freq')
    plt.ylabel('wave luminosity')
    plt.xlim(1e-6, 1e-4)
    plt.ylim(1e10, 1e30)
    plt.legend()
    plt.title(r'$\ell = $' + '{}'.format(ell))
    plt.axvline(0.2)
    plt.axvline(0.5)
    plt.show()
