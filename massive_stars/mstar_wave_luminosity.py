import h5py
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'

from d3_stars.simulations.star_builder import find_core_cz_radius
from astropy import units as u
from astropy import constants
import mesa_reader as mr


def read_mesa(mesa_file_path):
    file_dict = dict()

    print("Reading MESA file {}".format(mesa_file_path))
    p = mr.MesaData(mesa_file_path)
    file_dict['mass']           = (p.mass[::-1] * u.M_sun).cgs
    file_dict['r']              = (p.radius[::-1] * u.R_sun).cgs
    file_dict['rho']            = 10**p.logRho[::-1] * u.g / u.cm**3
    file_dict['P']              = p.pressure[::-1] * u.g / u.cm / u.s**2
    file_dict['T']              = p.temperature[::-1] * u.K
    file_dict['R_gas']          = eval('P / (rho * T)', file_dict)
    file_dict['cp']             = p.cp[::-1]  * u.erg / u.K / u.g
    file_dict['opacity']        = p.opacity[::-1] * (u.cm**2 / u.g)
    file_dict['Luminosity']     = (p.luminosity[::-1] * u.L_sun).cgs
    file_dict['conv_L_div_L']   = p.lum_conv_div_L[::-1]
    file_dict['csound']         = p.csound[::-1] * u.cm / u.s
    file_dict['N2'] = N2_mesa   = p.brunt_N2[::-1] / u.s**2
    file_dict['N2_structure']   = p.brunt_N2_structure_term[::-1] / u.s**2
    file_dict['N2_composition'] = p.brunt_N2_composition_term[::-1] / u.s**2
    file_dict['lamb_freq'] = lambda ell : np.sqrt(ell*(ell + 1)) * csound/r
    file_dict['Lstar'] = file_dict['Luminosity'].max()
    file_dict['Rcore'] = find_core_cz_radius(mesa_file_path)*u.cm
    r = file_dict['r']
    L = file_dict['Luminosity']
    Lconv = file_dict['Luminosity']*file_dict['conv_L_div_L']
    rho = file_dict['rho']
    cs = file_dict['csound']
    good = r.value < file_dict['Rcore'].value
    file_dict['Lconv_cz'] = np.sum((4*np.pi*r**2*np.gradient(r)*Lconv)[good])/(4*np.pi*file_dict['Rcore']**3/3)
    file_dict['rho_cz'] = np.sum((4*np.pi*r**2*np.gradient(r)*rho)[good])/(4*np.pi*file_dict['Rcore']**3/3)
    file_dict['u_cz'] = np.sum((4*np.pi*r**2*np.gradient(r)*(Lconv/(4*np.pi*r**2*rho))**(1/3))[good])/(4*np.pi*file_dict['Rcore']**3/3)
    file_dict['cs_cz'] = np.sum((4*np.pi*r**2*np.gradient(r)*cs)[good])/(4*np.pi*file_dict['Rcore']**3/3)
    file_dict['Ma_cz'] = eval('u_cz/cs_cz', file_dict)
    file_dict['f_cz'] = eval('u_cz/Rcore', file_dict)
    print(file_dict['Ma_cz'].cgs, file_dict['f_cz'].cgs)
    return file_dict



rv = '1.25'
hz_to_invday = 24*60*60

with h5py.File('twoRcore_re1e4_damping/wave_flux/wave_luminosities.h5', 'r') as f:
    freqs_msol15 = f['cgs_freqs'][()]*hz_to_invday
    ells_msol15  = f['ells'][()].ravel()
#    power = f['vel_power(r=1.1)'][0,:]
    lum_msol15 = f['cgs_wave_luminosity(r={})'.format(rv)][0,:]
    msol15_file_dict = read_mesa('gyre/15msol/LOGS/profile47.data')

with h5py.File('other_stars/msol40_twoRcore_re1e4_damping/wave_flux/wave_luminosities.h5', 'r') as f:
    freqs_msol40 = f['cgs_freqs'][()]*hz_to_invday
    ells_msol40  = f['ells'][()].ravel()
    lum_msol40 = f['cgs_wave_luminosity(r={})'.format(rv)][0,:]
    msol40_file_dict = read_mesa('gyre/40msol/LOGS/profile53.data')

with h5py.File('other_stars/msol3_twoRcore_re1e4_damping/wave_flux/wave_luminosities.h5', 'r') as f:
    freqs_msol3 = f['cgs_freqs'][()]*hz_to_invday
    ells_msol3  = f['ells'][()].ravel()
    lum_msol3 = f['cgs_wave_luminosity(r={})'.format(rv)][0,:]
    msol3_file_dict = read_mesa('gyre/3msol/LOGS/profile43.data')



#Plot up power of CZ velocities.
#ells, freqs = np.meshgrid(ells_msol15, freqs_msol15)
#print(ells.shape, freqs.shape, np.log10(power).max())
#plt.pcolormesh(freqs, ells, np.log10(power), vmin=-40, vmax=-6, cmap='tab20b')
#plt.xscale('log')
#plt.yscale('log')
#plt.ylim(1, msol15)
#plt.xlim(1e-4, 1)
#plt.show()

#power_v_ell = np.sum(power, axis=0)
#plt.loglog(ells_msol15, power_v_ell)
#plt.loglog(ells_msol15, power_v_ell[1]*ells_msol15**(-5/3))
#plt.show()
#

from palettable.colorbrewer.sequential import Oranges_7_r 
#for freq in np.array([3e-6, 5e-6, 1e-5])*hz_to_invday:
#    print('f = {}'.format(freq))
#    plt.loglog(ells_msol3, lum_msol3[freqs_msol3 > freq, :][0,:],                 color=Oranges_7_r.mpl_colors[3], label=r'3 $M_{\odot}$')
#    plt.loglog(ells_msol15, lum_msol15[freqs_msol15 > freq, :][0,:],              color=Oranges_7_r.mpl_colors[1], label=r'15 $M_{\odot}$')
#    plt.loglog(ells_msol40, lum_msol40[freqs_msol40 > freq, :][0,:],  color=Oranges_7_r.mpl_colors[0], label=r'40 $M_{\odot}$')
#    kh = np.sqrt(ells_msol15*(ells_msol15+1))
#    plt.loglog(ells_msol15, 3e-15*(freq/hz_to_invday)**(-6.5)*kh**4, c='lightgrey', label=r'$(3\times10^{-15})f^{-6.5}\left[\sqrt{\ell(\ell+1)}\,\right]^{4}$')
#    plt.loglog(ells_msol15, 3e-11*(freq/hz_to_invday)**(-6.5)*kh**4, c='lightgrey', label=r'$(3\times10^{-11})f^{-6.5}\left[\sqrt{\ell(\ell+1)}\,\right]^{4}$')
#    plt.loglog(ells_msol15, 3e-10*(freq/hz_to_invday)**(-6.5)*kh**4, c='k', label=r'$(3\times10^{-10})f^{-6.5}\left[\sqrt{\ell(\ell+1)}\,\right]^{4}$')
#    plt.ylabel('wave luminosity')
#    plt.xlabel(r'$\ell$')
#    plt.legend()
#    plt.title(r'$f = $' + '{}'.format(freq))
#    plt.savefig('wave_luminosity_comparison/mass_freq{:0.2e}.png'.format(freq))
#    plt.clf()
#
#for ell in range(1, 4):
#    print('ell = {}'.format(ell))
#    plt.loglog(freqs_msol3,      np.abs(lum_msol3[:, ells_msol3 == ell]),           color=Oranges_7_r.mpl_colors[3], label=r'3 $M_{\odot}$')
#    plt.loglog(freqs_msol15,     np.abs(lum_msol15[:, ells_msol15 == ell]),         color=Oranges_7_r.mpl_colors[1], label=r'$15 M_{\odot}$')
#    plt.loglog(freqs_msol40, np.abs(lum_msol40[:, ells_msol40 == ell]), color=Oranges_7_r.mpl_colors[0], label=r'40 $M_{\odot}$')
##    plt.loglog(freqs_msol15, 2.14e-28*freqs_msol15**(-6.5)*ell**2, c='k')
#    kh = np.sqrt(ell*(ell+1))
#    plt.loglog(freqs_msol15, 3e-15*(freqs_msol15/hz_to_invday)**(-6.5)*kh**4, c='lightgrey', label=r'$(3\times10^{-15})(f/Hz)^{-6.5}\left[\sqrt{\ell(\ell+1)}\,\right]^{4}$')
#    plt.loglog(freqs_msol15, 3e-11*(freqs_msol15/hz_to_invday)**(-6.5)*kh**4, c='lightgrey', label=r'$(3\times10^{-11})(f/Hz)^{-6.5}\left[\sqrt{\ell(\ell+1)}\,\right]^{4}$')
#    plt.loglog(freqs_msol15, 3e-10*(freqs_msol15/hz_to_invday)**(-6.5)*kh**4, c='k', label=r'$(3\times10^{-10})(f/Hz)^{-6.5}\left[\sqrt{\ell(\ell+1)}\,\right]^{4}$')
#    plt.xlabel('freq (1/day)')
#    plt.ylabel('wave luminosity')
#    plt.xlim(1e-2, 1e1)
#    plt.ylim(1e12, 1e32)
#    plt.legend()
#    plt.title(r'$\ell = $' + '{}'.format(ell))
#    plt.savefig('wave_luminosity_comparison/mass_ell{:03d}.png'.format(ell))
#    plt.clf()
##    plt.show()

fig = plt.figure(figsize=(7.5, 4.5))
ax1 = fig.add_axes([0.025, 0.55, 0.4, 0.4])
ax2 = fig.add_axes([0.575, 0.55, 0.4, 0.4])
ax3 = fig.add_axes([0.025, 0.00, 0.4, 0.4])
ax4 = fig.add_axes([0.575, 0.00, 0.4, 0.4])
axs = [ax1, ax2, ax3, ax4]
axleft = [ax1, ax3]
axright = [ax2, ax4]
axtop = [ax1, ax2]
axbot = [ax3, ax4]





ell = 1
kh = np.sqrt(ell*(ell+1))
ax3.fill_between(freqs_msol15, 4e-45*(freqs_msol15/hz_to_invday)**(-6.5)*kh**4, 4e-47*(freqs_msol15/hz_to_invday)**(-6.5)*kh**4, color='lightgrey')
print('ell = {}'.format(ell))
ax1.loglog(freqs_msol3,  np.abs(lum_msol3[:, ells_msol3 == ell]),           color=Oranges_7_r.mpl_colors[3], label=r'3 $M_{\odot}$')
ax1.loglog(freqs_msol15, np.abs(lum_msol15[:, ells_msol15 == ell]),         color=Oranges_7_r.mpl_colors[1], label=r'$15 M_{\odot}$')
ax1.loglog(freqs_msol40, np.abs(lum_msol40[:, ells_msol40 == ell]), color=Oranges_7_r.mpl_colors[0], label=r'40 $M_{\odot}$')
ax3.loglog(freqs_msol3,  np.abs(lum_msol3[:, ells_msol3 == ell])/eval('Ma_cz*Lconv_cz', msol3_file_dict).value,           color=Oranges_7_r.mpl_colors[3], label=r'3 $M_{\odot}$')
ax3.loglog(freqs_msol15, np.abs(lum_msol15[:, ells_msol15 == ell])/eval('Ma_cz*Lconv_cz', msol15_file_dict).value,         color=Oranges_7_r.mpl_colors[1], label=r'$15 M_{\odot}$')
ax3.loglog(freqs_msol40, np.abs(lum_msol40[:, ells_msol40 == ell])/eval('Ma_cz*Lconv_cz', msol40_file_dict).value, color=Oranges_7_r.mpl_colors[0], label=r'40 $M_{\odot}$')
#    plt.loglog(freqs_msol15, 2.14e-28*freqs_msol15**(-6.5)*ell**2, c='k')
ax3.loglog(freqs_msol15, 4e-45*(freqs_msol15/hz_to_invday)**(-6.5)*kh**4,lw=0.5, c='lightgrey')
ax3.loglog(freqs_msol15, 4e-46*(freqs_msol15/hz_to_invday)**(-6.5)*kh**4, c='k', label=r'$(4 \times 10^{-46})(f/Hz)^{-6.5}\left[\sqrt{\ell(\ell+1)}\,\right]^{4}$')
ax3.loglog(freqs_msol15, 4e-47*(freqs_msol15/hz_to_invday)**(-6.5)*kh**4,lw=0.5, c='lightgrey')
for ax in axleft:
    ax.set_xlabel('$f$ (d$^{-1}$)')
    ax.set_xlim(1e-2, 1e1)
    ax.text(0.99, 0.98, r'$\ell = 1$', ha='right', va='top', transform=ax.transAxes)
for ax in axright:
    ax.set_xlabel('$\ell$')
    ax.set_xlim(1, 1e2)
    ax.text(0.01, 0.98, r'$f = 0.8$ d$^{-1}$', ha='left', va='top', transform=ax.transAxes)
for ax in axtop:
    ax.set_ylabel(r'$L_w$ (erg$\,\,$s$^{-1}$)')
    ax.set_ylim(1e10, 1e32)
for ax in axbot:
    ax.set_ylabel(r'$L_w / (\mathscr{M} L_*)$')
    ax.set_ylim(1e-19, 1e-4)

ax1.text(8e-2, 1e19, r'$3 M_{\odot}$', c=Oranges_7_r.mpl_colors[3], ha='left', va='center')
ax1.text(3e-2, 1e24, r'$15 M_{\odot}$', c=Oranges_7_r.mpl_colors[1], ha='left', va='center')
ax1.text(2e-2, 2e27, r'$40 M_{\odot}$', c=Oranges_7_r.mpl_colors[0], ha='left', va='center')
ax2.text(9, 1e20, r'$3 M_{\odot}$', c=Oranges_7_r.mpl_colors[3], ha='left', va='center')
ax2.text(16, 3e24, r'$15 M_{\odot}$', c=Oranges_7_r.mpl_colors[1], ha='left', va='center')
ax2.text(50, 2e28, r'$40 M_{\odot}$', c=Oranges_7_r.mpl_colors[0], ha='left', va='center')

#    plt.show()

freq = 0.8 #invday
kh = np.sqrt(ells_msol15*(ells_msol15+1))
ax4.fill_between(ells_msol15, 4e-45*(freq/hz_to_invday)**(-6.5)*kh**4, 4e-47*(freq/hz_to_invday)**(-6.5)*kh**4, color='lightgrey')
print('f = {}'.format(freq))
ax2.loglog(ells_msol3,  lum_msol3[ freqs_msol3  > freq, :][0,:],    color=Oranges_7_r.mpl_colors[3], label=r'3 $M_{\odot}$')
ax2.loglog(ells_msol15, lum_msol15[freqs_msol15 > freq, :][0,:],  color=Oranges_7_r.mpl_colors[1], label=r'15 $M_{\odot}$')
ax2.loglog(ells_msol40, lum_msol40[freqs_msol40 > freq, :][0,:],  color=Oranges_7_r.mpl_colors[0], label=r'40 $M_{\odot}$')
ax4.loglog(ells_msol3,  lum_msol3[ freqs_msol3  > freq, :][0,:]/eval('Ma_cz*Lconv_cz', msol3_file_dict).value,   color=Oranges_7_r.mpl_colors[3], label=r'3 $M_{\odot}$')
ax4.loglog(ells_msol15, lum_msol15[freqs_msol15 > freq, :][0,:]/eval('Ma_cz*Lconv_cz', msol15_file_dict).value,  color=Oranges_7_r.mpl_colors[1], label=r'15 $M_{\odot}$')
ax4.loglog(ells_msol40, lum_msol40[freqs_msol40 > freq, :][0,:]/eval('Ma_cz*Lconv_cz', msol40_file_dict).value,  color=Oranges_7_r.mpl_colors[0], label=r'40 $M_{\odot}$')
kh = np.sqrt(ells_msol15*(ells_msol15+1))
ax4.loglog(ells_msol15, 4e-45*(freq/hz_to_invday)**(-6.5)*kh**4, lw=0.5, c='lightgrey', label=r'$(3\times10^{-15})f^{-6.5}\left[\sqrt{\ell(\ell+1)}\,\right]^{4}$')
ax4.loglog(ells_msol15, 4e-46*(freq/hz_to_invday)**(-6.5)*kh**4, c='k', label=r'$(3\times10^{-11})f^{-6.5}\left[\sqrt{\ell(\ell+1)}\,\right]^{4}$')
ax4.loglog(ells_msol15, 4e-47*(freq/hz_to_invday)**(-6.5)*kh**4, lw=0.5, c='lightgrey', label=r'$(3\times10^{-10})f^{-6.5}\left[\sqrt{\ell(\ell+1)}\,\right]^{4}$')
plt.savefig('wave_luminosity_comparison/mstar_waveflux.png', bbox_inches='tight', dpi=300)
plt.savefig('wave_luminosity_comparison/mstar_waveflux.pdf', bbox_inches='tight', dpi=300)
plt.clf()
