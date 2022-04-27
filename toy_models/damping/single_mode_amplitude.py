"""
Calculate transfer function to get horizontal velocities at the top of the simulation.

"""
import traceback
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy import interpolate
from pathlib import Path
from scipy.interpolate import interp1d
from configparser import ConfigParser


from d3_stars.simulations.anelastic_functions import make_bases
from d3_stars.simulations.parser import parse_std_config

leading_normalization = 1e-2
ell_force = 2
freq_force = 0.2
om_force = 2*np.pi*freq_force

r_transition = 1
force_radial = lambda r: np.exp(-(r - r_transition)**2/0.1**2)

def transfer_function(eigvalues, u_sample, u_dual, r_range, ell):
    #dimensionality is [eig_omega, rf]
    u_dual           = u_dual(r_range)[:, :]
    dr               = np.gradient(r_range)[None, :]
    r_range          = r_range[None, :]
    radial_integ = np.sum(r_range**2*force_radial(r_range)*np.conj(u_dual)*dr, axis=1)
    Amp = 1j * np.sqrt(2) * np.pi * np.sqrt(ell*(ell+1)) * leading_normalization * radial_integ / (om_force - eigvalues)
    T = Amp * u_sample
    return np.sum(T)

ell_list = np.arange(ell_force, ell_force+1)
print(ell_list)

dir = 'eigenvalues'
for ell in ell_list:
    plt.figure()
    print("ell = %i" % ell)

    xmin = 1e99
    xmax = -1e99

    for r_str in ['1.25', '1.4', '1.5', '1.6', '1.75']:

        with h5py.File('{:s}/duals_ell{:03d}_eigenvalues.h5'.format(dir, ell), 'r') as f:
            velocity_duals = f['velocity_duals'][()]
            values = f['good_evalues'][()]
            velocity_eigenfunctions = f['velocity_eigenfunctions'][()]
            r = f['r'][()].ravel()

        good = np.ones_like(values, dtype=bool)
        values = values[good]
        velocity_eigenfunctions = velocity_eigenfunctions[good]
        velocity_duals = velocity_duals[good]

        r0 = 0.7
        r1 = 1.3
        r_range = np.linspace(r0, r1, num=1000, endpoint=True)

        uphi_dual_interp = interpolate.interp1d(r, velocity_duals[:,0,:], axis=-1)

        r_sample = float(r_str)
        uphi_sample = interpolate.interp1d(r, velocity_eigenfunctions[:,0,:], axis=-1)(r_sample)
        utheta_sample = interpolate.interp1d(r, velocity_eigenfunctions[:,1,:], axis=-1)(r_sample)
        ur_sample = interpolate.interp1d(r, velocity_eigenfunctions[:,2,:], axis=-1)(r_sample)

        Tphi   = transfer_function(values, uphi_sample,   uphi_dual_interp, r_range, ell)
        Ttheta = transfer_function(values, utheta_sample, uphi_dual_interp, r_range, ell)
        Tr     = transfer_function(values, ur_sample,     uphi_dual_interp, r_range, ell)
        Th_pow = np.conj(Tphi)*Tphi# + np.conj(Ttheta)*Ttheta
        Tr_pow = np.conj(Tr)*Tr

#        Th_pow /= r_sample
#        Tr_pow /= r_sample

#        print('{:.3e}, {:.3e}'.format(Th_pow, Tr_pow))
        with h5py.File('pulled_power/power_spectra.h5', 'r') as f:
            ells = f['ells'][()].ravel()
            freqs = f['freqs'][()]
            good_ell = np.where(ells == ell_force)[0]
            good_f = np.where((freqs > freq_force*0.97)*(freqs < freq_force*1.03))[0]
            df = np.gradient(freqs)
            u_pow = f['u(r={})_power_per_ell'.format(r_str)][()]
    #        print(u_pow[0, good_f, good_ell])
#            u_pow = u_pow[:, good_f, good_ell]
            uphi_pow   = np.sum(u_pow[0,:])#*df[:,None])
            utheta_pow = np.sum(u_pow[1,:])#*df[:,None])
            ur_pow     = np.sum(u_pow[2,:])#*df[:,None])
            uh_pow = uphi_pow + utheta_pow

    #        print(uphi_pow, utheta_pow, ur_pow)
#            print('{:.3e}, {:.3e}'.format(uh_pow, ur_pow))
            print('r = {} ratios: {:.6f}, {:.6f}'.format(r_str, (Th_pow/uh_pow).real, (Tr_pow/ur_pow).real))
