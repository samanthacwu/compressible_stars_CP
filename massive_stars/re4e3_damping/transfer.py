"""
Calculate transfer function to get horizontal velocities at the top of the simulation.

"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy import interpolate
from pathlib import Path
from scipy.interpolate import interp1d
from configparser import ConfigParser

import dedalus.public as d3


from d3_stars.simulations.anelastic_functions import make_bases

from d3_stars.simulations.parser import name_star
from d3_stars.defaults import config

star_dir, star_file = name_star()

Lmax = config.eigenvalue['Lmax']

with h5py.File(star_file, 'r') as f:
    stitch_radii = f['r_stitch'][()]
    radius = f['r_outer'][()]

ntheta = config.eigenvalue['Lmax'] + 1
nphi = 4
resolutions = []
for nr in config.star['nr']:
    resolutions.append((nphi, ntheta, nr))
dealias = 1
coords, dist, bases, bases_keys = make_bases(resolutions, stitch_radii, radius, dealias=dealias, dtype=np.complex128, mesh=None)

s_surf = dist.Field(bases=bases[bases_keys[-1]].S2_basis())
s_surf_pow_op = d3.Average(np.conj(s_surf)*s_surf, coords.S2coordsys)

with h5py.File(star_file, 'r') as f:
    tau_s = f['tau_nd'][()]
    tau = tau_s/(60*60*24)
    rs = []
    rhos = []
    for bk in bases_keys:
        rs.append(f['r_{}'.format(bk)][()])
        rhos.append(np.exp(f['ln_rho_{}'.format(bk)][()]))
    r = np.concatenate(rs, axis=-1)
    rho = np.concatenate(rhos, axis=-1)
rho = interpolate.interp1d(r.flatten(), rho.flatten())

def transfer_function(om, values, u_dual, u_outer, r_range, ell):
    #The none's expand dims
    #dimensionality is [eigenvalue, rf, omega_force, phi, theta, r]
    dr = np.gradient(r_range)[None, :, None]
    r_range         = r_range[None, :, None]
    om                   = om[None, None, :] 
    values           = values[:, None, None]
    u_dual           = u_dual[:, :,    None]
    k_h = np.sqrt(ell * (ell + 1)) / r_range
    Forcing = (np.pi/2) * (om / k_h**2) #technically times ur at r_rcb
    Amp = np.sum(dr * (2 * np.pi * r_range**2 * rho(r_range) * u_dual * Forcing /(om - values) ), axis=1)/np.sum(dr)

    u_outer         = u_outer[:, None,:]
    T = Amp[:,:,None,None,None] * u_outer
    return np.sum(T, axis=0)

def refine_peaks(om, T, *args):
    print(om.shape, T.shape)
    i_peaks = []
    for i in range(1,len(om)-1):
        Ti = np.sum(np.abs(T[i]))
        Timinus = np.sum(np.abs(T[i-1]))
        Tiplus = np.sum(np.abs(T[i+1]))
        if (Ti > Timinus) and (Ti > Tiplus):
            delta_m = np.abs(Ti-Timinus)/Ti
            delta_p = np.abs(Ti-Tiplus)/Ti
            if delta_m > 0.01 or delta_p > 0.01:
                i_peaks.append(i)

    print("number of peaks: %i" %(len(i_peaks)))

    om_new = np.array([])
    for i in i_peaks:
        om_low = om[i-1]
        om_high = om[i+1]
        om_new = np.concatenate([om_new,np.linspace(om_low,om_high,10)])


    T_new = transfer_function(om_new, values, *args)

    om = np.concatenate([om,om_new])
    T = np.concatenate([T,T_new])

    om, sort = np.unique(om, return_index=True)
    T = T[sort]

    return om, T, len(i_peaks)


ell_list = np.arange(1, Lmax+1)

dir = 'eigenvalues'

for ell in ell_list:
    plt.figure()
    print("ell = %i" % ell)

    xmin = 1e99
    xmax = -1e99

    transfers = []
    oms = []
    depth_list = [1,]
    for j, d_filter in enumerate(depth_list):
        with h5py.File('{:s}/duals_ell{:03d}_eigenvalues.h5'.format(dir, ell), 'r') as f:
            velocity_duals = f['velocity_duals'][()]
            values = f['good_evalues'][()]
            values_inv_day = f['good_evalues_inv_day'][()]
            velocity_eigenfunctions = f['velocity_eigenfunctions'][()]
            s1_amplitudes = f['s1_amplitudes'][()]
            depths = f['depths'][()]

            rs = []
            for bk in bases_keys:
                rs.append(f['r_{}'.format(bk)][()].flatten())
            r = np.concatenate(rs)
            smooth_oms = f['smooth_oms'][()]
            smooth_depths = f['smooth_depths'][()]
            depthfunc = interp1d(smooth_oms, smooth_depths, bounds_error=False, fill_value='extrapolate')

#        if j == 0:
#            for value in values:
#                plt.axvline(value.real/(2*np.pi), c='k')

        good = depths < d_filter

        values = values[good]
        s1_amplitudes = s1_amplitudes[good]
        velocity_eigenfunctions = velocity_eigenfunctions[good]
        velocity_duals = velocity_duals[good]
        print('good values: {}'.format(values))

        om0 = np.abs(values.real[-1])
        om1 = np.abs(values.real[0]*1.1)
        if om0 < xmin: xmin = om0
        if om1 > xmax: xmax = om1
        if j == 0:
            om0/= 10**(1)
        om = np.exp( np.linspace(np.log(om0), np.log(om1), num=5000, endpoint=True) )

        r0 = 0.5
        r1 = 1.1
        r_range = np.linspace(r0, r1, num=100, endpoint=True)
#        r_range = np.linspace(r.min(), r.max(), num=100, endpoint=True)
        uphi_dual_interp = interpolate.interp1d(r, velocity_duals[:,0,:], axis=-1)(r_range)


        T = transfer_function(om, values, uphi_dual_interp, s1_amplitudes, r_range, ell)

        peaks = 1
        while peaks > 0:
            om, T, peaks = refine_peaks(om, T, uphi_dual_interp, s1_amplitudes, r_range, ell)

        print(om.shape, T.shape)

        power = np.zeros_like(om)
        for i, this_om in enumerate(om):
            s_surf['g'] = T[i,:]
            power[i] = s_surf_pow_op.evaluate()['g'].ravel()[0].real


#        plt.loglog(om/(2*np.pi), np.exp(-depthfunc(om))*power*om**(-13/2), lw=1+0.5*(len(depth_list)-j), label='depth filter = {}'.format(d_filter))
        oms.append(om)
        transfers.append(power)

    good_om = np.sort(np.concatenate(oms))
    good_pow = np.zeros_like(good_om)

    from scipy.interpolate import interp1d
    interps = []
    for om, power in zip(oms, transfers):
        interps.append(interp1d(om, power, bounds_error=False, fill_value=np.inf))

    for i, omega in enumerate(good_om):
        if omega.real < 0:
            continue
        vals = []
        for f in interps:
            vals.append(f(omega))
        good_pow[i] = np.min(vals)

    with h5py.File('{:s}/transfer_ell{:03d}_eigenvalues.h5'.format(dir, ell), 'w') as f:
        f['om'] = good_om
        f['om_inv_day'] = good_om / tau
        f['transfer'] = good_pow
#    plt.loglog(good_om/(2*np.pi), np.exp(-depthfunc(good_om))*np.abs(good_T)**2*good_om**(-13/2), lw=1, label='combined')
    plt.loglog(good_om/(2*np.pi), good_pow*good_om**(-13/2), lw=1, label='combined')

#    maxval = (np.exp(-depthfunc(good_om))*np.abs(good_T)**2*good_om**(-13/2)).max()
#    plt.ylim(maxval/1e15, maxval*2)
    plt.xlabel('frequency (sim units)')
    plt.legend()
    plt.title("ell = %i" % ell)
    plt.xlim(0.7*xmin/(2*np.pi), 1.2*xmax/(2*np.pi))
    plt.show()

#plt.loglog(good_om/(2*np.pi), np.exp(-depthfunc(good_om))*np.abs(good_T)**2*good_om**(-13/2), lw=1, label='combined')
#plt.xlim(0.7*xmin/(2*np.pi), 1.2*xmax/(2*np.pi))
#plt.ylim(maxval/1e15, maxval*2)
#plt.show()
