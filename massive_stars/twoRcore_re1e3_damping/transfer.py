"""
Calculate transfer function to get surface response of convective forcing.
Outputs a function which, when multiplied by sqrt(wave flux), gives you the surface response.
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy import interpolate
from pathlib import Path
from scipy.interpolate import interp1d
from configparser import ConfigParser


from d3_stars.defaults import config
from d3_stars.simulations.parser import name_star
out_dir, out_file = name_star()

def transfer_function(om, values, u_dual, field_outer, r_range, rho, ell):
    """
    rho is a function of (r) at each point for r_range.
    k_r is eqn 13 of Lecoanet+2015 PRE 91, 063016.

    Multiply the output of this function by sqrt(wave luminosity) to get surface response.
    """
    #The none's expand dims
    #dimensionality is [omega', rf, omega]
    dr = np.gradient(r_range)[None, :, None]
    r_range         = r_range[None, :, None]
    rho                 = rho[None, :, None]
    om                   = om[None, None, :] 
    values           = values[:, None, None]
    u_dual           = u_dual[:, :,    None]
    field_outer = field_outer[:, None, None]
    k_h = np.sqrt(ell * (ell + 1)) / r_range
    leading_amp = 2*np.pi*r_range**2*rho
    bulk_to_bound_force = np.sqrt(2) * om / k_h #times ur -> comes later.
    Amp = leading_amp * bulk_to_bound_force * u_dual / (om - values)
    T = np.sum( np.abs(np.sum(Amp * field_outer * dr, axis=0)), axis=0) / np.sum(dr)
    return T

def calculate_refined_transfer(om, *args):

    T = transfer_function(om, *args)

    peaks = 1
    while peaks > 0:
        i_peaks = []
        for i in range(2,len(om)-2):
            if (T[i]>T[i-1] and T[i] > T[i-2]) and (T[i]>T[i+1] and T[i] > T[i+2]):
                delta_m = np.abs(T[i]-T[i-1])/T[i]
                delta_p = np.abs(T[i]-T[i+1])/T[i]
                if delta_m > 0.01 or delta_p > 0.01:
                    i_peaks.append(i)

        peaks = len(i_peaks)
        print("number of peaks: %i" % (peaks))

        om_new = np.array([])
        for i in i_peaks:
            om_low = om[i-1]
            om_high = om[i+1]
            om_new = np.concatenate([om_new,np.linspace(om_low,om_high,10)])

        T_new = transfer_function(om_new, *args)

#        print([om[i] for i in i_peaks])
        om = np.concatenate([om,om_new])
        T = np.concatenate([T,T_new])

        om, sort = np.unique(om, return_index=True)
        T = T[sort]
#        if args[-1] > 0:
#            plt.loglog(om, T)
#            plt.show()

    return om, T


Lmax = config.eigenvalue['Lmax']

with h5py.File(out_file, 'r') as f:
    stitch_radii = f['r_stitch'][()]
    radius = f['r_outer'][()]

resolutions = []
for nr in config.star['nr']:
    resolutions.append((1, 1, nr))
dealias = 1

with h5py.File(out_file, 'r') as f:
    tau_s = f['tau_nd'][()]
    tau = tau_s/(60*60*24)
    rs = []
    rhos = []
    chi_rads = []
    N2s = []
    for bk in ['B', 'S1', 'S2']:
        rs.append(f['r_{}'.format(bk)][()])
        rhos.append(np.exp(f['ln_rho0_{}'.format(bk)][()]))
        chi_rads.append(f['chi_rad_{}'.format(bk)][()])
        #-g[2] * grad_s[2] / cp
        N2s.append(-f['g_{}'.format(bk)][2,:]*f['grad_s0_{}'.format(bk)][2,:]/f['Cp'][()])
    rs = np.concatenate(rs, axis=-1)
    rhos = np.concatenate(rhos, axis=-1)
    chi_rads = np.concatenate(chi_rads, axis=-1)
    N2s = np.concatenate(N2s, axis=-1)
rho = interpolate.interp1d(rs.flatten(), rhos.flatten())
chi_rad = interpolate.interp1d(rs.flatten(), chi_rads.flatten())
N2 = interpolate.interp1d(rs.flatten(), N2s.flatten())




forcing_radius = 1.02
ell_list = np.arange(1, Lmax+1)

dir = 'eigenvalues'

for ell in ell_list:
    plt.figure()
    print("ell = %i" % ell)

    xmin = 1e99
    xmax = -1e99

    transfers = []
    oms = []
    depth_list = [10, 1, 0.01]
    for j, d_filter in enumerate(depth_list):
        with h5py.File('{:s}/duals_ell{:03d}_eigenvalues.h5'.format(dir, ell), 'r') as f:
            velocity_duals = f['velocity_duals'][()]
            values = f['good_evalues'][()]
            velocity_eigenfunctions = f['velocity_eigenfunctions'][()]
            s1_amplitudes = f['s1_amplitudes'][()].squeeze()
            enth_amplitudes = f['enth_amplitudes'][()].squeeze()
            depths = f['depths'][()]

            rs = []
            for bk in ['B', 'S1', 'S2']:
                rs.append(f['r_{}'.format(bk)][()].flatten())
            r = np.concatenate(rs)
            smooth_oms = f['smooth_oms'][()]
            smooth_depths = f['smooth_depths'][()]
            depthfunc = interp1d(smooth_oms, smooth_depths, bounds_error=False, fill_value='extrapolate')

#        if j == 0:
#            for value in values:
#                plt.axvline(value.real/(2*np.pi), c='k')
        print('depth cutoff: {}'.format(d_filter))
        good = depths < d_filter

        mingood = np.min(np.abs(values[good].real))
#        good *= (np.abs(values.real) < mingood * 10)
        values = values[good]
        s1_amplitudes = s1_amplitudes[good]
        enth_amplitudes = enth_amplitudes[good]
        velocity_eigenfunctions = velocity_eigenfunctions[good]
        velocity_duals = velocity_duals[good]
#        print('first 20 good values: {}'.format(values[:20]))
#        print('first 20 depths: {}'.format(depths[good][:20]))

        om0 = np.min(np.abs(values.real))
        om1 = np.max(values.real)*5
        if om0 < xmin: xmin = om0
        if om1 > xmax: xmax = om1
        if j == 0:
            om0/= 10**(1)
            stitch_om = np.abs(values[depths[good] <= 1][-1].real)
        om = np.exp( np.linspace(np.log(om0), np.log(om1), num=5000, endpoint=True) )

        r0 = forcing_radius
        r1 = r0 + 0.05*(r.max())
        r_range = np.linspace(r0, r1, num=100, endpoint=True)
#        r_range = np.linspace(r.min(), r.max(), num=100, endpoint=True)
        uphi_dual_interp = interpolate.interp1d(r, velocity_duals[:,0,:], axis=-1)(r_range)
        
        om, T = calculate_refined_transfer(om, values, uphi_dual_interp, s1_amplitudes, r_range, rho(r_range), ell)

        oms.append(om)
        transfers.append(T)


    #right now we have a transfer function for each optical depth filter
    # We want to just get one transfer function value at each om
    # use the minimum T value from all filters we've done.
    good_om = np.sort(np.concatenate(oms))
    good_T = np.zeros_like(good_om)

    from scipy.interpolate import interp1d
    interps = []
    for om, T in zip(oms, transfers):
        interps.append(interp1d(om, T, bounds_error=False, fill_value=np.inf))

    for i, omega in enumerate(good_om):
        vals = []
        for f in interps:
            vals.append(f(omega))
        good_T[i] = np.min(vals)

    #Do WKB at low frequency -- exponentially attenuate by exp(-om/om_{tau=1}) [assume transfer does everything right up to tau=1]
#    good_T *= np.exp(-depthfunc(good_om))
    good_T[good_om <= stitch_om] *= np.exp(-depthfunc(good_om[good_om <= stitch_om]) + depthfunc(stitch_om))


    # Right now the transfer function gets us from ur (near RCB) -> surface. We want wave luminosity (near RCB) -> surface.
    chi_rad_u = chi_rad(forcing_radius)
    brunt_N2_u = N2(forcing_radius)
    rho_u = rho(forcing_radius)

    k_h = np.sqrt(ell*(ell+1))/forcing_radius 
    k_r_low_diss  = -np.sqrt(brunt_N2_u/good_om**2 - 1).real*k_h
    k_r_high_diss = (((-1)**(3/4) / np.sqrt(2))\
                          *np.sqrt(-2*1j*k_h**2 - (good_om/chi_rad_u) + np.sqrt((good_om)**3 + 4*1j*k_h**2*chi_rad_u*brunt_N2_u)/(chi_rad_u*np.sqrt(good_om)) )).real
    k_r_err = np.abs(1 - k_r_low_diss/k_r_high_diss)

    k_r = np.copy(k_r_high_diss)
    if np.sum(k_r_err < 1e-3) > 0:
        om_switch = good_om.ravel()[k_r_err < 1e-3][0]
        k_r[good_om.ravel() >= om_switch] = k_r_low_diss[good_om.ravel() >= om_switch]
#
    k2 = k_r**2 + k_h**2
    root_lum_to_ur = np.sqrt(1/(4*np.pi*forcing_radius**2*rho_u))*np.sqrt(np.array(1/(-(good_om + 1j*chi_rad_u*k2)*k_r / k_h**2).real, dtype=np.complex128))

#    plt.loglog(good_om/(2*np.pi), good_T)
#    plt.loglog(good_om/(2*np.pi), (root_lum_to_ur*good_T).real, color='orange')
#    plt.loglog(good_om/(2*np.pi), (root_lum_to_ur*good_T).imag, color='orange', ls='--')
#    plt.axvline(stitch_om/(2*np.pi))
#    plt.xlabel('frequency')
#    plt.ylabel('T')
#    plt.show()
#
    with h5py.File('{:s}/transfer_ell{:03d}_eigenvalues.h5'.format(dir, ell), 'w') as f:
        f['om'] = good_om
        f['transfer_ur'] = good_T
        f['transfer_root_lum'] = root_lum_to_ur*good_T 


