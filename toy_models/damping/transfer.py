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

Lmax = 10
radius = 2
dealias = 1

rho = lambda radius : 1

def transfer_function(om, values, u_dual, u_outer, r_range, ell):
    #The none's expand dims
    #dimensionality is [omega', rf, omega]
    dr = np.gradient(r_range)[None, :, None]
    r_range         = r_range[None, :, None]
    om                   = om[None, None, :] 
    values           = values[:, None, None]
    u_dual           = u_dual[:, :,    None]
    u_outer         = u_outer[:, None, None]
    k_h = np.sqrt(ell * (ell + 1)) / r_range
    Forcing = (np.pi/2) * (om / k_h**2)  #technically times ur at r_rcb
    Amp = 2 * np.pi * r_range**2 * rho(r_range) * u_dual * Forcing /(om - values)
    T = Amp * u_outer
    return np.sum(np.abs(np.sum(T*dr, axis=0)), axis=0)/np.sum(dr)

def refine_peaks(om, T, *args):
    i_peaks = []
    for i in range(1,len(om)-1):
        if (T[i]>T[i-1]) and (T[i]>T[i+1]):
            delta_m = np.abs(T[i]-T[i-1])/T[i]
            delta_p = np.abs(T[i]-T[i+1])/T[i]
            print(i, om[i]/(2*np.pi), delta_m, delta_p)
            if delta_m > 1e-2 or delta_p > 1e-2:
                i_peaks.append(i)
    print(i_peaks)

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
print(ell_list)

dir = 'eigenvalues'

for ell in ell_list:
    try:
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
                velocity_eigenfunctions = f['velocity_eigenfunctions'][()]
                T_amplitudes = f['T_amplitudes'][()]
    #            depths = f['depths'][()]

                r = f['r'][()].ravel()
    #            smooth_oms = f['smooth_oms'][()]
    #            smooth_depths = f['smooth_depths'][()]
    #            depthfunc = interp1d(smooth_oms, smooth_depths, bounds_error=False, fill_value='extrapolate')

    #        if j == 0:
    #            for value in values:
    #                plt.axvline(value.real/(2*np.pi), c='k')

    #        good = depths < d_filter
            good = np.ones_like(values, dtype=bool)

            values = values[good]
            T_amplitudes = T_amplitudes[good]
            velocity_eigenfunctions = velocity_eigenfunctions[good]
            velocity_duals = velocity_duals[good]
            print('good values: {}'.format(values))

            om0 = values.real[-1]
            om1 = values.real[0]*1.1
            if om0 < xmin: xmin = om0
            if om1 > xmax: xmax = om1
            if j == 0:
                om0/= 10**(1)
                print(values.real[-1], om0)
            om = np.exp( np.linspace(np.log(om0), np.log(om1), num=5000, endpoint=True) )

            r0 = 1.02
            r1 = r0 + 0.05*(r.max())
            r_range = np.linspace(r0, r1, num=100, endpoint=True)
    #        r_range = np.linspace(r.min(), r.max(), num=100, endpoint=True)
            uphi_dual_interp = interpolate.interp1d(r, velocity_duals[:,0,:], axis=-1)(r_range)
            print(T_amplitudes)

            T = transfer_function(om, values, uphi_dual_interp, T_amplitudes, r_range, ell)

            peaks = 1
            while peaks > 0:
                om, T, peaks = refine_peaks(om, T, uphi_dual_interp, T_amplitudes, r_range, ell)

    #        plt.loglog(om/(2*np.pi), np.exp(-depthfunc(om))*np.abs(T)**2*om**(-13/2), lw=1+0.5*(len(depth_list)-j), label='depth filter = {}'.format(d_filter))
            oms.append(om)
            transfers.append(T)

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

        with h5py.File('{:s}/transfer_ell{:03d}_eigenvalues.h5'.format(dir, ell), 'w') as f:
            f['om'] = good_om
            f['transfer'] = good_T

        plt.loglog(good_om/(2*np.pi), good_T, lw=1, label='combined')
#        plt.ylim(maxval/1e15, maxval*2)
        plt.xlabel('frequency (sim units)')
        plt.ylabel('T')
        plt.legend()
        plt.title("ell = %i" % ell)
        plt.xlim(0.7*xmin/(2*np.pi), 1.2*xmax/(2*np.pi))
        plt.show()

    except Exception:
        print(traceback.format_exc())
    #    print('{:.3e}'.format((np.abs(good_T)**2*good_om**(-13/2)).max()))
    #    plt.loglog(good_om/(2*np.pi), np.exp(-depthfunc(good_om))*np.abs(good_T)**2*good_om**(-13/2), lw=1, label='combined')
    
    #plt.loglog(good_om/(2*np.pi), np.exp(-depthfunc(good_om))*np.abs(good_T)**2*good_om**(-13/2), lw=1, label='combined')
    #plt.xlim(0.7*xmin/(2*np.pi), 1.2*xmax/(2*np.pi))
    #plt.ylim(maxval/1e15, maxval*2)
    #plt.show()
