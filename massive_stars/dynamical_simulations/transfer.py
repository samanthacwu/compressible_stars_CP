"""
Calculate transfer function to get horizontal velocities at the top of the simulation.

Usage:
    transfer_ballShell.py <root_dir> [options]
    transfer_ballShell.py <root_dir> <config>

Options:
    --Lmax=<L>      Maximum L value to calculate [default: 1]
    --ncc_file=<f>      path to a .h5 file of ICCs, curated from a MESA model

"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy import interpolate
from pathlib import Path
from scipy.interpolate import interp1d

from docopt import docopt
from configparser import ConfigParser
args = docopt(__doc__)

# Read in parameters and create output directory
args   = docopt(__doc__)
if args['<config>'] is not None: 
    config_file = Path(args['<config>'])
    config = ConfigParser()
    config.read(str(config_file))
    for n, v in config.items('parameters'):
        for k in args.keys():
            if k.split('--')[-1].lower() == n:
                if v == 'true': v = True
                args[k] = v

#TODO: fix rho
if args['--ncc_file'] is not None:
    with h5py.File(args['--ncc_file'], 'r') as mf:
        tau_s = mf['tau_nd'][()]
        print(tau_s)
        tau = tau_s/(60*60*24)
        r_B = mf['r_B']
        r_S1 = mf['r_S1']
        r_S2 = mf['r_S2']
        rho_B = np.exp(mf['ln_rho_B'])
        rho_S1 = np.exp(mf['ln_rho_S1'])
        rho_S2 = np.exp(mf['ln_rho_S2'])
        r = np.concatenate((r_B, r_S1, r_S2), axis=-1)
        rho = np.concatenate((rho_B, rho_S1, rho_S2), axis=-1)
else:
    raise ValueError("Must specify ncc_file")
rho = interpolate.interp1d(r.flatten(), rho.flatten())

def transfer_function(om, values, u_dual, u_outer, r_range):
    #The none's expand dims
    #dimensionality is [omega', rf, omega]
    dr = np.gradient(r_range)[None, :, None]
    r_range         = r_range[None, :, None]
    om                   = om[None, None, :] 
    values           = values[:, None, None]
    u_dual           = u_dual[:, :,    None]
    u_outer         = u_outer[:, None, None]
    T = (2*np.pi**2*rho(r_range)*r_range**3*om)*u_dual*u_outer/(om-values)
    return np.sum(np.abs(np.sum(T*dr, axis=0)), axis=0)/np.sum(dr)

def refine_peaks(om, T, *args):
    i_peaks = []
    for i in range(1,len(om)-1):
        if (T[i]>T[i-1]) and (T[i]>T[i+1]):
            delta_m = np.abs(T[i]-T[i-1])/T[i]
            delta_p = np.abs(T[i]-T[i+1])/T[i]
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


ell_list = np.arange(1, int(args['--Lmax'])+1)
print(ell_list)

dir = args['<root_dir>']

for ell in ell_list:
    plt.figure()
    print("ell = %i" % ell)

    xmin = 1e99
    xmax = -1e99

    transfers = []
    oms = []
    depth_list = [10, 1, 0.1, 0.05, 0.01]
    for j, d_filter in enumerate(depth_list):
        with h5py.File('{:s}/duals_ell{:03d}_eigenvalues.h5'.format(dir, ell), 'r') as f:
            velocity_duals = f['velocity_duals'][()]
            values = f['good_evalues'][()]
            values_inv_day = f['good_evalues_inv_day'][()]
            velocity_eigenfunctions = f['velocity_eigenfunctions'][()]
            s1_amplitudes = f['s1_amplitudes'][()]
            depths = f['depths'][()]
            rB = f['r_B'][()].flatten()
            rS1 = f['r_S1'][()].flatten()
            rS2 = f['r_S2'][()].flatten()
            r = np.concatenate((rB, rS1, rS2))
            smooth_oms = f['smooth_oms'][()]
            smooth_depths = f['smooth_depths'][()]
            depthfunc = interp1d(smooth_oms, smooth_depths, bounds_error=False, fill_value='extrapolate')

#            omega = np.logspace(-1, 3, 1000)
#            plt.scatter(values.real, np.exp(-np.array(depths)))
#            plt.loglog(omega, np.exp(-depthfunc(omega)))
#            plt.ylim(1e-10, 1e1)
#            plt.show()
#            import sys
#            sys.exit()


        if j == 0:
            for value in values:
                plt.axvline(value.real/(2*np.pi), c='k')

        good = depths < d_filter

        values = values[good]
        s1_amplitudes = s1_amplitudes[good]
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
        print(s1_amplitudes)

        T = transfer_function(om, values, uphi_dual_interp, s1_amplitudes, r_range)

        peaks = 1
        while peaks > 0:
            om, T, peaks = refine_peaks(om, T, uphi_dual_interp, s1_amplitudes, r_range)

        plt.loglog(om/(2*np.pi), np.exp(-depthfunc(om))*np.abs(T)**2*om**(-13/2), lw=1+0.5*(len(depth_list)-j), label='depth filter = {}'.format(d_filter))
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
        f['om_inv_day'] = good_om / tau
        f['transfer'] = good_T
#    print('{:.3e}'.format((np.abs(good_T)**2*good_om**(-13/2)).max()))
    plt.loglog(good_om/(2*np.pi), np.exp(-depthfunc(good_om))*np.abs(good_T)**2*good_om**(-13/2), lw=1, label='combined')
#    plt.loglog(good_om/(2*np.pi), np.abs(good_T)**2*good_om**(-13/2), lw=1, label='combined')

    maxval = (np.exp(-depthfunc(good_om))*np.abs(good_T)**2*good_om**(-13/2)).max()
    plt.ylim(maxval/1e15, maxval*2)
    plt.xlabel('frequency (sim units)')
    plt.legend()
    plt.title("ell = %i" % ell)
    plt.xlim(0.7*xmin/(2*np.pi), 1.2*xmax/(2*np.pi))
    plt.show()

plt.loglog(good_om/(2*np.pi), np.exp(-depthfunc(good_om))*np.abs(good_T)**2*good_om**(-13/2), lw=1, label='combined')
plt.xlim(0.7*xmin/(2*np.pi), 1.2*xmax/(2*np.pi))
plt.ylim(maxval/1e15, maxval*2)
plt.show()
