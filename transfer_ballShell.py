"""
Calculate transfer function to get horizontal velocities at the top of the simulation.

Usage:
    transfer_ballShell.py <root_dir> [options]

Options:
    --Lmax=<L>      Maximum L value to calculate [default: 1]
    --mesa_file=<f>      path to a .h5 file of ICCs, curated from a MESA model

"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy import interpolate

from docopt import docopt
args = docopt(__doc__)

#TODO: fix rho
if args['--mesa_file'] is not None:
    with h5py.File(args['--mesa_file'], 'r') as mf:
        r_mesa = mf['r_mesa'][()]/mf['L'][()]
        N2_mesa = mf['N2_mesa'][()]*((60*60*24)/(2*np.pi))**2
        S1_mesa = mf['S1_mesa'][()]*(60*60*24)/(2*np.pi)
        N2_mesa_full = mf['N2_mesa'][()]
        g_mesa  = mf['g_mesa'][()]
        cp_mesa = mf['cp_mesa'][()]
        pre_PE = g_mesa**2/cp_mesa**2/N2_mesa_full * (mf['tau'][()]*mf['s_c'][()]/mf['L'][()])**2
        rB = mf['rB']
        rS = mf['rS']
        ρB = np.exp(mf['ln_ρB'])
        ρS = np.exp(mf['ln_ρS'])
        r = np.concatenate((rB, rS), axis=-1)
        ρ = np.concatenate((ρB, ρS), axis=-1)
else:
    raise ValueError("Must specify mesa_file")
rho = interpolate.interp1d(r.flatten(), ρ.flatten())

def transfer_function(om, values, u_dual, u_outer, r_range):
    #The none's expand dims
    #dimensionality is [omega', rf, omega]
    dr = r_range[None, :, None]
    om = om[None, None, :] #The None's expand dims.
    T = (2*np.pi**2*rho(r_range[None,:,None])*r_range[None,:,None]**3*om)*u_dual[:,:, None]*u_outer[:, None, None]/(om-values[:, None, None])
    return np.sum(np.abs(np.sum(T*dr, axis=0)), axis=0)/np.sum(dr)

def refine_peaks(om, T, *args):
    i_peaks = []
    for i in range(1,len(om)-1):
        if (T[i]>T[i-1]) and (T[i]>T[i+1]):
            delta_m = np.abs(T[i]-T[i-1])/T[i]
            delta_p = np.abs(T[i]-T[i+1])/T[i]
            if delta_m > 0.05 or delta_p > 0.05:
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
    print("ell = %i" % ell)

    with h5py.File('{:s}/duals_ell{:03d}_eigenvalues.h5'.format(dir, ell), 'r') as f:
        velocity_duals = f['velocity_duals'][()]
        values = f['good_evalues'][()]
        velocity_eigenfunctions = f['velocity_eigenfunctions'][()]
        rB = f['rB'][()].flatten()
        rS = f['rS'][()].flatten()
        r = np.concatenate((rB, rS))
 
#    values = data['values']
#    u_dual = data['u_dual']
#    u = data['u']
#    z = data['z']

    om0 = values.real[-2]
    print(om0)
    om1 = values.real[0]*2
    om = np.exp( np.linspace(np.log(om0), np.log(om1), num=5000, endpoint=True) )

    r0 = 1
    r1 = 1 + 0.02*r.max()
    r_range = np.linspace(r0, r1, num=100, endpoint=True)
    uphi_dual_interp = interpolate.interp1d(r, velocity_duals[:,0,:], axis=-1)(r_range)
    u_surf = velocity_eigenfunctions[:,0,-1]

    T = transfer_function(om, values, uphi_dual_interp, u_surf, r_range)

    peaks = 1
    while peaks > 0:
        om, T, peaks = refine_peaks(om, T, uphi_dual_interp, u_surf, r_range)

    with h5py.File('{:s}/transfer_ell{:03d}_eigenvalues.h5'.format(dir, ell), 'w') as f:
        f['om'] = om
        f['transfer'] = T

    plt.loglog(om, T)
    plt.show()

