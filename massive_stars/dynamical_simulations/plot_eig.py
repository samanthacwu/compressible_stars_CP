"""
Plot all eigenfunctions from an EVP run

Usage:
    plot_eig.py <root_dir> [options]
    plot_eig.py <root_dir> <config> 

Options:
    --Lmax=<ell>      Maximum ell value to plot
    --ncc_file=<f>      path to a .h5 file of ICCs, curated from a MESA model
"""
import glob
import os
import sys
from collections import OrderedDict
from pathlib import Path
from configparser import ConfigParser

import h5py
import numpy as np
from docopt import docopt
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

import logging
logger = logging.getLogger(__name__)

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


root_dir = args['<root_dir>']
Lmax = args['--Lmax']
if Lmax is not None:
    Lmax = int(Lmax)

ell_files = glob.glob("{}/ell*.h5".format(root_dir))
ell_values = [int(f.split('ell')[-1].split('_')[0]) for f in ell_files]
zipped_files = [[f, ell] for f, ell in zip(ell_files, ell_values)]
ell_files, ell_values = zip(*sorted(zipped_files, key=lambda x: x[1]))

fig, axs = plt.subplots(3,2, figsize=(8,8), sharex=True)
((ax1, ax2), (ax3, ax4), (ax5, ax6)) = axs

for f, ell in zip(ell_files, ell_values):
    if Lmax is not None and ell > Lmax:
        break
    logger.info('plotting from {}'.format(f))

    if args['--ncc_file'] is not None:
        with h5py.File(args['--ncc_file'], 'r') as mf:
            r_mesa = mf['r_mesa'][()]/mf['L_nd'][()]
            N2_mesa = mf['N2_mesa'][()]*((60*60*24)/(2*np.pi))**2
            S1_mesa = mf['S1_mesa'][()]*(60*60*24)/(2*np.pi)
            N2_mesa_full = mf['N2_mesa'][()]
            g_mesa  = mf['g_mesa'][()]
            cp_mesa = mf['cp_mesa'][()]
            pre_PE = g_mesa**2/cp_mesa**2/N2_mesa_full * (mf['tau_nd'][()]*mf['s_nd'][()]/mf['L_nd'][()])**2
            rhoB = np.exp(mf['ln_rho_B'])
            rhoS1 = np.exp(mf['ln_rho_S1'])
            rhoS2 = np.exp(mf['ln_rho_S2'])
            rho = np.concatenate((rhoB, rhoS1, rhoS2), axis=-1)
    N2_of_r = interp1d(r_mesa, N2_mesa)
    S_of_r = interp1d(r_mesa, S1_mesa*np.sqrt(ell*(ell+1))/np.sqrt((1*(1+1))) )
    pre_PE_of_r = interp1d(r_mesa, pre_PE)

    out_dir = f.split('.h5')[0] + '_figures/'
    if not os.path.exists('{:s}/'.format(out_dir)):
        os.makedirs('{:s}/'.format(out_dir))
    with h5py.File('{:s}'.format(f), 'r') as f:
        evalues = f['good_evalues'][()]
        s1_amplitudes = f['s1_amplitudes'][()]
        integ_energies = f['integ_energies'][()]
        velocity_eigenfunctions = f['velocity_eigenfunctions'][()]
        entropy_eigenfunctions  = f['entropy_eigenfunctions'][()]
        wave_flux_eigenfunctions  = f['wave_flux_eigenfunctions'][()]
        rB  = f['r_B'][()]
        rS1  = f['r_S1'][()]
        rS2  = f['r_S2'][()]
        r = np.concatenate((rB.flatten(), rS1.flatten(), rS2.flatten()))
#
#    plt.figure()
##    plt.scatter(evalues.real, s1_amplitudes, c='r')
##    plt.scatter(evalues.real, np.sum(velocity_eigenfunctions*np.conj(velocity_eigenfunctions), axis=1)[:,0,0,-1], c='b')
#    plt.scatter(evalues.real, (entropy_eigenfunctions*np.conj(entropy_eigenfunctions))[:,0,0,-1]/np.sum(velocity_eigenfunctions*np.conj(velocity_eigenfunctions), axis=1)[:,0,0,-1], c='b')
#    plt.ylabel('|s1|^2 / surface_energy')
#    plt.xscale('log')
#    plt.yscale('log')
#    plt.xlabel('frequency (1/day)')
#    plt.xlim(1e-1, 1e1)
#    plt.ylim(1e6, 1e9)
#    plt.show()
#    import sys
#    sys.exit()

    for i in range(len(evalues)):
        print('plotting {}/{}'.format(i+1, len(evalues)))
        ax1.plot(r, velocity_eigenfunctions[i,0,:].real/np.abs(velocity_eigenfunctions[i,2,:]).max(), label='real')
        ax2.plot(r, velocity_eigenfunctions[i,1,:].real/np.abs(velocity_eigenfunctions[i,2,:]).max())
        ax3.plot(r, velocity_eigenfunctions[i,2,:].real/np.abs(velocity_eigenfunctions[i,2,:]).max())
        ax4.plot(r,    entropy_eigenfunctions[i,:].real/np.abs(   entropy_eigenfunctions[i,:]).max())
        ax1.plot(r, velocity_eigenfunctions[i,0,:].imag/np.abs(velocity_eigenfunctions[i,2,:]).max(), label='imag')
        ax2.plot(r, velocity_eigenfunctions[i,1,:].imag/np.abs(velocity_eigenfunctions[i,2,:]).max())
        ax3.plot(r, velocity_eigenfunctions[i,2,:].imag/np.abs(velocity_eigenfunctions[i,2,:]).max())
        ax4.plot(r,    entropy_eigenfunctions[i,:].imag/np.abs(   entropy_eigenfunctions[i,:]).max())
#        ke = r**3/np.sqrt(N2_of_r(r)) * (rho.flatten()*np.sum(velocity_eigenfunctions*np.conj(velocity_eigenfunctions), axis=1)[i,:])
#        pe = np.sqrt(2) * r**3/np.sqrt(N2_of_r(r)) * (rho.flatten()*pre_PE_of_r(r)) * np.abs(entropy_eigenfunctions[i,:])**2
#        ax5.plot(r[N2_of_r(r) > 0], ke[N2_of_r(r) > 0], label='ke')
#        ax5.plot(r[N2_of_r(r) > 0], pe[N2_of_r(r) > 0], label='pe')
#        ax5.plot(r[N2_of_r(r) > 0], (ke+pe)[N2_of_r(r) > 0], label='sum')
        ax5.plot(r, 4*np.pi*r**2*wave_flux_eigenfunctions[i,:].real)
        ax5.plot(r, wave_flux_eigenfunctions[i,:].imag)
        ax6.plot(r, np.sqrt(N2_of_r(r)), c='r', label='N')
        ax6.plot(r, S_of_r(r), c='k', label='S')
        ax6.axhline(evalues[i].real/(2*np.pi), c='orange', label='this mode')
        ax1.set_ylabel(r'$u_\phi$'   + '/{:.2e}'.format(np.abs(velocity_eigenfunctions[i,2,:]).max()))
        ax2.set_ylabel(r'$u_\theta$' + '/{:.2e}'.format(np.abs(velocity_eigenfunctions[i,2,:]).max()))
        ax3.set_ylabel(r'$u_r$'      + '/{:.2e}'.format(np.abs(velocity_eigenfunctions[i,2,:]).max()))
        ax4.set_ylabel(r'$s_1$'      + '/{:.2e}'.format(np.abs(   entropy_eigenfunctions[i,:]).max()))
        ax5.set_ylabel(r'$r^3 \rho |u|^2/N$')
        ax6.set_ylabel(r'frequencies')
        ax6.set_yscale('log')
        fig.suptitle('eigenvalue = {:.3e} /  KE = {:.3e}  /  surface entropy = {:.3e}'.format(evalues[i], integ_energies[i], s1_amplitudes[i]))
        ax1.legend(loc='best')
        ax5.legend(loc='best')
        ax6.legend(loc='best')

        for ax_t in axs:
            for ax in ax_t:
                ax.set_xlabel('r')
                ax.set_xlim(r.min(), r.max())
        for ax in [ax2, ax4, ax6]:
            ax.yaxis.set_ticks_position('right')
            ax.yaxis.set_label_position('right')
        fig.savefig('{}/eigfunctions_{:04d}.png'.format(out_dir, i+1), dpi=300, bbox_inches='tight')
        for ax_t in axs:
            for ax in ax_t:
                ax.cla()
