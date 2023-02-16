import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import h5py

from dedalus.tools.general import natural_sort
from d3_stars.defaults import config

files = natural_sort(glob.glob('eigenvalues/dual*ell*.h5'))

from scipy.special import erf
def one_to_zero(x, x0, width=0.1):
    return (1 - erf( (x - x0)/width))/2

def zero_to_one(*args, **kwargs):
    return -(one_to_zero(*args, **kwargs) - 1)




fig = plt.figure(figsize=(8.5,6))
ncol = 3
hwidth = 0.25
pad = (1 - 3*hwidth)/(ncol-1)
ax1 = fig.add_axes([0.00,       0.55, hwidth, 0.4])
ax2 = fig.add_axes([0.00,       0.0, hwidth, 0.4])
ax3 = fig.add_axes([hwidth+pad, 0.55, hwidth, 0.4])
ax4 = fig.add_axes([hwidth+pad, 0.0, hwidth, 0.4])
ax5 = fig.add_axes([1-hwidth,   0.55, hwidth, 0.4])
ax6 = fig.add_axes([1-hwidth,   0.0, hwidth, 0.4])

axs = [ax1,ax2,ax3, ax4, ax5, ax6]
axs_left = [ax1, ax4]
axs_bot = [ax4, ax5, ax6]

brunt_pow_adj = 0

Lmax = config.eigenvalue['Lmax']
ell = 0
for file in files:
    ell += 1
    if ell > Lmax: break
    print('plotting from {}'.format(file))
    out_dir = file.split('.h5')[0]+'_duals'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)


    with h5py.File(file, 'r') as f:
        r = f['r'][()].ravel().real
        evalues = f['good_evalues'][()]
        efs_u = f['velocity_eigenfunctions'][()]
        duals_u = f['velocity_duals'][()]
        efs_enth = f['enthalpy_fluc_eigenfunctions'][()]
        rho = f['rho_full'][()].ravel().real
        bruntN2 = f['bruntN2'][()].ravel().real
        N2_scale = np.copy(bruntN2)
        N2_scale[N2_scale < 1] = 1
        print(N2_scale)


    for i, ev in enumerate(evalues):
        print('plotting {:.7e}'.format(ev))
        ax1.plot(r, efs_u[i,1,:].real, c='black', label='real')
        ax1.plot(r, efs_u[i,1,:].imag, c='orange', label='imag')
        ax1.set_ylabel(r'$u_\theta$')
        ax1.legend()
        ax2.plot(r, efs_u[i,2,:].real, c='black', label='real')
        ax2.plot(r, efs_u[i,2,:].imag, c='orange', label='imag')
        ax2.set_ylabel(r'$u_r$')
        ax3.plot(r, duals_u[i,1,:].real, c='black', label='real')
        ax3.plot(r, duals_u[i,1,:].imag, c='orange', label='imag')
        ax3.set_ylabel(r'$u_\theta^{\dagger}$')
        ax4.plot(r, duals_u[i,2,:].real, c='black', label='real')
        ax4.plot(r, duals_u[i,2,:].imag, c='orange', label='imag')
        ax4.set_ylabel(r'$u_r^{\dagger}$')
        ax5.plot(r, rho*np.conj(duals_u[i,1,:])*efs_u[i,1,:], c='blue')
        ax5.set_ylabel(r'$\rho u_\theta^{\dagger,*}u_\theta$')
        ax6.plot(r, rho*np.conj(duals_u[i,2,:])*efs_u[i,2,:], c='blue')
        ax6.set_ylabel(r'$\rho u_r^{\dagger,*}u_r$')



        plt.suptitle('ev = {:.3e}'.format(ev))


        for ax in axs:
            ax.set_xlim(0, r.max())

        for ax in axs_bot:
            ax.set_xlabel('r')
        
        plt.savefig('{}/ef_{:03d}.png'.format(out_dir, i), bbox_inches='tight')
        for ax in axs:
            ax.clear()



