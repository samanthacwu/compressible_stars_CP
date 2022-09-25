import numpy as np
import matplotlib.pyplot as plt
import h5py

from scipy.interpolate import interp1d


for ell in [1,5,10]:
    with h5py.File('eigenvalues/transfer_ell{:03d}_eigenvalues.h5'.format(ell),'r') as f:
        om = f['om'][()]
        transfer = f['transfer'][()]
        freq = om/(2*np.pi)
    with h5py.File('eigenvalues/duals_ell{:03d}_eigenvalues.h5'.format(ell),'r') as f:
        smooth_oms = f['smooth_oms'][()]
        smooth_depths = f['smooth_depths'][()]
        depths = interp1d(smooth_oms/(2*np.pi), smooth_depths)(freq)

    plt.loglog(freq,np.exp(-depths)*transfer, label='ell={}'.format(ell))
plt.ylim(1e-1, 1e5)
plt.xlim(1e-2,None)


plt.legend()
plt.xlabel('f')
plt.ylabel('transfer')
plt.show()

