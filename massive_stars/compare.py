import h5py
import numpy as np
import matplotlib.pyplot as plt

theory_file = 're4e3_damping/damping_theory_power/simulated_powers.h5'
sim_file = 're4e3_waves/surface_power/power_spectra.h5'

with h5py.File(theory_file, 'r') as f:
    theory_freqs = f['freqs'][()]
    theory_powers = f['powers'][()]


with h5py.File(sim_file, 'r') as f:
    sim_freqs = f['freqs'][()]
    sim_powers = f['shell(s1_S2,r=R)_power_per_ell'][()]

print(sim_freqs.shape, sim_powers.shape)

for i in range(theory_powers.shape[0]):
    plt.loglog(sim_freqs, sim_powers[:,i+1], label='sim', c='k')
    plt.loglog(theory_freqs, theory_powers[i,:], label='theory', c='orange')
    plt.title('ell = {}'.format(i+1))
    plt.ylim(1e-30, 1e-14)
    plt.xlim(3e-3, 1e0)
    plt.legend()
    plt.show()
