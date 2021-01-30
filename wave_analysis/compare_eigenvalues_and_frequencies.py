"""
Compare EVP output frequencies to a wave spectrum.

Usage:
    compare_eigenvalues_and_frequences.py <sh_spectrum_file> <evp_data_file> [options]
    compare_eigenvalues_and_frequences.py <sh_spectrum_file> <evp_data_file_lores> <evp_data_file_hires> [options]

Options:
    --freq_power=<p>    Power law exponent for convective wave driving [default: -13/2]

"""
from fractions import Fraction

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from docopt import docopt
args = docopt(__doc__)

sh_spectrum_file = args['<sh_spectrum_file>']
evp_data_file = args['<evp_data_file>']
evp_data_file_lores = args['<evp_data_file_lores>']
evp_data_file_hires = args['<evp_data_file_hires>']

if evp_data_file_hires is None:
    ell = int(evp_data_file.split('ell')[-1].split('_')[0])
    with h5py.File(evp_data_file, 'r') as ef:
        complex_eigenvalues = ef['good_evalues'][()]
        good_values = complex_eigenvalues.real >= 0
        complex_eigenvalues = complex_eigenvalues[good_values]
        s1_amplitudes = ef['good_s1_amplitudes'][()][good_values]
        mode_energies = ef['good_integ_energies'][()][good_values]

        #TODO: come up with a better way of filtering out bad eigenvalues.
    #    good_values = np.abs(1/(complex_eigenvalues.imag/(2*np.pi))) > 10
        good_values = np.isfinite(complex_eigenvalues)
else:
    ell = int(evp_data_file_lores.split('ell')[-1].split('_')[0])
    with h5py.File(evp_data_file_lores, 'r') as ef:
        lores_complex_eigenvalues = ef['good_evalues'][()]
        lores_good_values = lores_complex_eigenvalues.real >= 0
        lores_complex_eigenvalues = lores_complex_eigenvalues[lores_good_values]
        lores_s1_amplitudes = ef['good_s1_amplitudes'][()][lores_good_values]
        lores_mode_energies = ef['good_integ_energies'][()][lores_good_values]
    with h5py.File(evp_data_file_hires, 'r') as ef:
        hires_complex_eigenvalues = ef['good_evalues'][()]
        hires_good_values = hires_complex_eigenvalues.real >= 0
        hires_complex_eigenvalues = hires_complex_eigenvalues[hires_good_values]
        hires_s1_amplitudes = ef['good_s1_amplitudes'][()][hires_good_values]
        hires_mode_energies = ef['good_integ_energies'][()][hires_good_values]

    lores_freqs = lores_complex_eigenvalues.real/(2*np.pi)
    hires_freqs = hires_complex_eigenvalues.real/(2*np.pi)
    good_lores_eigenvalues = []
    good_lores_s1_amplitudes = []
    good_lores_mode_energies = []
    good_hires_eigenvalues = []
    good_hires_s1_amplitudes = []
    good_hires_mode_energies = []
    for i, f in enumerate(lores_freqs):
        if f < 1e-2 or f > 1e2:
            continue
        else:
            hires_error = np.abs(hires_freqs - f)/f
            if hires_error.min() < 1e-2:
                good_lores_eigenvalues.append(lores_complex_eigenvalues[i])
                good_lores_s1_amplitudes.append(lores_s1_amplitudes[i])
                good_lores_mode_energies.append(lores_mode_energies[i])
                good_hires_eigenvalues.append(hires_complex_eigenvalues[np.argmin(hires_error)])
                good_hires_s1_amplitudes.append(hires_s1_amplitudes[np.argmin(hires_error)])
                good_hires_mode_energies.append(hires_mode_energies[np.argmin(hires_error)])

    good_lores_eigenvalues = np.array(good_lores_eigenvalues)
    good_hires_eigenvalues = np.array(good_hires_eigenvalues)
    good_hires_s1_amplitudes = np.array(good_hires_s1_amplitudes)
    good_hires_mode_energies = np.array(good_hires_mode_energies)

    print(good_lores_eigenvalues.real/(2*np.pi))
    plt.scatter(good_lores_eigenvalues.real/(2*np.pi), good_lores_mode_energies, c='r', label='lores')
    plt.scatter(good_hires_eigenvalues.real/(2*np.pi), good_hires_mode_energies, c='k', label='hires')
    plt.xlim(1e-1, 1e1)
    plt.yscale("log")
    plt.xscale('log')
    plt.legend(loc='best')
    plt.xlabel('frequency (1/day)')
    plt.ylabel('eigenvalue KE')
    plt.title(r'$\ell = ${}'.format(ell))
    plt.savefig('scratch/ell{:03d}_lores_hires_energies.png'.format(ell), dpi=300)




with h5py.File(sh_spectrum_file, 'r') as sf:
    power_per_ell = sf['power_per_ell'][()]
    ells = sf['ells'][()]
    freqs = sf['freqs_inv_day'][()]
    power = power_per_ell[:,ells.flatten() == ell]

fig = plt.figure()
for f in good_hires_eigenvalues:
    plt.axvline(np.abs(f.real)/(2*np.pi), c='b', lw=0.5)

plt.loglog(freqs, power, c='k')
plt.xlim(1e-1, 1e2)
plt.xlabel('frequency (day$^{-1}$)')
plt.ylabel('power')
plt.title(r'$\ell = ${}'.format(ell))
plt.savefig('scratch/ell{:03d}_identified_eigenmodes.png'.format(ell), dpi=300)

fig = plt.figure()
for f in good_hires_eigenvalues:
    plt.scatter(np.abs(f.real)/(2*np.pi), np.abs(1/f.imag)*(2*np.pi), c='b')
plt.yscale('log')
plt.xscale('log')

plt.xlim(1e-1, 1e1)
plt.ylim(1e-2, 1e7)
plt.xlabel('frequency (day$^{-1}$)')
plt.ylabel('decay time (day)')
plt.title(r'$\ell = ${}'.format(ell))
plt.savefig('scratch/ell{:03d}_decay_times.png'.format(ell), dpi=300)

good_omegas = good_hires_eigenvalues.real
domegas = np.abs(np.gradient(good_omegas))
Nm = good_omegas/domegas
Nm_func = interp1d(good_omegas, Nm)
power_slope = lambda om: om**(float(Fraction(args['--freq_power'])))

adjusted_energies = np.zeros_like(good_hires_eigenvalues, dtype=np.float64)
for i, e in enumerate(good_hires_eigenvalues):
    this_om = np.abs(e.real)
    decay   = np.abs(e.imag)
    shiode_energy = Nm_func(this_om)*power_slope(this_om)/decay
    adjusted_energies[i] = good_hires_s1_amplitudes[i]**2 * (shiode_energy / good_hires_mode_energies[i])

match_freq_guess = 2.9e-1
if ell == 2:
    match_freq_guess = 4.1e-1
elif ell == 3:
    match_freq_guess = 5e-1
elif ell == 4:
    match_freq_guess = 6.4e-1

real_frequencies = good_omegas/(2*np.pi)
match_freq = real_frequencies[real_frequencies > match_freq_guess][0]
match_freq_sim = np.argmin(np.abs(freqs - match_freq))
peak_found = False
while not peak_found:
    if   power[match_freq_sim-10:match_freq_sim].max()    < power[match_freq_sim]\
     and power[match_freq_sim+1:match_freq_sim+11].max()  < power[match_freq_sim]:
        peak_found = True
    else:
        match_freq_sim -= 1
    print(freqs[match_freq_sim], match_freq)
print(real_frequencies)

adjusted_energies *= power[match_freq_sim]/adjusted_energies[real_frequencies == match_freq][0]
#print(real_power[real_frequencies >= match_freq], real_frequencies, match_freq)



fig = plt.figure()
plt.loglog(freqs, power, c='k')
plt.scatter(real_frequencies, adjusted_energies, c='blue')#np.abs(eig_freqs[good_eig_freqs].real), E_of_om)
plt.xlim(1e-1, 1e1)
plt.xlabel('frequency (day$^{-1}$)')
plt.ylabel('power')
plt.title(r'$\ell = ${}'.format(ell))
plt.ylim(1e-12, 1e0)
plt.savefig('scratch/ell{:03d}_shiode_eqn_9.png'.format(ell), dpi=300)
