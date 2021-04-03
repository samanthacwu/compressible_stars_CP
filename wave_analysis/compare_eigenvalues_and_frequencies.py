"""
Compare EVP output frequencies to a wave spectrum.

Usage:
    compare_eigenvalues_and_frequences.py <sh_spectrum_file> <evp_data_file> [options]
    compare_eigenvalues_and_frequences.py <sh_spectrum_file> <evp_data_file> <wave_flux_file> [options]

Options:
    --freq_power=<p>     Power law exponent for convective wave driving [default: -13/2]
    --transfer_file=<f>  File to get the transfer function from

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
wave_flux_file = args['<wave_flux_file>']
transfer_file = args['--transfer_file']

ell = int(evp_data_file.split('ell')[-1].split('_')[0])
with h5py.File(evp_data_file, 'r') as ef:
    complex_eigenvalues = ef['good_evalues'][()]
    s1_amplitudes = ef['s1_amplitudes'][()]
    integ_energies = ef['integ_energies'][()]

with h5py.File(sh_spectrum_file, 'r') as sf:
    try:
        power_per_ell = sf['power_per_ell'][()]
    except:
        power_per_ell = sf['s1_power_per_ell'][()]
    ells = sf['ells'][()]
    freqs = sf['freqs_inv_day'][()]
    power = power_per_ell[:,ells.flatten() == ell]

if transfer_file is not None:
    with h5py.File(transfer_file, 'r') as f:
        transfer = f['transfer'][()]
        transfer_om = f['om'][()]
else:
    transfer = transfer_om = None
        

fig = plt.figure()
for f in complex_eigenvalues:
    print(f.real)
    plt.axvline(np.abs(f.real)/(2*np.pi), c='b', lw=0.5)

plt.loglog(freqs, power, c='k')
plt.xlim(1e-1, 1e2)
plt.xlabel('frequency (day$^{-1}$)')
plt.ylabel('power')
plt.title(r'$\ell = ${}'.format(ell))
plt.savefig('scratch/ell{:03d}_identified_eigenmodes.png'.format(ell), dpi=300)

fig = plt.figure()
for f in complex_eigenvalues:
    plt.scatter(np.abs(f.real)/(2*np.pi), np.abs(1/f.imag)*(2*np.pi), c='b')
plt.yscale('log')
plt.xscale('log')

plt.xlim(1e-1, 1e1)
plt.ylim(1e-2, 1e7)
plt.xlabel('frequency (day$^{-1}$)')
plt.ylabel('decay time (day)')
plt.title(r'$\ell = ${}'.format(ell))
plt.savefig('scratch/ell{:03d}_decay_times.png'.format(ell), dpi=300)

omegas = complex_eigenvalues.real
real_frequencies = omegas/(2*np.pi)
domegas = np.abs(np.gradient(omegas))
Nm = omegas/domegas

if wave_flux_file is None:
    power_slope = lambda om: om**(float(Fraction(args['--freq_power'])))
    shiode_energies = Nm*power_slope(complex_eigenvalues.real)/complex_eigenvalues.imag
    adjusted_energies = s1_amplitudes**2 * (shiode_energies / integ_energies)

    match_freq_guess = 4.5e-1
    if ell == 2:
        match_freq_guess = 4.1e-1
    elif ell == 3:
        match_freq_guess = 5e-1
    elif ell == 4:
        match_freq_guess = 6.4e-1
    elif ell == 5:
        match_freq_guess = 6e-1
    elif ell == 6:
        match_freq_guess = 7.5e-1
    elif ell == 7:
        match_freq_guess = 8e-1
    elif ell == 8:
        match_freq_guess = 9.5e-1
    elif ell == 9:
        match_freq_guess = 1

    match_freq = real_frequencies[real_frequencies > match_freq_guess][-1]
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

    transfer_power = transfer**2*power_slope(transfer_om)
    transfer_power_func = interp1d(transfer_om/(2*np.pi), transfer_power)
    sim_power_func = interp1d(freqs, power.squeeze())
    transfer_power *= sim_power_func(match_freq_guess)/transfer_power_func(match_freq_guess)
else:
    radius = 1.15
    with h5py.File(wave_flux_file, 'r') as f:
        ells = np.expand_dims(f['ells'][()].flatten(), axis=0)
        freqs_inv_day = np.expand_dims(f['real_freqs_inv_day'][()], axis=1)
        freqs_sim = np.expand_dims(f['real_freqs'][()], axis=1)
        d2F_dell_df = f['wave_flux'][()]/radius**2 #wave flux calculation is currently r^2 * rho(r) * hat(ur) * conj(hat(p))
        d2F_dlnell_dlnf = ells*freqs_sim*d2F_dell_df
        sim_omegas = 2*np.pi*freqs_inv_day
        this_ell_flux = d2F_dlnell_dlnf[:, ells.flatten() == ell]
        wave_flux_func = interp1d(sim_omegas.flatten(), this_ell_flux.flatten())
        tau = freqs_sim.max()/freqs_inv_day.max()

    fudge_factor = 1e-5
    shiode_energies = fudge_factor * (0.5 * Nm*wave_flux_func(complex_eigenvalues.real)/np.abs(tau*complex_eigenvalues.imag))
    adjusted_energies = np.abs(s1_amplitudes**2/integ_energies) * shiode_energies

    transfer_power = fudge_factor*transfer**2*wave_flux_func(transfer_om)
#    transfer_power_func = interp1d(transfer_om/(2*np.pi), transfer_power)
#    sim_power_func = interp1d(freqs, power.squeeze())
#    transfer_power *= sim_power_func(match_freq_guess)/transfer_power_func(match_freq_guess)


fig = plt.figure()
plt.loglog(freqs, power, c='k', label='IVP')
plt.scatter(real_frequencies, adjusted_energies, c='blue', label='shiode')
if transfer_file is not None:
    plt.loglog(transfer_om/(2*np.pi), transfer_power, c='orange', alpha=0.8, label='transfer')
plt.xlim(1e-1, 1e1)
plt.xlabel('frequency (day$^{-1}$)')
plt.ylabel('power')
plt.title(r'$\ell = ${}'.format(ell))
plt.ylim(1e-14, 1e0)
plt.legend()
plt.savefig('scratch/ell{:03d}_data_theory_comparison.png'.format(ell), dpi=300)

fig = plt.figure()
plt.scatter(complex_eigenvalues.real/(2*np.pi), s1_amplitudes**2, c='blue', label='|s1|$^2$')#np.abs(eig_freqs[good_eig_freqs].real), E_of_om)
plt.scatter(complex_eigenvalues.real/(2*np.pi), integ_energies, c='red', label='energy')#np.abs(eig_freqs[good_eig_freqs].real), E_of_om)
plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlim(1e-1, 1e1)
plt.xlabel('frequency (day$^{-1}$)')
plt.ylabel('|s1|$^2$ / integ energy')
plt.title(r'$\ell = ${}'.format(ell))
plt.savefig('scratch/ell{:03d}_energy_s1_ratio.png'.format(ell), dpi=300)
