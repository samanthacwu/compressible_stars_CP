"""
Compare EVP output frequencies to a wave spectrum.

Usage:
    compare_eigenvalues_and_frequences.py <sh_spectrum_file> <evp_data_file> [options]
    compare_eigenvalues_and_frequences.py <sh_spectrum_file> <evp_data_file> <wave_flux_file> [options]

Options:
    --freq_power=<p>     Power law exponent for convective wave driving [default: -15/2]
    --transfer_file=<f>  File to get the transfer function from
    --mesa_file=<f>      MESA-derived NCC file

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
    complex_eigenvalues_inv_day = ef['good_evalues_inv_day'][()]
    s1_amplitudes = ef['s1_amplitudes'][()]
    integ_energies = ef['integ_energies'][()]

with h5py.File(sh_spectrum_file, 'r') as sf:
    try:
        power_per_ell = sf['power_per_ell'][()]
    except:
        power_per_ell = sf['s1_power_per_ell'][()]
    ells = sf['ells'][()]
    freqs = sf['freqs'][()]
    freqs_inv_day = sf['freqs_inv_day'][()]
    power = power_per_ell[:,ells.flatten() == ell]

if transfer_file is not None:
    with h5py.File(transfer_file, 'r') as f:
        transfer = f['transfer'][()]
        transfer_om = f['om'][()]
        transfer_om_inv_day = f['om_inv_day'][()]
else:
    transfer = transfer_om = None
        

fig = plt.figure()
for f in complex_eigenvalues_inv_day:
    print(f.real)
    plt.axvline(np.abs(f.real)/(2*np.pi), c='b', lw=0.5)

plt.loglog(freqs_inv_day, power, c='k')
plt.xlim(1e-1, 1e2)
plt.xlabel('frequency (day$^{-1}$)')
plt.ylabel('power')
plt.title(r'$\ell = ${}'.format(ell))
plt.savefig('scratch/ell{:03d}_identified_eigenmodes.png'.format(ell), dpi=300)

fig = plt.figure()
for f in complex_eigenvalues_inv_day:
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
    with h5py.File(args['--mesa_file'], 'r') as f:
        r_mesa = f['r_mesa'][()]/f['L'][()]
        N2_mesa = f['N2_mesa'][()]*f['tau'][()]**2
        N2_mesa_func = interp1d(r_mesa, N2_mesa)

        sim_grad_T = np.concatenate( (f['grad_TB'][()][2,:].squeeze(), f['grad_TS'][()][2,:].squeeze()) )
        sim_grad_s0 = np.concatenate( (f['grad_s0B'][()][2,:].squeeze(), f['grad_s0S'][()][2,:].squeeze()) )
        sim_grad_ln_ρ = np.concatenate( (f['grad_ln_ρB'][()][2,:].squeeze(), f['grad_ln_ρS'][()][2,:].squeeze()) )
        sim_ln_ρ = np.concatenate( (f['ln_ρB'][()].squeeze(), f['ln_ρS'][()].squeeze()) )
        sim_radius = np.concatenate( (f['rB'][()].squeeze(), f['rS'][()].squeeze()) )

        grad_s0_func = interp1d(sim_radius, sim_grad_s0)
        grad_T_func = interp1d(sim_radius, sim_grad_T)
        grad_ln_ρ_func = interp1d(sim_radius, sim_grad_ln_ρ)
        ln_ρ_func = interp1d(sim_radius, sim_ln_ρ)
        N2_sim = lambda r: -grad_T_func(r)*grad_s0_func(r)
        
       
    radius = 1.15

    with h5py.File(wave_flux_file, 'r') as f:
        ells = np.expand_dims(f['ells'][()].flatten(), axis=0)
        waveflux_freqs_inv_day = np.expand_dims(f['real_freqs_inv_day'][()], axis=1)
        waveflux_freqs = np.expand_dims(f['real_freqs'][()], axis=1)
        d2F_dell_df = 4 * np.pi * f['wave_flux'][()]  #wave flux calculation is currently r^2 * rho(r) * hat(ur) * conj(hat(p)) \propto dF / d ell / d f
        #delta F = rho * hat(ur) * conj(hat(p))
        #sum[delta F] = F
        #dF / d ell d f = delta F / (delta ell delta f); delta ell = 1; delta f = 1/T where T is total time of sample
        #delta f = np.diff(waveflux_freqs)
        d2F_dell_dlnf = waveflux_freqs*d2F_dell_df
        this_ell_d2F_dell_dlnf = ell**3*d2F_dell_dlnf[:, ells.flatten() == ell]
        waveflux_omegas = 2*np.pi*waveflux_freqs
        this_ell_flux = d2F_dell_df[:, ells.flatten() == ell]
        tau = waveflux_freqs.max()/waveflux_freqs_inv_day.max()

    #wave flux vs omega fit
    #fit a omega^{freq_power} function to omega between the 3rd highest freq eigenvalue and the median eigenvalue
    fit_power = -13/2
    nvals = len(complex_eigenvalues)
    good = (waveflux_omegas.flatten() > complex_eigenvalues.real[int(nvals/2)])*(waveflux_omegas.flatten() < complex_eigenvalues.real[2])
    x = np.log10(waveflux_omegas.flatten())
    y = np.log10(this_ell_d2F_dell_dlnf.flatten()/waveflux_omegas.flatten()**(fit_power))
    mean_offset = np.sum(np.gradient(x[good])*y[good])/np.sum(np.gradient(x[good]))
    wave_lum_func = lambda omega: 10**(mean_offset)*omega**(fit_power)
    print('wave flux: {:.3e} * omega ^ {}'.format(10**(mean_offset), fit_power))

    #Shiode eqn 9
    fudge_factor_shiode = 3e1*(complex_eigenvalues.real/(2*np.pi))**2 / ell**3
    shiode_energies = fudge_factor_shiode * (0.5 * Nm**(-1)*wave_lum_func(complex_eigenvalues.real)/np.abs(complex_eigenvalues.imag) / (4*np.pi*radius**2))
    adjusted_energies = np.abs(s1_amplitudes**2/integ_energies) * shiode_energies


    #Transfer function
    #wave flux vs omega fit
    #fit a omega^{freq_power} function to omega between the 3rd highest freq eigenvalue and the median eigenvalue
    fit_power = -15/2
    nvals = len(complex_eigenvalues)
    good = (waveflux_omegas.flatten() > complex_eigenvalues.real[int(nvals/2)])*(waveflux_omegas.flatten() < complex_eigenvalues.real[2])
    x = np.log10(waveflux_omegas.flatten())
    y = np.log10(this_ell_flux.flatten()/waveflux_omegas.flatten()**(fit_power))
    mean_offset = np.sum(np.gradient(x[good])*y[good])/np.sum(np.gradient(x[good]))
    wave_lum_func = lambda omega: 10**(mean_offset)*omega**(fit_power)
    print('wave flux: {:.3e} * omega ^ {}'.format(10**(mean_offset), fit_power))


    #TODO: check to make sure that both N^2 and om^2 are in angular frequency units
#    H = -1/grad_ln_ρ_func(radius)
    k_h = np.sqrt(ell*(ell+1))/radius
#    k_r = np.sqrt((N2_mesa_func(radius)/transfer_om**2 - 1)*k_h**2 - 1/(4*H**2))
#    print(N2_sim(radius)/transfer_om**2 - 1, k_h, 1/(2*H))
#    k_r = np.sqrt((N2_sim(radius)/transfer_om**2 - 1)*k_h**2 - 1/(4*H**2))
#    k = np.sqrt(k_r**2 + k_h**2)

    print(transfer_om.max(), transfer_om_inv_day.max())
    u_r2 = k_h/(N2_mesa_func(radius) - transfer_om**2)*wave_lum_func(transfer_om) / (4*np.pi*radius**2)
#    u_r2 *= transfer_om * k**2 / k_r
#    u_r2 /= 4*np.pi*radius**2 * np.exp(ln_ρ_func(1)) * N2_sim(1)
#    u_r = np.sqrt( ( (transfer_om_sim)*k**2/(k_r) ) * (1/((4*np.pi*radius**2) * np.exp(ln_ρ_func(radius))*N2_sim(radius))) * wave_lum_func(transfer_om) )
    u_r = np.sqrt(u_r2)
    fudge_factor_transfer =  1e2 / ell**2#1e-2# / k_h**2
    transfer_power = fudge_factor_transfer*(transfer*u_r)**2


fig = plt.figure()
plt.loglog(freqs_inv_day, power, c='k', label='IVP')
plt.scatter(complex_eigenvalues_inv_day.real/(2*np.pi), adjusted_energies, c='blue', label='shiode')
if transfer_file is not None:
    plt.loglog(transfer_om_inv_day/(2*np.pi), transfer_power, c='orange', alpha=0.8, label='transfer')
plt.xlim(1e-1, 1e1)
plt.xlabel('frequency (day$^{-1}$)')
plt.ylabel('power')
plt.title(r'$\ell = ${}'.format(ell))
plt.ylim(1e-14, 1e2)
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



