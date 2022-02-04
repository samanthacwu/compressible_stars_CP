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

if args['--mesa_file'] is None:
    raise ValueError("Must provide MESA file")
with h5py.File(args['--mesa_file'], 'r') as f:
    rB = f['rB'][()]
    rS = f['rS'][()]
    ρB = np.exp(f['ln_ρB'][()])
    ρS = np.exp(f['ln_ρS'][()])
    r = np.concatenate((rB.flatten(), rS.flatten()))
    ρ = np.concatenate((ρB.flatten(), ρS.flatten()))
    rho_func = interp1d(r,ρ)
    tau_sec= f['tau'][()]
    tau = tau_sec/(60*60*24)
    r_outer = f['r_outer'][()]
    L_sim = f['L'][()] #cm
    radius = r_outer * L_sim
    #Entropy units are erg/K/g
    s_c = f['s_c'][()]
    N2_plateau_sec = f['N2plateau'][()]
    N2_plateau = N2_plateau_sec * (60*60*24)**2
    N2max_shell = f['N2max_shell'][()]
    r_rcb = 1
    ρ_rcb = rho_func(r_rcb)
    N2_plateau_sim = N2_plateau_sec * tau_sec**2

def fit_waveflux_spectrum(ells, freqs, flux, f_norm_pow=-17/2, ell_norm_pow=4):
        detrended = flux/(ells**ell_norm_pow * freqs**f_norm_pow)
        window = np.abs(detrended[(ells > 0)*(ells <= 10)*(freqs > 2)*(freqs <= 8)])
        good = np.isfinite(np.log10(window))
        A0 = 10**(np.mean(np.log10(window[good])))
        A0_ur2 = 2*A0/(r_rcb*ρ_rcb*np.sqrt(N2_plateau_sim))
        wave_flux = lambda f, ell: A0 * f**f_norm_pow * ell**ell_norm_pow
        ur_squared = lambda f, ell: A0_ur2 * f**f_norm_pow * ell**(ell_norm_pow) * np.sqrt(1+1/ell)
        return wave_flux, ur_squared

sh_spectrum_file = args['<sh_spectrum_file>']
evp_data_file = args['<evp_data_file>']
wave_flux_file = args['<wave_flux_file>']
transfer_file = args['--transfer_file']

# Read in eigenvalues from EVP file
ell = int(evp_data_file.split('ell')[-1].split('_')[0])
with h5py.File(evp_data_file, 'r') as ef:
    complex_eigenvalues = ef['good_evalues'][()]
    complex_eigenvalues_inv_day = ef['good_evalues_inv_day'][()]
    s1_amplitudes = ef['s1_amplitudes'][()]
    integ_energies = ef['integ_energies'][()]

# Read in power spectrum from long simulation
with h5py.File(sh_spectrum_file, 'r') as sf:
    try:
        power_per_ell = sf['power_per_ell'][()]
    except:
        power_per_ell = sf['s1_surf_power_per_ell'][()]
    ells = sf['ells'][()]
    freqs = sf['freqs'][()]
    freqs_inv_day = sf['freqs_inv_day'][()]
    power = power_per_ell[:,ells.flatten() == ell]

# Read in transfer function
if transfer_file is not None:
    with h5py.File(transfer_file, 'r') as f:
        transfer = f['transfer'][()]
        transfer_om = f['om'][()]
        transfer_om_inv_day = f['om_inv_day'][()]
else:
    transfer = transfer_om = None
        
# Plot identified eigemodes
fig = plt.figure()
for f in complex_eigenvalues:
    print(f.real)
    plt.axvline(np.abs(f.real)/(2*np.pi), c='b', lw=0.5)

plt.loglog(freqs, power, c='k')
plt.xlim(1e-1, 1e2)
plt.xlabel('frequency ( sim units )')
plt.ylabel('power')
plt.title(r'$\ell = ${}'.format(ell))
plt.savefig('scratch/ell{:03d}_identified_eigenmodes.png'.format(ell), dpi=300)

# Plot decay timescales
fig = plt.figure()
for f in complex_eigenvalues:
    plt.scatter(np.abs(f.real)/(2*np.pi), np.abs(1/f.imag)*(2*np.pi), c='b')
plt.yscale('log')
plt.xscale('log')

plt.xlim(1e-1, 1e1)
plt.ylim(1e-2, 1e7)
plt.xlabel('frequency (sim units)')
plt.ylabel('decay time (sim units)')
plt.title(r'$\ell = ${}'.format(ell))
plt.savefig('scratch/ell{:03d}_decay_times.png'.format(ell), dpi=300)

omegas = complex_eigenvalues.real
real_frequencies = omegas/(2*np.pi)
domegas = np.abs(np.gradient(omegas))
Nm = omegas/domegas

if wave_flux_file is None:
    raise ValueError("Must provide wave flux file")
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
        waveflux_freqs = np.expand_dims(f['real_freqs'][()], axis=1)
        wave_flux = f['wave_flux'][()]  #wave flux calculation is currently r^2 * rho(r) * hat(ur) * conj(hat(p)) \propto dF / d ell / d f
#        wave_flux = 4 * np.pi * f['wave_flux'][()]  #wave flux calculation is currently r^2 * rho(r) * hat(ur) * conj(hat(p)) \propto dF / d ell / d f
#        #dF / d ell d f = delta F / (delta ell delta f); delta ell = 1; delta f = 1/T where T is total time of sample
#        #delta f = np.diff(waveflux_freqs)
#        d2F_dell_dlnf = waveflux_freqs*d2F_dell_df
#        this_ell_d2F_dell_dlnf = ell**3*d2F_dell_dlnf[:, ells.flatten() == ell]
#        waveflux_omegas = 2*np.pi*waveflux_freqs
#        this_ell_flux = d2F_dell_df[:, ells.flatten() == ell]
#        tau = waveflux_freqs.max()/waveflux_freqs_inv_day.max()

    wave_flux_func, ur2_func = fit_waveflux_spectrum(ells, waveflux_freqs, wave_flux)

    #wave flux vs omega fit
    #fit a omega^{freq_power} function to omega between the 3rd highest freq eigenvalue and the median eigenvalue
#    fit_power = -13/2
#    nvals = len(complex_eigenvalues)
#    good = (waveflux_omegas.flatten() > complex_eigenvalues.real[int(nvals/2)])*(waveflux_omegas.flatten() < complex_eigenvalues.real[2])
#    x = np.log10(waveflux_omegas.flatten())
#    y = np.log10(this_ell_d2F_dell_dlnf.flatten()/waveflux_omegas.flatten()**(fit_power))
#    mean_offset = np.sum(np.gradient(x[good])*y[good])/np.sum(np.gradient(x[good]))
#    wave_lum_func = lambda omega: 10**(mean_offset)*omega**(fit_power)
#    print('wave flux: {:.3e} * omega ^ {}'.format(10**(mean_offset), fit_power))
#
#    #Shiode eqn 9
#    fudge_factor_shiode = 3e1*(complex_eigenvalues.real/(2*np.pi))**2 / ell**3
#    shiode_energies = fudge_factor_shiode * (0.5 * Nm**(-1)*wave_lum_func(complex_eigenvalues.real)/np.abs(complex_eigenvalues.imag) / (4*np.pi*radius**2))
#    shiode_energies = fudge_factor_shiode * (0.5 * Nm**(-1)*wave_flux_func(complex_eigenvalues.real, ell)/np.abs(complex_eigenvalues.imag) / (4*np.pi*radius**2))
    fudge_shiode = 1e-1 / (ell*(ell+1))
    gamma = complex_eigenvalues.imag
    R = np.abs(s1_amplitudes**2/integ_energies)
    freq_eigenvalues = complex_eigenvalues.real / (2*np.pi)
    adjusted_energies = fudge_shiode * (wave_flux_func(freq_eigenvalues, ell) * domegas / gamma**2) * R


#    #Transfer function
#    #wave flux vs omega fit
#    #fit a omega^{freq_power} function to omega between the 3rd highest freq eigenvalue and the median eigenvalue
#    fit_power = -15/2
#    nvals = len(complex_eigenvalues)
#    good = (waveflux_omegas.flatten() > complex_eigenvalues.real[int(nvals/2)])*(waveflux_omegas.flatten() < complex_eigenvalues.real[2])
#    x = np.log10(waveflux_omegas.flatten())
#    y = np.log10(this_ell_flux.flatten()/waveflux_omegas.flatten()**(fit_power))
#    mean_offset = np.sum(np.gradient(x[good])*y[good])/np.sum(np.gradient(x[good]))
#    wave_lum_func = lambda omega: 10**(mean_offset)*omega**(fit_power)
#    print('wave flux: {:.3e} * omega ^ {}'.format(10**(mean_offset), fit_power))
#
#
#    #TODO: check to make sure that both N^2 and om^2 are in angular frequency units
##    H = -1/grad_ln_ρ_func(radius)
#    k_h = np.sqrt(ell*(ell+1))/radius
##    k_r = np.sqrt((N2_mesa_func(radius)/transfer_om**2 - 1)*k_h**2 - 1/(4*H**2))
##    print(N2_sim(radius)/transfer_om**2 - 1, k_h, 1/(2*H))
##    k_r = np.sqrt((N2_sim(radius)/transfer_om**2 - 1)*k_h**2 - 1/(4*H**2))
##    k = np.sqrt(k_r**2 + k_h**2)
#
#    print(transfer_om.max(), transfer_om_inv_day.max())
#    u_r2 = k_h/(N2_mesa_func(radius) - transfer_om**2)*wave_lum_func(transfer_om) / (4*np.pi*radius**2)
##    u_r2 *= transfer_om * k**2 / k_r
##    u_r2 /= 4*np.pi*radius**2 * np.exp(ln_ρ_func(1)) * N2_sim(1)
##    u_r = np.sqrt( ( (transfer_om_sim)*k**2/(k_r) ) * (1/((4*np.pi*radius**2) * np.exp(ln_ρ_func(radius))*N2_sim(radius))) * wave_lum_func(transfer_om) )
#    u_r = np.sqrt(u_r2)
    fudge_factor_transfer =  1e-1 / (ell*(ell+1))
    transfer_power = fudge_factor_transfer*ur2_func(transfer_om/(2*np.pi), ell)*transfer**2


fig = plt.figure()
plt.loglog(freqs_inv_day, power, c='k', label='IVP')
plt.scatter(complex_eigenvalues_inv_day.real/(2*np.pi), adjusted_energies, c='blue', label='shiode')
if transfer_file is not None:
    plt.loglog(transfer_om_inv_day/(2*np.pi), transfer_power, c='orange', alpha=0.8, label='transfer')
plt.xlim(1e-1, 1e1)
plt.xlabel('frequency (day$^{-1}$)')
plt.ylabel('power')
plt.title(r'$\ell = ${}'.format(ell))
plt.ylim(1e-18, 1e2)
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



