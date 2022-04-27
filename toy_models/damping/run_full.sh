python3 clean.py
mpiexec_mpt -n 16 python3 boussinesq_waveconv.py
mpiexec_mpt -n 28 python3 post_ivp_SH_transform.py
#python3 post_ivp_SH_wave_flux.py
python3 post_ivp_SH_power_spectrum.py
python3 single_mode_amplitude.py
