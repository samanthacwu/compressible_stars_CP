for freq in 0.1 0.2 0.4 0.8
do
    for ell in `seq 1 11`
    do
        echo $ell $freq
        python3 clean.py
        mpiexec_mpt -n 16 python3 boussinesq_waveconv.py --ell=$ell --freq=$freq
        mpiexec_mpt -n 28 python3 post_ivp_SH_transform.py
        #python3 post_ivp_SH_wave_flux.py
        python3 post_ivp_SH_power_spectrum.py
        python3 single_mode_amplitude.py  --ell=$ell --freq=$freq
        mv shells/ shells_ell$ell\_freq$freq\/
    done
done
