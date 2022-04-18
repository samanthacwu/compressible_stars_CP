#!/bin/bash

#If we need to string format, do this.
#ellplots=()
#for ell in `seq 0 11`
#do
#    echo $ell
#    ellfmt=$(printf "%03d" $ell)
#    ellplots=(${ellplots[@]} $ellfmt)
#done
#for p in ${ellplots[@]}
#do
#    echo $p
#done

pleiades_dir=/nobackup/eanders/d3_stars/massive_stars/re4e3_damping/SH_wave_flux_spectra/
local_dir=wave_flux
files=()
mkdir $local_dir
for ell in `seq 1 10`
do
    if [[ $ell -eq 1 ]]
    then
        files=freq_spectrum_ell$ell\.png
    else
        files=(${files[@]},freq_spectrum_ell$ell\.png)
    fi
done

for freq in 0.01 0.05 0.08
do
    files=(${files[@]},ell_spectrum_freq$freq\.png)
done


files={${files[@]}}
scp pfe:$pleiades_dir/$files ./$local_dir/

