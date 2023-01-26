#!/bin/bash

pleiades_dir=/nobackup/eanders/d3_stars/massive_stars/compressible_re3e4_damping/FT_SH_transform_wave_shells/
local_dir=wave_flux/
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

for freq in 0.03 0.05 0.1 0.2 0.5
do
    files=(${files[@]},ell_spectrum_freq$freq\.png)
done


files={${files[@]}}
scp pfe:$pleiades_dir/$files ./$local_dir/

