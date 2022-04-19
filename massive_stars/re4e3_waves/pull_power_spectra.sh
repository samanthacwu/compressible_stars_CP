#!/bin/bash

pleiades_dir=/nobackup/eanders/d3_stars/massive_stars/re4e3_waves/SH_power_shells/
local_dir=surface_power
files=()
mkdir $local_dir
for ell in `seq 1 20`
do
    ellfmt=$(printf "%03d" $ell)
    if [[ $ell -eq 1 ]]
    then
        files=shell_s1_S2_rR_v0_ell_$ellfmt\.png
    else
        files=(${files[@]},shell_s1_S2_rR_v0_ell_$ellfmt\.png)
    fi
done

files=(${files[@]},shell_s1_S2_rR_v0_summed_power.png,power_spectra.h5)
files={${files[@]}}
scp pfe:$pleiades_dir/$files ./$local_dir/

