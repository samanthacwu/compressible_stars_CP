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

pleiades_dir=/nobackup/eanders/d3_stars/massive_stars/re4e3_damping/damping_theory_power/
local_dir=damping_theory_power
files=()
mkdir $local_dir
for ell in `seq 1 20`
do
    if [[ $ell -eq 1 ]]
    then
        files=s1_simulated_freq_spectrum_ell$ell\.png
    else
        files=(${files[@]},s1_simulated_freq_spectrum_ell$ell\.png)
    fi
done

files=(${files[@]},s1_simulated_freq_spectrum_summed_ells.png)


files={${files[@]}}
scp pfe:$pleiades_dir/$files ./$local_dir/

