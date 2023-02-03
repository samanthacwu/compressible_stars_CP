#!/bin/bash

pleiades_dir=/nobackup/eanders/d3_stars/massive_stars/other_stars/msol3_twoRcore_re4e3_damping/
local_dir=basics
files=(snapshots_equatorial.mp4,flux_plots.mp4)
mkdir $local_dir
for sf in energy_fluc mass
do
    files=(${files[@]},traces/$sf\.png)
done

files={${files[@]}}
echo $files
scp pfe:$pleiades_dir/$files ./$local_dir/

