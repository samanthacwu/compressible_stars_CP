#!/bin/bash

# A POSIX variable
OPTIND=1         # Reset in case getopts has been used previously in the shell.

# Initialize our own variables:
DIR=""
NCORE=""

while getopts ":d:n:h?:" opt; do
    case "$opt" in
    h|\?)
        echo "specify dir with -d and core number with -n" 
        exit 0
        ;;
    d)  DIR=$OPTARG
        ;;
    n)  NCORE=$OPTARG
        ;;
    esac
done
echo $DIR
echo $NCORE
echo "Processing wave flux $DIR on $NCORE cores"

mpiexec_mpt -n $NCORE python3 spherical_harmonic_transform.py $DIR --data_dir=wave_shell_slices --shell_basis
mpiexec_mpt -n 1 python3 spherical_harmonic_wave_flux.py $DIR
