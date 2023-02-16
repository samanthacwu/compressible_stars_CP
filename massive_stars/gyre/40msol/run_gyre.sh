export MESASDK_ROOT="/Applications/mesasdk"
source "$MESASDK_ROOT/bin/mesasdk_init.sh"
export MESA_DIR="/Users/evananders/software/mesa-r21.12.1"
export OMP_NUM_THREADS="8"
export GYRE_DIR="/Users/evananders/software/gyre-6.0.1/"

mkdir gyre_output/
rm gyre_output/*
$GYRE_DIR/bin/gyre  gyre_ell01-05.in


