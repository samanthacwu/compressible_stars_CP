#!/bin/zsh
#
#Run this file from the ncc_creation/ folder.
python3 make_ball_2shells_nccs.py --Re=4e3 --NB=128 --NS1=128 --NS2=32 --dealias=1.5
python3 make_ball_2shells_nccs.py --Re=4e3 --NB=128 --NS1=128 --NS2=32 --dealias=1
python3 make_ball_2shells_nccs.py --Re=4e3 --NB=192 --NS1=192 --NS2=48 --dealias=1
python3 make_ball_2shells_nccs.py --Re=1e4 --NB=256 --NS1=256 --NS2=48 --dealias=1.5
python3 make_ball_2shells_nccs.py --Re=1e4 --NB=256 --NS1=256 --NS2=48 --dealias=1
python3 make_ball_2shells_nccs.py --Re=1e4 --NB=384 --NS1=384 --NS2=64 --dealias=1
#python3 make_ball_2shells_nccs.py --Re=2e4 --NB=512 --NS1=256 --NS2=64 --dealias=1.5
#python3 make_ball_2shells_nccs.py --Re=2e4 --NB=512 --NS1=256 --NS2=64 --dealias=1
#python3 make_ball_2shells_nccs.py --Re=2e4 --NB=768 --NS1=384 --NS2=96 --dealias=1
