"""
Script for plotting equatorial cuts of a joint ball-sphere simulation.

Usage:
    plot_ballSphere_equatorial_slices.py <root_dir> [options]

Options:
    --fig_name=<fig_name>               Name of figure output directory & base name of saved figures [default: averaged_profiles]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<num_files>               Number of files to plot
    --dpi=<dpi>                         Image pixel density [default: 200]
    --avg_writes=<n_writes>             Number of output writes to average over [default: 40]

    --col_inch=<in>                     Number of inches / column [default: 6]
    --row_inch=<in>                     Number of inches / row [default: 3]

    --mesa_file=<f>                     NCC file for making full flux plot
    --polytrope                         Use polytropic background
    --r_inner=<r>                       linking shell-ball radius [default: 1.2]
    --r_outer=<r>                       outer shell radius [default: 2]
"""
import numpy as np
import matplotlib.pyplot as plt
from docopt import docopt
args = docopt(__doc__)
from plotpal.profiles import ProfilePlotter
from plotpal.plot_grid import PlotGrid as PG

# Read in master output directory
root_dir    = args['<root_dir>']
data_dir   = 'profiles'
fig_name   = 'avg_profile'
if root_dir is None:
    print('No dedalus output dir specified, exiting')
    import sys
    sys.exit()

start_file  = int(args['--start_file'])
avg_writes  = int(args['--avg_writes'])
fig_name    = args['--fig_name']
n_files     = args['--n_files']
if n_files is not None: 
    n_files = int(n_files)

plotter = ProfilePlotter(root_dir, file_dir=data_dir, fig_name=fig_name, start_file=start_file, n_files=n_files)

plotter.add_profile('TB',  avg_writes, basis='r_0')
plotter.add_profile('TS',  avg_writes, basis='r_1')

plotter_kwargs = { 'col_in' : int(args['--col_inch']), 'row_in' : int(args['--row_inch']) }
plotter.plot_avg_profiles(dpi=int(args['--dpi']), **plotter_kwargs)
