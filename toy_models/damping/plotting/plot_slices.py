"""
This script plots snapshots of the evolution of a 2D slice through the equator of a BallBasis simulation.

Usage:
    plot_slices.py <root_dir> [options]

Options:
    --data_dir=<dir>                    Name of data handler directory [default: slices]
    --fig_name=<fig_name>               Name of figure output directory & base name of saved figures [default: snapshots_equatorial]
    --start_fig=<fig_start_num>         Number of first figure file [default: 1]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<num_files>               Total number of files to plot
    --dpi=<dpi>                         Image pixel density [default: 200]

    --col_inch=<in>                     Number of inches / column [default: 3]
    --row_inch=<in>                     Number of inches / row [default: 3]
"""
from docopt import docopt
args = docopt(__doc__)
from plotpal.slices import SlicePlotter

# Read in master output directory
root_dir    = args['<root_dir>']
data_dir    = args['--data_dir']
if root_dir is None:
    print('No dedalus output dir specified, exiting')
    import sys
    sys.exit()

# Read in additional plot arguments
start_fig   = int(args['--start_fig'])
start_file  = int(args['--start_file'])
fig_name    = args['--fig_name']
n_files     = args['--n_files']
if n_files is not None: 
    n_files = int(n_files)

# Create Plotter object, tell it which fields to plot
plotter = SlicePlotter(root_dir, data_dir, fig_name, start_file=start_file, n_files=n_files)
plotter_kwargs = { 'col_inch' : int(args['--col_inch']), 'row_inch' : int(args['--row_inch']) }

plotter.setup_grid(num_rows=1, num_cols=2, polar=True, **plotter_kwargs)
kwargs = {'colatitude_basis' : 'theta', 'radial_basis' : 'r', 'r_inner' : 0, 'r_outer' : 2}
plotter.add_meridional_colormesh(left='T(phi=0)', right='T(phi=pi)', **kwargs)
plotter.add_meridional_colormesh(left='F', right='F', cmap='PuOr_r', vector_ind=0, **kwargs)
plotter.plot_colormeshes(start_fig=start_fig, dpi=int(args['--dpi']))
