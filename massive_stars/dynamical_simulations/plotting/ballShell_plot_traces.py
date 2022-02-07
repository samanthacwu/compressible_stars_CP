"""
Script for plotting traces of evaluated scalar quantities vs. time from a BallShell simulation.

Usage:
    ballShell_plot_traces.py <root_dir> [options]

Options:
    --data_dir=<dir>                    Name of data handler directory [default: scalars]
    --fig_name=<fig_name>               Output directory for figures [default: traces]
    --start_file=<start_file>           Dedalus output file to start at [default: 1]
    --n_files=<num_files>               Number of files to plot [default: 100000]
    --dpi=<dpi>                         Image pixel density [default: 150]
"""
from docopt import docopt
args = docopt(__doc__)
from plotpal.scalars import ScalarFigure, ScalarPlotter

# Read in master output directory
root_dir    = args['<root_dir>']
data_dir    = args['--data_dir']
if root_dir is None:
    print('No dedalus output dir specified, exiting')
    import sys
    sys.exit()

fig_name    = args['--fig_name']
start_file  = int(args['--start_file'])
n_files     = args['--n_files']
if n_files is not None: 
    n_files = int(n_files)

figs = []

this_fig = ScalarFigure(num_rows=1, num_cols=1, col_inch=6, fig_name='reynolds')
this_fig.add_field(0, 'Re_avg_ball')
figs.append(this_fig)

this_fig = ScalarFigure(num_rows=1, num_cols=1, col_inch=6, fig_name='energy')
this_fig.add_field(0, 'TE_ball')
this_fig.add_field(0, 'KE_ball')
this_fig.add_field(0, 'TE_shell')
this_fig.add_field(0, 'KE_shell')
figs.append(this_fig)

this_fig = ScalarFigure(num_rows=2, num_cols=1, col_inch=6, fig_name='angular_momentum')
this_fig.add_field(0, 'Lx_AM_ball')
this_fig.add_field(0, 'Ly_AM_ball')
this_fig.add_field(0, 'Lz_AM_ball')
this_fig.add_field(1, 'Lx_AM_shell')
this_fig.add_field(1, 'Ly_AM_shell')
this_fig.add_field(1, 'Lz_AM_shell')
figs.append(this_fig)


# Load in figures and make plots
plotter = ScalarPlotter(root_dir, data_dir, fig_name, start_file=start_file, n_files=n_files)
plotter.load_figures(figs)
plotter.plot_figures(dpi=int(args['--dpi']))
plotter.plot_convergence_figures(dpi=int(args['--dpi']))
