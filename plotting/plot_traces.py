"""
Script for plotting traces of evaluated scalar quantities vs. time.

Usage:
    plot_traces.py <root_dir> [options]

Options:
    --data_dir=<dir>                    Name of data handler directory [default: scalar]
    --fig_name=<fig_name>               Output directory for figures [default: traces]
    --start_file=<start_file>           Dedalus output file to start at [default: 1]
    --n_files=<num_files>               Number of files to plot [default: 100000]
    --dpi=<dpi>                         Image pixel density [default: 150]
    --lite                              Flag for a lite output run
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

fig2 = ScalarFigure(1, 1, col_in=6, fig_name='energy')
fig2.add_field(0, 'KE')
fig2.add_field(0, 'TE')
figs.append(fig2)


fig3 = ScalarFigure(1, 1, col_in=6, fig_name='re')
if 'cartesian' in root_dir or 'annulus' in root_dir:
    fig3.add_field(0, 'Re_rms')
else:
    fig3.add_field(0, 'Re_avg')
figs.append(fig3)


#fig4 = ScalarFigure(1, 1, col_in=6, fig_name='s')
#fig4.add_field(0, 's1')
#figs.append(fig4)


# Load in figures and make plots
plotter = ScalarPlotter(root_dir, file_dir=data_dir, fig_name=fig_name, start_file=start_file, n_files=n_files)
plotter.load_figures(figs)
plotter.plot_figures(dpi=int(args['--dpi']))
plotter.plot_convergence_figures(dpi=int(args['--dpi']))
