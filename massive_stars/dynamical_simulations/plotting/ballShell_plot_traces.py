"""
Script for plotting traces of evaluated scalar quantities vs. time from a BallShell simulation.

Usage:
    ballShell_plot_traces.py <root_dir> [options]

Options:
    --data_dir=<dir>                    Name of data handler directory [default: scalar]
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

scalars = solver.evaluator.add_file_handler('{:s}/scalars'.format(out_dir), sim_dt=scalar_dt, max_writes=np.inf)
scalars.add_task(vol_avg_B(Re*(u_B_squared)**(1/2)), name='Re_avg_ball',  layout='g')
scalars.add_task(vol_avg_S(Re*(u_S_squared)**(1/2)), name='Re_avg_shell', layout='g')
scalars.add_task(vol_avg_B(ρ_B*u_B_squared/2), name='KE_ball',   layout='g')
scalars.add_task(vol_avg_S(ρ_S*u_S_squared/2), name='KE_shell',  layout='g')
scalars.add_task(vol_avg_B(ρ_B*T_B*s1_B), name='TE_ball',  layout='g')
scalars.add_task(vol_avg_S(ρ_S*T_S*s1_S), name='TE_shell', layout='g')
scalars.add_task(vol_avg_B(Lx_AM_B), name='Lx_AM_ball', layout='g')
scalars.add_task(vol_avg_B(Ly_AM_B), name='Ly_AM_ball', layout='g')
scalars.add_task(vol_avg_B(Lz_AM_B), name='Lz_AM_ball', layout='g')
scalars.add_task(vol_avg_S(Lx_AM_S), name='Lx_AM_shell', layout='g')
scalars.add_task(vol_avg_S(Ly_AM_S), name='Ly_AM_shell', layout='g')
scalars.add_task(vol_avg_S(Lz_AM_S), name='Lz_AM_shell', layout='g')
analysis_tasks.append(scalars)

n_files = int(n_files)

figs = []

this_fig = ScalarFigure(1, 1, col_in=6, fig_name='reynolds')
this_fig.add_field(0, 'Re_avg_ball')
figs.append(this_fig)

this_fig = ScalarFigure(1, 1, col_in=6, fig_name='energy')
this_fig.add_field(0, 'TE_ball')
this_fig.add_field(0, 'KE_ball')
this_fig.add_field(0, 'TE_shell')
this_fig.add_field(0, 'KE_shell')
figs.append(this_fig)

this_fig = ScalarFigure(2, 1, col_in=6, fig_name='angular_momentum')
this_fig.add_field(0, 'Lx_AM_ball')
this_fig.add_field(0, 'Ly_AM_ball')
this_fig.add_field(0, 'Lz_AM_ball')
this_fig.add_field(1, 'Lx_AM_shell')
this_fig.add_field(1, 'Ly_AM_shell')
this_fig.add_field(1, 'Lz_AM_shell')
figs.append(this_fig)


# Load in figures and make plots
plotter = ScalarPlotter(root_dir, file_dir=data_dir, fig_name=fig_name, start_file=start_file, n_files=n_files)
plotter.load_figures(figs)
plotter.plot_figures(dpi=int(args['--dpi']))
plotter.plot_convergence_figures(dpi=int(args['--dpi']))
