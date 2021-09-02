"""
Script for plotting equatorial cuts of a joint ball-sphere simulation.

Usage:
    plot_ballSphere_equatorial_slices.py <root_dir> [options]

Options:
    --fig_name=<fig_name>               Name of figure output directory & base name of saved figures [default: snapshots_eq]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<num_files>               Number of files to plot
    --dpi=<dpi>                         Image pixel density [default: 200]
    --avg_writes=<n_writes>             Number of output writes to average over [default: 100]

    --col_inch=<in>                     Number of inches / column [default: 6]
    --row_inch=<in>                     Number of inches / row [default: 3]

    --mesa_file=<f>                     NCC file for making full flux plot
    --polytrope                         Use polytropic background
    --r_inner=<r>                       linking shell-ball radius
    --r_outer=<r>                       outer shell radius

    --entropy_only
"""
import numpy as np
import matplotlib.pyplot as plt
from docopt import docopt
args = docopt(__doc__)
from dedalus.core import coords, distributor, basis, field, operators, problems, solvers, timesteppers, arithmetic
from plotpal.file_reader import SingleFiletypePlotter as SFP
from plotpal.file_reader import match_basis
from plotpal.plot_grid import ColorbarPlotGrid as CPG

# Read in master output directory
root_dir    = args['<root_dir>']
data_dir   = 'slices'
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

if args['--entropy_only']:
    fig_name = 'entropy_' + fig_name

plotter = SFP(root_dir, file_dir=data_dir, fig_name=fig_name, start_file=start_file, n_files=n_files, distribution='even')

fields = ['s1B_eq', 's1S_eq', 'uB_eq', 'uS_eq']
bases  = ['φ', 'r']

if args['--entropy_only']:
    plot_grid = CPG(1, 1, polar=True, col_in=5, row_in=5) 
    axs  = plot_grid.axes
    caxs = plot_grid.cbar_axes
    ax1 = axs['ax_0-0']
    cax1 = caxs['ax_0-0']
else:
    plot_grid = CPG(2, 2, polar=True, col_in=3, row_in=3) 
    axs  = plot_grid.axes
    caxs = plot_grid.cbar_axes
    ax1 = axs['ax_0-0']
    ax2 = axs['ax_1-0']
    ax3 = axs['ax_0-1']
    ax4 = axs['ax_1-1']
    cax1 = caxs['ax_0-0']
    cax2 = caxs['ax_1-0']
    cax3 = caxs['ax_0-1']
    cax4 = caxs['ax_1-1']

r_inner = args['--r_inner']
r_outer = args['--r_outer']
mesa_file = args['--mesa_file']
if r_inner is not None and r_outer is not None:
    r_inner = float(r_inner)
    r_outer = float(r_outer)
elif mesa_file is not None:
    import h5py
    with h5py.File(args['--mesa_file'], 'r') as f:
        r_inner = f['r_inner'][()]
        r_outer = f['r_outer'][()]
else:
    raise ValueError("Must specify r_inner & r_outer OR mesa_file")

shell_basis_grids = None
ball_basis_grids = None
if not plotter.idle:
    while plotter.files_remain(fields):
        dsets = plotter.read_next_file()
        time_dset = dsets['uB_eq'].dims[0]
        φB = match_basis(dsets['uB_eq'], 'φ')
        rB = match_basis(dsets['uB_eq'], 'r')
        φS = match_basis(dsets['uS_eq'], 'φ')
        rS = match_basis(dsets['uS_eq'], 'r')

        φB_plot = np.append(φB, 2*np.pi)
        φS_plot = np.append(φS, 2*np.pi)

        rB_plot = np.pad(rB, ((1,1)), mode='constant', constant_values=(0, r_inner))
        rS_plot = np.pad(rS, ((1,1)), mode='constant', constant_values=(r_inner, r_outer))

        rrB, φφB = np.meshgrid(rB_plot.flatten(),  φB_plot)
        rrS, φφS = np.meshgrid(rS_plot.flatten(),  φS_plot)

        for i, n in enumerate(time_dset['write_number'][()]):
            print('writing filenum {}'.format(n))
            s1B = dsets['s1B_eq'][i,:,0,:]
            s1S = dsets['s1S_eq'][i,:,0,:]
            uφB = dsets['uB_eq'][i,0,:,0,:]
            uφS = dsets['uS_eq'][i,0,:,0,:]
            uθB = dsets['uB_eq'][i,1,:,0,:]
            uθS = dsets['uS_eq'][i,1,:,0,:]
            urB = dsets['uB_eq'][i,2,:,0,:]
            urS = dsets['uS_eq'][i,2,:,0,:]

            s1B -= np.mean(s1B, axis=0)
            s1S -= np.mean(s1S, axis=0)
            s1B /= np.mean(np.abs(s1B), axis=0)
            s1S /= np.mean(np.abs(s1S), axis=0)

            if args['--entropy_only']:
                zip_list = zip((ax1,), (cax1,), (s1B,), (s1S,), ('RdBu_r',))
            else:
                zip_list = zip((ax1, ax2, ax3, ax4), \
                               (cax1, cax2, cax3, cax4), \
                               (s1B, uθB, uφB, urB), \
                               (s1S, uθS, uφS, urS), \
                               ('RdBu_r', 'PuOr_r', 'PuOr_r', 'PuOr_r'))

            for ax, cax, fB, fS, cmap in zip_list:

                fB = np.pad(fB, ((0, 0), (1, 0)), mode='edge')
                fS = np.pad(fS, ((0, 0), (1, 0)), mode='edge')

                vals = np.sort(np.abs(np.concatenate((fB.flatten(), fS.flatten()))))
                vmax  = vals[int(0.998*len(vals))]
                vmin  = -vmax

                p = ax.pcolormesh(φφB, rrB, fB, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)#tasksB['s1_B'][0,:,0,:])
                ax.pcolormesh(φφS, rrS, fS, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)#tasksB['s1_B'][0,:,0,:])
                plt.colorbar(p, cax, orientation='horizontal')
            plt.suptitle('t = {:.2f}'.format(time_dset['sim_time'][i]))
            for k, ax in axs.items():
                ax.set_xticks([])
                ax.set_rticks([])
                ax.set_aspect(1)
            cax1.text(0.5, 0.5, r'$s_1$', ha='center', va='center', transform=cax1.transAxes)
            if not args['--entropy_only']:
                cax2.text(0.5, 0.5, r'$u_\theta$', ha='center', va='center', transform=cax2.transAxes)
                cax3.text(0.5, 0.5, r'$u_\phi$',   ha='center', va='center', transform=cax3.transAxes)
                cax4.text(0.5, 0.5, r'$u_r$',     ha='center', va='center', transform=cax4.transAxes)
            plt.savefig('{:s}/{:s}/{:s}_{:06d}.png'.format(root_dir, fig_name, fig_name, int(n)), dpi=float(args['--dpi']))#, bbox_inches='tight')
            for k, ax in axs.items():
                ax.cla()
            for k, cax in caxs.items():
                cax.cla()
