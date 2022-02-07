"""
A shell of a script for making nice DFD talk slices

Usage:
    plot_talk_slices.py <root_dir> [options]

Options:
    --fig_name=<fig_name>               Name of figure output directory & base name of saved figures [default: talk_snapshots]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<num_files>               Number of files to plot
    --dpi=<dpi>                         Image pixel density [default: 200]

    --col_inch=<in>                     Number of inches / column [default: 6]
    --row_inch=<in>                     Number of inches / row [default: 3]
"""
import numpy as np
import matplotlib.pyplot as plt
from docopt import docopt
args = docopt(__doc__)
from plotpal.file_reader import SingleFiletypePlotter as SFP
from plotpal.plot_grid   import ColorbarPlotGrid as CPG
import cartopy.crs as ccrs

import h5py

with h5py.File('../mesa_stars/nccs_40msol/ballShell_nccs_B255_S127_Re1e4.h5', 'r') as f:
    tau = f['tau'][()]
    time_day = tau/60/60/24


# Read in master output directory
root_dir = args['<root_dir>']
data_dir = 'slices'
if root_dir is None:
    print('No dedalus output dir specified, exiting')
    import sys
    sys.exit()

start_file  = int(args['--start_file'])
fig_name    = args['--fig_name']
n_files     = args['--n_files']
if n_files is not None: 
    n_files = int(n_files)

plotter    = SFP(root_dir, file_dir=data_dir, fig_name=fig_name, start_file=start_file, n_files=n_files, distribution='even')

shell_field_keys = ['s1_r0.5', 's1_r1', 's1_r_near_surf']
shell_bases_keys  = ['φ', 'θ']


fig = plt.figure()
ax1 = fig.add_axes([   0.03,   0.3,   0.94, 0.7], polar=True)
ax2 = fig.add_axes([   0.20  - 0.015, 0.02,   0.20, 0.20], projection=ccrs.Orthographic(180, 45))
ax3 = fig.add_axes([   0.355 - 0.015, 0.02,   0.25, 0.25], projection=ccrs.Orthographic(180, 45))
ax4 = fig.add_axes([   0.55  - 0.015, 0.02,   0.30, 0.30], projection=ccrs.Orthographic(180, 45))

plotterEqB.set_read_fields(['r', 'φ'], ['s1_B',])
plotterEqS.set_read_fields(['r', 'φ'], ['s1_S',])
with plotter.my_sync:
    if not plotter.idle:
        while plotter.files_remain(shell_bases_keys, shell_field_keys):
            bases, tasks, write_num, sim_time = plotter.read_next_file()
            basesEqB, tasksEqB, write_numEqB, sim_timeEqB = plotterEqB.read_next_file()
            basesEqS, tasksEqS, write_numEqS, sim_timeEqS = plotterEqS.read_next_file()

            theta = bases['θ']
            phi   = bases['φ']
            dtheta = np.expand_dims(np.gradient(theta.flatten()), axis=0)
            dphi = np.expand_dims(np.gradient(phi.flatten()), axis=1)
            theta_orig = np.copy(theta).squeeze()

            theta *= 180/np.pi
            theta -= 90

            phi *= 180/np.pi
            phi -= 180


            lats, lons = np.meshgrid(theta.flatten(), phi.flatten())

            r_inner  = 1.1
            r_outer  = 2.59

            r   = basesEqB['r']
            phi = basesEqB['φ']
            phi_plot = np.append(phi.flatten(), 2*np.pi)
            r_plot   = np.pad(r, ((0,0), (0,0), (1,1)), mode='constant', constant_values=(0, r_inner))
            rrB, phisB = np.meshgrid(r_plot.flatten(),  phi_plot)

            r   = basesEqS['r']
            phi = basesEqS['φ']
            phi_plot = np.append(phi.flatten(), 2*np.pi)
            r_plot   = np.pad(r, ((0,0), (0,0), (1,1)), mode='constant', constant_values=(r_inner, r_outer))
            rrS, phisS = np.meshgrid(r_plot.flatten(),  phi_plot)



            for i, num in enumerate(write_num):
                num = int(num)
                print('writing {}/{}'.format(num, len(write_num)))
                entropy_shell_cz   = tasks['s1_r0.5'][i,:].squeeze()
                entropy_shell_int  = tasks['s1_r1'][i,:].squeeze()
                entropy_shell_surf = tasks['s1_r_near_surf'][i,:].squeeze()
                entropy_shell_cz   -= np.sum(np.sin(theta_orig)*dtheta*dphi*entropy_shell_cz)/np.sum(np.sin(theta_orig)*dtheta*dphi)
                entropy_shell_int  -= np.sum(np.sin(theta_orig)*dtheta*dphi*entropy_shell_int)/np.sum(np.sin(theta_orig)*dtheta*dphi)
                entropy_shell_surf -= np.sum(np.sin(theta_orig)*dtheta*dphi*entropy_shell_surf)/np.sum(np.sin(theta_orig)*dtheta*dphi)

                entropy_eqB    = tasksEqB['s1_B'][i,:].squeeze()
                entropy_eqB    -= np.mean(entropy_eqB, axis=0)
                entropy_eqB    /= np.mean(np.abs(entropy_eqB), axis=0)
                entropy_eqB     = np.pad(entropy_eqB, ((0, 0), (1, 0)), mode='edge')
                entropy_eqS    = tasksEqS['s1_S'][i,:].squeeze()
                entropy_eqS    -= np.mean(entropy_eqS, axis=0) 
                entropy_eqS    /= np.mean(np.abs(entropy_eqS), axis=0)
                entropy_eqS     = np.pad(entropy_eqS, ((0, 0), (1, 0)), mode='edge')

                eq_valsB = np.sort(np.abs(entropy_eqB.flatten()))
                eq_minmaxB = eq_valsB[int(0.98*len(eq_valsB))]
                eq_valsS = np.sort(np.abs(entropy_eqS.flatten()))
                eq_minmaxS = eq_valsS[int(0.98*len(eq_valsS))]
                eq_minmax = np.max((eq_minmaxB, eq_minmaxS))

                shc_minmax = 2*np.std(entropy_shell_cz)
                shi_minmax = 2*np.std(entropy_shell_int)
                shs_minmax = 2*np.std(entropy_shell_surf)

                p1B = ax1.pcolormesh(phisB, rrB,   entropy_eqB,    cmap='RdBu_r', vmin=-eq_minmax, vmax=eq_minmax, rasterize=True)
                p1S = ax1.pcolormesh(phisS, rrS,   entropy_eqS,    cmap='RdBu_r', vmin=-eq_minmax, vmax=eq_minmax, rasterize=True)
                p2 = ax2.pcolormesh(lons, lats, entropy_shell_cz,   cmap='RdBu_r', vmin=-shc_minmax, vmax=shc_minmax, transform=ccrs.PlateCarree(), rasterize=True)
                p3 = ax3.pcolormesh(lons, lats, entropy_shell_int,  cmap='RdBu_r', vmin=-shi_minmax, vmax=shi_minmax, transform=ccrs.PlateCarree(), rasterize=True)
                p4 = ax4.pcolormesh(lons, lats, entropy_shell_surf, cmap='RdBu_r', vmin=-shs_minmax, vmax=shs_minmax, transform=ccrs.PlateCarree(), rasterize=True)

#                ax1.plot(phi_plot[0]*np.ones(1000), np.linspace(0, 1, 1000))

                shell_color='k'
                ax1.plot(np.linspace(-np.pi/2, np.pi/2, 1000), 0.5*np.ones(1000),          lw=2.5, c=shell_color)
                ax1.plot(np.linspace(-np.pi/2, np.pi/2, 1000), 1.0*np.ones(1000),          lw=2.5, c=shell_color)
                ax1.plot(np.linspace(-np.pi/2, np.pi/2, 1000), 0.95*r_outer*np.ones(1000), lw=2.5, c=shell_color)

                ax1.set_theta_zero_location("S")


                for ax, color in zip([ax2, ax3, ax4], [shell_color, shell_color, shell_color]):
                    ax.spines['geo'].set_color(color)
                    ax.gridlines(color=color, alpha=0.3)
                    ax.plot(np.linspace(-180, 180, 1000), np.zeros(1000), transform=ccrs.PlateCarree(), c=color, lw=2)
#                    ax.plot(((phi_plot[0])*180/np.pi-180)*np.ones(1000), np.linspace(-90, 90, 1000), transform=ccrs.PlateCarree(), c='k')

                ax1.set_xticks([])
                ax1.set_rticks([])
                ax1.set_aspect(1)
                ax1.text( -0.03, 0.97, 't = {:.2f} days'.format(sim_time[i]*time_day), transform=ax1.transAxes)
#                plt.colorbar(p, cax2, orientation='horizontal')

#                cax2.text(0.5, 0.5, 'z vorticity', ha='center', va='center', transform=cax2.transAxes)
                plt.savefig('{:s}/{:s}_{:04d}.png'.format(plotter.out_dir, fig_name, num), dpi=float(args['--dpi']), bbox_inches='tight')
                for ax in [ax1, ax2, ax3, ax4]:# cax2,]:
                    ax.cla()
