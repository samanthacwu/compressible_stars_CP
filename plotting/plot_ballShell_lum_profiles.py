"""
Script for plotting flux profiles of a joint ball-sphere simulation.

Usage:
    plot_ballSphere_flux_profiles.py <root_dir> [options]

Options:
    --fig_name=<fig_name>               Name of figure output directory & base name of saved figures [default: fluxes_profiles]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<num_files>               Number of files to plot
    --dpi=<dpi>                         Image pixel density [default: 200]
    --avg_writes=<n_writes>             Number of output writes to average over [default: 100]

    --col_inch=<in>                     Number of inches / column [default: 8]
    --row_inch=<in>                     Number of inches / row [default: 2.5]

    --mesa_file=<f>                     NCC file for making full flux plot
    --polytrope                         Use polytropic background
    --r_inner=<r>                       linking shell-ball radius [default: 1.1]
    --r_outer=<r>                       outer shell radius [default: 2.59]
"""
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from docopt import docopt
args = docopt(__doc__)
from plotpal.file_reader import SingleFiletypePlotter as SFP
from plotpal.plot_grid import PlotGrid as PG

# Read in master output directory
root_dir    = args['<root_dir>']
data_dirB   = 'profilesB'
data_dirS   = 'profilesS'
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

plotterS = SFP(root_dir, file_dir=data_dirS, fig_name=fig_name, start_file=start_file, n_files=n_files, distribution='even')
plotterB = SFP(root_dir, file_dir=data_dirB, fig_name=fig_name, start_file=start_file, n_files=n_files, distribution='even')

fields = ['enth_lum', 'visc_lum', 'cond_lum', 'KE_lum']
fieldsB = ['{}B'.format(f) for f in fields]
fieldsS = ['{}S'.format(f) for f in fields]
bases  = ['r']

plot_grid = PG(5, 1, col_in=float(args['--col_inch']), row_in=float(args['--row_inch'])) 
axs  = plot_grid.axes
ax1 = axs['ax_0-0']
ax2 = axs['ax_0-1']
ax3 = axs['ax_0-2']
ax4 = axs['ax_0-3']
ax5 = axs['ax_0-4']
axs = [ax1, ax2, ax3, ax4, ax5]

plotterS.set_read_fields(bases, fieldsS)
count = 0
r_inner = float(args['--r_inner'])
r_outer = float(args['--r_outer'])
if args['--mesa_file'] is not None:
    import h5py
    with h5py.File(args['--mesa_file'], 'r') as f:
        r_inner = f['r_inner'][()]
        r_outer = f['r_outer'][()]
        H = f['H_effB'][()]
        r = f['rB'][()]
        ρ = np.exp(f['ln_ρB'][()])
        T = np.exp(f['ln_TB'][()])
        r_dense  = np.linspace(r.min(), r.max(), 2048)
        dr_dense = np.gradient(r_dense)
        H_dense  = np.interp(r_dense, r.flatten(), H.flatten())
        T_dense  = np.interp(r_dense, r.flatten(), T.flatten())
        ρ_dense  = np.interp(r_dense, r.flatten(), ρ.flatten())
        d_L_S_dense = 4*np.pi*r_dense**2*ρ_dense*T_dense*H_dense*dr_dense
        L_S_dense   = np.zeros_like(d_L_S_dense)
        for i in range(L_S_dense.shape[-1]-1):
            L_S_dense[i+1] = L_S_dense[i] + d_L_S_dense[i]
        L_S = np.interp(r.flatten(), r_dense, L_S_dense)
        r_plot = r.flatten()
        L_plot = L_S
        r_plot_shell = f['rS'][()].flatten()
        L_plot_shell = np.zeros_like(r_plot_shell)

else:
    L_plot = None

data = OrderedDict()
for f in fieldsB: data[f] = []
for f in fieldsS: data[f] = []
data['sim_time'] = []
with plotterB.my_sync:
    if not plotterB.idle:
        while plotterB.files_remain(bases, fieldsB):
            basesB, tasksB, write_numB, sim_timeB = plotterB.read_next_file()
            basesS, tasksS, write_numS, sim_timeS = plotterS.read_next_file()

            for f in fieldsB:
                data[f].append(tasksB[f].squeeze())
            for f in fieldsS:
                data[f].append(tasksS[f].squeeze())
            data['sim_time'].append(sim_timeB)
                

            rB = basesB['r']
            rS = basesS['r']

        for f in fieldsB:
            data[f] = np.concatenate(data[f], axis=0)
        for f in fieldsS:
            data[f] = np.concatenate(data[f], axis=0)
        print(data['sim_time'])
        data['sim_time'] = np.concatenate(data['sim_time'], axis=0)
        n_chunks = np.ceil(data['sim_time'].shape[0]/int(args['--avg_writes']))

        data['sim_time'] = np.array_split(data['sim_time'], n_chunks, axis=0)
        for f in fieldsB:
            data[f] = np.array_split(data[f], n_chunks, axis=0)
        for f in fieldsS:
            data[f] = np.array_split(data[f], n_chunks, axis=0)
        for i in range(len(data['sim_time'])):
            if L_plot is not None:
                axs[0].plot(r_plot.flatten(), L_plot, c='orange', lw=3)
                axs[0].plot(r_plot_shell.flatten(), L_plot_shell, c='orange', lw=3)
            sum_ball = np.zeros_like(rB.flatten())
            sum_shell = np.zeros_like(rS.flatten())
            for j, f in enumerate(fieldsB):
                ball = np.mean(data[f][i], axis=0)
                sum_ball += ball
                axs[j+1].plot(rB.flatten(), ball, label=f)
                axs[0].plot(rB.flatten(), ball, label=f)
            for j, f in enumerate(fieldsS):
                shell = np.mean(data[f][i], axis=0)
                sum_shell += shell
                axs[j+1].plot(rS.flatten(), shell, label=f)
                axs[0].plot(rS.flatten(), shell, label=f)
                axs[j+1].set_ylabel(f[:-1])
            axs[0].plot(rB.flatten(), sum_ball,  c='k')
            axs[0].plot(rS.flatten(), sum_shell, c='k')
                

            plt.savefig('{:s}/{:s}/{:s}_{:04d}.png'.format(root_dir, fig_name, fig_name, i+1), dpi=float(args['--dpi']), bbox_inches='tight')
            for ax in axs:
                ax.cla()

            count += 1

        
plotterS = SFP(root_dir, file_dir=data_dirS, fig_name='wave_lum', start_file=start_file, n_files=n_files, distribution='even')
plotterB = SFP(root_dir, file_dir=data_dirB, fig_name='wave_lum', start_file=start_file, n_files=n_files, distribution='even')

fields = ['wave_lum',]
fieldsB = ['{}B'.format(f) for f in fields]
fieldsS = ['{}S'.format(f) for f in fields]
bases  = ['r']

plot_grid = PG(1, 1, col_in=float(args['--col_inch']), row_in=float(args['--row_inch'])) 
axs  = plot_grid.axes
ax1 = axs['ax_0-0']
axs = [ax1,]

plotterS.set_read_fields(bases, fieldsS)
count = 0
r_inner = float(args['--r_inner'])
r_outer = float(args['--r_outer'])

with plotterB.my_sync:
    if not plotterB.idle:
        while plotterB.files_remain(bases, fieldsB):
            basesB, tasksB, write_numB, sim_timeB = plotterB.read_next_file()
            basesS, tasksS, write_numS, sim_timeS = plotterS.read_next_file()

            rB = basesB['r']
            rS = basesS['r']

            for i, t in enumerate(sim_timeB):
                ax1.plot(rB.flatten(), tasksB['wave_lumB'][i,0,0,:], c='k')
                ax1.plot(rS.flatten(), tasksS['wave_lumS'][i,0,0,:], c='k')
                ax1.plot(rB.flatten(), -tasksB['wave_lumB'][i,0,0,:], c='k', ls='--')
                ax1.plot(rS.flatten(), -tasksS['wave_lumS'][i,0,0,:], c='k', ls='--')
                ax1.set_yscale('log')
                ax1.set_xlabel('r')
                ax1.set_ylabel(r'wave lum = $4\pi r^2 u_r (h + \phi - T_0 s_1)$')
                     
                plt.savefig('{:s}/wave_lum/wave_lum_{:06d}.png'.format(root_dir, int(write_numB[i])), dpi=float(args['--dpi']), bbox_inches='tight')
                ax1.cla()

            count += 1

        

