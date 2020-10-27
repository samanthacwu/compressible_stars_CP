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
    --r_inner=<r>                       linking shell-ball radius [default: 1.2]
    --r_outer=<r>                       outer shell radius [default: 2]
"""
import numpy as np
import matplotlib.pyplot as plt
from docopt import docopt
args = docopt(__doc__)
from plotpal.file_reader import SingleFiletypePlotter as SFP

# Read in master output directory
root_dir    = args['<root_dir>']
data_dirB   = 'eq_sliceB'
data_dirS   = 'eq_sliceS'
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

plotterS = SFP(root_dir, file_dir=data_dirS, fig_name=fig_name, start_file=start_file, n_files=n_files)
plotterB = SFP(root_dir, file_dir=data_dirB, fig_name=fig_name, start_file=start_file, n_files=n_files)

fields = ['s1', 'ur', 'uφ', 'uθ']
fieldsB = ['{}_B'.format(f) for f in fields]
fieldsS = ['{}_S'.format(f) for f in fields]
bases  = ['φ', 'r']

fig = plt.figure()
ax1 = fig.add_subplot(2,2,1, polar=True)
ax2 = fig.add_subplot(2,2,2, polar=True)
ax3 = fig.add_subplot(2,2,3, polar=True)
ax4 = fig.add_subplot(2,2,4, polar=True)

plotterS.set_read_fields(bases, fieldsS)
count = 0
r_inner = float(args['--r_inner'])
r_outer = float(args['--r_outer'])
if args['--mesa_file'] is not None:
    import h5py
    with h5py.File(args['--mesa_file'], 'r') as f:
        r_inner = f['r_inner'][()]
        r_outer = f['r_outer'][()]
        
while plotterB.files_remain(bases, fieldsB):
    basesB, tasksB, write_numB, sim_timeB = plotterB.read_next_file()
    basesS, tasksS, write_numS, sim_timeS = plotterS.read_next_file()

    rB = basesB['r']
    rS = basesS['r']
    φB = basesB['φ']
    φS = basesB['φ']

    φB_plot = np.append(φB.flatten(), 2*np.pi)
    φS_plot = np.append(φS.flatten(), 2*np.pi)

    rB_plot = np.pad(rB, ((0,0), (0,0), (1,1)), mode='constant', constant_values=(0, r_inner))
    rS_plot = np.pad(rS, ((0,0), (0,0), (1,1)), mode='constant', constant_values=(r_inner, r_outer))

    rrB, φφB = np.meshgrid(rB_plot.flatten(),  φB_plot)
    rrS, φφS = np.meshgrid(rS_plot.flatten(),  φS_plot)

    for i in range(tasksB['s1_B'].shape[0]):
        s1B = tasksB['s1_B'][i,:,0,:]
        s1S = tasksS['s1_S'][i,:,0,:]
        urB = tasksB['ur_B'][i,:,0,:]
        urS = tasksS['ur_S'][i,:,0,:]
        uφB = tasksB['uφ_B'][i,:,0,:]
        uφS = tasksS['uφ_S'][i,:,0,:]
        uθB = tasksB['uθ_B'][i,:,0,:]
        uθS = tasksS['uθ_S'][i,:,0,:]

        s1B -= np.mean(s1B, axis=0)
        s1S -= np.mean(s1S, axis=0)
        s1B /= np.mean(np.abs(s1B), axis=0)
        s1S /= np.mean(np.abs(s1S), axis=0)

        for ax, fB, fS in zip((ax1, ax2, ax3, ax4), (s1B, uθB, uφB, urB), (s1S, uθS, uφS, urS)):

            fB = np.pad(fB, ((0, 0), (1, 0)), mode='edge')
            fS = np.pad(fS, ((0, 0), (1, 0)), mode='edge')

            vals = np.sort(np.abs(np.concatenate((fB.flatten(), fS.flatten()))))
            vmax  = vals[int(0.998*len(vals))]
            vmin  = -vmax

            ax.pcolormesh(φφB, rrB, fB, cmap='RdBu_r', vmin=vmin, vmax=vmax)#tasksB['s1_B'][0,:,0,:])
            ax.pcolormesh(φφS, rrS, fS, cmap='RdBu_r', vmin=vmin, vmax=vmax)#tasksB['s1_B'][0,:,0,:])
        plt.suptitle('t = {}'.format(sim_timeB[i]))
        plt.savefig('{:s}/{:s}/{:s}_{:04d}.png'.format(root_dir, fig_name, fig_name, count+1), dpi=100, bbox_inches='tight')
        for ax in [ax1, ax2, ax3, ax4]:
            ax.cla()
        count += 1

    

