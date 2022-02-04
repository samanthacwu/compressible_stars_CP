"""
Script for plotting equatorial cuts of a joint ball-sphere simulation.

Usage:
    mean_flow_examination.py <root_dir> [options]

Options:
    --fig_name=<fig_name>               Name of figure output directory & base name of saved figures [default: mean_flows]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<num_files>               Number of files to plot
    --dpi=<dpi>                         Image pixel density [default: 200]
    --avg_writes=<n_writes>             Number of output writes to average over [default: 100]

    --col_inch=<in>                     Number of inches / column [default: 6]
    --row_inch=<in>                     Number of inches / row [default: 3]

    --r_inner=<r>                       linking shell-ball radius [default: 1.1]
    --r_outer=<r>                       outer shell radius [default: 2.59]
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

plotterS = SFP(root_dir, file_dir=data_dirS, fig_name=fig_name, start_file=start_file, n_files=n_files, distribution='even')
plotterB = SFP(root_dir, file_dir=data_dirB, fig_name=fig_name, start_file=start_file, n_files=n_files, distribution='even')

fields = ['ur', 'uφ', 'uθ']
fieldsB = ['{}B'.format(f) for f in fields]
fieldsS = ['{}S'.format(f) for f in fields]
bases  = ['φ', 'r']

fig = plt.figure(figsize=(float(args['--col_inch']), float(args['--row_inch'])*3))
ax1 = fig.add_subplot(3,1,1)
ax2 = fig.add_subplot(3,1,2)
ax3 = fig.add_subplot(3,1,3)

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
            φB = basesB['φ']
            rS = basesS['r']
            dr = np.expand_dims(np.gradient(rB.flatten()), axis=0)
            dphi = np.expand_dims(np.gradient(φB.flatten()), axis=1)

            for i, n in enumerate(write_numB):
                print('writing filenum {}'.format(n))
                urB = tasksB['urB'][i,:,0,:]
                urS = tasksS['urS'][i,:,0,:]
                uφB = tasksB['uφB'][i,:,0,:]
                uφS = tasksS['uφS'][i,:,0,:]
                uθB = tasksB['uθB'][i,:,0,:]
                uθS = tasksS['uθS'][i,:,0,:]

                u_core = np.sqrt(urB**2 + uφB**2 + uθB**2)

                core_cz_mag = np.sum(u_core[:, rB.flatten() < 0.9]*dr[:,rB.flatten() < 0.9]*dphi) / (np.pi*(0.9)**2)
                uperpB = np.mean(np.sqrt(uφB**2 + uθB**2), axis=0)
                uperpS = np.mean(np.sqrt(uφS**2 + uθS**2), axis=0)

                ax1.plot(rB.flatten(), uperpB, c='k')
                ax1.plot(rS.flatten(), uperpS, c='k')
                ax1.set_yscale('log')
                ax2.plot(rB.flatten(), uperpB/core_cz_mag, c='k')
                ax2.plot(rS.flatten(), uperpS/core_cz_mag, c='k')
                ax2.set_yscale('log')
                ax3.plot(rB.flatten(), uperpB/rB.flatten(), c='k')
                ax3.plot(rS.flatten(), uperpS/rS.flatten(), c='k')
                ax3.set_yscale('log')
                ax1.set_ylabel(r'$u_\perp$')
                ax2.set_ylabel(r'$u_\perp/u_{\mathrm{core}}$')
                ax3.set_ylabel(r'$u_\perp/r = \Omega$')
                for ax in [ax1, ax2, ax3]:
                    ax.set_xlabel('radius')
                plt.suptitle('t = {:.2f}'.format(sim_timeB[i]))
                plt.savefig('{:s}/{:s}/{:s}_{:04d}.png'.format(root_dir, fig_name, fig_name, int(n)), dpi=float(args['--dpi']))#, bbox_inches='tight')
                for ax in [ax1, ax2, ax3]:
                    ax.cla()


                count += 1
