"""
Script for plotting equatorial cuts of a joint ball-sphere simulation.

Usage:
    plot_ballSphere_equatorial_slices.py <root_dir> [options]

Options:
    --fig_name=<fig_name>               Name of figure output directory & base name of saved figures [default: snapshots_eq]
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
data_dirB   = 'profilesB'
data_dirS   = 'profilesS'
fig_nameB   = 'avg_profileB'
fig_nameS   = 'avg_profileS'
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

plotterS = ProfilePlotter(root_dir, file_dir=data_dirS, fig_name=fig_nameS, start_file=start_file, n_files=n_files)
plotterB = ProfilePlotter(root_dir, file_dir=data_dirB, fig_name=fig_nameB, start_file=start_file, n_files=n_files)

for plotter in [plotterS, plotterB]:
    plotter.add_profile('enth_flux',  avg_writes, basis='r')
    plotter.add_profile('visc_flux', avg_writes, basis='r')
    plotter.add_profile('cond_flux', avg_writes, basis='r')
    plotter.add_profile('KE_flux', avg_writes, basis='r')
    plotter.add_profile('s1', avg_writes, basis='r')

    plotter_kwargs = { 'col_in' : int(args['--col_inch']), 'row_in' : int(args['--row_inch']) }
    plotter.plot_avg_profiles(dpi=int(args['--dpi']), **plotter_kwargs)


fig = plt.figure(figsize=(4,3))
for fig_name in [fig_nameB, fig_nameS]:
    import h5py
    import numpy as np
    from scipy.interpolate import interp1d
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with h5py.File('{:s}/{:s}/averaged_{:s}.h5'.format(root_dir, fig_name, fig_name), 'r') as f:
        enth_flux = f['enth_flux'][()][-1,:]
        visc_flux = f['visc_flux'][()][-1,:]
        cond_flux = f['cond_flux'][()][-1,:]
        KE_flux   = f['KE_flux'][()][-1,:]
        r_flat    = f['r'][()].flatten()

    d_r_flat = np.gradient(r_flat)
    L_enth = 4*np.pi*r_flat**2*enth_flux
    L_visc = 4*np.pi*r_flat**2*visc_flux
    L_cond = 4*np.pi*r_flat**2*cond_flux
    L_KE   = 4*np.pi*r_flat**2*KE_flux

    L_tot   = L_enth + L_visc + L_cond + L_KE

    plt.axhline(0, c='k', lw=0.25)
    plt.plot(r_flat.flatten(),  L_tot.flatten(),  label='L tot',  c='k')
    plt.plot(r_flat.flatten(), L_enth.flatten(), label='L enth', lw=0.5)
    plt.plot(r_flat.flatten(), L_visc.flatten(), label='L visc', lw=0.5)
    plt.plot(r_flat.flatten(), L_cond.flatten(), label='L cond', lw=0.5)
    plt.plot(r_flat.flatten(),   L_KE.flatten(),   label='L KE',   lw=0.5)
    plt.xlabel('r')
    plt.ylabel('Luminosity')
    plt.legend(loc='best')

fig.savefig('{:s}/final_fluxes.png'.format(root_dir, fig_name), dpi=400, bbox_inches='tight')


