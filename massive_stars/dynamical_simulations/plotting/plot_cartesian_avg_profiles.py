"""
Script for plotting time-averaged 1D profiles vs. a basis of a dedalus simulation.  

The profiles specified in the "--fig_type" flag are averaged over the number of time outputs specified in "--avg_writes"

Usage:
    plot_avg_profiles.py <root_dir> [options]

Options:
    --data_dir=<dir>                    Name of data handler directory [default: profiles]
    --fig_name=<fig_name>               Name of figure output directory & base name of saved figures [default: avg_profiles]
    --start_file=<file_start_num>       Number of Dedalus output file to start plotting at [default: 1]
    --n_files=<num_files>               Number of files to plot
    --dpi=<dpi>                         Image pixel density [default: 200]
    --avg_writes=<n_writes>             Number of output writes to average over [default: 100]

    --col_inch=<in>                     Number of inches / column [default: 6]
    --row_inch=<in>                     Number of inches / row [default: 3]

    --fig_type=<fig_type>               Type of figure to plot
                                            1 - T, velocities, fluxes
                                        [default: 1]
    --mesa_file=<f>                     NCC file for making full flux plot
"""
from docopt import docopt
args = docopt(__doc__)
from plotpal.profiles import ProfilePlotter

# Read in master output directory
root_dir    = args['<root_dir>']
data_dir    = args['--data_dir']
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

plotter = ProfilePlotter(root_dir, file_dir=data_dir, fig_name=fig_name, start_file=start_file, n_files=n_files)
if int(args['--fig_type']) == 1:
    plotter.add_profile('enth_flux',  avg_writes, basis='z')
    plotter.add_profile('visc_flux', avg_writes,  basis='z')
    plotter.add_profile('cond_flux', avg_writes,  basis='z')
    plotter.add_profile('KE_flux', avg_writes,    basis='z')
    plotter.add_profile('s1', avg_writes,    basis='z')

plotter_kwargs = { 'col_in' : int(args['--col_inch']), 'row_in' : int(args['--row_inch']) }
plotter.plot_avg_profiles(dpi=int(args['--dpi']), **plotter_kwargs)


if args['--mesa_file'] is not None:
    import h5py
    import numpy as np
    from scipy.interpolate import interp1d
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with h5py.File(args['--mesa_file'], 'r') as f:
        ρ = np.exp(f['ln_ρ'][()])
        T = np.exp(f['ln_T'][()])
        H = f['H_eff'][()]
        r = f['r'][()]


    with h5py.File('{:s}/{:s}/averaged_{:s}.h5'.format(root_dir, fig_name, fig_name), 'r') as f:
        enth_flux = f['enth_flux'][()][-1,:]
        visc_flux = f['visc_flux'][()][-1,:]
        cond_flux = f['cond_flux'][()][-1,:]
        KE_flux   = f['KE_flux'][()][-1,:]
        r_flat    = f['z'][()].flatten()

    d_r_flat = np.gradient(r_flat)
    L_enth = enth_flux
    L_visc = visc_flux
    L_cond = cond_flux
    L_KE   = KE_flux


    if 'polytrope' in args['--mesa_file']:
        L_S  = -(1 - 4*(r_flat - 0.5)**2)
    else:
        dr = np.gradient(r.flatten()).reshape(r.shape)
        d_L_S = (-4*np.pi*r**2*ρ*T*H*dr).flatten()

        L_S      = np.zeros_like(d_L_S)
        L_S[0] = d_L_S[0]
        for i in range(L_S.shape[-1]-1):
            L_S[i+1] = L_S[i] + d_L_S[i+1]

    L_diff  = L_S + L_enth + L_visc + L_cond + L_KE
    L_tot   = L_enth + L_visc + L_cond + L_KE

    fig = plt.figure(figsize=(4,3))
    plt.axhline(0, c='k', lw=0.25)
    plt.plot(r_flat, L_tot,  label='L tot',  c='k')
    plt.plot(r_flat, L_diff,  label='L diff',  c='k', lw=0.5, ls='--')
    plt.plot(r_flat, -L_S,   label='-(L source)', c='grey',  lw=1)
    plt.plot(r_flat, L_enth, label='L enth', lw=0.5)
    plt.plot(r_flat, L_visc, label='L visc', lw=0.5)
    plt.plot(r_flat, L_cond, label='L cond', lw=0.5)
    plt.plot(r_flat, L_KE,   label='L KE',   lw=0.5)
    plt.xlabel('r')
    plt.xlim(r_flat.min(), r_flat.max())
    plt.ylabel('Luminosity')
    plt.legend(loc='best')

    fig.savefig('{:s}/{:s}/final_fluxes.png'.format(root_dir, fig_name), dpi=400, bbox_inches='tight')


