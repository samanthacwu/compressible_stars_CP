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
    --nr=<nr>                           Number of r-coeffs in problem [default: 64]
"""
from docopt import docopt
args = docopt(__doc__)
from plotpal.profiles import ProfilePlotter
import dedalus.public as de

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
    for l in ['B', 'S']:
        plotter.add_profile('{}{}'.format('enth_lum',l),  avg_writes, basis='r')
        plotter.add_profile('{}{}'.format('visc_lum',l), avg_writes,  basis='r')
        plotter.add_profile('{}{}'.format('cond_lum',l), avg_writes,  basis='r')
        plotter.add_profile('{}{}'.format('KE_lum',l), avg_writes,    basis='r')
        plotter.add_profile('{}{}'.format('wave_lum',l), avg_writes,    basis='r')

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
        ρB = np.exp(f['ln_ρB'][()])
        TB = np.exp(f['ln_TB'][()])
        HB = f['H_effB'][()]


    with h5py.File('{:s}/{:s}/averaged_{:s}.h5'.format(root_dir, fig_name, fig_name), 'r') as f:
        enth_lum = f['enth_lumB'][()][-1,:]
        visc_lum = f['visc_lumB'][()][-1,:]
        cond_lum = f['cond_lumB'][()][-1,:]
        KE_lum   = f['KE_lumB'][()][-1,:]
        r_flat    = f['r'][()].flatten()

    if 'polytrope' in args['--mesa_file']:
        rbot = 0.5
        L_S = -(1 - 4*(r_flat-(0.5+rbot))**2)
    else:
        Lbot = 0.5
        Lr   = 1
        nr = int(args['--nr'])
        r_basis = de.Chebyshev('r', nr, interval = [Lbot, Lbot+Lr], dealias=1)

        domain = de.Domain([r_basis,], grid_dtype=np.float64, mesh=None)
        r = z = domain.grid(-1)
        dr = np.gradient(r.flatten()).reshape(r.shape)

        H_eff = domain.new_field()
        d_L_S = domain.new_field()
        L_S   = domain.new_field()
        print(r-Lbot)
        H_factor   = 4*np.pi*(r-Lbot)**2 / (2 * np.pi * r)
        H_eff['g'] = H_factor*H
        d_L_S['g'] = ρ*T*(2 * np.pi * r) * H_eff['g']
        d_L_S.antidifferentiate('r', ('left', 0), out=L_S)

        L_S = -L_S['g']
 

    L_tot   = enth_lum + visc_lum + cond_lum + KE_lum
    L_diff  = L_S + L_tot

    fig = plt.figure(figsize=(4,3))
    plt.axhline(0, c='k', lw=0.25)
    plt.plot(r_flat, L_tot,  label='L tot',  c='k')
    plt.plot(r_flat, L_diff,  label='L diff',  c='k', lw=0.5, ls='--')
    plt.plot(r_flat, -L_S,   label='-(L source)', c='grey',  lw=0.75)
    plt.plot(r_flat, enth_lum, label='L enth', lw=0.5)
    plt.plot(r_flat, visc_lum, label='L visc', lw=0.5)
    plt.plot(r_flat, cond_lum, label='L cond', lw=0.5)
    plt.plot(r_flat, KE_lum,   label='L KE',   lw=0.5)
    plt.xlabel('r')
    plt.xlim(r_flat.min(), r_flat.max())
    plt.ylabel('Luminosity')
    plt.legend(loc='best')

    fig.savefig('{:s}/{:s}/final_lums.png'.format(root_dir, fig_name), dpi=400, bbox_inches='tight')


