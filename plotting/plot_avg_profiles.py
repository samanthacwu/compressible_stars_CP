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
    --polytrope                         Use polytropic background
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
    plotter.add_profile('enth_flux',  avg_writes, basis='r')
    plotter.add_profile('visc_flux', avg_writes, basis='r')
    plotter.add_profile('cond_flux', avg_writes, basis='r')
    plotter.add_profile('KE_flux', avg_writes, basis='r')
    plotter.add_profile('s1', avg_writes, basis='r')
    plotter.add_profile('ρ_ur', avg_writes, basis='r')

plotter_kwargs = { 'col_in' : int(args['--col_inch']), 'row_in' : int(args['--row_inch']) }
plotter.plot_avg_profiles(dpi=int(args['--dpi']), **plotter_kwargs)


if args['--mesa_file'] is not None or args['--polytrope']:
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



    if args['--polytrope']:
        n_rho = 1
        gamma = 5/3
        gradT = np.exp(n_rho * (1 - gamma)) - 1

        #Gaussian luminosity -- zero at r = 0 and r = 1
        mu = 0.5
        sig = 0.15
        L_S  = -np.exp(-(r_flat - mu)**2/(2*sig**2))#1 - 4 * (rg - 0.5)**2

        r = r_flat
    else:
        with h5py.File(args['--mesa_file'], 'r') as f:
            ρ = np.exp(f['ln_ρ'][()])
            T = np.exp(f['ln_T'][()])
            H = f['H_eff'][()]
            r = f['r'][()]
        r_dense  = np.linspace(r.min(), r.max(), 2048)
        dr_dense = np.gradient(r_dense)
        H_dense  = np.interp(r_dense, r.flatten(), H.flatten())
        T_dense  = np.interp(r_dense, r.flatten(), T.flatten())
        ρ_dense  = np.interp(r_dense, r.flatten(), ρ.flatten())
        d_L_S_dense = -4*np.pi*r_dense**2*ρ_dense*T_dense*H_dense*dr_dense

        L_S_dense      = np.zeros_like(d_L_S_dense)
        for i in range(L_S_dense.shape[-1]-1):
            L_S_dense[i+1] = L_S_dense[i] + d_L_S_dense[i]
        L_S    = np.interp(r.flatten(), r_dense, L_S_dense)
        L_enth = interp1d(r_flat, L_enth, bounds_error=False, fill_value='extrapolate')(r.flatten())
        L_visc = interp1d(r_flat, L_visc, bounds_error=False, fill_value='extrapolate')(r.flatten())
        L_cond = interp1d(r_flat, L_cond, bounds_error=False, fill_value='extrapolate')(r.flatten())
        L_KE   = interp1d(r_flat, L_KE, bounds_error=False, fill_value='extrapolate')(r.flatten())


    L_diff  = L_S + L_enth + L_visc + L_cond + L_KE
    L_tot   = L_enth + L_visc + L_cond + L_KE

    fig = plt.figure(figsize=(4,3))
    plt.axhline(0, c='k', lw=0.25)
    plt.plot(r.flatten(),  L_tot.flatten(),  label='L tot',  c='k')
    plt.plot(r.flatten(), L_diff.flatten(),  label='L diff',  c='k', lw=0.5, ls='--')
    plt.plot(r.flatten(),   -L_S.flatten(),   label='-(L source)', c='grey',  lw=1)
    plt.plot(r.flatten(), L_enth.flatten(), label='L enth', lw=0.5)
    plt.plot(r.flatten(), L_visc.flatten(), label='L visc', lw=0.5)
    plt.plot(r.flatten(), L_cond.flatten(), label='L cond', lw=0.5)
    plt.plot(r.flatten(),   L_KE.flatten(),   label='L KE',   lw=0.5)
    plt.xlabel('r')
    plt.xlim(r_flat.min(), r_flat.max())
    plt.ylabel('Luminosity')
    plt.legend(loc='best')

    fig.savefig('{:s}/{:s}/final_fluxes.png'.format(root_dir, fig_name), dpi=400, bbox_inches='tight')


