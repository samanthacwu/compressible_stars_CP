powers_per_ell = OrderedDict()
with h5py.File('{}/power_spectra.h5'.format(full_out_dir), 'r') as out_f:
    for f in fields:
        powers_per_ell[f] = out_f['{}_power_per_ell'.format(f)][()]
    ells = out_f['ells'][()]
    freqs = out_f['freqs'][()]


good = freqs >= 0
min_freq = 3e-1
max_freq = freqs.max()
for k, powspec in powers_per_ell.items():
    good_axis = np.arange(len(powspec.shape))[np.array(powspec.shape) == len(ells.flatten())][0]
    print(good_axis, powspec.shape, len(ells.flatten()))
    sum_power = np.sum(powspec, axis=good_axis).squeeze()
        
    ymin = sum_power[(freqs > min_freq)*(freqs < max_freq)][-1].min()/2
    ymax = sum_power[(freqs > min_freq)*(freqs <= max_freq)].max()*2

    plt.figure()
    if len(sum_power.shape) > 1:
        for i in range(sum_power.shape[1]):
            plt.plot(freqs[good], sum_power[good, i], c = 'k', label=r'axis {}, sum over $\ell$ values'.format(i))
    else:
        plt.plot(freqs[good], sum_power[good], c = 'k', label=r'sum over $\ell$ values')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'Power ({})'.format(k))
    plt.xlabel(r'Frequency (sim units)')
    plt.axvline(np.sqrt(N2plateau)/(2*np.pi), c='k')
    plt.xlim(min_freq, max_freq)
    plt.ylim(ymin, ymax)
    plt.legend(loc='best')
    k_out = k.replace('(', '_').replace(')', '_').replace('=', '')

    plt.savefig('{}/{}_summed_power.png'.format(full_out_dir, k_out), dpi=600)

    if k == 'uB(r=0.5)':
        scalar_plotter = SFP(root_dir, file_dir='scalars', fig_name=out_dir, start_file=1, n_files=np.inf, distribution='single')
        with h5py.File(scalar_plotter.files[0], 'r') as f:
            re_ball = f['tasks']['Re_avg_ball'][()]
            re_ball_avg = np.mean(re_ball.flatten()[int(len(re_ball.flatten())/2):])
            re_ball_avg *= (1.1/1.0)**3 #r_ball/r_cz cubed
            Re_input = float(root_dir.split('Re')[-1].split('_')[0])
            u_ball_avg = re_ball_avg / Re_input

        grid_dir = data_dir.replace('SH_transform_', '')
        start_file = int(plotter.files[0].split('.h5')[0].split('_s')[-1])
        n_files = len(plotter.files)
        grid_plotter = SFP(root_dir, file_dir=grid_dir, fig_name=out_dir, start_file=start_file, n_files=n_files, distribution='single')
        dealias = 1
        Lmax = powspec.squeeze().shape[-1] - 1
        c = coords.SphericalCoordinates('φ', 'θ', 'r')
        d = distributor.Distributor((c,), mesh=None, comm=MPI.COMM_SELF)
        b = basis.SphericalShellBasis(c, (2*(Lmax+2), Lmax+1, 1), radii=(0.5-1e-6, 0.5), dtype=np.float64)
        φ, θ, r = b.local_grids((dealias, dealias, dealias))
        weight_φ = np.gradient(φ.flatten()).reshape(φ.shape)
        weight_θ = b.local_colatitude_weights(dealias)
        weight = weight_θ * weight_φ
        volume = np.sum(weight)

        time_weight = np.expand_dims(weight, axis=0)

        uB_vals = []
        while grid_plotter.files_remain([], [k,]):
            bases, tasks, write_num, sim_time = grid_plotter.read_next_file()
            uB = tasks['uB(r=0.5)']
            uB_mag = np.sqrt(np.sum(uB**2, axis=1))
            uB_val = np.sum(np.sum(uB_mag*time_weight, axis=2), axis=1).squeeze()/volume
            for v in uB_val:
                uB_vals.append(v)

        avg_u_ball = np.mean(uB_vals) 
        avg_u_ball_perday1 = avg_u_ball / tau
        avg_u_ball_perday2 = u_ball_avg / tau

            
        plt.figure()
        KE_v_f = np.sum(np.sum(powspec.squeeze(), axis=2), axis=1)
#        plt.loglog(freqs[freqs > 0], (KE_v_f)[freqs > 0])
        plt.loglog(freqs[freqs > 0], (freqs*KE_v_f)[freqs > 0])
#        plt.loglog(freqs[freqs > 0], freqs[freqs > 0]**(-5/3)/1.2e5, c='k')
        plt.axvline(avg_u_ball_perday1, c='k')
        plt.axvline(avg_u_ball_perday2, c='grey')
#        plt.ylabel(r'$\frac{\partial (KE)}{\partial f}$ (cz)')
        plt.ylabel(r'$f\,\, \frac{\partial (KE)}{\partial f}$ (cz)')
        plt.xlabel(r'Frequency (day$^{-1}$)')
        plt.ylim(1e-6, 1e-4)
        plt.savefig('{}/fke_spec.png'.format(full_out_dir), dpi=600)

        plt.figure()
        df = np.gradient(freqs)[:,None]
        KE_v_ell = np.sum(np.sum(powspec.squeeze(), axis=1), axis=0)
        plt.loglog(ells.flatten()[ells.flatten() > 0], (ells.flatten()*KE_v_ell)[ells.flatten() > 0])
        plt.loglog(ells.flatten()[ells.flatten() > 0], ells.flatten()[ells.flatten() > 0]**(-2/3)/3, c='k')
        plt.ylabel(r'$\ell\,\, \frac{\partial (KE)}{\partial ell}$ (cz)')
        plt.xlabel(r'$\ell$')
        plt.ylim(1e-3, 1e0)
        plt.savefig('{}/ellke_spec.png'.format(full_out_dir), dpi=600)
        

        

